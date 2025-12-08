# app.py
import io
import math
import re
import hashlib
import random
from collections import defaultdict, deque, Counter

import numpy as np
import pandas as pd
import streamlit as st

from docx import Document
from sentence_transformers import SentenceTransformer

import faiss


# =============================
# 1. Config & model loading
# =============================

st.set_page_config(page_title="Text Dedup & Clustering (FAISS / MinHash / SimHash)",
                   layout="wide")

@st.cache_resource
def load_st_model():
    """
    In the original notebook, embeddings were loaded from a precomputed .npy.
    The exact model wasn't encoded, so here I choose a very common one:
    'sentence-transformers/all-MiniLM-L6-v2'.
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    model = load_st_model()
    # normalize_embeddings=True so that dot product = cosine similarity
    X = model.encode(texts, batch_size=64,
                     show_progress_bar=True,
                     normalize_embeddings=True)
    return X.astype("float32")


# =============================
# 2. Helpers: docx & text
# =============================

def load_docx_paragraphs(file) -> list[str]:
    """
    Read a .docx file and return non-empty paragraphs as a list of strings.
    Matches the structure of data_demo.docx:
    - each paragraph is one sentence.
    """
    doc = Document(file)
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return texts

def normalize_and_dedup(texts: list[str]) -> list[str]:
    return list(dict.fromkeys(texts))


# =============================
# 3. Graph + clustering helpers
# =============================

def connected_components(adj):
    """
    Standard BFS connected components on an adjacency list.
    Returns:
      labels: np.array of component IDs per node
      n_comp: number of components
    """
    comp_id = [-1] * len(adj)
    cid = 0
    for s in range(len(adj)):
        if comp_id[s] != -1:
            continue
        dq = deque([s])
        comp_id[s] = cid
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if comp_id[v] == -1:
                    comp_id[v] = cid
                    dq.append(v)
        cid += 1
    return np.array(comp_id), cid

def duplicates_to_labels(n_items: int, duplicates: list[tuple[int, int]]) -> np.ndarray:
    """
    Convert a list of duplicate pairs (edges) into cluster labels using
    connected components. If duplicates is empty, each point is its own
    singleton cluster.
    """
    adj = [[] for _ in range(n_items)]
    for a, b in duplicates:
        if a == b:
            continue
        adj[a].append(b)
        adj[b].append(a)
    labels, _ = connected_components(adj)
    return labels


# =============================
# 4. CSV writer (adapted from notebook)
# =============================

def save_clustering_results(questions, labels, embeddings, filename: str) -> pd.DataFrame:
    """
    Equivalent to the notebook's save_clustering_results:
    - original_id: position in questions list
    - text: sentence text
    - pred_cluster_id: component ID
    - pred_cluster_rank: rank by cluster size (0 = largest)
    - is_representative: 1 if central sentence of cluster (cosine centroid)
    """
    st.write(f"Generating CSV: `{filename}` ...")

    n_samples = len(questions)
    labels = np.asarray(labels)

    # 1. Cluster sizes & ranks (ignore label == -1 if used, but we never use -1 here)
    valid_indices = [i for i, label in enumerate(labels) if label != -1]
    valid_labels = labels[valid_indices]

    cluster_sizes = Counter(valid_labels)
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
    rank_map = {cid: rank for rank, (cid, _) in enumerate(sorted_clusters)}

    df = pd.DataFrame({
        "original_id": range(n_samples),
        "text": questions,
        "pred_cluster_id": labels,
    })
    df["pred_cluster_rank"] = df["pred_cluster_id"].map(lambda x: rank_map.get(x, -1))

    # 2. Representative per cluster (semantic centroid)
    is_rep = np.zeros(n_samples, dtype=int)
    X = embeddings

    for cid, _size in sorted_clusters:
        indices = np.where(labels == cid)[0]
        if len(indices) == 0:
            continue
        if len(indices) == 1:
            is_rep[indices[0]] = 1
            continue

        cluster_vecs = X[indices]
        centroid = cluster_vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        sims = np.dot(cluster_vecs, centroid)
        best_local_idx = np.argmax(sims)
        best_global_idx = indices[best_local_idx]
        is_rep[best_global_idx] = 1

    df["is_representative"] = is_rep

    # 3. Sort & return (largest clusters first, reps at top)
    df = df.sort_values(
        by=["pred_cluster_rank", "is_representative"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df


# =============================
# 5. MinHash (from notebook)
# =============================

def clean(text):
    # Lowercase, remove punctuation, remove extra whitespace
    return re.sub(r"[^\w\s]", "", text.lower()).strip()

def get_shingles(text, k=5):
    text = clean(text)
    text = text.lower().replace(" ", "")
    return {text[i:i + k] for i in range(max(1, len(text) - k + 1))}

def create_minhash(shingles, A, B, max_hash):
    if not shingles:
        return np.zeros_like(A)
    hashes = np.array([hash(s) % max_hash for s in shingles], dtype=np.uint32)
    sigs = np.min((A[:, None] * hashes + B[:, None]) % max_hash, axis=1)
    return sigs

def deduplicate_minhash(
    texts,
    num_perm=64,        # same as notebook
    threshold=0.6,      # same as notebook
    bands=8             # same as notebook
):
    import time
    from tqdm import tqdm

    t0 = time.time()
    max_hash = (1 << 32) - 1

    rng = np.random.RandomState(42)
    A = rng.randint(1, max_hash, size=num_perm, dtype=np.uint32)
    B = rng.randint(0, max_hash, size=num_perm, dtype=np.uint32)

    # 1. Build signatures
    signatures = np.zeros((len(texts), num_perm), dtype=np.uint32)
    for i, text in enumerate(tqdm(texts, desc="Building MinHash", leave=False)):
        shingles = get_shingles(text, k=5)  # same k-shingle = 5
        signatures[i] = create_minhash(shingles, A, B, max_hash)

    # 2. LSH bucketing
    rows_per_band = num_perm // bands
    buckets = defaultdict(list)
    for i in tqdm(range(len(texts)), desc="LSH Bucketing", leave=False):
        sig = signatures[i]
        for b in range(bands):
            start, end = b * rows_per_band, (b + 1) * rows_per_band
            band_hash = hashlib.sha1(sig[start:end].tobytes()).hexdigest()
            buckets[(b, band_hash)].append(i)

    # 3. Candidate & duplicate pairs
    candidates = set()
    for group in tqdm(buckets.values(), desc="Bucket comparisons", leave=False):
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                sim = np.mean(signatures[a] == signatures[b])
                if sim >= threshold:
                    candidates.add((min(a, b), max(a, b)))

    duplicates = list(candidates)
    elapsed = time.time() - t0
    print(f"[MinHash] Completed. Total time: {elapsed:.2f} seconds")
    return duplicates


# =============================
# 6. SimHash (from notebook)
# =============================

def hamming_distance(a, b):
    return np.count_nonzero(a ^ b)

def simhash_matrix(X: np.ndarray, nbits=64, seed=42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    rand_proj = rng.randn(X.shape[1], nbits)
    proj = np.dot(X, rand_proj) > 0
    return proj.astype(np.uint8)

def deduplicate_simhash(
    X: np.ndarray,
    nbits=64,           # same as notebook
    threshold=6,        # same as notebook (max Hamming distance)
    prefix=16           # same as notebook (bucket prefix length)
):
    import time
    from tqdm import tqdm

    t0 = time.time()
    fingerprints = simhash_matrix(X, nbits=nbits)
    N = fingerprints.shape[0]

    # Convert bit-vectors to ints for bucketing
    ints = [int("".join(map(str, row.tolist())), 2) for row in fingerprints]

    buckets = defaultdict(list)
    for i, val in tqdm(enumerate(ints), total=N, desc="Bucketing", leave=False):
        bucket_id = val >> (nbits - prefix)
        buckets[bucket_id].append(i)

    duplicates = []
    for group in tqdm(buckets.values(), desc="Comparing", leave=False):
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                dist = hamming_distance(fingerprints[group[i]], fingerprints[group[j]])
                if dist <= threshold:
                    duplicates.append((group[i], group[j]))

    elapsed = time.time() - t0
    print(f"[SimHash] Completed. Total time: {elapsed:.2f} seconds")
    return duplicates


# =============================
# 7. FAISS (IVF, same hyperparams)
# =============================

def clusters_faiss_ivf(
    X: np.ndarray,
    tau=0.9,           # TAU = 0.9 in notebook
    k_neighbors=10     # K_NEIGHBORS = 10 in notebook
) -> np.ndarray:
    """
    Use FAISS IVF-Flat with cosine similarity threshold TAU
    to build a graph of "duplicate" edges, then connected components.
    Hyperparams clone the notebook:
      - nlist = max(10, int(sqrt(N)))
      - train_sz = min(20000, N)
      - nprobe = 16
      - K_NEIGHBORS = 10
      - TAU = 0.9
    """
    import time

    N, d = X.shape
    if N == 0:
        return np.array([])

    start_time = time.time()

    # nlist ~ sqrt(N)
    nlist = max(10, int(math.sqrt(N)))
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train IVF on a sample
    train_sz = min(20000, N)
    train_ix = np.random.choice(N, size=train_sz, replace=False)
    index.train(X[train_ix])

    index.add(X)
    index.nprobe = 16

    K = k_neighbors
    D, I = index.search(X, K + 1)  # rank 0 is self

    # Build edge set by threshold on cosine similarity
    edges_pred = set()
    for i in range(N):
        for sim, j in zip(D[i], I[i]):
            if j < 0 or j == i:
                continue
            if sim >= tau:
                a, b = (i, j) if i < j else (j, i)
                edges_pred.add((a, b))

    adj = [[] for _ in range(N)]
    for a, b in edges_pred:
        adj[a].append(b)
        adj[b].append(a)

    labels, n_comps = connected_components(adj)
    total_time = time.time() - start_time
    print(f"[FAISS] Predicted components: {n_comps}, edges: {len(edges_pred)}")
    print(f"[FAISS] TOTAL TIME: {total_time:.2f}s")

    return labels


# =============================
# 8. Streamlit UI
# =============================

st.title("QQP-style Text Clustering: FAISS vs MinHash vs SimHash")

st.markdown(
    """
    - Upload a `.docx` file with one sentence per paragraph  
    - (Optionally) confirm you've translated it to **English** first if you want a pure-English example  
    - The app will compute sentence embeddings and create **three CSV files**:
      - `faiss_clusters.csv`
      - `minhash_clusters.csv`
      - `simhash_clusters.csv`
    """
)

uploaded_file = st.file_uploader("Upload a .docx file", type=["docx"])

use_manual_text = st.checkbox("Or paste raw text (one sentence per line) instead of file")

raw_text_input = ""
if use_manual_text:
    raw_text_input = st.text_area("Paste sentences here (one per line):", height=200)

col_run, col_info = st.columns([1, 2])
run_button = col_run.button("Run clustering")

with col_info:
    st.info(
        "Note: The app itself does **not** call any online translator. "
        "If you want a pure-English example from your Vietnamese `data_demo.docx`, "
        "please translate it first (e.g. with ChatGPT or another tool), "
        "save to a new `.docx`, then upload here."
    )

if run_button:
    # 1. Get texts
    texts: list[str] = []

    if use_manual_text and raw_text_input.strip():
        texts = [line.strip() for line in raw_text_input.splitlines() if line.strip()]
    elif uploaded_file is not None:
        texts = load_docx_paragraphs(uploaded_file)
    else:
        st.error("Please upload a .docx file or paste sentences.")
        st.stop()

    st.write(f"Loaded **{len(texts)}** sentences from input.")
    if len(texts) == 0:
        st.error("No non-empty sentences found.")
        st.stop()

    # 2. Optional small preview
    st.subheader("Sample sentences")
    st.write(pd.DataFrame({"text": texts[:10]}))

    # 3. De-duplicate identical texts (as in notebook using set())
    texts = normalize_and_dedup(texts)
    st.write(f"After removing exact duplicates: **{len(texts)}** unique sentences.")

    # 4. Compute embeddings
    st.subheader("Embedding sentences (SentenceTransformer)")
    X = embed_texts(texts)
    st.write(f"Embedding matrix shape: {X.shape}")

    # 5. Run FAISS clustering
    st.subheader("FAISS IVF-Flat clustering")
    labels_faiss = clusters_faiss_ivf(X, tau=0.9, k_neighbors=10)
    df_faiss = save_clustering_results(texts, labels_faiss, X, filename="faiss_clusters.csv")

    # 6. Run MinHash-based clustering
    st.subheader("MinHash-based clustering")
    duplicates_minhash = deduplicate_minhash(texts, num_perm=64, threshold=0.6, bands=8)
    labels_minhash = duplicates_to_labels(len(texts), duplicates_minhash)
    df_minhash = save_clustering_results(texts, labels_minhash, X, filename="minhash_clusters.csv")

    # 7. Run SimHash-based clustering
    st.subheader("SimHash-based clustering")
    duplicates_simhash = deduplicate_simhash(X, nbits=64, threshold=6, prefix=16)
    labels_simhash = duplicates_to_labels(len(texts), duplicates_simhash)
    df_simhash = save_clustering_results(texts, labels_simhash, X, filename="simhash_clusters.csv")

    # 8. Show previews and download buttons
    st.subheader("Download CSVs")

    def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**FAISS clusters**")
        st.dataframe(df_faiss.head(20))
        st.download_button(
            "Download FAISS CSV",
            data=df_to_csv_bytes(df_faiss),
            file_name="faiss_clusters.csv",
            mime="text/csv",
        )

    with c2:
        st.markdown("**MinHash clusters**")
        st.dataframe(df_minhash.head(20))
        st.download_button(
            "Download MinHash CSV",
            data=df_to_csv_bytes(df_minhash),
            file_name="minhash_clusters.csv",
            mime="text/csv",
        )

    with c3:
        st.markdown("**SimHash clusters**")
        st.dataframe(df_simhash.head(20))
        st.download_button(
            "Download SimHash CSV",
            data=df_to_csv_bytes(df_simhash),
            file_name="simhash_clusters.csv",
            mime="text/csv",
        )

    st.success("Done! All three methods have been run and CSVs are ready.")
