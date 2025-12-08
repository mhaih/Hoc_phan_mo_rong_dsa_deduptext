import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
import mmh3
import re
import hashlib
from collections import defaultdict
import time

# ==== PAGE CONFIG ====
st.set_page_config(
    page_title="Question Cluster Explorer",
    page_icon="🔍",
    layout="wide"
)

# ==== CONFIG ====
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "X_float32.npy"
# CSV Filenames
FAISS_CSV = "qqp_pred_clusters_faiss_grouped.csv"
GT_CSV = "qqp_clusters_grouped.csv"
MINHASH_CSV = "results_minhash.csv"
SIMHASH_CSV = "results_simhash.csv"

# Algorithm Parameters
MINHASH_PERM = 64
MINHASH_BANDS = 8
SIMHASH_BITS = 64
SIMHASH_PREFIX = 16
BLOOM_FP_RATE = 0.01

# ==== ALGORITHM CLASSES ====

class BloomFilter:
    def __init__(self, n_items: int, fp_rate: float = 0.01):
        self.size = int(-(n_items * np.log(fp_rate)) / (np.log(2) ** 2))
        self.hash_count = int((self.size / n_items) * np.log(2))
        self.bit_array = np.zeros(self.size, dtype=bool)

    def _get_hash_indices(self, item: str):
        h1 = mmh3.hash(item, seed=42)
        h2 = mmh3.hash(item, seed=101)
        indices = []
        for i in range(self.hash_count):
            idx = (h1 + i * h2) % self.size
            indices.append(idx)
        return indices

    def add(self, item: str):
        for idx in self._get_hash_indices(item):
            self.bit_array[idx] = True

    def check(self, item: str) -> bool:
        return all(self.bit_array[idx] for idx in self._get_hash_indices(item))

def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

def get_shingles(text, k=5):
    text = clean_text(text)
    text = text.replace(" ", "")
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(max(1, len(text) - k + 1))}

def create_minhash_sig(shingles, A, B, max_hash):
    if not shingles:
        return np.zeros(len(A), dtype=np.uint32)
    hashes = np.array([hash(s) % max_hash for s in shingles], dtype=np.uint32)
    sigs = np.min((A[:, None] * hashes + B[:, None]) % max_hash, axis=1)
    return sigs

def hamming_distance(a, b):
    return np.count_nonzero(a ^ b)

# ==== CACHE LOADING FUNCTIONS ====

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_data(show_spinner=False)
def load_data():
    """
    Load Embeddings and ALL CSVs (FAISS, GT, MinHash, SimHash).
    Strictly renames columns to standard format: 
    [original_id, text, pred_cluster_id, pred_cluster_rank, is_representative]
    """
    X = np.load(EMBEDDINGS_PATH)
    
    # 1. Load Ground Truth (Keep as is, it uses 'cluster_id' not 'pred_cluster_id')
    df_gt = pd.read_csv(GT_CSV)
    
    # 2. Load FAISS Results & Normalize Columns
    df_faiss = pd.read_csv(FAISS_CSV)
    # Rename specifically for FAISS schema
    df_faiss = df_faiss.rename(columns={
        'id': 'original_id',  # If present
        'mean_semantic_remain': 'is_representative' # Normalize this
    })
    # Ensure all required columns exist
    required_cols = ['original_id', 'text', 'pred_cluster_id', 'pred_cluster_rank', 'is_representative']
    for col in required_cols:
        if col not in df_faiss.columns and col == 'original_id':
             # If original_id missing in CSV, assume it matches index if aligned
             df_faiss['original_id'] = df_faiss.index
    
    # 3. Load MinHash/SimHash Results (Handle missing files gracefully)
    try:
        df_mh = pd.read_csv(MINHASH_CSV)
        df_mh = df_mh.rename(columns={
            'id': 'original_id', 
            'cluster_id': 'pred_cluster_id', 
            'cluster_rank': 'pred_cluster_rank'
        })
    except FileNotFoundError:
        st.error(f"Missing {MINHASH_CSV}. Please run personal.py to generate it.")
        df_mh = pd.DataFrame(columns=required_cols)

    try:
        df_sh = pd.read_csv(SIMHASH_CSV)
        df_sh = df_sh.rename(columns={
            'id': 'original_id', 
            'cluster_id': 'pred_cluster_id', 
            'cluster_rank': 'pred_cluster_rank'
        })
    except FileNotFoundError:
        st.error(f"Missing {SIMHASH_CSV}. Please run personal.py to generate it.")
        df_sh = pd.DataFrame(columns=required_cols)

    return X, df_gt, df_faiss, df_mh, df_sh

@st.cache_resource(show_spinner=False)
def build_indexes(texts, X):
    """Builds search indexes live."""
    # MinHash Params
    max_hash = (1 << 32) - 1
    np.random.seed(42)
    A = np.random.randint(1, max_hash, size=MINHASH_PERM, dtype=np.uint32)
    B = np.random.randint(0, max_hash, size=MINHASH_PERM, dtype=np.uint32)
    
    mh_buckets = defaultdict(list)
    mh_signatures = np.zeros((len(texts), MINHASH_PERM), dtype=np.uint32)
    rows_per_band = MINHASH_PERM // MINHASH_BANDS
    
    # SimHash Params
    rng = np.random.RandomState(42)
    rand_proj = rng.randn(X.shape[1], SIMHASH_BITS)
    proj = np.dot(X, rand_proj) > 0
    fingerprints = proj.astype(np.uint8)
    sh_buckets = defaultdict(list)
    
    # Bloom Params
    bf = BloomFilter(n_items=len(texts), fp_rate=BLOOM_FP_RATE)
    
    # Build Loop
    for i, text in enumerate(texts):
        # MinHash
        shingles = get_shingles(text)
        sig = create_minhash_sig(shingles, A, B, max_hash)
        mh_signatures[i] = sig
        
        for b in range(MINHASH_BANDS):
            start, end = b * rows_per_band, (b + 1) * rows_per_band
            band_hash = hashlib.sha1(sig[start:end].tobytes()).hexdigest()
            mh_buckets[(b, band_hash)].append(i)
            
        # SimHash
        fp_int = int("".join(map(str, fingerprints[i].tolist())), 2)
        sh_bucket_id = fp_int >> (SIMHASH_BITS - SIMHASH_PREFIX)
        sh_buckets[sh_bucket_id].append(i)
        
        # Bloom
        bf.add(clean_text(text))
            
    return {
        'minhash': {'buckets': mh_buckets, 'signatures': mh_signatures, 'A': A, 'B': B, 'max_hash': max_hash, 'rows_per_band': rows_per_band},
        'simhash': {'buckets': sh_buckets, 'fingerprints': fingerprints, 'rand_proj': rand_proj},
        'bloom': bf
    }

# ==== LOAD DATA & BUILD INDEXES ====
with st.spinner("Loading System..."):
    model = load_model()
    X, df_gt, df_faiss, df_mh, df_sh = load_data()
    
    # Align text with Embeddings (Sort by original_id)
    # We use df_faiss as the 'Master' text list since it aligns with X
    df_sorted = df_faiss.sort_values('original_id').reset_index(drop=True)
    all_texts = df_sorted['text'].tolist()
    
    # Build Indexes
    indexes = build_indexes(all_texts, X)
    minhash_idx = indexes['minhash']
    simhash_idx = indexes['simhash']
    bloom_idx = indexes['bloom']

# ==== SEARCH FUNCTIONS ====

def search_faiss_logic(query_text, top_k=5):
    """Semantic Search using FAISS CSV Clusters"""
    query_vec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
    
    # Predicted Clusters (FAISS) - Updated to use 'is_representative'
    reps_pred = df_sorted[df_sorted['is_representative'] == 1].copy()
    sims_pred = X[reps_pred['original_id'].values] @ query_vec
    top_idxs_pred = np.argsort(sims_pred)[::-1][:top_k]
    
    top_clusters_pred = []
    for idx in top_idxs_pred:
        row = reps_pred.iloc[idx]
        top_clusters_pred.append({
            'text': row['text'],
            'similarity': sims_pred[idx],
            'cluster_id': row['pred_cluster_id']
        })
    
    # Ground Truth Clusters
    reps_gt = df_gt.groupby('cluster_id').first().reset_index()
    # Handle cases where GT ID > len(X) due to subsets
    valid_mask = reps_gt['original_id'].values < len(X)
    reps_gt = reps_gt[valid_mask]
    
    sims_gt = X[reps_gt['original_id'].values] @ query_vec
    top_idxs_gt = np.argsort(sims_gt)[::-1][:top_k]
    
    top_clusters_gt = []
    for idx in top_idxs_gt:
        row = reps_gt.iloc[idx]
        top_clusters_gt.append({
            'text': row['text'],
            'similarity': sims_gt[idx],
            'cluster_id': row['cluster_id']
        })

    return {
        'pred': {'top': top_clusters_pred, 'best': top_clusters_pred[0]},
        'gt': {'top': top_clusters_gt, 'best': top_clusters_gt[0]}
    }

def search_minhash(query_text, top_k=5):
    """MinHash Search using MinHash CSV Clusters"""
    shingles = get_shingles(query_text)
    query_sig = create_minhash_sig(shingles, minhash_idx['A'], minhash_idx['B'], minhash_idx['max_hash'])
    
    candidates = set()
    for b in range(MINHASH_BANDS):
        band_hash = hashlib.sha1(query_sig[b*minhash_idx['rows_per_band']:(b+1)*minhash_idx['rows_per_band']].tobytes()).hexdigest()
        candidates.update(minhash_idx['buckets'][(b, band_hash)])
        
    results = []
    for idx in candidates:
        cand_sig = minhash_idx['signatures'][idx]
        score = np.mean(query_sig == cand_sig)
        
        # Look up Cluster ID from MinHash CSV
        cluster_id = -1
        try:
            # Finding the row in df_mh where original_id matches the candidate idx
            cluster_id = df_mh.loc[df_mh['original_id'] == idx, 'pred_cluster_id'].values[0]
        except:
            pass
            
        results.append({'text': all_texts[idx], 'score': score, 'cluster_id': cluster_id, 'id': idx})
            
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def search_simhash(query_vec, top_k=5):
    """SimHash Search using SimHash CSV Clusters"""
    proj = np.dot(query_vec, simhash_idx['rand_proj']) > 0
    query_fp = proj.astype(np.uint8)
    
    fp_int = int("".join(map(str, query_fp.tolist())), 2)
    bucket_id = fp_int >> (SIMHASH_BITS - SIMHASH_PREFIX)
    candidates = simhash_idx['buckets'][bucket_id]
    
    results = []
    for idx in candidates:
        cand_fp = simhash_idx['fingerprints'][idx]
        dist = hamming_distance(query_fp, cand_fp)
        score = 1 - (dist / SIMHASH_BITS)
        
        # Look up Cluster ID from SimHash CSV
        cluster_id = -1
        try:
            cluster_id = df_sh.loc[df_sh['original_id'] == idx, 'pred_cluster_id'].values[0]
        except:
            pass

        results.append({'text': all_texts[idx], 'score': score, 'dist': dist, 'cluster_id': cluster_id, 'id': idx})
        
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# ==== UI ====
st.title("🔍 Question Cluster Explorer")
st.markdown("Find similar question clusters using semantic search on the Quora Question Pairs dataset")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of top similar clusters to show", 1, 10, 5)
    
    st.markdown("---")
    st.header("📊 Dataset Info")
    st.metric("Total Questions", len(df_faiss))
    st.metric("GT Clusters", df_gt['cluster_id'].nunique())
    st.metric("Predicted Clusters", df_faiss['pred_cluster_id'].nunique())
    st.markdown("---")
    st.header("🗂️ Index Stats")
    st.text(f"MinHash Buckets: {len(minhash_idx['buckets'])}")
    st.text(f"SimHash Buckets: {len(simhash_idx['buckets'])}")
    
    st.markdown("---")
    st.markdown("### How it works\n1. Enter query\n2. Bloom Filter check\n3. Search via FAISS, MinHash, SimHash")

# Search Bar
st.header("🔎 Search Query")
col1, col2 = st.columns([3, 1])
with col1:
    query_input = st.text_input("Enter your question:", placeholder="Type here...", label_visibility="collapsed", value=st.session_state.get('query', ""))
with col2:
    if st.button("🔍 Search", type="primary", use_container_width=True):
        st.session_state.query = query_input

# Examples
example_queries = ["How do I lose weight?", "What are good programming languages?", "How can I make money online?", "Best way to learn guitar?", "How to improve memory?"]
cols = st.columns(len(example_queries))
for i, ex in enumerate(example_queries):
    with cols[i]:
        if st.button(f"💡 {ex[:15]}...", help=ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.query = ex
            st.rerun()

if 'query' in st.session_state and st.session_state.query:
    query = st.session_state.query
    st.markdown("---")
    st.subheader(f"Results for: *\"{query}\"*")
    
    # Bloom Check
    if bloom_idx.check(clean_text(query)):
        st.warning(f"🔒 **Bloom Filter:** Exact string likely exists in database.")
    else:
        st.success(f"✅ **Bloom Filter:** Unique string.")

    # Tabs
    tab_faiss, tab_mh, tab_sh = st.tabs(["🧠 Semantic Cluster", "📝 MinHash (Lexical)", "🔢 SimHash (Semantic)"])

    # --- TAB 1: FAISS (SEMANTIC) ---
    with tab_faiss:
        with st.spinner("Searching clusters..."):
            res = search_faiss_logic(query, top_k)
        
        col_gt, col_pred = st.columns(2)
        
        # Predicted
        with col_pred:
            st.markdown("### 🤖 Predicted Clusters (FAISS)")
            best = res['pred']['best']
            st.success(f"**Best Match:** Cluster {best['cluster_id']} (Sim: {best['similarity']:.4f})")
            
            st.markdown(f"#### 📋 Top {top_k} Similar Clusters")
            for rep in res['pred']['top']:
                cid = rep['cluster_id']
                with st.expander(f"Cluster {cid} (Sim: {rep['similarity']:.4f}) - {rep['text']}"):
                    # Fetch from FAISS CSV
                    mems = df_faiss[df_faiss['pred_cluster_id'] == cid].sort_values('original_id')
                    for _, r in mems.iterrows():
                        marker = "⭐" if r['is_representative'] == 1 else "▪️"
                        st.markdown(f"{marker} `[{r['original_id']}]` {r['text']}")

        # Ground Truth
        with col_gt:
            st.markdown("### ✅ Ground Truth Clusters")
            best_gt = res['gt']['best']
            st.success(f"**Best Match:** Cluster {best_gt['cluster_id']} (Sim: {best_gt['similarity']:.4f})")
            
            st.markdown(f"#### 📋 Top {top_k} Similar Clusters")
            for rep in res['gt']['top']:
                cid = rep['cluster_id']
                with st.expander(f"Cluster {cid} (Sim: {rep['similarity']:.4f}) - {rep['text']}"):
                    # Fetch from GT CSV
                    mems = df_gt[df_gt['cluster_id'] == cid].sort_values('original_id')
                    for _, r in mems.iterrows():
                        st.markdown(f"▪️ `[{r['original_id']}]` {r['text']}")

    # --- TAB 2: MINHASH ---
    with tab_mh:
        st.markdown("### 📝 MinHash LSH Results")
        st.caption(f"Top {top_k} lexical matches. Clusters fetched from `{MINHASH_CSV}`.")
        
        mh_results = search_minhash(query, top_k=top_k)
        if mh_results:
            for i, r in enumerate(mh_results, 1):
                color = "green" if r['score'] > 0.8 else "orange"
                st.markdown(f"**{i}. Match {r['score']*100:.1f}%** :{color}[●] — {r['text']}")
                
                # Expand Cluster from MinHash CSV
                cid = r['cluster_id']
                if not pd.isna(cid) and cid != -1 and not df_mh.empty:
                    with st.expander(f"See MinHash Cluster {int(cid)} Members"):
                        mems = df_mh[df_mh['pred_cluster_id'] == cid].sort_values('original_id')
                        for _, row in mems.iterrows():
                             marker = "⭐" if row['is_representative'] == 1 else "▪️"
                             st.markdown(f"{marker} `[{row['original_id']}]` {row['text']}")
        else:
            st.warning("No matches in LSH buckets.")

    # --- TAB 3: SIMHASH ---
    with tab_sh:
        st.markdown("### 🔢 SimHash LSH Results")
        st.caption(f"Top {top_k} semantic hash matches. Clusters fetched from `{SIMHASH_CSV}`.")
        
        query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sh_results = search_simhash(query_vec, top_k=top_k)
        
        if sh_results:
            for i, r in enumerate(sh_results, 1):
                color = "green" if r['dist'] <= 3 else "orange"
                st.markdown(f"**{i}. Dist {r['dist']}** (Score {r['score']:.2f}) :{color}[●] — {r['text']}")
                
                # Expand Cluster from SimHash CSV
                cid = r['cluster_id']
                if not pd.isna(cid) and cid != -1 and not df_sh.empty:
                    with st.expander(f"See SimHash Cluster {int(cid)} Members"):
                        mems = df_sh[df_sh['pred_cluster_id'] == cid].sort_values('original_id')
                        for _, row in mems.iterrows():
                             marker = "⭐" if row['is_representative'] == 1 else "▪️"
                             st.markdown(f"{marker} `[{row['original_id']}]` {row['text']}")
        else:
            st.warning("No matches in LSH buckets.")
