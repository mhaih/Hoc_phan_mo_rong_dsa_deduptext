
# Project: Deduplication with Embeddings

## Overview
This project implements text deduplication on the Quora Question Pairs (QQP) dataset. Our approach is to group similar questions together to make clusters. It compares some clustering approach:
1. **Ground Truth (GT)**: Based on manual duplicate labels from the dataset
2. **Predicted using FAISS**: Using FAISS (Facebook AI Similarity Search) with cosine similarity
3. **SimHash**
4. **MinHash**
## Links
Link to the github repo: https://github.com/mhaih/Hoc_phan_mo_rong_dsa_deduptext

Link to the github page: https://mhaih.github.io/Hoc_phan_mo_rong_dsa_deduptext
## 📁 Project Structure

```
project/
│
├── gen_csv_file_and_eval.ipynb                # Main clustering pipeline + evaluation (faiss + hashing methods)
├── hpmrt.ipynb                                # code implement faiss (seperated)
├── personal.ipynb                             # code implementing hashing methods (seperated)
├──  web_demo/
│   └──app.py                                  # web using for demo (gen csv files from docx)
└── web_for_checking_clustering/               # Streamlit web application
   ├── app.py                                  # Python script used for web application
    ├── qqp_clusters_grouped.csv               # Ground truth clusters
    ├── qqp_pred_clusters_faiss_grouped.csv    # Predicted clusters (FAISS) (used for this app but also is the resulted files generated from gen_csv_file_and_eval.ipynb)
    ├── results_minhash.csv                    # Predicted clusters (MinHash) (used for this app but also is the resulted files generated from gen_csv_file_and_eval.ipynb)
    ├── results_simhash.csv                    # Predicted clusters (SimHash) (used for this app but also is the resulted files generated from gen_csv_file_and_eval.ipynb)
    └── requirements.txt                       # Python dependencies

```
We cant include file X_float32.npy here because the file itself is pretty large (700 MB)
## 📊 Data Files Explained

### 1. `X_float32.npy`
**What it is:** Pre-computed sentence embeddings for all questions in the dataset.

**Format:** NumPy array of shape `(N, 384)` where:
- `N` = number of questions (e.g., 493874 questions)
- `384` = embedding dimension (from `all-MiniLM-L6-v2` model)

**Purpose:** 
- Stores vector representations of each question
- Each row `i` corresponds to the embedding of question with `original_id = i`
- Used for fast similarity search without re-embedding

**How it's created:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(questions, normalize_embeddings=True)
np.save("X_float32.npy", embeddings.astype(np.float32))
```

**Usage:**
```python
X = np.load("X_float32.npy")
question_42_embedding = X[42]  # Get embedding for original_id=42
```

---

### 2. `qqp_clusters_grouped.csv`
**What it is:** Ground truth clusters based on manual duplicate labels from Quora Question Pairs dataset.

**Columns:**
- `original_id` (int): Original index in the dataset (0 to N-1), maps to row in `X_float32.npy`
- `text` (str): The actual question text
- `cluster_id` (int): Ground truth cluster ID
- `cluster_rank` (int): Cluster rank by size (0 = largest cluster)

**Format:**
```csv
original_id,text,cluster_id,cluster_rank
91,How can I lose weight?,91,0
1663,What are the best ways to shed pounds?,91,0
334586,What is the fastest way to lose fat?,91,0
```

**Purpose:**
- Represents the "correct" clustering based on human labels
- Used as baseline for evaluating predicted clustering
- Each `cluster_id` groups questions that are duplicates/paraphrases

**Key characteristics:**
- Sorted by `cluster_rank` so similar questions appear together
- `cluster_rank=0` means this is the largest cluster
- Multiple questions can have the same `cluster_id` (they're in the same cluster)

---

### 3. `qqp_pred_clusters_faiss_grouped.csv`
**What it is:** Predicted clusters generated using FAISS approximate nearest neighbor search.

**Columns:**
- `original_id` (int): Original index in dataset, maps to row in `X_float32.npy`
- `text` (str): The actual question text
- `pred_cluster_id` (int): Predicted cluster ID
- `pred_cluster_rank` (int): Cluster rank by size (0 = largest)
- `mean_semantic_remain` (int): 1 if this is the cluster representative, 0 otherwise

**Format:**
```csv
original_id,text,pred_cluster_id,pred_cluster_rank,mean_semantic_remain
284576,What are some good ways to lose weight?,743,0,1
748,What are the best diets for weight loss?,743,0,0
2234,What are healthy foods to eat?,743,0,0
```

**Purpose:**
- Shows results of automated clustering algorithm
- `mean_semantic_remain=1` marks the most representative sentence (closest to cluster centroid)
- Used to compare automated vs manual clustering

**Key characteristics:**
- Sorted by `pred_cluster_rank` so predicted clusters appear together
- Each cluster has exactly ONE sentence with `mean_semantic_remain=1`
- Representative sentence is the one closest to the cluster's semantic mean (centroid)

---

### 4. `results_minhash.csv`
**What it is:** Predicted clusters generated using the **MinHash** algorithm (Lexical Deduplication).

**Columns:**
* `id` (int): Original index in the dataset (0 to N-1).
* `text` (str): The actual question text.
* `cluster_id` (int): Predicted cluster ID based on Jaccard similarity of text shingles.
* `cluster_rank` (int): Cluster rank by size.
* `is_representative` (int): 1 if this is the centroid of the cluster, 0 otherwise.

**Purpose:**
* Groups questions based on **word overlap** and surface-level textual similarity.
* Excellent for catching near-duplicates with slight spelling variations or reordering.
* Acts as a "Lexical Search" baseline.

---

### 5. `results_simhash.csv`
**What it is:** Predicted clusters generated using the **SimHash** algorithm (Semantic Hashing).

**Columns:**
* `id` (int): Original index in the dataset.
* `text` (str): The actual question text.
* `cluster_id` (int): Predicted cluster ID based on Hamming distance of binary fingerprints.
* `cluster_rank` (int): Cluster rank by size.
* `is_representative` (int): 1 if this is the centroid, 0 otherwise.

**Purpose:**
* Groups questions based on the **geometry of their embeddings**, compressed into 64-bit binary fingerprints.
* Optimized for speed and memory efficiency compared to raw FAISS search.

---

## 🛠️ How to Use This Codebase

### 1. Generating Data
If you plan to run the code again:
1.  Push the file `gen_csv_file_and_eval.ipynb` to Google Colab.
2.  Click "Run All".
3.  This will generate the CSV files and evaluation metrics.

### 2. Running the Demo (Notebook)
If you want to run the demo code without the web interface, run `query_clusters.ipynb`.

### 3. About the Embeddings (`X_float32.npy`)
This project uses pre-computed embeddings using the **sentence-transformers/all-MiniLM-L6-v2** model.
- Generating this from scratch takes a long time on CPU.
- It takes about 10 mins on a T4 GPU (Google Colab).

**Option A: Download Pre-computed (Recommended)**
```python
import gdown
import numpy as np

url = "[https://drive.google.com/uc?id=177zbL5sW2mUb4n8TVoviw7dnpyOLfhPn](https://drive.google.com/uc?id=177zbL5sW2mUb4n8TVoviw7dnpyOLfhPn)"
output = "X_float32.npy"
gdown.download(url, output, quiet=False)

X = np.load("X_float32.npy")
```

**Option B: Recompute**
```python
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBED_MODEL)
texts = list(questions)
X = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
X = X.astype("float32")
```

---
## 🚀 How to Run the Web Application

This project includes a **Streamlit** dashboard that allows you to interactively search and visualize the clusters using all four methods.

### Prerequisites
1.  Ensure you have Python installed (3.8+).
2.  Ensure you have the `X_float32.npy` file placed inside the `web/` folder.

### Installation & Execution

1.  **Navigate to the web directory:**
    ```bash
    cd project/web
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

4.  **Access the Dashboard:**
    The application will automatically open at `http://localhost:8501`.
---
## About the Data
---
This project uses a pre-computed embedding file: **`X_float32.npy`** which is the embeding using **sentence-transformers/all-MiniLM-L6-v2** model on the Quora Question Pairs dataset

- Generating this file from scratch takes a **long time** (if using cpu) and requires a **GPU** (actually takes only about 10 mins on GPU T4 of GG Colab free trial) to process efficiently.
- To save time and avoid recomputing, we provide the pre-computed embeddings here.
- This way, anyone who wants to run the project can **directly load the embeddings** instead of re-running the entire model.

## Usage Notes

- If you want to use the project **without recomputing embeddings**, simply load the file:

  ```python
  import gdown
  url = "https://drive.google.com/uc?id=177zbL5sW2mUb4n8TVoviw7dnpyOLfhPn"
  output = "X_float32.npy"
  gdown.download(url, output, quiet=False)

  import numpy as np
  X = np.load("X_float32.npy")
  print(X.shape)
  #...
- Or if you want to recomputing from scratch then just code something like this:
  ```python
  import numpy as np
  from sentence_transformers import SentenceTransformer


  EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
  model = SentenceTransformer(EMBED_MODEL)
  texts = list(questions)
  X = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
  X = X.astype("float32")
  #...
