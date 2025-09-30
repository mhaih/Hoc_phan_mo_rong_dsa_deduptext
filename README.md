Ai giỏi văn với rành viết README thì viết giùm lun i cũng được =)))  

Link file X_float32 đây ha (file này 700MB lận nên không push lên đây được): https://drive.google.com/drive/folders/1YvF88XJ0ctALA6JLRxlnarrYCe-Vroqt?usp=sharing  
Với đây là ycau của thầy về readme ha: <img width="1058" height="278" alt="image" src="https://github.com/user-attachments/assets/b8574c35-6b19-44a7-a977-cf233fb456c4" />  
tui tự viết vài cái để đây th

# Project: Deduplication with Embeddings

## About the Data

This project uses a pre-computed embedding file: **`X_float32.npy`** which is the embeding using **sentence-transformers/all-MiniLM-L6-v2** model on the Quora Question Pairs dataset

- Generating this file from scratch takes a **long time** (if using cpu) and requires a **GPU** to process efficiently.
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
