import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# =========================================================
# LOAD DATA
# =========================================================
with open(
    "data/normalized_assessments.json",
    "r",
    encoding="utf-8"
) as f:

    DATA = json.load(f)

print(f"Loaded {len(DATA)} assessments.")

# =========================================================
# MODEL
# =========================================================
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)

# =========================================================
# CLEAN
# =========================================================
def clean(text):

    if not text:
        return ""

    return (
        text.replace("\n", " ")
        .replace("Login", "")
        .replace("Careers", "")
        .replace("Contact", "")
        .replace("Practice Tests", "")
        .replace("Buy Online", "")
        .strip()
    )

# =========================================================
# BUILD TEXTS
# =========================================================
texts = []

for item in DATA:

    combined = (
        item.get("name", "")
        + " "
        + clean(item.get("description", ""))[:400]
        + " "
        + item.get("test_type", "")
        + " "
        + " ".join(item.get("keys", []))
    )

    texts.append(combined)

# =========================================================
# CREATE EMBEDDINGS
# =========================================================
print("Creating embeddings...")

embeddings = np.array(
    model.encode(
        texts,
        show_progress_bar=True
    )
).astype("float32")

print("Embeddings created.")

# =========================================================
# CREATE FAISS INDEX
# =========================================================
index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

# =========================================================
# SAVE INDEX
# =========================================================
faiss.write_index(
    index,
    "data/faiss.index"
)

print("FAISS index saved.")

# =========================================================
# SAVE EMBEDDINGS
# =========================================================
np.save(
    "data/embeddings.npy",
    embeddings
)

print("Embeddings saved.")