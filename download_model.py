from sentence_transformers import SentenceTransformer

# This downloads and caches the model during the build phase
print("Downloading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model downloaded successfully!")
