import os
from sentence_transformers import SentenceTransformer

# Force Hugging Face to save the model inside the project directory
os.environ['HF_HOME'] = './hf_cache'

print("Downloading model to local folder...")
SentenceTransformer('all-MiniLM-L6-v2')
print("Model cached locally successfully!")
