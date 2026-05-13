from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

try:
    from semantic_rag import generate_response
    # Add this import here!
    from retriever import initialize_retriever 
    print("semantic_rag imported")
except Exception as e:
    print("IMPORT ERROR:", e)
    raise e

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI()

# =========================================================
# STARTUP EVENT (The Fix!)
# =========================================================
@app.on_event("startup")
def startup_event():
    print("Waking up AI models... This might take a couple of minutes on the Free Tier.")
    initialize_retriever()
    print("Models successfully loaded into memory. Ready for traffic!")
    
# =========================================================
# REQUEST MODEL
# =========================================================
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():

    return {
        "message": "SHL Recommendation API running"
    }

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

# =========================================================
# CHAT
# =========================================================
@app.post("/chat")
def chat(req: ChatRequest):

    messages = [m.dict() for m in req.messages]

    response = generate_response(messages)

    return response
