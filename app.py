from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

try:
    from semantic_rag import generate_response 
    print("semantic_rag imported")
except Exception as e:
    print("IMPORT ERROR:", e)
    raise e

import threading

# =========================================================
# FASTAPI
# =========================================================
app = FastAPI()

# =========================================================
# STARTUP EVENT
# =========================================================
@app.on_event("startup")
def startup_event():
    print("Opening port immediately for Render...")
    from retriever import initialize_retriever
    
    # Spin off the heavy 3-minute load into a background thread
    load_thread = threading.Thread(target=initialize_retriever)
    load_thread.start()
    
    print("Background load started! Server is ready to accept health checks.")
    
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
