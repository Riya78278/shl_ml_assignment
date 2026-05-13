from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from semantic_rag import generate_response
from retriever import initialize_retriever

app = FastAPI()

# Initialize immediately on startup. Since it's just reading a JSON and FAISS file, it takes 0.1 seconds.
@app.on_event("startup")
def startup_event():
    initialize_retriever()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/")
def root():
    return {"message": "SHL Recommendation API running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [m.dict() for m in req.messages]
    # Await the response since the core logic is now async
    response = await generate_response(messages)
    return response
