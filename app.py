from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from semantic_rag import generate_response

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/chat")
def chat(req: ChatRequest):

    messages = [m.dict() for m in req.messages]

    response = generate_response(messages)

    return response