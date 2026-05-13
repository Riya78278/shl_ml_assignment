from fastapi import FastAPI
from semantic_rag import initialize_system

app = FastAPI()

# =========================================================
# STARTUP
# =========================================================
@app.on_event("startup")
async def startup_event():

    print("Initializing AI system...")

    initialize_system()

    print("AI system ready.")

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def home():

    return {
        "status": "working"
    }
