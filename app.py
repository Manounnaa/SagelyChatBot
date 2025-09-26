from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_chatbot import rag_chat  # import your function
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="University RAG Chatbot with Gemini")

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = rag_chat(request.query)
    return {"question": request.query, "answer": answer}

if __name__ == "_main_":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8075, reload=True)