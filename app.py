from fastapi import FastAPI
from pydantic import BaseModel
from rag_chatbot import rag_chat  # import your function

app = FastAPI(title="University RAG Chatbot with Gemini")

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask_question(request: QueryRequest):
    answer = rag_chat(request.query)
    return {"question": request.query, "answer": answer}
