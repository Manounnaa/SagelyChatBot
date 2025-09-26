import os
import pandas as pd
import numpy as np
import faiss
import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ---------------------------
# 1. Load .env
# ---------------------------
load_dotenv()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")  # optional if your Ollama setup requires it

print("✅ Ollama configured")

# ---------------------------
# 2. Load Embedding Model
# ---------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return embed_model.encode(text).astype("float32")

# ---------------------------
# 3. Load CSV Data
# ---------------------------
CSV_FILE = "university_chatbot_dataset.csv"
df = pd.read_csv(CSV_FILE)

if "Question" not in df.columns or "Answer" not in df.columns:
    raise RuntimeError("❌ CSV must contain 'Question' and 'Answer' columns")

questions = df["Question"].astype(str).tolist()
answers = df["Answer"].astype(str).tolist()

print("✅ Loaded", len(questions), "Q&A pairs")

# ---------------------------
# 4. Build FAISS Index
# ---------------------------
print("🔍 Computing embeddings locally...")
embeddings = np.array([get_embedding(q) for q in questions])
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

print("✅ FAISS index built with", index.ntotal, "entries")

# ---------------------------
# 5. RAG Chat Function with Ollama
# ---------------------------
def rag_chat(query: str, top_k=3):
    # Retrieve similar Q&A
    q_emb = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(q_emb, top_k)

    retrieved_context = "\n".join([f"Q: {questions[i]}\nA: {answers[i]}" for i in indices[0]])

    prompt = f"""
    You are a helpful university assistant.
    Use the following context to answer the question accurately.
    If you don't know, say "I don't know, Your Request Is Escalated To Admins ,Now! ."

    Context:
    {retrieved_context}

    Question: {query}
    Answer:
    """

    try:
        response = ollama.chat(
            model='llama3.2',  # Ollama model
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()

    except Exception as e:
        print("⚠️ Ollama API error:", e)
        # Fallback → return first retrieved FAISS answer
        return f"(Fallback) {answers[indices[0][0]]}"

# ---------------------------
# 6. Interactive Loop
# ---------------------------
if __name__ == "__main__":
    print("🤖 RAG Chatbot is ready! Type 'exit' to quit")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        answer = rag_chat(user_input)
        print("Bot:", answer)
