# 🎓 SagelyChatBot (RAG-based University Assistant)
SagelyChatBot is an intelligent university assistant built by Retrieval-Augmented Generation (RAG).
It retrieves relevant answers from a dataset using FAISS and generates responses using Ollama + LLM.

# 🚀 Features
📚 Uses a CSV dataset (Questions & Answers)
🔍 Fast similarity search with FAISS
🧠 Embeddings powered by Sentence Transformers
🤖 Response generation using Ollama (LLM)
🔄 Fallback mechanism if the model fails
💬 Interactive command-line interface

# 🧠 How It Works
Load Q&A data from a CSV file
Convert questions into embeddings
Store embeddings in a FAISS index
When a user asks a question:
Convert it into an embedding
Retrieve the top 3 most similar questions
Send them as context to the LLM
Generate a final answer

# 📂 Project Structure
SagelyChatBot/
│
├── app.py
├── university_chatbot_dataset.csv
├── requirements.txt
├── .env
└── README.md

# ⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/SagelyChatBot.git
cd SagelyChatBot
2. Install dependencies
pip install -r requirements.txt
3. Setup environment variables (optional)

Create a .env file:

OLLAMA_API_KEY=your_key_if_needed

▶️ Run the Chatbot

python app.py

# 💬 Usage
After running the program:

🤖 RAG Chatbot is ready! Type 'exit' to quit

You: What are university admission requirements?
Bot: ...

# 📊 Dataset Format
Your CSV file must look like this:

Question	Answer
What is GPA?	GPA is ...
How to apply?	You can apply by ...

# 🛠️ Technologies Used
Python
FAISS
SentenceTransformers (all-MiniLM-L6-v2)
Ollama (LLM - llama3.2)
Pandas / NumPy

# ⚠️ Notes
Make sure Ollama is running locally
Ensure the model is available:
ollama run llama3.2

# 🔮 Future Improvements
🌐 Web interface (Streamlit / Flask)
🗂️ Support PDF instead of CSV
🧾 Improve prompt engineering
🔐 Add authentication

# 👩‍💻 Author
Menna Hany
AI Student | Data Science Enthusiast

Menna Hany
AI Student | Data Science Enthusiast
