# AmbedkarGPT - Intern Assignment (Kalpit Pvt Ltd UK)

A simple RAG-based Q&A system that answers questions from Dr. B.R. Ambedkar's speech.

### 🚀 Tech Stack
- Python 3.11
- LangChain (RAG orchestration)
- ChromaDB (local vector DB)
- HuggingFace Sentence Transformers (Embeddings)
- Ollama + Mistral 7B (LLM)

### 🛠 Setup Instructions
```bash
git clone https://github.com/YourUserName/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
ollama pull mistral
python main.py
