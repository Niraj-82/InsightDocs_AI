"""
Configuration module for AmbedkarGPT
Separates configuration from main logic for better maintainability
"""

import os
from typing import Dict, Any

# File paths
SPEECH_FILE = "speech.txt"
CHROMA_DIR = "chroma_db"
LOG_FILE = "ambedkar_gpt.log"

# RAG Configuration
RAG_CONFIG: Dict[str, Any] = {
    # Document Processing
    "chunk_size": 400,
    "chunk_overlap": 100,
    "separators": ["\n\n", "\n", ". ", " ", ""],
    
    # Embeddings
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_device": "cpu",  # Change to "cuda" if GPU available
    "normalize_embeddings": True,
    
    # LLM Configuration
    "ollama_model": "mistral",
    "temperature": 0.0,  # 0 = deterministic, 1 = creative
    "max_tokens": 500,   # Maximum response length
    
    # Retriever Settings
    "retriever_k": 3,           # Number of chunks to retrieve
    "retriever_type": "mmr",    # "mmr" or "similarity"
    "mmr_diversity": 0.3,       # MMR diversity score (0-1)
    "similarity_threshold": 0.7, # Minimum similarity score
    
    # Performance
    "batch_size": 32,     # Embedding batch size
    "show_progress": True # Show progress bars
}

# Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """You are AmbedkarGPT, an AI assistant specializing in Dr. B.R. Ambedkar's writings on social reform and caste.

Context from the speech:
{context}

Instructions:
- Answer ONLY based on the provided context
- If the answer is not in the context, say "I cannot find that information in the provided speech excerpt."
- Be concise but thorough
- Quote relevant passages when it strengthens the answer
- Maintain an educational and respectful tone

Question: {question}

Answer:"""

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "file": LOG_FILE,
    "console": True
}

# CLI Colors (ANSI codes)
COLORS = {
    "HEADER": '\033[95m',
    "OKBLUE": '\033[94m',
    "OKCYAN": '\033[96m',
    "OKGREEN": '\033[92m',
    "WARNING": '\033[93m',
    "FAIL": '\033[91m',
    "ENDC": '\033[0m',
    "BOLD": '\033[1m',
    "UNDERLINE": '\033[4m'
}

# Validation Rules
VALIDATION_CONFIG = {
    "min_query_length": 3,
    "max_query_length": 500,
    "allowed_commands": ["exit", "quit", "help", "sources on", "sources off"],
}

# Help Text
HELP_TEXT = """
Available Commands:
  exit, quit    - Exit the application
  help          - Show this help message
  sources on    - Display source chunks with answers
  sources off   - Hide source chunks
  
Tips:
  - Ask specific questions about the speech content
  - Use natural language
  - Try rephrasing if you don't get the answer you expect
  
Example Questions:
  - What is the main argument of the speech?
  - What does the speaker say about the shastras?
  - How does the speaker describe social reform work?
"""

# Environment-specific overrides
def load_environment_config():
    """Load configuration from environment variables if present"""
    config = RAG_CONFIG.copy()
    
    if os.getenv("CHUNK_SIZE"):
        config["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
    
    if os.getenv("OLLAMA_MODEL"):
        config["ollama_model"] = os.getenv("OLLAMA_MODEL")
    
    if os.getenv("RETRIEVER_K"):
        config["retriever_k"] = int(os.getenv("RETRIEVER_K"))
    
    if os.getenv("TEMPERATURE"):
        config["temperature"] = float(os.getenv("TEMPERATURE"))
    
    return config

# Export main configuration
def get_config() -> Dict[str, Any]:
    """Get the current configuration, including environment overrides"""
    return load_environment_config()
