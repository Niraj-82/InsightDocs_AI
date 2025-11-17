"""
Tech Stack:
- LangChain for RAG orchestration
- ChromaDB for local vector storage
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Ollama with Mistral 7B LLM
"""

import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"  

import logging
from typing import Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys
import logging
from typing import Optional, Dict, Any
import time

# Configuration
SPEECH_FILE = "speech.txt"
CHROMA_DIR = "chroma_db"
LOG_FILE = "ambedkar_gpt.log"

# RAG Configuration
CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 100,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "ollama_model": "mistral:7b-instruct-q4_0",
    "temperature": 0.0,
    "retriever_k": 3,
    "retriever_type": "similarity",  # mmr or similarity
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text to terminal"""
    print(f"{color}{text}{Colors.ENDC}")


def check_ollama_availability() -> bool:
    """Check if Ollama is running and the model is available"""
    try:
        llm = Ollama(model=CONFIG["ollama_model"], temperature=0.0)
        # Test with a simple query
        llm.invoke("test")
        return True
    except Exception as e:
        logger.error(f"Ollama check failed: {e}")
        return False


def build_vectorstore(speech_path: str, persist_dir: str = CHROMA_DIR) -> Chroma:
    """
    Build vector store from speech text file
    
    Args:
        speech_path: Path to the speech.txt file
        persist_dir: Directory to persist ChromaDB
        
    Returns:
        Chroma vector store instance
    """
    try:
        # 1) Load document
        print_colored("[1/4] Loading document...", Colors.OKCYAN)
        loader = TextLoader(speech_path, encoding="utf-8")
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} document(s) from {speech_path}")
        print_colored(f"✓ Loaded {len(docs)} document(s)", Colors.OKGREEN)

        # 2) Split into chunks with better splitter
        print_colored("[2/4] Splitting into chunks...", Colors.OKCYAN)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunked_docs = splitter.split_documents(docs)
        logger.info(f"Split into {len(chunked_docs)} chunks")
        print_colored(f"✓ Created {len(chunked_docs)} chunks", Colors.OKGREEN)

        # 3) Create embeddings
        print_colored("[3/4] Loading embedding model...", Colors.OKCYAN)
        embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["embedding_model"],
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Created embeddings with model: {CONFIG['embedding_model']}")
        print_colored("✓ Embedding model loaded", Colors.OKGREEN)

        # 4) Create and persist vector store
        print_colored("[4/4] Building vector database...", Colors.OKCYAN)
        vectordb = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectordb.persist()
        logger.info(f"Persisted ChromaDB to {persist_dir}")
        print_colored(f"✓ Vector database created at {persist_dir}\n", Colors.OKGREEN)
        
        return vectordb
    
    except Exception as e:
        logger.error(f"Error building vectorstore: {e}")
        print_colored(f"✗ Error: {e}", Colors.FAIL)
        raise


def load_or_build(
    speech_path: str = SPEECH_FILE,
    persist_dir: str = CHROMA_DIR,
    rebuild: bool = False
) -> Chroma:
    """
    Load existing vector store or build new one
    
    Args:
        speech_path: Path to speech.txt
        persist_dir: ChromaDB persist directory
        rebuild: Force rebuild even if DB exists
        
    Returns:
        Chroma vector store instance
    """
    if rebuild and os.path.isdir(persist_dir):
        print_colored("[*] Rebuilding: Removing existing ChromaDB", Colors.WARNING)
        import shutil
        shutil.rmtree(persist_dir)
        logger.info("Removed existing ChromaDB for rebuild")

    if os.path.isdir(persist_dir) and not rebuild:
        print_colored("[*] Loading existing vector database...", Colors.OKCYAN)
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=CONFIG["embedding_model"],
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vectordb = Chroma(
                persist_directory=persist_dir,
                embedding_function=embeddings
            )
            print_colored("✓ Loaded existing database\n", Colors.OKGREEN)
            logger.info("Loaded existing ChromaDB")
            return vectordb
        except Exception as e:
            logger.warning(f"Failed to load existing DB: {e}. Building new one.")
            print_colored(f"⚠ Could not load existing DB: {e}", Colors.WARNING)
            print_colored("Building new database...\n", Colors.OKCYAN)
    
    return build_vectorstore(speech_path, persist_dir)


def create_custom_prompt() -> PromptTemplate:
    """Create a custom prompt template for better answers"""
    template = """You are AmbedkarGPT, an AI assistant that answers questions based exclusively on Dr. B.R. Ambedkar's speech excerpt provided below.

Context from the speech:
{context}

Instructions:
- Answer ONLY based on the provided context
- If the answer is not in the context, say "I cannot find that information in the provided speech excerpt."
- Be concise but complete
- Quote relevant parts when appropriate

Question: {question}

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def create_qa_chain(vectordb: Chroma, show_sources: bool = False) -> RetrievalQA:
    """
    Create RetrievalQA chain with LLM and retriever
    
    Args:
        vectordb: ChromaDB vector store
        show_sources: Whether to return source documents
        
    Returns:
        RetrievalQA chain instance
    """
    # Create retriever
    retriever = vectordb.as_retriever(
        search_type=CONFIG["retriever_type"],
        search_kwargs={"k": CONFIG["retriever_k"]}
    )
    logger.info(f"Created retriever with k={CONFIG['retriever_k']}, type={CONFIG['retriever_type']}")

    # Initialize Ollama LLM
    try:
        llm = Ollama(
            model=CONFIG["ollama_model"],
            temperature=CONFIG["temperature"]
        )
        logger.info(f"Initialized Ollama with model: {CONFIG['ollama_model']}")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}")
        print_colored(f"✗ Error: Could not connect to Ollama", Colors.FAIL)
        print_colored("  Make sure Ollama is running: 'ollama serve'", Colors.WARNING)
        print_colored(f"  And model is pulled: 'ollama pull {CONFIG['ollama_model']}'", Colors.WARNING)
        sys.exit(1)

    # Create custom prompt
    prompt = create_custom_prompt()

    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=show_sources,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def validate_query(query: str) -> Optional[str]:
    """
    Validate and preprocess user query
    
    Returns:
        Cleaned query or None if invalid
    """
    query = query.strip()
    
    if len(query) < 3:
        return None
    
    if len(query) > 500:
        print_colored("⚠ Query too long. Please keep it under 500 characters.", Colors.WARNING)
        return None
    
    return query

def interactive_loop(chain: RetrievalQA, show_sources: bool = False):
    """
    Interactive CLI loop for asking questions
    
    Args:
        chain: RetrievalQA chain
        show_sources: Display source documents
    """
    print_colored("\n" + "="*60, Colors.HEADER)
    print_colored("  AmbedkarGPT - RAG Q&A System", Colors.HEADER + Colors.BOLD)
    print_colored("="*60, Colors.HEADER)
    print_colored("\nAsk questions about Dr. Ambedkar's speech on caste.", Colors.OKBLUE)
    print_colored("Commands: 'exit', 'quit', 'help', 'sources on/off'\n", Colors.OKCYAN)
    
    while True:
        try:
            query = input(f"{Colors.BOLD}Q:{Colors.ENDC} ").strip()
        except (KeyboardInterrupt, EOFError):
            print_colored("\n\nGoodbye! 👋", Colors.OKGREEN)
            break
        
        # Handle commands
        if query.lower() in ("exit", "quit"):
            print_colored("\nGoodbye! 👋", Colors.OKGREEN)
            break
        
        if query.lower() == "help":
            print_colored("\nCommands:", Colors.OKCYAN)
            print("  exit/quit    - Exit the program")
            print("  help         - Show this help message")
            print("  sources on   - Show source chunks with answers")
            print("  sources off  - Hide source chunks")
            print()
            continue
        
        if query.lower() == "sources on":
            show_sources = True
            print_colored("✓ Source display enabled\n", Colors.OKGREEN)
            continue
        
        if query.lower() == "sources off":
            show_sources = False
            print_colored("✓ Source display disabled\n", Colors.OKGREEN)
            continue
        
        # Validate query
        query = validate_query(query)
        if not query:
            continue
        
        # Process query
        try:
            print_colored("\n⏳ Processing...", Colors.OKCYAN)
            start_time = time.time()
                        
            if show_sources:
                result = chain({"query": query})
                answer = result.get("result", "")
                sources = result.get("source_documents", [])
            else:
                result = chain.invoke({"query": query})
                if isinstance(result, str):
                    answer = result
                else:
                    answer = result.get("result", "")
                sources = []


            
            elapsed = time.time() - start_time
            
            print_colored(f"\n{Colors.BOLD}A:{Colors.ENDC} {answer}", Colors.OKGREEN)
            
            if sources:
                print_colored(f"\n📄 Sources ({len(sources)} chunks):", Colors.OKCYAN)
                for i, doc in enumerate(sources, 1):
                    print_colored(f"\n[{i}] {doc.page_content[:200]}...", Colors.OKCYAN)
            
            logger.info(f"Query processed in {elapsed:.2f}s: {query[:50]}...")
            print_colored(f"\n⏱ Response time: {elapsed:.2f}s\n", Colors.OKCYAN)
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print_colored(f"\n✗ Error: {e}\n", Colors.FAIL)

def main():
    """Main entry point"""
    print_colored("\n🚀 Starting AmbedkarGPT...\n", Colors.HEADER + Colors.BOLD)
    
    # Check speech file exists
    if not os.path.exists(SPEECH_FILE):
        print_colored(f"✗ Error: {SPEECH_FILE} not found", Colors.FAIL)
        print_colored(f"  Please ensure {SPEECH_FILE} exists in the current directory", Colors.WARNING)
        sys.exit(1)
    
    # Parse arguments
    rebuild = "--rebuild" in sys.argv or "-r" in sys.argv
    show_sources = "--sources" in sys.argv or "-s" in sys.argv
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check Ollama availability
    print_colored("Checking Ollama availability...", Colors.OKCYAN)
    if not check_ollama_availability():
        print_colored("✗ Ollama is not available", Colors.FAIL)
        print_colored("\nPlease ensure:", Colors.WARNING)
        print_colored("  1. Ollama is installed: https://ollama.ai", Colors.WARNING)
        print_colored(f"  2. Model is pulled: ollama pull {CONFIG['ollama_model']}", Colors.WARNING)
        print_colored("  3. Ollama is running: ollama serve", Colors.WARNING)
        sys.exit(1)
    print_colored("✓ Ollama is ready\n", Colors.OKGREEN)
    
    # Load or build vector store
    try:
        vectordb = load_or_build(SPEECH_FILE, CHROMA_DIR, rebuild=rebuild)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    
    # Create QA chain
    print_colored("Initializing QA system...", Colors.OKCYAN)
    qa_chain = create_qa_chain(vectordb, show_sources=show_sources)
    print_colored("✓ QA system ready\n", Colors.OKGREEN)
    
    # Start interactive loop
    interactive_loop(qa_chain, show_sources=show_sources)
    
    logger.info("Session ended")


if __name__ == "__main__":
    main()
