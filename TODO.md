# TODO: Fix ChromaDB Telemetry Errors and MMR Warning

## Steps to Complete

- [x] Update requirements.txt: Upgrade chromadb to 0.5.23, langchain to 0.3.0, langchain-community to 0.3.0
- [x] Edit main.py: Change retriever_type from "mmr" to "similarity" in CONFIG
- [x] Install updated dependencies: Run pip install -r requirements.txt
- [ ] Delete existing chroma_db directory (incompatible with new ChromaDB version)
- [ ] Test the application: Run python main.py and verify no telemetry errors or warnings
