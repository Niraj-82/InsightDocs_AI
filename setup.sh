#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Ensure Ollama is installed and 'ollama pull mistral' has been executed."
