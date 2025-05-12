#!/bin/bash

# Only install if not already installed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment and installing dependencies..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
    echo "⬇️ Downloading NLTK data..."
    mkdir -p ./nltk_data
    python -m nltk.downloader -d ./nltk_data punkt averaged_perceptron_tagger wordnet
else
    source venv/bin/activate
fi