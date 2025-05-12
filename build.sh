#!/bin/bash

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "⬇️ Downloading NLTK data..."
python -m nltk.downloader -d ./nltk_data punkt averaged_perceptron_tagger wordnet
