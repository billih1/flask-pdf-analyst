# AI-Powered PDF Summarizer & Q&A Tool

## Overview
This project is a local web application that leverages Natural Language Processing (NLP) to analyze PDF documents. It allows users to upload a PDF, generates an intelligent summary of the content, and provides a chat interface to ask specific questions about the document.

Unlike cloud-based solutions, this tool runs entirely locally using Hugging Face Transformers, ensuring data privacy and offline capability.

## Key Features
- Automatic Summarization: Uses the facebook/bart-large-cnn model to condense long documents into concise summaries.
- Intelligent Q&A: Implements distilbert-base-cased-distilled-squad to perform extractive Question Answering.
- Long Document Support: Features a custom sliding window chunking algorithm to process documents larger than the model's token limit.
- Web Interface: A clean, responsive UI built with HTML/CSS and deployed via Flask.
- PDF Parsing: Extracts and cleans raw text from PDF files using PyPDF2 and Regex.

## Tech Stack
- Backend: Python, Flask
- AI/ML: Hugging Face Transformers, PyTorch
- Data Processing: PyPDF2, Regular Expressions (Re)
- Frontend: HTML5, CSS3

## How It Works
1. Upload: The user uploads a PDF via the Flask web interface.
2. Preprocessing: The app extracts text and cleans formatting.
3. Chunking: Large texts are split into overlapping chunks to fit within the Transformer model's context window.
4. Inference:
   - The Summarization Pipeline processes the chunks to create a synopsis.
   - The Q&A Pipeline scans the text to locate the precise answer to user queries.
5. Result: The summary or answer is rendered back to the user in the browser.

## Installation & Usage
1. Clone the repository:
   git clone https://github.com/YOUR-USERNAME/flask-pdf-summarizer.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   python app.py

4. Open your browser and go to:
   http://127.0.0.1:5000
