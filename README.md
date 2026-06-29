# AI Document Assistant

An AI-powered PDF question-answering application built with Python and Streamlit.

Users can upload a PDF, ask questions in natural language, and receive answers generated from relevant sections of the document with page-level source references.

## Features

- PDF text extraction
- Overlapping document chunking
- TF-IDF and cosine-similarity retrieval
- Retrieval of neighboring pages for additional context
- AI-generated answers grounded in document content
- Page-level source references
- Expandable supporting passages
- Streamlit web interface

## Technology Stack

- Python
- Streamlit
- Groq API
- OpenAI-compatible Python SDK
- PyPDF
- Scikit-learn
- TF-IDF
- Cosine similarity

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
