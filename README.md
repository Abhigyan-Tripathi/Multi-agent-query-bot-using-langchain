# Multi-Agent Astronomy Query Bot

This project demonstrates how to build a modular, intelligent, and multi-modal AI chatbot that answers astronomy-related queries using documents, images, and web content. It is powered by **LangGraph**, **LangChain**, **Streamlit**, **ChromaDB** and a Hugging Face LLM.

## Features

- Extracts information from uploaded PDFs
- Scrapes web content for relevant answers
- Uses OCR to process image-based text
- Leverages a multi-agent architecture to route queries smartly
- Retrieval-augmented generation (RAG) using Chroma and sentence-transformers
- Clean Streamlit UI for interaction

## Technologies Used

- [Streamlit](https://streamlit.io) (for deploying and endpoint creation)
- [LangGraph](https://docs.langgraph.dev) (multiagent framework creation)
- [LangChain](https://www.langchain.com) (for agent creation)
- [Hugging Face Transformers](https://huggingface.co) (Utilising LLM)
- [Chroma Vector DB](https://docs.trychroma.com/) (creating the vector database for rag)
- [sentence-transformers](https://www.sbert.net/) (for vector embeddings)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) (For image extraction)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) (for pdf parsing)

<img width="500" alt="Screenshot 2025-05-22 at 9 10 17 PM" src="https://github.com/user-attachments/assets/f93dcc50-65c5-4caf-8224-e0cb3a11d56a" />
## final agentic workflow 
---
