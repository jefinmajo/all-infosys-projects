This repository contains all the projects i completed during my infosys internship, it contains a separate folder for the final project, which is:

Medical AI Chat Application 🏥
This is a Streamlit-based application that enables users to interact with a medical knowledge assistant. It provides functionality to query a medical knowledge base derived from PDFs, URLs, or a default medical encyclopedia.

Features:

    Chat Interface: Ask medical questions in natural language and get AI-powered responses.
    Multiple Knowledge Sources:
    
      Default Medical Encyclopedia: Pre-loaded for immediate use.
      
      PDF Upload: Upload medical documents for analysis and Q&A.
      
      Website Content: Process and query medical information from URLs.
      
    Pinecone Integration: Efficiently stores and retrieves contextual information using a vector database.
    
    Customizable Models: Leverages HuggingFace embeddings and LLaMA API for question-answering.
    
    Debug Mode: View contextual information retrieved for each query.


Application Architecture:
      Front-end: Streamlit for a simple, interactive web interface.
      
      Back-end:
      
        HuggingFace Transformers for embeddings.
        
        Pinecone for vector storage and retrieval.
        
        LLaMA API for large language model-powered Q&A.
        
  
Requirements:
      streamlit>=1.25.0
      
      langchain>=0.0.300
      
      pinecone-client>=2.3.1
      
      requests>=2.31.0
      
      python-dotenv>=1.0.0
      
      sentence-transformers>=2.2.2

License:

    This project is licensed under MIT License.
