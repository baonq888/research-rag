# ResearchRAG

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Models Used](#models-used)  
- [Architecture](#architecture)  

## Overview
ResearchRAG is a Retrieval-Augmented Generation (RAG) system designed for answering complex queries from unstructured research documents such as PDFs. 

The system supports multimodal content including text, tables, and images. It features both full-document and section-level summaries, powered by LangChain and Transformer-based models.

Built with FastAPI and Redis, providing persistent document storage and scalable inference.

## Features

**Multimodal Input Support**  
Handles PDF content including text, tables, and embedded images.

**Graph + Vector Hybrid Retrieval**  
- Vector DB captures semantic similarity via embeddings.  
- Graph DB (Neo4j) encodes entities, citations, and relationships between documents.  
- A hybrid retriever merges semantic and structural signals for richer context.  

**Summarization (Full & Section-Level)**  
Generates summaries from both entire documents and specific sections.

**Vector-Based Retrieval with Re-ranking**  
Retrieves top-k candidates via dense vector search, followed by CrossEncoder reranking.

**Metadata Filtering**  
Dynamically filters documents using extracted metadata.

**Persistent Storage with Redis**  
Stores raw and structured document data in Redis for efficient access.

## Tech Stack

- **Programming Language**: Python  
- **Backend Framework**: FastAPI  
- **Vector Database**: ChromaDB  
- **Graph Database**: Neo4j  
- **Orchestration & RAG Framework**: LangChain  
- **Cache/Storage**: Redis  
- **UI**: Streamlit  

## Models Used

- **Large Language Models (LLMs)**  
  - `deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free`  

- **Image Model**  
  - `meta-llama/Llama-Vision-Free`  

- **Embedding Model**  
  - `intfloat/e5-large-v2`  

- **Re-ranking Model**  
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`  

## Architecture

### Preprocessing
1. **Vector Database Indexing**  
   - User documents → chunking → embeddings → Vector DB  

2. **Knowledge Graph Construction**  
   - User documents → entity/relation extraction → Graph DB (Neo4j)  

### Query Flow
1. User submits query.  
2. Query is rewritten/expanded for better recall.  
3. Query router decides:  
   - Vector DB retrieval (semantic similarity)  
   - Graph DB retrieval (entity + relation search)  
   - Or hybrid combination.  
4. Retrieved results are reranked (CrossEncoder).  
5. Augmented query + context passed to LLM.  
6. LLM generates final response.  

### Output
- Accurate answers grounded in both semantic similarity and structural knowledge.  
- Summaries or detailed explanations depending on context.  

## Summary Settings

- **Temperature**: `0.5`  
- **Max Concurrency**: `3`  
