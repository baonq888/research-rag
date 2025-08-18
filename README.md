# ResearchRAG

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Models Used](#models-used)  
- [Architecture](#architecture)  
- [Installation](#installation)  
- [Usage](#usage)  
- [API Endpoints](#api-endpoints)  
- [Development](#development)  
- [License](#license)

## Overview
**ResearchRAG** is a Retrieval-Augmented Generation (RAG) system designed for answering complex queries from unstructured research documents such as PDFs. It combines vector search, summarization, reranking, and metadata filtering to extract accurate and context-aware answers.

The system supports multimodal content including text, tables, and images. It features both full-document and section-level summaries, powered by LangChain and Transformer-based models.

Built with FastAPI and Redis, providing persistent document storage and scalable inference.

## Features

**Multimodal Input Support**  
Handles PDF content including text, tables, and embedded images.

**Summarization (Full & Section-Level)**  
Uses zero-shot classification to route queries to full or section summaries.

**Vector-Based Retrieval with Re-ranking**  
Retrieves top-k documents via dense vector search, followed by CrossEncoder-based reranking.

**Metadata Filtering**  
Dynamically filters documents using extracted metadata of document types.

**Persistent Storage with Redis**  
Stores raw and structured document data in Redis for efficient access.

## Models Used

- **Large Language Models (LLMs)**  
  - `deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free`  
  Used to generate answers based on retrieved and summarized content.

- **Image Model**  
  - `meta-llama/Llama-Vision-Free`  
  Used for generating summaries or understanding image content from PDFs.

- **Embedding Model**  
  - `intfloat/e5-large-v2`  
  Used to generate dense vector representations for retrieval.

- **Re-ranking Model**  
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`  
  Used to re-rank retrieved documents based on semantic relevance.

## Summary Settings

- **Temperature**: `0.5`  
  Controls the creativity of the summarization outputs.

- **Max Concurrency**: `3`  
  Limits parallel summarization tasks to prevent resource exhaustion.
