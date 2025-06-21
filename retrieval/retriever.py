from langchain.schema.document import Document
from langchain_core.runnables import RunnableLambda
from typing import List
import json


class Retriever:
    def __init__(self, vectorstore, docstore, embedding_function, id_key="doc_id"):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.embedding_function = embedding_function
        self.id_key = id_key

    def retrieve(self, query: str, top_k=5, metadata_filter: dict = None):
        """
        Perform vector retrieval with optional metadata filtering.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=top_k, filter=metadata_filter)
        
        # Check confidence scores and fallback if needed
        if results and results[0][1] > 0.8:
            print("Low confidence, fallback logic can be triggered here")
            # Placeholder: call web search 

        # Get full doc from Redis using doc_id
        enriched_docs = []
        for doc, score in results:
            doc_id = doc.metadata.get(self.id_key)
            redis_raw = self.docstore.mget([doc_id])[0]
            if redis_raw:
                try:
                    parsed = json.loads(redis_raw)
                    enriched_docs.append(Document(**parsed))
                except Exception as e:
                    print(f"Error parsing doc_id={doc_id}: {e}")
                    enriched_docs.append(doc)  # fallback to summary

        return enriched_docs

    def hybrid_retrieve(self, query: str, top_k=5):
        """
        Combine vector search + keyword match (optional stub).
        """
        vector_results = self.vectorstore.similarity_search(query, k=top_k)
        # Placeholder: keyword search logic (e.g., full-text filter, regex, or BM25)
        return vector_results  # Combine both later