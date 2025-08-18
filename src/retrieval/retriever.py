from langchain.schema.document import Document
import json
from langchain_community.vectorstores import Chroma
from src.config.retrieval import TOP_K_RETRIEVAL
from transformers import pipeline
from src.config.models import RERANKER_MODEL, ZERO_SHOT_MODEL
from src.retrieval.reranker import Reranker
from src.config.constants import (
    SUMMARY_INTENT_FULL,
    QUERY_INTENT_DETAIL,
    SUMMARY_INTENT_SECTION
)
from sentence_transformers import CrossEncoder

QUERY_INTENT_LABELS = [SUMMARY_INTENT_FULL, SUMMARY_INTENT_SECTION, QUERY_INTENT_DETAIL]

class Retriever:
    def __init__(
        self,
        vectorstore: Chroma,
        docstore,
        embedding_function,
        id_key="doc_id"
    ):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.embedding_function = embedding_function
        self.reranker = Reranker()
        self.id_key = id_key

        # Load zero-shot classification model
        try:
            self.classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
        except Exception as e:
            print(f"[Retriever] Failed to load zero-shot model: {e}")
            self.classifier = None

        # Load cross-encoder reranker model
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"[Retriever] Failed to load reranker model: {e}")
            self.reranker = None



    def _format_filter(self, metadata_filter):
        """
        Format metadata filter for ChromaDB vector search.
        """
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            return metadata_filter
        return {"$and": [{k: v} for k, v in metadata_filter.items()]}


    def retrieve(self, query: str, metadata_filter: dict = None):
        """
        Main entry point for document retrieval:
        - If full summary is requested → delegate to SummaryRetriever
        - If section summary is requested → delegate to SummaryRetriever
        - Else → do top-k similarity search with optional metadata filter and rerank
        """

        formatted_filter = self._format_filter(metadata_filter)
        if formatted_filter:
            print(f"[Retriever] Applying metadata filter: {formatted_filter}")

        results = self.vectorstore.similarity_search_with_score(
            query,
            k=TOP_K_RETRIEVAL,
            filter=formatted_filter
        )

        enriched_docs = []
        for doc, score in results:
            doc_id = doc.metadata.get(self.id_key)
            redis_raw = self.docstore.mget([doc_id])[0]
            if redis_raw:
                try:
                    parsed = json.loads(redis_raw)
                    enriched_docs.append(Document(**parsed))
                except Exception as e:
                    print(f"[Retriever] Failed to parse doc_id={doc_id}: {e}")
                    enriched_docs.append(doc)

       # Re-rank only for detail queries
      
        return self.reranker.rerank(query, enriched_docs)
