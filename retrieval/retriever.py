from langchain.schema.document import Document
import json
from langchain_community.vectorstores import Chroma
from config.retrieval import TOP_K_RETRIEVAL
from transformers import pipeline
from config.models import ZERO_SHOT_MODEL
from retrieval.retriever_summary import SummaryRetriever 
from config.constants import (
    SUMMARY_INTENT_GENERAL,
    SUMMARY_INTENT_LABELS,
    SUMMARY_INTENT_FULL,
    SUMMARY_INTENT_SECTION,
    SUMMARY_INTENT_DETAIL
)

QUERY_INTENT_LABELS = ["full summary", "section summary", "detail"]

class Retriever:
    def __init__(
        self,
        vectorstore: Chroma,
        docstore,
        embedding_function,
        summary_retriever: SummaryRetriever, 
        id_key="doc_id"
    ):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.embedding_function = embedding_function
        self.summary_retriever = summary_retriever 
        self.id_key = id_key

        # Load zero-shot classification model
        try:
            self.classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
        except Exception as e:
            print(f"[Retriever] Failed to load zero-shot model: {e}")
            self.classifier = None

    def _classify_query_intent(self, query: str) -> str:
        """
        Determine if the user wants a full summary, a section summary, or detailed info.
        """
        if not self.classifier:
            return "detail"

        result = self.classifier(query, candidate_labels=QUERY_INTENT_LABELS)
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        print(f"[Zero-shot] Summary intent: {top_label} (score: {top_score:.2f})")

        
        if top_label == SUMMARY_INTENT_GENERAL:
            return SUMMARY_INTENT_GENERAL

        return SUMMARY_INTENT_DETAIL

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
        - Else → do top-k similarity search with optional metadata filter
        """
        summary_type = self._classify_query_intent(query)

        if summary_type == SUMMARY_INTENT_GENERAL:
            print(f"[Retriever] Delegating {summary_type} retrieval to SummaryRetriever...")
            return self.summary_retriever.retrieve(query)

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

        return enriched_docs