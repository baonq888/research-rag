from langchain.schema.document import Document
import json
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from src.config.constants import SUMMARY_INTENT_FULL, SUMMARY_INTENT_SECTION
from src.config.models import ZERO_SHOT_MODEL
from rapidfuzz import fuzz

from src.loader.pdf_loader import UnstructuredPDFLoader

class SummaryRetriever:
    def __init__(self, vectorstore: Chroma, docstore, embedding_function, pdf_loader: UnstructuredPDFLoader, id_key="doc_id"):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.embedding_function = embedding_function
        self.pdf_loader = pdf_loader
        self.id_key = id_key
        
        try:
            self.classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
        except Exception as e:
            print(f"[SummaryRetriever] Failed to load zero-shot model: {e}")
            self.classifier = None

    def _classify_summary_intent(self, query: str) -> str:
        if not self.classifier:
            return "detail"

        labels = [SUMMARY_INTENT_FULL, SUMMARY_INTENT_SECTION]
        result = self.classifier(query, candidate_labels=labels)
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        print(f"[Zero-shot] Summary intent: {top_label} (score: {top_score:.2f})")

        
        if top_label == SUMMARY_INTENT_FULL:
            return SUMMARY_INTENT_FULL

        return SUMMARY_INTENT_SECTION

    def _load_all_summary_docs(self):
        docs = []
        for key in self.docstore.keys("*"):
            raw = self.docstore.get(key)
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
                if parsed.get("metadata", {}).get("type") == "summary":
                    docs.append(Document(**parsed))
            except Exception as e:
                print(f"[SummaryRetriever] Failed to parse doc {key}: {e}")
        return docs
    

    def _extract_section_mentions(self, query: str) -> list[str]:
        """
        Match query against known section titles using fuzzy logic.
        """
        pdf_known_sections = self.pdf_loader.get_extracted_section_titles()

        query_lower = query.lower()
        matches = []

        for title in pdf_known_sections:
            score = fuzz.partial_ratio(query_lower, title.lower())
            if score > 50:
                matches.append((title, score))

        matches = sorted(matches, key=lambda x: -x[1])
        matched_titles = [title for title, _ in matches]

        print(f"[SummaryRetriever] Matched sections from known titles: {matched_titles}")
        return matched_titles

    def _load_section_summary_docs(self, query: str):
        query_lower = query.lower()
        candidates = []

        # Dynamically determine how many sections are mentioned
        top_k = self._extract_section_mentions(query)

        for key in self.docstore.keys("*"):
            raw = self.docstore.get(key)
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
                meta = parsed.get("metadata", {})
                if meta.get("type") != "summary":
                    continue

                section_title = (meta.get("section") or meta.get("heading") or "").lower().strip()
                if not section_title:
                    continue

                score = fuzz.partial_ratio(query_lower, section_title)
                if score > 50:
                    candidates.append((score, Document(**parsed)))

            except Exception as e:
                print(f"[SummaryRetriever] Failed to parse doc {key}: {e}")

        sorted_docs = sorted(candidates, key=lambda x: -x[0])[:top_k]

        if sorted_docs:
            print(f"[SummaryRetriever] Returning top {len(sorted_docs)} section summaries.")
        else:
            print("[SummaryRetriever] No matching section summaries found.")

        return [doc for _, doc in sorted_docs]

    def retrieve(self, query: str):
        summary_type = self._classify_summary_intent(query)

        if summary_type == SUMMARY_INTENT_FULL:
            print("[SummaryRetriever] Full summary requested.")
            return self._load_all_summary_docs()

        if summary_type == SUMMARY_INTENT_SECTION:
            print("[SummaryRetriever] Section summary requested.")
            return self._load_section_summary_docs(query)

        # Fallback to standard similarity-based retrieval
        print("[SummaryRetriever] Fallback. Running vector similarity search...")
        results = self.vectorstore.similarity_search_with_score(query, k=5)

        enriched_docs = []
        for doc, score in results:
            doc_id = doc.metadata.get(self.id_key)
            raw = self.docstore.mget([doc_id])[0]
            if raw:
                try:
                    parsed = json.loads(raw)
                    enriched_docs.append(Document(**parsed))
                except Exception as e:
                    print(f"[SummaryRetriever] Failed to parse doc {doc_id}: {e}")
                    enriched_docs.append(doc)

        return enriched_docs