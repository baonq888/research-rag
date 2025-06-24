from langchain.schema.document import Document
import json
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from config.models import ZERO_SHOT_MODEL
from rapidfuzz import fuzz


class SummaryRetriever:
    def __init__(self, vectorstore: Chroma, docstore, embedding_function, id_key="doc_id"):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.embedding_function = embedding_function
        self.id_key = id_key

        try:
            self.classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
        except Exception as e:
            print(f"[SummaryRetriever] Failed to load zero-shot model: {e}")
            self.classifier = None

    def _classify_summary_intent(self, query: str) -> str:
        if not self.classifier:
            return "detail"

        labels = ["full summary", "section summary", "detail"]
        result = self.classifier(query, candidate_labels=labels)
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        print(f"[Zero-shot] Summary intent: {top_label} (score: {top_score:.2f})")

        if top_score > 0.4:
            if top_label == "full summary":
                return "full_summary"
            elif top_label == "section summary":
                return "section_summary"

        return "detail"

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

    def _load_section_summary_docs(self, query: str):
        query_lower = query.lower()
        candidates = []
        best_section = None
        best_score = 0

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
                    if score > best_score:
                        best_section = section_title
                        best_score = score
            except Exception as e:
                print(f"[SummaryRetriever] Failed to parse doc {key}: {e}")

        if best_section:
            print(f"[SummaryRetriever] Best matched section: {best_section} (score: {best_score})")
        else:
            print("[SummaryRetriever] No matching section found.")

        return [doc for _, doc in sorted(candidates, key=lambda x: -x[0])]

    def retrieve(self, query: str):
        summary_type = self._classify_summary_intent(query)

        if summary_type == "full_summary":
            print("[SummaryRetriever] Full summary requested.")
            return self._load_all_summary_docs()

        if summary_type == "section_summary":
            print("[SummaryRetriever] Section summary requested.")
            return self._load_section_summary_docs(query)

        print("[SummaryRetriever] Running vector similarity search...")
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