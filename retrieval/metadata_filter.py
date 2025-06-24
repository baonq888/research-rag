import os
from typing import Optional
from transformers import pipeline
from config.models import ZERO_SHOT_MODEL
from rapidfuzz import fuzz

TYPE_LABELS = ["text", "table", "image"]

# Fuzzy rule-based matching
def rule_based_type(query: str, threshold: int = 85) -> Optional[str]:
    query_lower = query.lower()

    def fuzzy_contains(keywords):
        return any(fuzz.partial_ratio(query_lower, word) >= threshold for word in keywords)

    if fuzzy_contains(["text", "describe", "explain", "say", "content", "paragraph"]):
        return "text"
    if fuzzy_contains(["table", "tabular", "spreadsheet", "data table"]):
        return "table"
    if fuzzy_contains(["image", "figure", "diagram", "chart", "plot", "graph", "visual"]):
        return "image"

    return None

class MetadataFilterExtractor:
    def __init__(self):
        try:
            self.classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
        except Exception as e:
            print(f"[MetadataFilterExtractor] Failed to load zero-shot model: {e}")
            self.classifier = None

    def extract(self, query: str) -> dict:
        try:
            if self.classifier:
                result = self.classifier(query, candidate_labels=TYPE_LABELS)
                top_label = result["labels"][0]
                top_score = result["scores"][0]

                print(f"[Zero-Shot] Predicted: {top_label} (score: {top_score:.2f})")

                if top_score > 0.4:
                    return {"type": top_label.lower()}

        except Exception as e:
            print(f"[MetadataFilterExtractor] Zero-shot classification failed: {e}")

        # Fallback
        fallback_type = rule_based_type(query)
        if fallback_type:
            print(f"[Fallback] Rule-based matched type: {fallback_type}")
            return {"type": fallback_type}

        print("[MetadataFilterExtractor] No type matched.")