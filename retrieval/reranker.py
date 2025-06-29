from sentence_transformers import CrossEncoder
from langchain.schema import Document
from config.models import RERANKER_MODEL

class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            print(f"[Reranker] Failed to load reranker model: {e}")
            self.model = None

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        if not self.model:
            print("[Reranker] Model not available. Skipping reranking.")
            return docs

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        reranked = sorted(zip(docs, scores), key=lambda x: -x[1])
        return [doc for doc, _ in reranked]