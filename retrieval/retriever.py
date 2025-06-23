from langchain.schema.document import Document
import json
from langchain_community.vectorstores import Chroma
from config.retrieval import TOP_K_RETRIEVAL


class Retriever:
    def __init__(self, vectorstore: Chroma, docstore, embedding_function, id_key="doc_id"):
        self.vectorstore = vectorstore
        self.docstore = docstore
        self.embedding_function = embedding_function
        self.id_key = id_key


    def retrieve(self, query: str):
        """
        Perform vector retrieval
        """
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=TOP_K_RETRIEVAL, 
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
                    print(f"Error parsing doc_id={doc_id}: {e}")
                    enriched_docs.append(doc)  # fallback

        return enriched_docs