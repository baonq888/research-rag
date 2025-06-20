import uuid
from langchain.vectorstores import Chroma
from langchain_community.storage.redis import RedisStore
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

from config.models import EMBEDDING_MODEL
from config.redis import REDIS_URL

class VectorStoreManager:
    def __init__(self, collection_name="multi_modal_rag", persist_dir=None):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        # Store summarized chunks' embeddings 
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_dir  # Optional directory to persist Chroma DB to disk
        )

        # Store orignal contents
        self.docstore = RedisStore.from_url(REDIS_URL)
        # Set the key used to link vector entries to full documents
        self.id_key = "doc_id"

        # Retreive summarized chunks' embeddings along side with the original chunk via id_key
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
        )

    def add_chunks(self, chunks: list[str], parent_metadata: dict = None):
        """Embed and store summary chunks with parent metadata."""
        doc_id = str(uuid.uuid4())
        metadata = parent_metadata or {}

        docs = [
            Document(
                page_content=chunk,
                metadata={**metadata, "doc_id": doc_id}
            ) for chunk in chunks
        ]

        self.retriever.add_documents(documents=docs)
        print(f"Stored {len(docs)} chunks under doc_id={doc_id}")

    def query(self, query: str, top_k=5):
        """Retrieve top chunks relevant to the query."""
        results = self.retriever.invoke(query)
        return results[:top_k] if top_k else results