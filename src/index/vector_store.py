import uuid
from langchain_community.vectorstores import Chroma
from langchain_community.storage.redis import RedisStore
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from redis import Redis

from src.config.models import EMBEDDING_MODEL
from src.config.redis import REDIS_URL

class VectorStoreManager:
    def __init__(self, collection_name="multi_modal_rag", persist_dir="chroma_db"):
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
        self.docstore = RedisStore(client=Redis.from_url(REDIS_URL))        
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

        # Check if doc_id already exists in Redis
        redis_client = self.docstore.redis
        if redis_client.exists(doc_id):
            print(f"Doc_id already exists in Redis: {doc_id}")
            return

        docs = [
            Document(
                page_content=chunk,
                metadata={**metadata, "doc_id": doc_id}
            ) for chunk in chunks
        ]

        self.retriever.add_documents(documents=docs)
        print(f"Stored {len(docs)} chunks under doc_id={doc_id}")

    def add_documents(self, docs: list[Document]):
        self.vectorstore.add_documents(documents=docs)

    def get_vectorstore(self):
        return self.vectorstore

    def get_docstore(self):
        return self.docstore

    def get_retriever(self):
        return self.retriever
