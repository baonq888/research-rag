import os
import json
from typing import List, Dict, Tuple
from langchain.schema import Document

from src.loader.pdf_loader import UnstructuredPDFLoader
from src.loader.summarizer import Summarizer
from src.index.vector_store import VectorStoreManager
from src.retrieval.retriever import Retriever
from src.retrieval.retriever_summary import SummaryRetriever


def load_pdf(file_path: str, image_output_dir: str = "./data") -> UnstructuredPDFLoader:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return UnstructuredPDFLoader(
        file_path=file_path,
        image_output_dir=image_output_dir,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )


def summarize_content(texts: List[Document], tables: List[Document], images_b64: List[str]) -> Dict[str, List[Document]]:
    summarizer = Summarizer()
    return summarizer.summarize_all(texts, tables, images_b64)


def initialize_vector_stores(
    full_docs: List[Document],
    summary_docs: List[Document]
) -> Tuple[VectorStoreManager, VectorStoreManager]:
    full_store = VectorStoreManager(collection_name="full_content")
    summary_store = VectorStoreManager(collection_name="summary_content")

    full_store.add_documents(full_docs)
    summary_store.add_documents(summary_docs)

    return full_store, summary_store


def persist_to_docstore(store, documents: List[Document]):
    def serialize(doc: Document):
        return json.dumps({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })

    store.docstore.mset([
        (doc.metadata["doc_id"], serialize(doc)) for doc in documents
    ])


def initialize_retrievers(
    full_store: VectorStoreManager,
    summary_store: VectorStoreManager,
    pdf_loader: UnstructuredPDFLoader
) -> Tuple[Retriever, SummaryRetriever]:
    summary_retriever = SummaryRetriever(
        vectorstore=summary_store.get_vectorstore(),
        docstore=summary_store.get_docstore(),
        embedding_function=summary_store.embedding_model,
        pdf_loader=pdf_loader
    )

    detail_retriever = Retriever(
        vectorstore=full_store.get_vectorstore(),
        docstore=full_store.get_docstore(),
        embedding_function=full_store.embedding_model,
        summary_retriever=summary_retriever
    )

    return detail_retriever, summary_retriever