import os
from typing import Dict
from dotenv import load_dotenv

from retrieval.metadata_filter import MetadataFilterExtractor
from generator.generation import Generation
from retrieval.retriever import Retriever
from retrieval.retriever_summary import SummaryRetriever

from helper.pdf_utils import (
    load_pdf,
    summarize_content,
    initialize_vector_stores,
    persist_to_docstore,
    initialize_retrievers,
)

load_dotenv()

class QAService:
    def __init__(self):
        self.full_store = None
        self.summary_store = None
        self.detail_retriever: Retriever = None
        self.summary_retriever: SummaryRetriever = None

    def load_and_index_pdf(self, file_path: str, image_output_dir: str = "./data") -> Dict[str, int]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load and chunk
        pdf_loader = load_pdf(file_path, image_output_dir)
        full_texts, full_tables, images_b64 = pdf_loader.process_pdf_content()

        # Summarize
        summary_results = summarize_content(full_texts, full_tables, images_b64)
        summarized_texts = summary_results["texts"]
        summarized_tables = summary_results["tables"]
        summarized_images = summary_results["images"]

        # Combine documents
        full_docs = full_texts + full_tables
        summary_docs = summarized_texts + summarized_tables + summarized_images

        # Init vector stores and persist
        self.full_store, self.summary_store = initialize_vector_stores(full_docs, summary_docs)
        persist_to_docstore(self.full_store, full_docs)
        persist_to_docstore(self.summary_store, summary_docs)

        # Init retrievers
        self.detail_retriever, self.summary_retriever = initialize_retrievers(
            full_store=self.full_store,
            summary_store=self.summary_store,
            pdf_loader=pdf_loader
        )

        return {
            "texts": len(full_texts),
            "tables": len(full_tables),
            "images": len(summarized_images),
        }

    def answer_query(self, query: str) -> Dict:
        if self.detail_retriever is None:
            raise RuntimeError("PDF not loaded. Call load_and_index_pdf() first.")

        metadata_filter = MetadataFilterExtractor().extract(query)
        generator = Generation(retriever=self.detail_retriever)
        answer = generator.answer(query, metadata_filter)

        return {
            "query": query,
            "filter": metadata_filter,
            "answer": answer,
        }