import json
import os
from dotenv import load_dotenv

from retrieval.metadata_filter import MetadataFilterExtractor
from generator.generation import Generation
from loader.pdf_loader import UnstructuredPDFLoader
from loader.summarizer import Summarizer
from index.vector_store import VectorStoreManager
from retrieval.retriever import Retriever
from retrieval.retriever_summary import SummaryRetriever  
from langchain.schema import Document

load_dotenv()

def main():
    pdf_path = "./data/attention.pdf"
    image_output_dir = "./data"

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Load and chunk PDF
    loader = UnstructuredPDFLoader(
        file_path=pdf_path,
        image_output_dir=image_output_dir,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    full_texts, full_tables, images_b64 = loader.process_pdf_content()

    # Summarize content
    summarizer = Summarizer()
    print("\nSummarizing all content...")
    summary_results = summarizer.summarize_all(full_texts, full_tables, images_b64)

    summarized_texts = summary_results["texts"]
    summarized_tables = summary_results["tables"]
    summarized_images = summary_results["images"]

    # Create separate vector stores for full and summary content
    full_store = VectorStoreManager(collection_name="full_content")
    summary_store = VectorStoreManager(collection_name="summary_content")

    full_docs = full_texts + full_tables
    summary_docs = summarized_texts + summarized_tables + summarized_images

    # Add documents to their respective stores
    full_store.add_documents(full_docs)
    summary_store.add_documents(summary_docs)

    # Shared Redis docstore for both
    def serialize_doc(doc: Document):
        return json.dumps({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })

    full_store.docstore.mset([
        (doc.metadata["doc_id"], serialize_doc(doc)) for doc in full_docs
    ])
    summary_store.docstore.mset([
        (doc.metadata["doc_id"], serialize_doc(doc)) for doc in summary_docs
    ])

    print(f"\nStored:")
    print(f" - Full texts: {len(full_texts)}")
    print(f" - Full tables: {len(full_tables)}")
    print(f" - Summarized texts: {len(summarized_texts)}")
    print(f" - Summarized tables: {len(summarized_tables)}")
    print(f" - Summarized images: {len(summarized_images)}")

    # --- QA Demo ---
    print("\nTesting QA:")

    # Create SummaryRetriever
    summary_retriever = SummaryRetriever(
        vectorstore=summary_store.get_vectorstore(),
        docstore=summary_store.get_docstore(),
        embedding_function=summary_store.embedding_model
    )

    # Main Retriever with fallback + delegation
    test_retriever = Retriever(
        vectorstore=full_store.get_vectorstore(),
        docstore=full_store.get_docstore(),
        embedding_function=full_store.embedding_model,
        summary_retriever=summary_retriever
    )

    query = "Tell me more about the image of the attention model."

    # Metadata Filtering
    filter_extractor = MetadataFilterExtractor()
    metadata_filter = filter_extractor.extract(query)
    print("Generated Metadata Filter:", metadata_filter)

    # Generate answer
    generator = Generation(retriever=test_retriever)
    print("\nLLM Answer via Generation class:")
    answer = generator.answer(query, metadata_filter)
    print(answer)

if __name__ == "__main__":
    main()