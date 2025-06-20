import json
from loader.pdf_loader import UnstructuredPDFLoader
from loader.summarizer import Summarizer
from index.vector_store import VectorStoreManager
from langchain.schema import Document
import base64, os, uuid
from IPython.display import Image, display
from dotenv import load_dotenv

load_dotenv()


def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))


def main():
    pdf_path = "./data/attention.pdf"
    image_output_dir = "./data"

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Load content
    loader = UnstructuredPDFLoader(
        file_path=pdf_path,
        image_output_dir=image_output_dir,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    texts, tables, images_b64 = loader.process_pdf_content()

    # Summarization
    summarizer = Summarizer()
    print("\nSummarizing all content...")
    summaries = summarizer.summarize_all(texts, tables, images_b64)
    text_summaries = summaries["texts"]
    table_summaries = summaries["tables"]
    image_summaries = summaries["images"]

    # Setup retriever
    store = VectorStoreManager()
    retriever = store.retriever

    # Generate IDs
    text_ids = [str(uuid.uuid4()) for _ in texts]
    table_ids = [str(uuid.uuid4()) for _ in tables]
    image_ids = [str(uuid.uuid4()) for _ in images_b64]

    # Prepare documents
    summary_texts = [
        Document(page_content=summary, metadata={"doc_id": text_ids[i], "type": "text"})
        for i, summary in enumerate(text_summaries)
    ]
    summary_tables = [
        Document(page_content=summary, metadata={"doc_id": table_ids[i], "type": "table"})
        for i, summary in enumerate(table_summaries)
    ]
    summary_images = [
        Document(page_content=summary, metadata={"doc_id": image_ids[i], "type": "image"})
        for i, summary in enumerate(image_summaries)
    ]

    # Store in vector DB and doc store
    retriever.vectorstore.add_documents(summary_texts + summary_tables + summary_images)

    def serialize_document(doc: Document):
        return json.dumps({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })

    def serialize_table(table):
        return json.dumps({
            "page_content": table.page_content if hasattr(table, "page_content") else str(table),
            "metadata": table.metadata.to_dict() if hasattr(table.metadata, "to_dict") else table.metadata
        })

    # Store original texts as JSON
    retriever.docstore.mset([
        (doc_id, serialize_document(doc)) for doc_id, doc in zip(text_ids, texts)
    ])

    # Store tables as JSON
    retriever.docstore.mset([
        (doc_id, serialize_table(tbl)) for doc_id, tbl in zip(table_ids, tables)
    ])

    # Store base64 images as raw strings (they're already serializable)
    retriever.docstore.mset([
        (doc_id, json.dumps({"image_b64": b64})) for doc_id, b64 in zip(image_ids, images_b64)
    ])

    print(f"\nStored {len(summary_texts)} text, {len(summary_tables)} table, and {len(summary_images)} image summaries.")


if __name__ == "__main__":
    main()