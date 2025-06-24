import json
import base64
import os
import uuid
from dotenv import load_dotenv
from IPython.display import Image, display
from retrieval.metadata_filter import MetadataFilterExtractor
from generator.generation import Generation
from loader.pdf_loader import UnstructuredPDFLoader
from loader.summarizer import Summarizer
from index.vector_store import VectorStoreManager
from retrieval.retriever import Retriever
from langchain.schema import Document

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
    image_summaries = summarizer.summarize_images(images_b64)

    # Setup retriever
    store = VectorStoreManager()
    retriever = store.retriever

    # Generate IDs
    image_ids = [str(uuid.uuid4()) for _ in images_b64]

    # Prepare documents
    summary_images = [
        Document(page_content=summary, metadata={"doc_id": image_ids[i], "type": "image"})
        for i, summary in enumerate(image_summaries)
    ]


    # Add all to vector DB
    retriever.vectorstore.add_documents(summary_images + texts + tables)

    # --- Serialization Utilities ---
    def serialize_doc(doc: Document):
        return json.dumps({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })

    # Store in Redis (docstore)
    store.docstore.mset([
        (doc.metadata["doc_id"], serialize_doc(doc)) for doc in (texts + tables + summary_images)
    ])

    print(f"\nStored {len(texts)} text, {len(tables)} table, and {len(summary_images)} image summaries.")

    # --- QA Test ---
    print("\nTesting QA:")
    test_retriever = Retriever(
        vectorstore=store.get_vectorstore(),
        docstore=store.get_docstore(),
        embedding_function=store.embedding_model
    )

    query = "Tell me more about the image of the attention model."

    #  Metadata Filter Extraction 
    print("\nExtracting metadata filter...")

    filter_extractor = MetadataFilterExtractor()
    metadata_filter = filter_extractor.extract(query)
    print("Generated Metadata Filter:", metadata_filter)


    # Run QA
    generator = Generation(retriever=test_retriever)
    print("\nLLM Answer via Generation class:")
    answer = generator.answer(query, metadata_filter)
    print(answer)


if __name__ == "__main__":
    main()