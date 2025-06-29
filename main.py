from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.main_route import router as main_router  

app = FastAPI(
    title="PDF QA API",
    description="Upload a PDF and ask questions about its content",
    version="1.0.0"
)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(main_router)

# uvicorn main:app --reload --port 8000




# import json
# import os
# from dotenv import load_dotenv

# from retrieval.metadata_filter import MetadataFilterExtractor
# from generator.generation import Generation
# from loader.pdf_loader import UnstructuredPDFLoader
# from loader.summarizer import Summarizer
# from index.vector_store import VectorStoreManager
# from retrieval.retriever import Retriever
# from retrieval.retriever_summary import SummaryRetriever
# from langchain.schema import Document

# load_dotenv()

# def main():
#     pdf_path = "./data/attention.pdf"
#     image_output_dir = "./data"

#     if not os.path.exists(pdf_path):
#         print(f"File not found: {pdf_path}")
#         return

#     # Load and chunk PDF
#     pdf_loader = UnstructuredPDFLoader(
#         file_path=pdf_path,
#         image_output_dir=image_output_dir,
#         chunking_strategy="by_title",
#         max_characters=10000,
#         combine_text_under_n_chars=2000,
#         new_after_n_chars=6000,
#     )
#     full_texts, full_tables, images_b64 = pdf_loader.process_pdf_content()

#     # Summarize content
#     summarizer = Summarizer()
#     print("\nSummarizing all content...")
#     summary_results = summarizer.summarize_all(full_texts, full_tables, images_b64)

#     summarized_texts = summary_results["texts"]
#     summarized_tables = summary_results["tables"]
#     summarized_images = summary_results["images"]

#     # Create separate vector stores for full and summary content
#     full_store = VectorStoreManager(collection_name="full_content")
#     summary_store = VectorStoreManager(collection_name="summary_content")

#     full_docs = full_texts + full_tables
#     summary_docs = summarized_texts + summarized_tables + summarized_images

#     # Add documents to their respective stores
#     full_store.add_documents(full_docs)
#     summary_store.add_documents(summary_docs)

#     def serialize_doc(doc: Document):
#         return json.dumps({
#             "page_content": doc.page_content,
#             "metadata": doc.metadata
#         })

#     full_store.docstore.mset([
#         (doc.metadata["doc_id"], serialize_doc(doc)) for doc in full_docs
#     ])
#     summary_store.docstore.mset([
#         (doc.metadata["doc_id"], serialize_doc(doc)) for doc in summary_docs
#     ])

#     # --- QA Demo ---
#     print("\nTesting QA:")

#     # Create SummaryRetriever with access to the PDF loader
#     summary_retriever = SummaryRetriever(
#         vectorstore=summary_store.get_vectorstore(),
#         docstore=summary_store.get_docstore(),
#         embedding_function=summary_store.embedding_model,
#         pdf_loader=pdf_loader  
#     )

#     # Full content retriever with summary fallback
#     detail_retriever = Retriever(
#         vectorstore=full_store.get_vectorstore(),
#         docstore=full_store.get_docstore(),
#         embedding_function=full_store.embedding_model,
#         summary_retriever=summary_retriever
#     )

#     query = "Who is the author of this paper?"

#     filter_extractor = MetadataFilterExtractor()
#     metadata_filter = filter_extractor.extract(query)
#     print("Generated Metadata Filter:", metadata_filter)

#     # Generate the answer
#     generator = Generation(retriever=detail_retriever)
#     print("\nLLM Answer via Generation class:")
#     answer = generator.answer(query, metadata_filter)
#     print(answer)

# if __name__ == "__main__":
#     main()