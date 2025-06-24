import json, uuid
from loader.pdf_loader import UnstructuredPDFLoader
from loader.summarizer import Summarizer
from index.vector_store import VectorStoreManager
from retrieval.retriever import Retriever
from generator.generation import Generation
from retrieval.metadata_filter import MetadataFilterExtractor
from langchain.schema import Document

def load_and_index_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path, image_output_dir="./data")
    texts, tables, images_b64 = loader.process_pdf_content()

    summarizer = Summarizer()
    image_summaries = summarizer.summarize_images(images_b64)

    image_ids = [str(uuid.uuid4()) for _ in image_summaries]
    image_docs = [
        Document(page_content=summary, metadata={"doc_id": image_ids[i], "type": "image"})
        for i, summary in enumerate(image_summaries)
    ]

    store = VectorStoreManager()
    store.retriever.vectorstore.add_documents(image_docs + texts + tables)

    def serialize(doc: Document):
        return json.dumps({"page_content": doc.page_content, "metadata": doc.metadata})

    store.docstore.mset([
        (doc.metadata["doc_id"], serialize(doc)) for doc in (texts + tables + image_docs)
    ])

    return {"texts": len(texts), "tables": len(tables), "images": len(image_docs)}

def answer_query(query: str):
    store = VectorStoreManager()
    retriever = Retriever(
        vectorstore=store.get_vectorstore(),
        docstore=store.get_docstore(),
        embedding_function=store.embedding_model
    )
    metadata_filter = MetadataFilterExtractor().extract(query)


    generator = Generation(retriever=retriever)
    answer = generator.answer(query, metadata_filter)

    return {
        "query": query,
        "filter": metadata_filter,
        "answer": answer
    }