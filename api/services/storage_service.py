from langchain.schema import Document
import json

def store_documents_in_vector_and_docstore(
    texts, tables, images_b64,
    text_ids, table_ids, image_ids,
    image_summaries,
    retriever
):
    summary_images = [
        Document(page_content=summary, metadata={"doc_id": image_ids[i], "type": "image"})
        for i, summary in enumerate(image_summaries)
    ]

    full_texts = [
        Document(
            page_content=doc.page_content,
            metadata={"doc_id": text_ids[i], "type": "full", "source": "text"}
        )
        for i, doc in enumerate(texts)
    ]

    full_tables = [
        Document(
            page_content=str(tbl),
            metadata={"doc_id": table_ids[i], "type": "full", "source": "table"}
        )
        for i, tbl in enumerate(tables)
    ]

    # Store in vector DB
    retriever.vectorstore.add_documents(summary_images + full_texts + full_tables)

    # JSON serialization functions
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

    # Store in Redis
    retriever.docstore.mset([
        (doc_id, serialize_document(doc)) for doc_id, doc in zip(text_ids, texts)
    ])
    retriever.docstore.mset([
        (doc_id, serialize_table(tbl)) for doc_id, tbl in zip(table_ids, tables)
    ])
    retriever.docstore.mset([
        (doc_id, json.dumps({"image_b64": b64})) for doc_id, b64 in zip(image_ids, images_b64)
    ])

    return {
        "texts": len(full_texts),
        "tables": len(full_tables),
        "images": len(summary_images)
    }