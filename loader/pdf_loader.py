
from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document
from typing import List
from config.unstructured import (
    CHUNKING_STRATEGY,
    MAX_CHARS,
    COMBINE_UNDER,
    NEW_AFTER,
    COMPOSITE_BLOCK_TYPE,
    TABLE_BLOCK_TYPE,
    IMAGE_BLOCK_TYPES,
)

class UnstructuredPDFLoader:
    def __init__(
        self,
        file_path: str,
        image_output_dir: str = None,
        chunking_strategy: str = CHUNKING_STRATEGY,
        max_characters: int = MAX_CHARS,
        combine_text_under_n_chars: int = COMBINE_UNDER,
        new_after_n_chars: int = NEW_AFTER,
    ):
        self.file_path = file_path  # Path to the PDF file to be processed
        self.image_output_dir = image_output_dir  # Directory to save extracted images (if any)
        self.chunking_strategy = chunking_strategy  # Strategy for splitting the document into chunks (e.g., 'by_title' or 'basic')
        self.max_characters = max_characters  # Maximum characters allowed per chunk
        self.combine_text_under_n_chars = combine_text_under_n_chars  # Combine chunks smaller than this number of characters
        self.new_after_n_chars = new_after_n_chars  # Force new chunk if content exceeds this number of characters
    
    def load_chunks(self) -> List[Document]:
        chunks = partition_pdf(
            filename=self.file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=[IMAGE_BLOCK_TYPES],
            image_output_dir_path=self.image_output_dir,
            extract_image_block_to_payload=True,
            chunking_strategy=self.chunking_strategy,
            max_characters=self.max_characters,
            combine_text_under_n_chars=self.combine_text_under_n_chars,
            new_after_n_chars=self.new_after_n_chars,
        )

        return chunks
    
    def separate_tables_and_texts_from_chunks(self, chunks):
        tables = []
        texts = []
        for chunk in chunks:
            if TABLE_BLOCK_TYPE in str(type(chunk)):
                tables.append(chunk)
            elif COMPOSITE_BLOCK_TYPE in str(type(chunk)):
                texts.append(chunk)
        return tables, texts

    def get_images_from_chunks(self, chunks):
        images_b64 = []
        for chunk in chunks:
            if COMPOSITE_BLOCK_TYPE in str(type(chunk)):
                orig_elements = chunk.metadata.orig_elements
                if orig_elements:
                    for el in orig_elements:
                        if IMAGE_BLOCK_TYPES in str(type(el)):
                            images_b64.append(el.metadata.image_base64)
        return images_b64

    def process_pdf_content(self) -> List[Document]:
        chunks = self.load_chunks()
        tables, texts = self.separate_tables_and_texts_from_chunks(chunks)
        images = self.get_images_from_chunks(texts)

        documents = [
            Document(
                page_content=str(el),
                metadata=el.metadata.to_dict() if hasattr(el, "metadata") else {}
            )
            for el in texts
        ]

        return documents, tables, images