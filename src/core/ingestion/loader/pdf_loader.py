import re
import uuid
from typing import List
from src.config.client import client
from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
from src.config.unstructured import (
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
        self.file_path = file_path
        self.image_output_dir = image_output_dir
        self.chunking_strategy = chunking_strategy
        self.max_characters = max_characters
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.new_after_n_chars = new_after_n_chars
        self.section_titles = set()


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

    
    
    def process_pdf_content(self):
        chunks = self.load_chunks()
        tables_raw, texts_raw = self.separate_tables_and_texts_from_chunks(chunks)

        def clean_section_title(raw_title: str) -> str:
            # Remove leading numbers, dots, dashes, and colons (e.g., "1.", "1.1:", "2-")
            return re.sub(r"^\s*[\d\W_]+", "", raw_title).strip().lower()
        
        def get_metadata(el, content_type):
            md = el.metadata.to_dict() if hasattr(el, "metadata") else {}

            # Default section (fallback)
            section_title = md.get("section", "").strip().lower()

            # Extract title from orig_elements 
            if hasattr(el.metadata, "orig_elements"):
                orig_elements = el.metadata.orig_elements or []
                for orig in orig_elements:
                    if getattr(orig, "category", None) == "Title":
                        raw_title = orig.text.strip()
                        section_title = clean_section_title(raw_title)
                        break  # only use the first matching title

            # Add to section title list if valid
            if section_title and len(section_title) > 3:
                self.section_titles.add(section_title)

            return {
                "doc_id": str(uuid.uuid4()),
                "type": content_type,
                "heading": md.get("heading", "").strip().lower(),
                "section": section_title,
                "page_number": md.get("page_number", -1),
                "element_id": md.get("id", ""),
                "parent_id": md.get("parent_id", ""),
            }

        images_b64 = self.get_images_from_chunks(texts_raw)

        text_docs = [
            Document(
                page_content=str(el),
                metadata=get_metadata(el, "text")
            )
            for el in texts_raw
        ]

        table_docs = [
            Document(
                page_content=el.metadata.text_as_html,
                metadata=get_metadata(el, "table")
            )
            for el in tables_raw
        ]



        return text_docs, table_docs, images_b64
    
    def get_extracted_section_titles(self):
        return list(self.section_titles)