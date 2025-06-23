import uuid
from typing import List

from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf

from config.models import ZERO_SHOT_MODEL
from config.unstructured import (
    CHUNKING_STRATEGY,
    MAX_CHARS,
    COMBINE_UNDER,
    NEW_AFTER,
    COMPOSITE_BLOCK_TYPE,
    TABLE_BLOCK_TYPE,
    IMAGE_BLOCK_TYPES,
)

from transformers import pipeline

classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL)
SECTION_LABELS = [
    "introduction",
    "background",
    "overview",
    "methodology",
    "process",
    "results",
    "analysis",
    "discussion",
    "conclusion",
    "summary",
    "table",
    "figure",
    "reference",
    "appendix",
    "other"
]

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

    def classify_section(self, text: str) -> str:
        try:
            result = classifier(text[:512], candidate_labels=SECTION_LABELS)
            return result["labels"][0]
        except Exception as e:
            print("Classification error:", e)
            return "other"

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

        def get_metadata(el, content_type):
            md = el.metadata.to_dict() if hasattr(el, "metadata") else {}
            section = md.get("section", "").strip().lower()

            # Auto-classify section if missing
            if not section and content_type == "text":
                section = self.classify_section(str(el))

            return {
                "doc_id": str(uuid.uuid4()),
                "type": content_type,
                "section": section,
                "heading": md.get("heading", "").strip().lower(),
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