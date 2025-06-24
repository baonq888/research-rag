from config.models import LLM_MODEL, IMAGE_MODEL
from config.prompts import SUMMARY_PROMPT, IMAGE_SUMMARY_PROMPT
from helper.response_cleaner import ResponseCleaner
from config.client import client
from langchain.schema import Document
import uuid


class Summarizer:
    def __init__(self):
        self.model = LLM_MODEL
        self.image_model = IMAGE_MODEL
        self.prompt_template = SUMMARY_PROMPT

    def _format_prompt(self, content):
        return self.prompt_template.replace("{element}", content)

    def summarize_text(self, texts):
        results = []

        for doc in texts:
            text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            prompt = self._format_prompt(text)

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=4096,
                stream=False
            )
            raw_output = response.choices[0].message.content
            cleaned = ResponseCleaner.strip_think_block(raw_output)

            doc_id = str(uuid.uuid4())
            results.append(Document(
                page_content=cleaned,
                metadata={**doc.metadata, "doc_id": doc_id, "type": "summary"}
            ))

        return results

    def summarize_tables(self, tables):
        results = []

        for table in tables:
            html = table.metadata.text_as_html
            prompt = self._format_prompt(html)

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=4096,
                stream=False
            )
            raw_output = response.choices[0].message.content
            cleaned = ResponseCleaner.strip_think_block(raw_output)

            doc_id = str(uuid.uuid4())
            results.append(Document(
                page_content=cleaned,
                metadata={**table.metadata, "doc_id": doc_id, "type": "summary"}
            ))

        return results

    def summarize_images(self, images_b64):
        results = []

        for b64 in images_b64:
            response = client.chat.completions.create(
                model=self.image_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                max_tokens=1024,
                temperature=0.5,
                stream=False
            )

            content = response.choices[0].message.content
            cleaned = ResponseCleaner.strip_think_block(content)

            doc_id = str(uuid.uuid4())
            results.append(Document(
                page_content=cleaned,
                metadata={"doc_id": doc_id, "type": "image"}
            ))

        return results

    def summarize_all(self, texts, tables, images_b64):
        return {
            "texts": self.summarize_text(texts),
            "tables": self.summarize_tables(tables),
            "images": self.summarize_images(images_b64),
        }