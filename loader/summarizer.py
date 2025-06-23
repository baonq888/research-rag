from config.models import LLM_MODEL, IMAGE_MODEL
from config.prompts import (SUMMARY_PROMPT, IMAGE_SUMMARY_PROMPT)
from helper.response_cleaner import ResponseCleaner
from config.client import client



class Summarizer:
    def __init__(self):
        self.model = LLM_MODEL
        self.image_model = IMAGE_MODEL
        self.prompt_template = SUMMARY_PROMPT

    def _format_prompt(self, content):
        return self.prompt_template.replace("{element}", content)

    def summarize_text(self, texts, concurrency=3):
        results = []

        for doc in texts:
            # Safely extract string content
            text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            

            prompt = self._format_prompt(text)


            response = client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                temperature=0.5,
                max_tokens=4096,
                stop=None,
                stream=False
            )
            raw_output = response.choices[0].message.content

            # Clean the output
            cleaned = ResponseCleaner.strip_think_block(raw_output)

            results.append(cleaned)

        return results

    def summarize_tables(self, tables_html, concurrency=3):
        return self.summarize_text(tables_html, concurrency)

    def summarize_images(self, images_b64, concurrency=2):
        results = []

        for b64 in images_b64:
            response = client.chat.completions.create(
                model=self.image_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }
                        }
                    ]
                }],
                max_tokens=1024,
                temperature=0.5,
                stream=False
            )

            content = response.choices[0].message.content
            cleaned = ResponseCleaner.strip_think_block(content)
            results.append(cleaned)

        return results

    def summarize_all(self, texts, tables, images_b64):
        return {
            "texts": self.summarize_text(texts),
            "tables": self.summarize_tables([t.metadata.text_as_html for t in tables]),
            "images": self.summarize_images(images_b64),
        }