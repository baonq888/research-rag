from langchain.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.models import (
    LLM_MODEL,
    IMAGE_MODEL
)
from config.prompts import (
    SUMMARY_PROMPT
)

class Summarizer:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)

        self.text_model = ChatOllama(model=LLM_MODEL)
        self.image_model = ChatOllama(model=IMAGE_MODEL)

        self.text_chain = {"element": lambda x: x} | self.prompt | self.text_model | StrOutputParser()
        self.image_chain = {"element": lambda x: x} | self.prompt | self.image_model | StrOutputParser()

    def summarize_text(self, texts, concurrency=3):
        return self.text_chain.batch(texts, {"max_concurrency": concurrency})

    def summarize_tables(self, tables_html, concurrency=3):
        return self.summarize_text(tables_html, concurrency)

    def summarize_images(self, images_b64, concurrency=2):
        return self.image_chain.batch(images_b64, {"max_concurrency": concurrency})

    def summarize_all(self, texts, tables, images_b64):
        return {
            "texts": self.summarize_text(texts),
            "tables": self.summarize_tables([t.metadata.text_as_html for t in tables]),
            "images": self.summarize_images(images_b64),
        }