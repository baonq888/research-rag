
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from config.client import client
from config.models import LLM_MODEL
from config.prompts import QA_PROMPT
from helper.response_cleaner import ResponseCleaner
from retrieval.retriever import Retriever


class Generation:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.client = client
        self.model_name = LLM_MODEL

    def build_answer_prompt(self, question: str, docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in docs])
        return QA_PROMPT.format(context=context, question=question)

    def answer(self, query: str, metadata_filter: dict) -> str:
        # Step 1: Retrieve top-k documents
        top_k_results = self.retriever.retrieve(query, metadata_filter)
        if not top_k_results:
            return "No relevant context found."

        # Build QA prompt
        prompt = self.build_answer_prompt(query, top_k_results)

        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.5,
                stream=False
            )
            raw_output = response.choices[0].message.content.strip()
            # Clean the output
            cleaned = ResponseCleaner.strip_think_block(raw_output)
            
            return cleaned

        except Exception as e:
            return f"LLM error during answer generation: {str(e)}"