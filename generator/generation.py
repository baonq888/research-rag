import json
import os
from typing import List
from dotenv import load_dotenv
from langchain.schema import Document
from together import Together
from config.models import LLM_MODEL
from config.prompts import QA_PROMPT
from helper.response_cleaner import ResponseCleaner


load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

client = Together(api_key=api_key)

class Generation:
    def __init__(self, retriever):
        self.retriever = retriever
        self.client = client
        self.model_name = LLM_MODEL

    def build_answer_prompt(self, question: str, docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in docs])
        return QA_PROMPT.format(context=context, question=question)

    def answer(self, query: str) -> str:
        # Step 1: Retrieve top-k documents
        top_k_results = self.retriever.retrieve(query)
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