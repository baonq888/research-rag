SUMMARY_PROMPT = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Content:
{element}
"""

IMAGE_SUMMARY_PROMPT = """
Describe the image in detail.
"""

QA_PROMPT = """
You are a helpful assistant.

Answer the question based only on the context provided.

Context:
{context}

Question: {question}

Answer:
"""

