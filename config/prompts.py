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

FILTER_INSTRUCTION_PROMPT = """
You are a metadata extraction assistant.

Your task is to extract a JSON filter dictionary based on the provided Input Query that can be used to narrow down the search results based on metadata.

Only include fields if they are clearly implied by the query.

Supported fields:
- "section": sections in 
[   "introduction",
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
    "other" ]
- "type": One of "text", "table", or "image"
- "heading": Specific subsection heading if available (optional)

Eg..: Input Query "What is the introduction?" -> {{"section": "introduction", "type": "text"}}

**Input Query**:
"{query}"

Output:
Respond ONLY with a valid JSON object using the supported fields.
Example: {{ "section": "results", "type": "table" }}
"""

