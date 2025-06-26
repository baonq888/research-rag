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

SECTION_CLASSIFY_PROMPT = (
    "You are a section classifier. Given the following PDF text chunk, classify it "
    "into one of the following section labels, choose the closest labels as possible:\n\n"
    "{labels}\n\n"
    "Respond with ONLY the section name.\n\n"
    "Text:\n\"\"\"\n{input}\n\"\"\"\n\n"
    "Section:"
)

FILTER_INSTRUCTION_PROMPT = """
You are part of an information system that processes users queries.
Given a user query you extract information from it that matches a given list of metadata fields.
The information to be extracted from the query must match the semantics associated with the given metadata fields.
The information that you extracted from the query will then be used as filters to narrow down the search space
when querying an index.
Just include the value of the extracted metadata without including the name of the metadata field.
The extracted information in 'Extracted metadata' must be returned as a valid JSON structure.

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

Eg..:
Input Query "What is the introduction?" -> {{"section": "introduction", "type": "text"}}
Input Query "Tell me about the methodology." -> {{"section": "methodology", "type": "text"}}
Input Query "Show me the results." -> {{"section": "results"}}
Input Query "Where can I find the references?" -> {{"section": "reference"}}
Input Query "Summarize the conclusion." -> {{"section": "conclusion", "type": "text"}}


**Input Query**:
"{query}"

Output:
Respond ONLY with a valid JSON object using the supported fields.
Example: {{ "section": "results", "type": "table" }}
"""

LLM_SECTION_EXTRACTION_PROMPT = """Extract all the section titles explicitly or implicitly mentioned in this query.
Return them as a Python list of lowercase strings. Do not explain.

Query: "{query}"
Answer:"""

