from groq import Groq
from backend.retrieval import search_documents
import os


def answer_question(query: str):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    results = search_documents(query)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = ""

    for i, doc in enumerate(documents):

        source = metadatas[i]

        context += f"""
Source {i+1}
Document: {source['document']}
Page: {source['page']}

{doc}

"""

    prompt = f"""
You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not found, say:
"The answer is not found in the uploaded documents."

Always include citations.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content