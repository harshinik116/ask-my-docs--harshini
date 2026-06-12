import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from backend.config import CHROMA_PATH

embedding_fn = DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_fn
)


def add_chunks_to_vector_db(chunks, document_name: str):

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{document_name}_{i}"],
            documents=[chunk["text"]],
            metadatas=[{
                "document": document_name,
                "page": chunk["page"]
            }]
        )


def search_documents(query: str, top_k: int = 5):

    n = min(top_k, collection.count())
    if n == 0:
        return {"documents": [[]], "metadatas": [[]]}

    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    return results
