import chromadb
from sentence_transformers import SentenceTransformer
from backend.config import CHROMA_PATH

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name="documents"
)


def get_embedding(text: str):
    return embedding_model.encode(text).tolist()


def add_chunks_to_vector_db(chunks, document_name: str):

    for i, chunk in enumerate(chunks):

        embedding = get_embedding(chunk["text"])

        collection.add(
            ids=[f"{document_name}_{i}"],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "document": document_name,
                "page": chunk["page"]
            }]
        )


def search_documents(query: str, top_k: int = 5):

    query_embedding = get_embedding(query)

    n = min(top_k, collection.count())
    if n == 0:
        return {"documents": [[]], "metadatas": [[]]}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n
    )

    return results