"""
retrieval.py  –  Semantic search over the vector store
-------------------------------------------------------
Fixed: original imported non-existent bare `model` from embeddings.py.
Now correctly uses EmbeddingModel class.
"""

from embeddings import EmbeddingModel

# Single shared instance – model weights loaded once, reused on every query
_embedding_model = EmbeddingModel()


def retrieve_relevant_chunks(vector_store, query: str, top_k: int = 3) -> list:
    """
    Encode `query` and return the top_k most relevant text chunks
    from the vector store.

    Parameters
    ----------
    vector_store : VectorStore   – a populated VectorStore instance
    query        : str           – natural-language question
    top_k        : int           – number of results (default 3)

    Returns
    -------
    list[str]  –  most relevant text chunks, ranked by similarity
    """
    # create_embeddings returns a 2-D array; [0] gives the single query vector
    query_embedding = _embedding_model.create_embeddings([query])[0]
    results = vector_store.search(query_embedding, top_k)
    return results