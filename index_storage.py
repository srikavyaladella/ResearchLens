import faiss
import numpy as np
import pickle


class VectorStore:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.text_chunks = []

    def add_embeddings(self, embeddings, chunks):

        embeddings = np.array(embeddings).astype('float32')

        self.index.add(embeddings)

        self.text_chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):

        query_embedding = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for idx in indices[0]:

            if idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results

    def save(self, path):

        faiss.write_index(self.index, f"{path}/faiss.index")

        with open(f"{path}/chunks.pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)

    def load(self, path):

        self.index = faiss.read_index(f"{path}/faiss.index")

        with open(f"{path}/chunks.pkl", "rb") as f:
            self.text_chunks = pickle.load(f)