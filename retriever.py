# retriever.py
import faiss
from sentence_transformers import SentenceTransformer

class CustomRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatL2(self.embedder.get_sentence_embedding_dimension())
        self.documents = []

    def add_documents(self, docs):
        embeddings = self.embedder.encode(docs)
        self.index.add(embeddings)
        self.documents.extend(docs)

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]
