import faiss
import numpy as np
import pickle

class FaissIndexer:
    def __init__(self, dimension=128, nlist=100):
        self.dimension = dimension
        self.index = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(dimension),
            dimension, nlist, faiss.METRIC_L2
        )
        self.is_trained = False

    def train(self, embeddings):
        if not self.is_trained:
            np_embeddings = np.array(embeddings).astype('float32')
            self.index.train(np_embeddings)
            self.is_trained = True

    def add(self, embeddings):
        np_embeddings = np.array(embeddings).astype('float32')
        self.index.add(np_embeddings)

    def search(self, query_embeddings, k=5):
        distances, indices = self.index.search(
            np.array(query_embeddings).astype('float32'), 
            k
        )
        return indices

    def save(self, path):
        faiss.write_index(self.index, path)

    def load(self, path):
        self.index = faiss.read_index(path)
        self.is_trained = True