import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

class GloVeRetriever:
    def __init__(self):
        self.glove = KeyedVectors.load_word2vec_format(
            MODEL_PATHS["glove"], binary=False)
        self.doc_vectors = self._precompute_vectors()

    def _precompute_vectors(self):
        # Precompute document vectors during initialization
        vectors = []
        for doc_text in self.documents.values():
            words = [word for word in doc_text.split() if word in self.glove]
            if len(words) == 0:
                vectors.append(np.zeros(300))
            else:
                vectors.append(np.mean([self.glove[word] for word in words], axis=0))
        return np.array(vectors)

    def retrieve(self, question, top_k=5):
        # Convert question to vector
        q_words = [word for word in question.split() if word in self.glove]
        if not q_words:
            return []
        q_vector = np.mean([self.glove[word] for word in q_words], axis=0)
        
        # Calculate similarities
        similarities = cosine_similarity([q_vector], self.doc_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.doc_ids[i] for i in top_indices]