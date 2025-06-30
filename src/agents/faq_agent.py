import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class FAQAgent:
    """Agent to handle FAQ queries from the local dataset using FAISS vector DB."""
    def __init__(self, faq_data, model_name='all-MiniLM-L6-v2'):
        self.faq_data = faq_data
        self.model = SentenceTransformer(model_name)
        self.questions = faq_data['question'].tolist()
        self.answers = faq_data['answer'].tolist()
        self.categories = faq_data['category'].tolist()
        self.index = self._index_faqs()

    def _index_faqs(self):
        # Compute embeddings for all questions
        embeddings = self.model.encode(self.questions, show_progress_bar=True, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype('float32')
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def retrieve_answer(self, question: str, top_k: int = 1, threshold: float = 0.6) -> str:
        query_emb = self.model.encode([question], convert_to_numpy=True)
        query_emb = np.array(query_emb).astype('float32')
        D, I = self.index.search(query_emb, top_k)
        idx = int(I[0][0])
        score = float(D[0][0])
        # Lower L2 distance means higher similarity; threshold is for filtering
        if score > (1 - threshold):  # Adjust threshold as needed
            return "[FAQAgent] Sorry, I couldn't find a relevant FAQ answer."
        return self.answers[idx] 