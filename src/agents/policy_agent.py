from sentence_transformers import SentenceTransformer, util
import numpy as np

class PolicyAgent:
    """Agent to handle policy, terms, and returns queries from local documents."""
    def __init__(self, policy_chunks, model_name='all-MiniLM-L6-v2'):
        self.policy_chunks = policy_chunks
        self.model = SentenceTransformer(model_name)
        self.contents = [chunk['content'] for chunk in policy_chunks]
        self.documents = [chunk['document'] for chunk in policy_chunks]
        # Precompute embeddings for all policy chunks
        self.chunk_embeddings = self.model.encode(self.contents, convert_to_tensor=True)

    def answer_policy_query(self, question: str, top_k: int = 1, threshold: float = 0.5) -> str:
        # Embed the user question
        query_embedding = self.model.encode(question, convert_to_tensor=True)
        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(query_embedding, self.chunk_embeddings)[0].cpu().numpy()
        # Get top match
        top_idx = int(np.argmax(similarities))
        top_score = similarities[top_idx]
        if top_score < threshold:
            return "[PolicyAgent] Sorry, I couldn't find a relevant policy section."
        return self.contents[top_idx] 