from sentence_transformers import SentenceTransformer, util
import numpy as np

class ProductAgent:
    """Agent to handle product search and recommendations from the local catalog."""
    def __init__(self, product_data, model_name='all-MiniLM-L6-v2'):
        self.product_data = product_data
        self.model = SentenceTransformer(model_name)
        self.search_texts = product_data['search_text'].tolist()
        self.product_ids = product_data['product_id'].tolist()
        self.names = product_data['name'].tolist()
        self.descriptions = product_data['description'].tolist()
        # Precompute embeddings for all products
        self.product_embeddings = self.model.encode(self.search_texts, convert_to_tensor=True)

    def search_product(self, query: str, top_k: int = 3, threshold: float = 0.5) -> str:
        # Embed the user query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # Compute cosine similarities
        similarities = util.pytorch_cos_sim(query_embedding, self.product_embeddings)[0].cpu().numpy()
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] < threshold:
                continue
            results.append(f"{self.names[idx]}: {self.descriptions[idx]}")
        if not results:
            return "[ProductAgent] Sorry, I couldn't find any relevant products."
        return '\n\n'.join(results) 