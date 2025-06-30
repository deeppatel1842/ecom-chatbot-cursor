import openai
from typing import Optional

class RAGPipeline:
    """Retrieval-Augmented Generation (RAG) pipeline for answer generation."""
    def __init__(self, llm_model: str = "gpt-3.5-turbo", openai_api_key: Optional[str] = None):
        self.llm_model = llm_model
        if openai_api_key:
            openai.api_key = openai_api_key

    def generate_answer(self, user_query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate an answer using the LLM, given the user query and retrieved context.
        :param user_query: The user's question
        :param context: Retrieved context (from FAQ, product, or policy)
        :param system_prompt: Optional system prompt for LLM
        :return: Generated answer string
        """
        if not context:
            context = "No relevant context found."
        prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=messages,
                max_tokens=512,
                temperature=0.2,
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"[RAGPipeline] Error generating answer: {e}" 