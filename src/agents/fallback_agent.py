class FallbackAgent:
    """Agent to handle unknown or unanswerable queries."""
    def handle(self, question: str) -> str:
        return "[FallbackAgent] Sorry, I couldn't find an answer to your question. Please rephrase or contact support." 