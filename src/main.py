import pandas as pd
import json
import os
from langgraph_workflow import create_agents, create_workflow
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load processed data
def load_data():
    faq_data = pd.read_csv('processed_data/faq_processed.csv')
    product_data = pd.read_csv('processed_data/products_processed.csv')
    with open('processed_data/policy_chunks.json', 'r') as f:
        policy_chunks = json.load(f)
    return faq_data, product_data, policy_chunks

# Simple agent router (can be improved with intent classification)
def route_query(user_input):
    user_input_lower = user_input.lower()
    if any(word in user_input_lower for word in ['refund', 'return', 'policy', 'shipping', 'privacy', 'terms']):
        return 'policy'
    elif any(word in user_input_lower for word in ['product', 'laptop', 'phone', 'buy', 'price', 'catalog', 'recommend']):
        return 'product'
    elif any(word in user_input_lower for word in ['account', 'order', 'faq', 'question', 'support', 'help']):
        return 'faq'
    else:
        return 'faq'  # Default to FAQ for general queries


def main():
    # Load data
    faq_data, product_data, policy_chunks = load_data()
    # Create agents
    agents = create_agents(faq_data, product_data, policy_chunks)
    # Initialize RAG pipeline
    rag = RAGPipeline(openai_api_key=OPENAI_API_KEY)
    print("[System] Chatbot is ready.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"): break
        agent_type = route_query(user_input)
        if agent_type == 'faq':
            context = agents['faq'].retrieve_answer(user_input)
        elif agent_type == 'product':
            context = agents['product'].search_product(user_input)
        elif agent_type == 'policy':
            context = agents['policy'].answer_policy_query(user_input)
        else:
            context = agents['fallback'].handle(user_input)
        # If fallback, just print the fallback message
        if agent_type == 'fallback' or context.startswith('[FallbackAgent]'):
            print(f"Bot: {context}")
            continue
        # Generate answer using RAG pipeline
        answer = rag.generate_answer(user_input, context)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    main() 