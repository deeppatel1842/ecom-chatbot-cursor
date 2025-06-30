from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from agents.faq_agent import FAQAgent
from agents.product_agent import ProductAgent
from agents.policy_agent import PolicyAgent
from agents.fallback_agent import FallbackAgent

# Define the conversation state
class ChatState(TypedDict):
    messages: list
    context: dict
    next_step: str

# Initialize agents (to be passed actual data in main)
def create_agents(faq_data, product_data, policy_chunks):
    return {
        'faq': FAQAgent(faq_data),
        'product': ProductAgent(product_data),
        'policy': PolicyAgent(policy_chunks),
        'fallback': FallbackAgent()
    }

# Define the workflow (skeleton)
def create_workflow(agents):
    workflow = StateGraph(ChatState)
    # Add agent nodes (handlers)
    workflow.add_node('faq', agents['faq'].retrieve_answer)
    workflow.add_node('product', agents['product'].search_product)
    workflow.add_node('policy', agents['policy'].answer_policy_query)
    workflow.add_node('fallback', agents['fallback'].handle)
    # TODO: Add routing logic and transitions
    return workflow 