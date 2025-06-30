from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import json
import os
from langgraph_workflow import create_agents
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

faq_data, product_data, policy_chunks = load_data()
agents = create_agents(faq_data, product_data, policy_chunks)
rag = RAGPipeline(openai_api_key=OPENAI_API_KEY)

@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")
    agent_type = route_query(user_input)
    if agent_type == 'faq':
        context = agents['faq'].retrieve_answer(user_input)
    elif agent_type == 'product':
        context = agents['product'].search_product(user_input)
    elif agent_type == 'policy':
        context = agents['policy'].answer_policy_query(user_input)
    else:
        context = agents['fallback'].handle(user_input)
    if agent_type == 'fallback' or context.startswith('[FallbackAgent]'):
        return JSONResponse({"response": context})
    answer = rag.generate_answer(user_input, context)
    return JSONResponse({"response": answer})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 