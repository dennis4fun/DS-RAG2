# rag_website_explainer/config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# --- Pinecone Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # e.g., "us-east-1", "gcp-starter"
PINECONE_INDEX_NAME = "website-explainer-index" # Choose a unique name for your index

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo") # Or "gpt-4", etc.

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good balance of size and performance for local embedding
EMBEDDING_DIMENSION = 384 # Dimension for 'all-MiniLM-L6-v2'
