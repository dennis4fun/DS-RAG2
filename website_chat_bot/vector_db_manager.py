# rag_website_explainer/vector_db_manager.py
# Updated import for Pinecone v7+
from pinecone import Pinecone, PodSpec, ServerlessSpec
from typing import List, Dict, Any # Keep 'Any' here
# LangChain's Pinecone integration package
from langchain_pinecone import PineconeVectorStore as LangChainPineconeVectorStore
from langchain_core.documents import Document
# Removed explicit import of 'Index' class for type hinting
# as it is causing persistent ImportError despite correct installation.
# We will use 'Any' for type hints where 'Index' was used,
# relying on the runtime object from pc_client.Index(index_name).
# from pinecone.data import Index # This line is now commented out

# NEW: Import LangChain's SentenceTransformerEmbeddings wrapper
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Import EMBEDDING_MODEL_NAME directly from config for use here
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME # Added EMBEDDING_MODEL_NAME


def get_pinecone_client():
    """Initializes and returns a Pinecone client for v7+."""
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        print("Pinecone API key or environment not set. Cannot connect to Pinecone.")
        return None
    try:
        # Pinecone v7+ client initialization
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        print("Successfully connected to Pinecone.")
        return pc
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return None

# Changed return type hint from Index to Any
def create_pinecone_index(pc_client: Pinecone, index_name: str, dimension: int) -> Any:
    """Creates a Pinecone index if it doesn't exist, compatible with v7+."""
    # FIX: Call the .names() method to get the list of index names
    if index_name in pc_client.list_indexes().names():
        print(f"Index '{index_name}' already exists.")
        # If you need to delete and recreate for development, uncomment these lines:
        # print(f"Deleting existing index '{index_name}' for recreation.")
        # pc_client.delete_index(index_name)
        # import time
        # time.sleep(5) # Give some time for deletion to propagate if recreating immediately
    else:
        print(f"Creating Pinecone index '{index_name}' with dimension {dimension}...")
        # Use ServerlessSpec for the free tier (recommended)
        # Choose a cloud and region that is available in your Pinecone environment.
        # Common options: cloud='aws', region='us-east-1' or cloud='gcp', region='us-central1'
        pc_client.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1') # Adjust as per your Pinecone environment
        )
        print(f"Index '{index_name}' created.")
    
    # In Pinecone v7+, you get a handle to the specific index object like this:
    return pc_client.Index(index_name)

# Changed 'index: Index' type hint to 'index: Any'
def add_documents_to_pinecone(index: Any, documents: List[Document], embedding_model):
    """Adds documents (with embeddings) to Pinecone, compatible with v7+."""
    if not documents:
        print("No documents to add to Pinecone.")
        return

    batch_size = 100 # Pinecone recommends batches of up to 100 for upserts
    
    vectors_to_upsert = []
    for doc in documents:
        # Create a unique ID for each chunk. Ensure IDs are strings.
        # It's good practice to hash content or use a UUID for robustness
        import hashlib
        doc_id = hashlib.sha256(f"{doc.metadata.get('source_url', '')}_{doc.metadata.get('chunk_id', 0)}_{doc.page_content}".encode()).hexdigest()
        
        embedding = embedding_model.encode(doc.page_content).tolist()
        
        metadata = {
            "text": doc.page_content, # Store original text for retrieval/display
            "source_url": doc.metadata.get("source_url"),
            "chunk_id": doc.metadata.get("chunk_id")
        }
        vectors_to_upsert.append((doc_id, embedding, metadata))

    # Upsert in batches
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        try:
            # Upsert method is called directly on the Index object
            index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1) // batch_size}")
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {e}")
            
    # FIX: Use PINECONE_INDEX_NAME from config instead of index.name
    # index.name might not be directly accessible on the runtime Index object
    from config import PINECONE_INDEX_NAME
    print(f"Finished adding {len(documents)} documents to Pinecone index '{PINECONE_INDEX_NAME}'.")

def get_pinecone_retriever(pc_client: Pinecone, index_name: str, embedding_model):
    """Returns a LangChain Pinecone retriever, compatible with v7+."""
    # LangChain's PineconeVectorStore now directly takes the Index object.
    # It will internally use the embedding_model to convert queries to vectors.
    
    # Ensure the index object is correctly obtained
    pinecone_index_obj = pc_client.Index(index_name)

    # FIX: Wrap the SentenceTransformer model with LangChain's SentenceTransformerEmbeddings
    # This provides the .embed_query method that PineconeVectorStore expects.
    # Use the original EMBEDDING_MODEL_NAME string from config.
    langchain_embeddings_wrapper = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME) # Changed from embedding_model.model_name

    vectorstore = LangChainPineconeVectorStore(
        index=pinecone_index_obj, # Pass the Index object
        embedding=langchain_embeddings_wrapper, # Pass the wrapped embedding model
        text_key="text" # The key in your metadata that stores the original text
    )
    # Return the vectorstore as a retriever
    return vectorstore.as_retriever()

if __name__ == "__main__":
    from web_scraper import get_website_content_as_polars_df
    from embedder import get_embedding_model
    from config import EMBEDDING_MODEL_NAME, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION

    pc_client = get_pinecone_client()
    if pc_client:
        # Create/get the index
        pinecone_index = create_pinecone_index(pc_client, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION)
        
        test_url = "https://www.pinecone.io/learn/series/rag/rag-primer/" # A page about Pinecone/RAG
        print(f"Scraping content from: {test_url}")
        df = get_website_content_as_polars_df(test_url)
        
        # Convert Polars DataFrame rows to LangChain Document objects
        documents = [
            Document(page_content=row["text"], metadata={"source_url": row["source_url"], "chunk_id": row["chunk_id"]})
            for row in df.iter_rows(named=True)
        ]
        
        embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
        if embedding_model:
            add_documents_to_pinecone(pinecone_index, documents, embedding_model)
            
            # Test retrieval
            retriever = get_pinecone_retriever(pc_client, PINECONE_INDEX_NAME, embedding_model)
            query = "What is RAG in AI?"
            print(f"\nPerforming retrieval for query: '{query}'")
            results = retriever.get_relevant_documents(query)
            
            print(f"\nRetrieved {len(results)} documents for query '{query}':")
            for i, doc in enumerate(results):
                print(f"--- Document {i+1} ---")
                print(f"Content: {doc.page_content[:200]}...") # Print first 200 chars
                print(f"Source: {doc.metadata.get('source_url')}")
                print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
                print("-" * 20)
        else:
            print("Embedding model not loaded. Cannot proceed with Pinecone operations.")
    else:
        print("Pinecone client not initialized. Check API keys and environment.")
