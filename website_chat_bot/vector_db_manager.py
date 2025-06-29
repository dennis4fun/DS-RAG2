# rag_website_explainer/vector_db_manager.py
from pinecone import Pinecone, Index, PodSpec, ServerlessSpec
from typing import List, Dict, Any
from langchain_community.vectorstores import Pinecone as LangChainPineconeVectorStore
from langchain_core.documents import Document
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION

def get_pinecone_client():
    """Initializes and returns a Pinecone client."""
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        print("Pinecone API key or environment not set. Cannot connect to Pinecone.")
        return None
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
        print("Successfully connected to Pinecone.")
        return pc
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return None

def create_pinecone_index(pc_client: Pinecone, index_name: str, dimension: int):
    """Creates a Pinecone index if it doesn't exist."""
    if index_name in pc_client.list_indexes().names:
        print(f"Index '{index_name}' already exists.")
        # Optional: delete and recreate for clean slate during development
        # pc_client.delete_index(index_name)
        # print(f"Deleted existing index '{index_name}'.")
        # # Give some time for deletion to propagate if recreating immediately
        # # import time
        # # time.sleep(5) 
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
    return pc_client.Index(index_name)

def add_documents_to_pinecone(index: Index, documents: List[Document], embedding_model):
    """Adds documents (with embeddings) to Pinecone."""
    if not documents:
        print("No documents to add to Pinecone.")
        return

    batch_size = 100 # Pinecone recommends batches of up to 100 for upserts
    
    vectors_to_upsert = []
    for doc in documents:
        # Create a unique ID for each chunk. Ensure IDs are strings.
        doc_id = f"{doc.metadata.get('source_url', 'unknown_url')}_{doc.metadata.get('chunk_id', 0)}"
        embedding = embedding_model.encode(doc.page_content).tolist()
        
        # Prepare metadata (Pinecone accepts dictionary for metadata)
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
            index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1) // batch_size}")
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {e}")
            # Consider more robust error handling or retries here
            
    print(f"Finished adding {len(documents)} documents to Pinecone index '{index.name}'.")

def get_pinecone_retriever(pc_client: Pinecone, index_name: str, embedding_model):
    """Returns a LangChain Pinecone retriever."""
    # LangChain's PineconeVectorStore requires an embedding function for its operations.
    # We pass our local SentenceTransformer model's embedding function.
    
    # Note: Ensure the index is created with the correct dimension
    # before trying to use this retriever.
    
    # Initialize the LangChain Pinecone vector store
    vectorstore = LangChainPineconeVectorStore(
        index=pc_client.Index(index_name),
        embedding=embedding_model, # Pass the embedding model directly
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
