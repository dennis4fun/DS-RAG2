# rag_website_explainer/embedder.py
from sentence_transformers import SentenceTransformer
from typing import List

def get_embedding_model(model_name: str):
    """Loads a Sentence Transformer embedding model."""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading embedding model {model_name}: {e}")
        return None

def embed_texts(texts: List[str], model) -> List[List[float]]:
    """Embeds a list of texts using the given model."""
    if model is None:
        return []
    return model.encode(texts).tolist()

if __name__ == "__main__":
    from config import EMBEDDING_MODEL_NAME
    model = get_embedding_model(EMBEDDING_MODEL_NAME)
    if model:
        test_texts = ["This is a test sentence.", "Another sentence to embed for demonstration."]
        embeddings = embed_texts(test_texts, model)
        if embeddings:
            print(f"Embedding for '{test_texts[0]}' (first 5 dimensions): {embeddings[0][:5]}...")
            print(f"Dimension: {len(embeddings[0])}")
        else:
            print("No embeddings generated.")
    else:
        print("Embedding model could not be loaded.")
