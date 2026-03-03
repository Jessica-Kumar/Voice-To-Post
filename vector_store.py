import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dimension = model.get_sentence_embedding_dimension()

index = faiss.IndexFlatL2(embedding_dimension)
text_store: List[Tuple[str, str]] = []  # (text, user_id)

def add_text_to_index(text_list: List[str], user_id: str) -> None:
    if not text_list:
        return
    embeddings = model.encode(text_list)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
    for text in text_list:
        text_store.append((text, user_id))

def search_index(query_text: str, top_k: int = 3, user_id: str = None) -> List[Dict[str, Any]]:
    if index.ntotal == 0:
        return []
    query_embedding = model.encode([query_text])
    query_embedding = np.array(query_embedding).astype('float32')
    # ✅ Search a larger pool (50) to ensure we can find the user's docs after filtering
    k = min(50, index.ntotal)
    distances, indices = index.search(query_embedding, k)

    results = []
    for i in range(k):
        idx = indices[0][i]
        if idx != -1 and idx < len(text_store):
            text, stored_user_id = text_store[idx]
            if user_id is None or stored_user_id == user_id:
                results.append({
                    "text": text,
                    "distance": float(distances[0][i]),
                    "user_id": stored_user_id
                })
    # Now limit to top_k after filtering
    return results[:top_k]