import time
import faiss
import numpy as np
import pickle
import os
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dimension = model.get_sentence_embedding_dimension()

# FAISS index and text store
index = faiss.IndexFlatL2(embedding_dimension)
text_store: List[Tuple[str, str]] = []  # (text, user_id)

# FIXED: Use persistent storage path for HuggingFace Spaces
# HF Spaces mounts persistent storage at /data if available, fallback to /tmp
PERSISTENT_DIR = "/data/vector_store" if os.path.exists("/data") else "/tmp/vector_store"
VECTOR_STORE_DIR = PERSISTENT_DIR
INDEX_PATH = f"{VECTOR_STORE_DIR}/faiss_index.bin"
TEXT_STORE_PATH = f"{VECTOR_STORE_DIR}/text_store.pkl"

# Hugging Face persistence
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "voice-to-post-vectors")  # Make configurable

# Debounce HF uploads: uploading on every write hammers the Hub API.
UPLOAD_DEBOUNCE_SECONDS = float(os.getenv("VECTOR_UPLOAD_DEBOUNCE_SECONDS", "120"))
_last_upload_time = 0.0

# Ensure directory exists
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
logger.info(f"Vector store directory: {VECTOR_STORE_DIR}")


def save_vector_store_local():
    """Save FAISS index and text store to local disk."""
    try:
        # Save FAISS index
        faiss.write_index(index, INDEX_PATH)

        # Save text store
        with open(TEXT_STORE_PATH, 'wb') as f:
            pickle.dump(text_store, f)

        logger.info(f"✅ Vector store saved locally to {VECTOR_STORE_DIR} ({index.ntotal} vectors)")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving vector store locally: {e}")
        return False


def load_vector_store_local():
    """Load FAISS index and text store from local disk."""
    global index, text_store

    try:
        if os.path.exists(INDEX_PATH) and os.path.exists(TEXT_STORE_PATH):
            # Load FAISS index
            index = faiss.read_index(INDEX_PATH)

            # Load text store
            with open(TEXT_STORE_PATH, 'rb') as f:
                text_store = pickle.load(f)

            logger.info(f"✅ Vector store loaded from local disk. {index.ntotal} vectors, {len(text_store)} texts.")
            return True
        else:
            logger.warning(f"Local vector store files not found at {VECTOR_STORE_DIR}")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading vector store from local disk: {e}")
        return False


def download_vector_store_from_hf():
    """Download vector store files from Hugging Face Dataset."""
    if not HF_TOKEN:
        logger.warning("⚠️ HF_TOKEN not set. Skipping vector store download from HuggingFace.")
        return False

    try:
        logger.info(f"📥 Attempting to download vector store from HuggingFace Dataset: {HF_DATASET_REPO}...")

        # Ensure local directory exists
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

        # Download FAISS index
        index_file = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="faiss_index.bin",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=VECTOR_STORE_DIR,
            local_dir_use_symlinks=False  # FIXED: Don't use symlinks, copy actual files
        )
        logger.info(f"✅ Downloaded FAISS index to {index_file}")

        # Download text store
        text_file = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="text_store.pkl",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=VECTOR_STORE_DIR,
            local_dir_use_symlinks=False  # FIXED: Don't use symlinks
        )
        logger.info(f"✅ Downloaded text store to {text_file}")

        # Now load the downloaded files
        return load_vector_store_local()

    except EntryNotFoundError:
        logger.warning(f"⚠️ Vector store not found in HuggingFace Dataset '{HF_DATASET_REPO}'. Will create new one.")
        return False
    except Exception as e:
        logger.error(f"❌ Error downloading vector store from HuggingFace: {e}")
        return False


def upload_vector_store_to_hf():
    """Upload vector store files to Hugging Face Dataset."""
    if not HF_TOKEN:
        logger.warning("⚠️ HF_TOKEN not set. Skipping vector store upload to HuggingFace.")
        return False

    try:
        # Save locally first
        if not save_vector_store_local():
            logger.error("❌ Failed to save vector store locally before upload")
            return False

        api = HfApi(token=HF_TOKEN)

        # Check if dataset repo exists, create if not
        try:
            api.dataset_info(repo_id=HF_DATASET_REPO)
            logger.info(f"✅ HuggingFace Dataset repo exists: {HF_DATASET_REPO}")
        except Exception:
            logger.info(f"📝 Creating HuggingFace Dataset repo: {HF_DATASET_REPO}")
            api.create_repo(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                private=True,
                exist_ok=True
            )

        # Upload FAISS index
        if os.path.exists(INDEX_PATH):
            logger.info(f"📤 Uploading FAISS index to {HF_DATASET_REPO}...")
            api.upload_file(
                path_or_fileobj=INDEX_PATH,
                path_in_repo="faiss_index.bin",
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Update vector store FAISS index ({index.ntotal} vectors)"
            )
            logger.info("✅ FAISS index uploaded")

        # Upload text store
        if os.path.exists(TEXT_STORE_PATH):
            logger.info(f"📤 Uploading text store to {HF_DATASET_REPO}...")
            api.upload_file(
                path_or_fileobj=TEXT_STORE_PATH,
                path_in_repo="text_store.pkl",
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Update vector store text store ({len(text_store)} texts)"
            )
            logger.info("✅ Text store uploaded")

        logger.info("🎉 Vector store successfully uploaded to HuggingFace!")
        return True
    except Exception as e:
        logger.error(f"❌ Error uploading vector store to HuggingFace: {e}")
        return False


def add_text_to_index(text_list: List[str], user_id: str) -> None:
    """
    Add texts to the vector store and persist to HF.
    """
    if not text_list:
        return

    try:
        embeddings = model.encode(text_list)
        embeddings = np.array(embeddings).astype('float32')
        index.add(embeddings)

        for text in text_list:
            text_store.append((text, user_id))

        logger.info(f"✅ Added {len(text_list)} texts to vector store for user {user_id}")

        # Save locally immediately
        save_vector_store_local()

        # Debounced upload to HF (fully flushed on app shutdown)
        global _last_upload_time
        now = time.time()
        if now - _last_upload_time >= UPLOAD_DEBOUNCE_SECONDS:
            _last_upload_time = now
            upload_vector_store_to_hf()

    except Exception as e:
        logger.error(f"❌ Error adding texts to vector store: {e}")


def search_index(query_text: str, top_k: int = 3, user_id: str = None) -> List[Dict[str, Any]]:
    """
    Search the vector store for similar texts.
    """
    if index.ntotal == 0:
        logger.warning("⚠️ Vector store is empty, returning no results")
        return []

    try:
        query_embedding = model.encode([query_text])
        query_embedding = np.array(query_embedding).astype('float32')

        # Search a larger pool to ensure user filtering works
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

        # Limit to top_k after filtering
        logger.info(f"🔍 Search returned {len(results[:top_k])} results for query: {query_text[:50]}...")
        return results[:top_k]

    except Exception as e:
        logger.error(f"❌ Error searching vector store: {e}")
        return []


def get_vector_store_stats() -> Dict[str, Any]:
    """Get statistics about the vector store."""
    return {
        "total_vectors": index.ntotal,
        "total_texts": len(text_store),
        "embedding_dimension": embedding_dimension,
        "model": "all-MiniLM-L6-v2",
        "index_type": "FlatL2",
        "storage_path": VECTOR_STORE_DIR,
        "hf_repo": HF_DATASET_REPO,
        "hf_sync_enabled": bool(HF_TOKEN)
    }


def clear_vector_store():
    """Clear all vectors from the store (useful for testing)."""
    global index, text_store

    logger.warning("⚠️ Clearing entire vector store...")
    index = faiss.IndexFlatL2(embedding_dimension)
    text_store = []

    save_vector_store_local()
    upload_vector_store_to_hf()

    logger.info("✅ Vector store cleared")


# Initialize: Try to load from HF, then local, or start fresh
def initialize_vector_store():
    """Initialize vector store on startup."""
    logger.info("🚀 Initializing vector store...")

    # Priority 1: Try loading from HuggingFace Dataset
    if download_vector_store_from_hf():
        logger.info(f"✅ Vector store initialized from HuggingFace ({index.ntotal} vectors)")
        return

    # Priority 2: Try loading from local disk
    if load_vector_store_local():
        logger.info(f"✅ Vector store initialized from local disk ({index.ntotal} vectors)")
        # Upload to HF if local exists but HF doesn't
        if HF_TOKEN:
            upload_vector_store_to_hf()
        return

    # Priority 3: Start fresh
    logger.info("📝 Starting with fresh vector store")

    # Add sample seed data
    sample_data = [
        "Welcome to Voice-To-Post backend!",
        "Vector databases help in doing semantic similarity search.",
        "FastAPI is a fast, highly performant web framework for building APIs."
    ]
    add_text_to_index(sample_data, user_id="system")

    logger.info("✅ Vector store initialized with seed data")


# Auto-initialize when module is imported
initialize_vector_store()
