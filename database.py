import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from cryptography.fernet import Fernet
import base64
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# --- Writable Database Path for Hugging Face ---
# The /tmp/ directory is the only writable area in a HF Space container.
DB_FILENAME = "credentials.db"
DB_PATH = f"/tmp/{DB_FILENAME}"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create the SQLAlchemy engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False} # Needed for SQLite + FastAPI
)

# Create a SessionLocal class to spawn DB sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for the ORM models
Base = declarative_base()

class SocialCreds(Base):
    """
    SQLAlchemy model to store Social Media Client IDs and Secrets.
    Uses 'platform' (e.g., 'twitter', 'linkedin') as the unique identifier.
    """
    __tablename__ = "social_creds"
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String, unique=True, index=True, nullable=False)
    client_id = Column(String, nullable=False)
    encrypted_secret = Column(String, nullable=False)

# --- Encryption Logic ---
ENV_KEY = os.getenv("ENCRYPTION_KEY")
if ENV_KEY:
    FERNET_KEY = ENV_KEY.encode('utf-8')
else:
    FERNET_KEY = Fernet.generate_key()
    print("WARNING: ENCRYPTION_KEY not found. Using a temporary runtime key.")

cipher_suite = Fernet(FERNET_KEY)

def encrypt_secret(plain_text: str) -> str:
    """Encrypts a string and returns it as a string formatted for the DB."""
    return cipher_suite.encrypt(plain_text.encode('utf-8')).decode('utf-8')

def decrypt_secret(encrypted_text: str) -> str:
    """Decrypts a DB formatted string back to the original string."""
    return cipher_suite.decrypt(encrypted_text.encode('utf-8')).decode('utf-8')

# Create the database tables in the writable /tmp/ path
Base.metadata.create_all(bind=engine)

# --- Hugging Face Persistence Logic ---
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "JessicaKumar/voice-to-post-data"

def download_db():
    """Downloads credentials.db from HF Dataset into the writable /tmp/ folder on startup."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Skipping cloud database download.")
        return
    try:
        print(f"Attempting to download {DB_FILENAME} from dataset {HF_DATASET_REPO} to /tmp/...")
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=DB_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp/" # MUST be /tmp/ for write access
        )
        print(f"Successfully downloaded DB to {DB_PATH}")
    except EntryNotFoundError:
        print(f"Database file {DB_FILENAME} not found in the dataset. A new one will be created in /tmp/.")
    except Exception as e:
        print(f"Error downloading DB from Hugging Face: {e}")

def upload_db():
    """Uploads the writable /tmp/credentials.db to Hugging Face Dataset."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Skipping cloud database upload.")
        return
            
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} does not exist locally to upload.")
        return
    try:
        api = HfApi(token=HF_TOKEN)
        print(f"Uploading {DB_PATH} to dataset {HF_DATASET_REPO}...")
        api.upload_file(
            path_or_fileobj=DB_PATH,
            path_in_repo=DB_FILENAME,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update social credentials via backend"
        )
        print("Database successfully uploaded to Hugging Face!")
    except Exception as e:
        print(f"Error uploading DB to Hugging Face: {e}")

# Dependency to get a DB session in FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()