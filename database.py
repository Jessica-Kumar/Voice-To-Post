import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from cryptography.fernet import Fernet
import base64
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# Define the SQLite database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./credentials.db"

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

# Fetch or generate Fernet key
# In production, ENCRYPTION_KEY must be stored securely in the .env file.
# For demo purposes, we will generate a valid one if missing.
ENV_KEY = os.getenv("ENCRYPTION_KEY")
if ENV_KEY:
    FERNET_KEY = ENV_KEY.encode('utf-8')
else:
    FERNET_KEY = Fernet.generate_key()
    print("WARNING: ENCRYPTION_KEY not found. Using a temporary runtime key.")

# Initialize the cipher suite
cipher_suite = Fernet(FERNET_KEY)

def encrypt_secret(plain_text: str) -> str:
    """Encrypts a string and returns it as a string formatted for the DB."""
    return cipher_suite.encrypt(plain_text.encode('utf-8')).decode('utf-8')

def decrypt_secret(encrypted_text: str) -> str:
    """Decrypts a DB formatted string back to the original string."""
    return cipher_suite.decrypt(encrypted_text.encode('utf-8')).decode('utf-8')

# Create the database tables if they don't exist
Base.metadata.create_all(bind=engine)

# --- Hugging Face Persistence Logic ---
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "JessicaKumar/voice-to-post-data"
DB_FILENAME = "credentials.db"

def download_db():
    """Downloads credentials.db from Hugging Face Dataset on startup."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Skipping cloud database download.")
        return

    try:
        print(f"Attempting to download {DB_FILENAME} from dataset {HF_DATASET_REPO}...")
        downloaded_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=DB_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="." # Download directly to current directory
        )
        print(f"Successfully downloaded DB to {downloaded_path}")
    except EntryNotFoundError:
        print(f"Database file {DB_FILENAME} not found in the dataset. A new one will be created.")
    except Exception as e:
        print(f"Error downloading DB from Hugging Face: {e}")

def upload_db():
    """Uploads the local credentials.db to Hugging Face Dataset."""
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Skipping cloud database upload.")
        return
        
    if not os.path.exists("./" + DB_FILENAME):
        print(f"Error: {DB_FILENAME} does not exist locally to upload.")
        return

    try:
        api = HfApi(token=HF_TOKEN)
        print(f"Uploading {DB_FILENAME} to dataset {HF_DATASET_REPO}...")
        api.upload_file(
            path_or_fileobj="./" + DB_FILENAME,
            path_in_repo=DB_FILENAME,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update social credentials via backend"
        )
        print("Database highly successfully uploaded to Hugging Face!")
    except Exception as e:
        print(f"Error uploading DB to Hugging Face: {e}")

# Dependency to get a DB session in FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
