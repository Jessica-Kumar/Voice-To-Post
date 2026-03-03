import os
import stat
from sqlalchemy import create_engine, event, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from cryptography.fernet import Fernet
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# 1. UNLOCK THE FOLDER: Ensure the entire /tmp directory is fully open
os.makedirs('/tmp/', exist_ok=True)
try:
    os.chmod('/tmp/', 0o777)  # Give full permissions to the folder itself
except Exception:
    pass  # In case it fails (rarely), we proceed anyway

DB_FILENAME = "credentials.db"
DB_PATH = f"/tmp/{DB_FILENAME}"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# 2. TUNING THE ENGINE: Configure SQLite for better cloud compatibility
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30}
)

# 3. THE MAGIC PRAGMA: Switch to Write-Ahead Logging (WAL)
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SocialCreds(Base):
    __tablename__ = "social_creds"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)

    # Twitter
    twitter_access_token = Column(String, nullable=True)
    twitter_refresh_token = Column(String, nullable=True)
    twitter_bio = Column(String, nullable=True)

    # LinkedIn
    linkedin_access_token = Column(String, nullable=True)
    linkedin_vanity_name = Column(String, nullable=True)
    linkedin_headline = Column(String, nullable=True)

# Encryption setup
ENV_KEY = os.getenv("ENCRYPTION_KEY")
if not ENV_KEY:
    raise ValueError(
        "❌ CRITICAL: ENCRYPTION_KEY must be set in environment variables. "
        "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
    )
FERNET_KEY = ENV_KEY.encode('utf-8')
cipher_suite = Fernet(FERNET_KEY)

def encrypt_secret(plain_text: str) -> str:
    return cipher_suite.encrypt(plain_text.encode('utf-8')).decode('utf-8')

def decrypt_secret(encrypted_text: str) -> str:
    return cipher_suite.decrypt(encrypted_text.encode('utf-8')).decode('utf-8')

# Create tables
Base.metadata.create_all(bind=engine)

# Hugging Face persistence
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "JessicaKumar/voice-to-post-data"

def download_db():
    """Downloads credentials.db from HF Dataset into /tmp/ and ensures write permissions."""
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
            local_dir="/tmp/"
        )
        print(f"Successfully downloaded DB to {DB_PATH}")

        # Force file to be writable by the current process
        if os.path.exists(DB_PATH):
            os.chmod(DB_PATH, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            print("Permissions set to 0777 to avoid readonly errors.")
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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()