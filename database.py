import os
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
from cryptography.fernet import Fernet
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

DB_FILENAME = "credentials.db"
DB_PATH = f"/tmp/{DB_FILENAME}"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class SocialCreds(Base):
    __tablename__ = "social_creds"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False, default="demo_user")

    # Twitter tokens
    twitter_access_token = Column(String, nullable=True)   # encrypted
    twitter_refresh_token = Column(String, nullable=True)  # encrypted
    twitter_bio = Column(String, nullable=True)            # plain text (profile description)

    # LinkedIn tokens
    linkedin_access_token = Column(String, nullable=True)  # encrypted
    linkedin_vanity_name = Column(String, nullable=True)   # plain text
    linkedin_headline = Column(String, nullable=True)      # plain text

# Encryption setup
ENV_KEY = os.getenv("ENCRYPTION_KEY")
if ENV_KEY:
    FERNET_KEY = ENV_KEY.encode('utf-8')
else:
    FERNET_KEY = Fernet.generate_key()
    print("WARNING: ENCRYPTION_KEY not found. Using a temporary runtime key.")

cipher_suite = Fernet(FERNET_KEY)

def encrypt_secret(plain_text: str) -> str:
    return cipher_suite.encrypt(plain_text.encode('utf-8')).decode('utf-8')

def decrypt_secret(encrypted_text: str) -> str:
    return cipher_suite.decrypt(encrypted_text.encode('utf-8')).decode('utf-8')

# Create tables
Base.metadata.create_all(bind=engine)

# Hugging Face persistence (unchanged)
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = "JessicaKumar/voice-to-post-data"

def download_db():
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Skipping cloud database download.")
        return
    try:
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=DB_FILENAME,
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir="/tmp/"
        )
        print(f"Successfully downloaded DB to {DB_PATH}")
    except EntryNotFoundError:
        print(f"Database file {DB_FILENAME} not found. A new one will be created.")
    except Exception as e:
        print(f"Error downloading DB: {e}")

def upload_db():
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Skipping cloud database upload.")
        return
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} does not exist.")
        return
    try:
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=DB_PATH,
            path_in_repo=DB_FILENAME,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message="Update social credentials"
        )
        print("Database uploaded to Hugging Face.")
    except Exception as e:
        print(f"Error uploading DB: {e}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()