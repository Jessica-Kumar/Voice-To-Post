"""
Enhanced Authentication Service with Logging
Provides JWT and API key authentication with proper error tracking.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import Column, Integer, String, DateTime
from database import Base, SessionLocal, get_db
from sqlalchemy.orm import Session

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


class User(Base):
    """User authentication table."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Integer, default=1)  # 1=active, 0=disabled


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"❌ Error verifying password: {e}")
        return False


def get_password_hash(password: str) -> str:
    """Hash a password."""
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"❌ Error hashing password: {e}")
        raise


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    try:
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        logger.info(f"✅ Created access token for user: {data.get('sub', 'unknown')}")
        return encoded_jwt
    except Exception as e:
        logger.error(f"❌ Error creating access token: {e}")
        raise


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"⚠️ Invalid JWT token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password."""
    try:
        user = db.query(User).filter(User.email == email).first()

        if not user:
            logger.warning(f"⚠️ Authentication failed: User not found for email {email}")
            return None

        if not verify_password(password, user.hashed_password):
            logger.warning(f"⚠️ Authentication failed: Invalid password for email {email}")
            return None

        logger.info(f"✅ User authenticated successfully: {email}")
        return user
    except Exception as e:
        logger.error(f"❌ Error authenticating user: {e}")
        return None


def create_user(db: Session, email: str, password: str, full_name: Optional[str] = None) -> User:
    """Create a new user."""
    try:
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            logger.warning(f"⚠️ User creation failed: Email already registered {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Generate user_id (could be UUID or email-based)
        import hashlib
        user_id = hashlib.md5(email.encode()).hexdigest()[:16]

        # Create user
        hashed_password = get_password_hash(password)
        user = User(
            user_id=user_id,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            created_at=datetime.utcnow()
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"✅ User created successfully: {email} (user_id: {user_id})")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating user: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get the current authenticated user from JWT token.
    Use this in protected endpoints: user = Depends(get_current_user)
    """
    token = credentials.credentials

    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")

        if user_id is None:
            logger.warning("⚠️ Invalid token: No user_id in payload")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    except JWTError as e:
        logger.warning(f"⚠️ JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    # Get user from database
    user = db.query(User).filter(User.user_id == user_id).first()

    if user is None:
        logger.warning(f"⚠️ User not found: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    if not user.is_active:
        logger.warning(f"⚠️ Inactive user attempted access: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    return user


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Optional authentication - returns None if no token provided.
    Use for endpoints that work with or without auth.
    """
    if credentials is None:
        return None

    try:
        return get_current_user(credentials, db)
    except HTTPException:
        return None


# API Key authentication (alternative to JWT for mobile apps)
class APIKey(Base):
    """API Key table for programmatic access."""
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)  # Name/description of the key
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Integer, default=1)


def generate_api_key() -> str:
    """Generate a random API key."""
    import secrets
    return f"vtp_{secrets.token_urlsafe(32)}"


def create_api_key(db: Session, user_id: str, name: Optional[str] = None) -> APIKey:
    """Create a new API key for a user."""
    try:
        api_key_str = generate_api_key()

        api_key = APIKey(
            user_id=user_id,
            api_key=api_key_str,
            name=name,
            created_at=datetime.utcnow()
        )

        db.add(api_key)
        db.commit()
        db.refresh(api_key)

        logger.info(f"✅ API key created for user {user_id}: {name or 'Unnamed'}")
        return api_key
    except Exception as e:
        logger.error(f"❌ Error creating API key: {e}")
        db.rollback()
        raise


def verify_api_key(db: Session, api_key: str) -> Optional[str]:
    """
    Verify an API key and return the user_id if valid.
    Returns None if invalid.
    """
    try:
        key = db.query(APIKey).filter(APIKey.api_key == api_key).first()

        if not key:
            logger.warning(f"⚠️ Invalid API key attempted")
            return None

        if not key.is_active:
            logger.warning(f"⚠️ Inactive API key attempted: {key.name}")
            return None

        # Check expiration
        if key.expires_at and key.expires_at < datetime.utcnow():
            logger.warning(f"⚠️ Expired API key attempted: {key.name}")
            return None

        # Update last used
        key.last_used = datetime.utcnow()
        db.commit()

        logger.info(f"✅ API key verified for user {key.user_id}")
        return key.user_id
    except Exception as e:
        logger.error(f"❌ Error verifying API key: {e}")
        return None


def get_user_from_api_key(
    api_key: str = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
) -> str:
    """
    Dependency to authenticate via API key.
    Returns user_id if valid.
    """
    user_id = verify_api_key(db, api_key.credentials)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key"
        )

    return user_id


# Initialize logging
logger.info("✅ Authentication service initialized")
