"""
Authentication and Authorization Module

Provides JWT-based authentication for the API.
Includes role-based access control (RBAC).
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Configuration
SECRET_KEY = "your-secret-key-change-in-production-use-env-variable"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==================== Models ====================

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    user_id: str

class TokenData(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    user_id: str
    username: str
    role: str
    full_name: Optional[str] = None

class UserInDB(User):
    hashed_password: str

# ==================== User Database (Mock) ====================
# In production, this would be a real database

# Simple password storage for demo (in production, use proper bcrypt hashing)
SIMPLE_USERS = {
    "analyst": {"password": "analyst123", "role": "analyst", "user_id": "analyst_001", "full_name": "Fraud Analyst"},
    "manager": {"password": "manager123", "role": "manager", "user_id": "manager_001", "full_name": "Risk Manager"},
    "investigator": {"password": "investigator123", "role": "investigator", "user_id": "investigator_001", "full_name": "Fraud Investigator"},
    "admin": {"password": "admin123", "role": "admin", "user_id": "admin_001", "full_name": "System Administrator"}
}

# Legacy USERS_DB for compatibility (not used with simple auth)
USERS_DB = {}

# ==================== Helper Functions ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database"""
    user_dict = USERS_DB.get(username)
    if user_dict:
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate a user by username and password.

    DEMO VERSION: Uses simple password comparison.
    In production, use bcrypt password hashing.

    Returns:
        User object if authentication successful, None otherwise
    """
    # Simple authentication for demo
    user_data = SIMPLE_USERS.get(username)
    if not user_data:
        return None

    if user_data["password"] != password:
        return None

    # Return User object
    return User(
        user_id=user_data["user_id"],
        username=username,
        role=user_data["role"],
        full_name=user_data["full_name"]
    )

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> Optional[TokenData]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token to decode

    Returns:
        TokenData if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        role: str = payload.get("role")

        if user_id is None:
            return None

        return TokenData(user_id=user_id, role=role)
    except JWTError:
        return None

# ==================== Role-Based Access Control ====================

ROLE_PERMISSIONS = {
    "analyst": ["view_alerts", "view_transactions", "update_alerts"],
    "manager": ["view_alerts", "view_transactions", "view_analytics", "export_reports"],
    "investigator": ["view_alerts", "view_transactions", "update_alerts", "view_full_details", "add_notes"],
    "admin": ["*"]  # All permissions
}

def has_permission(role: str, permission: str) -> bool:
    """
    Check if a role has a specific permission.

    Args:
        role: User role
        permission: Permission to check

    Returns:
        True if role has permission, False otherwise
    """
    if role not in ROLE_PERMISSIONS:
        return False

    permissions = ROLE_PERMISSIONS[role]

    # Admin has all permissions
    if "*" in permissions:
        return True

    return permission in permissions
