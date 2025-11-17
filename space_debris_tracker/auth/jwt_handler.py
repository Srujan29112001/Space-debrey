"""
JWT Token Handler for Authentication
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt

logger = logging.getLogger(__name__)


@dataclass
class TokenPayload:
    """JWT token payload"""

    user_id: str
    username: str
    roles: list[str]
    exp: datetime
    iat: datetime


class JWTHandler:
    """Handle JWT token creation and validation"""

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
    ):
        """
        Initialize JWT handler

        Args:
            secret_key: Secret key for signing tokens
            algorithm: JWT algorithm
            access_token_expire_minutes: Token expiration time
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "your-secret-key-change-me")
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes

    def create_access_token(
        self,
        user_id: str,
        username: str,
        roles: list[str],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create JWT access token

        Args:
            user_id: User ID
            username: Username
            roles: List of user roles
            expires_delta: Custom expiration time

        Returns:
            JWT token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        payload = {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "exp": expire,
            "iat": datetime.utcnow(),
        }

        encoded_jwt = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created access token for user {username}")

        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """
        Verify and decode JWT token

        Args:
            token: JWT token string

        Returns:
            TokenPayload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            token_payload = TokenPayload(
                user_id=payload["user_id"],
                username=payload["username"],
                roles=payload["roles"],
                exp=datetime.fromtimestamp(payload["exp"]),
                iat=datetime.fromtimestamp(payload["iat"]),
            )

            return token_payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None


# Global handler instance
_jwt_handler = None


def get_jwt_handler() -> JWTHandler:
    """Get global JWT handler instance"""
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler


def create_access_token(user_id: str, username: str, roles: list[str]) -> str:
    """Helper function to create access token"""
    handler = get_jwt_handler()
    return handler.create_access_token(user_id, username, roles)


def verify_token(token: str) -> Optional[TokenPayload]:
    """Helper function to verify token"""
    handler = get_jwt_handler()
    return handler.verify_token(token)
