"""
Authentication and Authorization Module
JWT-based auth with role-based access control (RBAC)
"""

from .jwt_handler import JWTHandler, create_access_token, verify_token
from .rbac import RBACManager, Role, Permission, require_permission
from .api_key import APIKeyManager, generate_api_key, validate_api_key

__all__ = [
    'JWTHandler',
    'create_access_token',
    'verify_token',
    'RBACManager',
    'Role',
    'Permission',
    'require_permission',
    'APIKeyManager',
    'generate_api_key',
    'validate_api_key',
]
