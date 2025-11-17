"""
Role-Based Access Control (RBAC) for Space Debris Tracking System
"""

from enum import Enum
from typing import Set, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions"""
    # Data access
    READ_TLE = "read:tle"
    READ_DETECTIONS = "read:detections"
    READ_TRAJECTORIES = "read:trajectories"
    READ_CONJUNCTIONS = "read:conjunctions"

    # Data modification
    WRITE_TLE = "write:tle"
    WRITE_DETECTIONS = "write:detections"
    WRITE_TRAJECTORIES = "write:trajectories"

    # Model operations
    TRAIN_MODELS = "train:models"
    DEPLOY_MODELS = "deploy:models"
    EVALUATE_MODELS = "evaluate:models"

    # Agent operations
    START_AGENTS = "start:agents"
    STOP_AGENTS = "stop:agents"
    CONFIGURE_AGENTS = "configure:agents"

    # Admin operations
    MANAGE_USERS = "manage:users"
    MANAGE_ROLES = "manage:roles"
    VIEW_SYSTEM_METRICS = "view:system_metrics"
    CONFIGURE_SYSTEM = "configure:system"


class Role(Enum):
    """System roles with associated permissions"""
    ADMIN = "admin"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


# Role-Permission mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # Full access
        Permission.READ_TLE,
        Permission.READ_DETECTIONS,
        Permission.READ_TRAJECTORIES,
        Permission.READ_CONJUNCTIONS,
        Permission.WRITE_TLE,
        Permission.WRITE_DETECTIONS,
        Permission.WRITE_TRAJECTORIES,
        Permission.TRAIN_MODELS,
        Permission.DEPLOY_MODELS,
        Permission.EVALUATE_MODELS,
        Permission.START_AGENTS,
        Permission.STOP_AGENTS,
        Permission.CONFIGURE_AGENTS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_ROLES,
        Permission.VIEW_SYSTEM_METRICS,
        Permission.CONFIGURE_SYSTEM,
    },

    Role.OPERATOR: {
        # Operational access
        Permission.READ_TLE,
        Permission.READ_DETECTIONS,
        Permission.READ_TRAJECTORIES,
        Permission.READ_CONJUNCTIONS,
        Permission.WRITE_TLE,
        Permission.START_AGENTS,
        Permission.STOP_AGENTS,
        Permission.CONFIGURE_AGENTS,
        Permission.VIEW_SYSTEM_METRICS,
    },

    Role.ANALYST: {
        # Analysis and training
        Permission.READ_TLE,
        Permission.READ_DETECTIONS,
        Permission.READ_TRAJECTORIES,
        Permission.READ_CONJUNCTIONS,
        Permission.TRAIN_MODELS,
        Permission.EVALUATE_MODELS,
        Permission.VIEW_SYSTEM_METRICS,
    },

    Role.VIEWER: {
        # Read-only access
        Permission.READ_TLE,
        Permission.READ_DETECTIONS,
        Permission.READ_TRAJECTORIES,
        Permission.READ_CONJUNCTIONS,
    },

    Role.API_USER: {
        # API access only
        Permission.READ_TLE,
        Permission.READ_DETECTIONS,
        Permission.READ_TRAJECTORIES,
        Permission.READ_CONJUNCTIONS,
    },
}


class RBACManager:
    """Manage role-based access control"""

    @staticmethod
    def has_permission(user_roles: list[str], required_permission: Permission) -> bool:
        """
        Check if user has required permission

        Args:
            user_roles: List of user roles
            required_permission: Required permission

        Returns:
            True if user has permission
        """
        for role_name in user_roles:
            try:
                role = Role(role_name)
                if required_permission in ROLE_PERMISSIONS.get(role, set()):
                    return True
            except ValueError:
                logger.warning(f"Unknown role: {role_name}")
                continue

        return False

    @staticmethod
    def get_user_permissions(user_roles: list[str]) -> Set[Permission]:
        """
        Get all permissions for user

        Args:
            user_roles: List of user roles

        Returns:
            Set of permissions
        """
        permissions = set()

        for role_name in user_roles:
            try:
                role = Role(role_name)
                permissions.update(ROLE_PERMISSIONS.get(role, set()))
            except ValueError:
                logger.warning(f"Unknown role: {role_name}")
                continue

        return permissions


def require_permission(permission: Permission):
    """
    Decorator to require specific permission

    Args:
        permission: Required permission

    Usage:
        @require_permission(Permission.READ_TLE)
        def get_tle_data():
            pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract user roles from context
            # In practice, this would come from the request context
            user_roles = kwargs.get('user_roles', [])

            if not RBACManager.has_permission(user_roles, permission):
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing required permission: {permission.value}"
                )

            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            user_roles = kwargs.get('user_roles', [])

            if not RBACManager.has_permission(user_roles, permission):
                raise PermissionError(f"Missing required permission: {permission.value}")

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    # Check permissions
    manager = RBACManager()

    # Admin has all permissions
    admin_perms = manager.get_user_permissions(['admin'])
    print(f"Admin permissions: {len(admin_perms)}")

    # Viewer only has read permissions
    viewer_perms = manager.get_user_permissions(['viewer'])
    print(f"Viewer permissions: {len(viewer_perms)}")

    # Check specific permission
    can_train = manager.has_permission(['analyst'], Permission.TRAIN_MODELS)
    print(f"Analyst can train models: {can_train}")

    can_manage = manager.has_permission(['operator'], Permission.MANAGE_USERS)
    print(f"Operator can manage users: {can_manage}")
