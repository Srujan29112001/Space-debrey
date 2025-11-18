"""
API Key Management for External Integrations
"""

import hashlib
import hmac
import json
import logging
import secrets
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API Key data structure"""

    key_id: str
    key_hash: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime]
    permissions: list[str]
    rate_limit: int  # requests per minute
    is_active: bool = True


class APIKeyManager:
    """Manage API keys for external access"""

    def __init__(self, storage_path: str = "api_keys.json"):
        """
        Initialize API key manager

        Args:
            storage_path: Path to store API keys
        """
        self.storage_path = Path(storage_path)
        self.api_keys: Dict[str, APIKey] = {}
        self._load_keys()

    def generate_api_key(
        self,
        name: str,
        permissions: list[str],
        rate_limit: int = 100,
        expires_in_days: Optional[int] = None,
    ) -> tuple[str, str]:
        """
        Generate new API key

        Args:
            name: Key name/description
            permissions: List of permissions
            rate_limit: Rate limit in requests per minute
            expires_in_days: Expiration in days

        Returns:
            Tuple of (key_id, api_key)
        """
        # Generate random key
        api_key = f"sdk_{secrets.token_urlsafe(32)}"

        # Generate key ID
        key_id = f"key_{secrets.token_hex(16)}"

        # Hash the key for storage
        key_hash = self._hash_key(api_key)

        # Set expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create API key object
        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            permissions=permissions,
            rate_limit=rate_limit,
            is_active=True,
        )

        # Store
        self.api_keys[key_id] = api_key_obj
        self._save_keys()

        logger.info(f"Generated API key: {key_id} ({name})")

        return key_id, api_key

    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate API key

        Args:
            api_key: API key to validate

        Returns:
            APIKey object if valid, None otherwise
        """
        key_hash = self._hash_key(api_key)

        # Find matching key
        for key_obj in self.api_keys.values():
            if hmac.compare_digest(key_obj.key_hash, key_hash):
                # Check if active
                if not key_obj.is_active:
                    logger.warning(f"Inactive API key used: {key_obj.key_id}")
                    return None

                # Check expiration
                if key_obj.expires_at and datetime.utcnow() > key_obj.expires_at:
                    logger.warning(f"Expired API key used: {key_obj.key_id}")
                    return None

                logger.debug(f"Valid API key: {key_obj.key_id}")
                return key_obj

        logger.warning("Invalid API key")
        return None

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke API key

        Args:
            key_id: Key ID to revoke

        Returns:
            True if revoked successfully
        """
        if key_id in self.api_keys:
            self.api_keys[key_id].is_active = False
            self._save_keys()
            logger.info(f"Revoked API key: {key_id}")
            return True

        logger.warning(f"API key not found: {key_id}")
        return False

    def list_api_keys(self) -> list[Dict]:
        """
        List all API keys (without hashes)

        Returns:
            List of API key information
        """
        keys = []
        for key_obj in self.api_keys.values():
            key_dict = asdict(key_obj)
            del key_dict["key_hash"]  # Don't expose hash
            keys.append(key_dict)

        return keys

    def _hash_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _save_keys(self):
        """Save API keys to storage"""
        data = {
            key_id: {
                **asdict(key_obj),
                "created_at": key_obj.created_at.isoformat(),
                "expires_at": (key_obj.expires_at.isoformat() if key_obj.expires_at else None),
            }
            for key_id, key_obj in self.api_keys.items()
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_keys(self):
        """Load API keys from storage"""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            for key_id, key_data in data.items():
                key_data["created_at"] = datetime.fromisoformat(key_data["created_at"])
                key_data["expires_at"] = (
                    datetime.fromisoformat(key_data["expires_at"])
                    if key_data["expires_at"]
                    else None
                )

                self.api_keys[key_id] = APIKey(**key_data)

            logger.info(f"Loaded {len(self.api_keys)} API keys")

        except Exception as e:
            logger.error(f"Error loading API keys: {e}")


# Global manager instance
_api_key_manager = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager"""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def generate_api_key(name: str, permissions: list[str], **kwargs) -> tuple[str, str]:
    """Helper to generate API key"""
    manager = get_api_key_manager()
    return manager.generate_api_key(name, permissions, **kwargs)


def validate_api_key(api_key: str) -> Optional[APIKey]:
    """Helper to validate API key"""
    manager = get_api_key_manager()
    return manager.validate_api_key(api_key)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = APIKeyManager("test_api_keys.json")

    # Generate key
    key_id, api_key = manager.generate_api_key(
        name="Test Integration",
        permissions=["read:tle", "read:detections"],
        rate_limit=1000,
        expires_in_days=365,
    )

    print("Generated API Key:")
    print(f"  Key ID: {key_id}")
    print(f"  API Key: {api_key}")
    print("\n⚠️  Save this API key securely - it won't be shown again!")

    # Validate
    key_obj = manager.validate_api_key(api_key)
    if key_obj:
        print("\n✓ Valid API key")
        print(f"  Name: {key_obj.name}")
        print(f"  Permissions: {key_obj.permissions}")
        print(f"  Rate limit: {key_obj.rate_limit} req/min")

    # List keys
    print("\nAll API keys:")
    for key_info in manager.list_api_keys():
        print(f"  {key_info['key_id']}: {key_info['name']}")
