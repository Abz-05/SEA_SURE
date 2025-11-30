"""
User data model.

This module defines the User data class with validation and utility methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config.constants import UserRole


@dataclass
class User:
    """User data model."""
    
    user_id: int
    name: str
    phone: str
    role: str
    verified: bool = False
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    password_hash: Optional[str] = None
    
    def __post_init__(self):
        """Validate user data after initialization."""
        if self.role not in [r.value for r in UserRole]:
            raise ValueError(f"Invalid role: {self.role}")
    
    @property
    def is_fisher(self) -> bool:
        """Check if user is a fisher."""
        return self.role == UserRole.FISHER.value
    
    @property
    def is_buyer(self) -> bool:
        """Check if user is a buyer."""
        return self.role == UserRole.BUYER.value
    
    @property
    def is_admin(self) -> bool:
        """Check if user is an admin."""
        return self.role == UserRole.ADMIN.value
    
    @property
    def display_role(self) -> str:
        """Get display-friendly role name."""
        role_map = {
            UserRole.FISHER.value: "Fisher",
            UserRole.BUYER.value: "Buyer",
            UserRole.ADMIN.value: "Administrator"
        }
        return role_map.get(self.role, self.role.title())
    
    def to_dict(self) -> dict:
        """Convert user to dictionary (excluding password)."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "phone": self.phone,
            "role": self.role,
            "verified": self.verified,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        """Create User from dictionary."""
        # Convert datetime strings to datetime objects
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('last_login'), str):
            data['last_login'] = datetime.fromisoformat(data['last_login'])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
