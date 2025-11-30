"""
Order data model.

This module defines the Order data class with validation and utility methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config.constants import OrderStatus


@dataclass
class Order:
    """Order data model."""
    
    order_id: str
    catch_id: str
    buyer_user_id: int
    buyer_name: str
    quantity_kg: float
    total_price: float
    status: str = OrderStatus.PENDING.value
    created_at: datetime = field(default_factory=datetime.now)
    buyer_latitude: Optional[float] = None
    buyer_longitude: Optional[float] = None
    
    def __post_init__(self):
        """Validate order data after initialization."""
        if self.quantity_kg <= 0:
            raise ValueError("Quantity must be positive")
        if self.total_price <= 0:
            raise ValueError("Total price must be positive")
        if self.status not in [s.value for s in OrderStatus]:
            raise ValueError(f"Invalid status: {self.status}")
    
    @property
    def is_pending(self) -> bool:
        """Check if order is pending."""
        return self.status == OrderStatus.PENDING.value
    
    @property
    def is_approved(self) -> bool:
        """Check if order is approved."""
        return self.status == OrderStatus.APPROVED.value
    
    @property
    def is_delivered(self) -> bool:
        """Check if order is delivered."""
        return self.status == OrderStatus.DELIVERED.value
    
    @property
    def is_rejected(self) -> bool:
        """Check if order is rejected."""
        return self.status == OrderStatus.REJECTED.value
    
    @property
    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.status == OrderStatus.CANCELLED.value
    
    @property
    def can_be_approved(self) -> bool:
        """Check if order can be approved."""
        return self.is_pending
    
    @property
    def can_be_cancelled(self) -> bool:
        """Check if order can be cancelled."""
        return self.is_pending or self.is_approved
    
    @property
    def price_per_kg(self) -> float:
        """Calculate price per kg."""
        return self.total_price / self.quantity_kg if self.quantity_kg > 0 else 0
    
    def to_dict(self) -> dict:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "catch_id": self.catch_id,
            "buyer_user_id": self.buyer_user_id,
            "buyer_name": self.buyer_name,
            "quantity_kg": self.quantity_kg,
            "total_price": self.total_price,
            "price_per_kg": self.price_per_kg,
            "status": self.status,
            "is_pending": self.is_pending,
            "is_approved": self.is_approved,
            "is_delivered": self.is_delivered,
            "is_rejected": self.is_rejected,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "buyer_latitude": self.buyer_latitude,
            "buyer_longitude": self.buyer_longitude
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Order':
        """Create Order from dictionary."""
        # Convert datetime strings to datetime objects
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
