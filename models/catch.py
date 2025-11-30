"""
Catch data model.

This module defines the Catch data class with freshness calculation
and validation methods.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

from config.constants import (
    CatchStatus,
    FreshnessCategory,
    FRESHNESS_THRESHOLDS,
    DECAY_RATES,
    STORAGE_TEMP_THRESHOLDS
)


@dataclass
class Catch:
    """Catch data model."""
    
    catch_id: str
    fisher_name: str
    species: str
    weight_g: int
    price_per_kg: float
    location: str
    latitude: float
    longitude: float
    freshness_days: float
    storage_temp: float
    hours_since_catch: float
    created_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[int] = None
    species_tamil: str = "மீன்"
    freshness_category: str = FreshnessCategory.GOOD.value
    status: str = CatchStatus.AVAILABLE.value
    area_temperature: float = 25.0
    qr_code: Optional[str] = None
    qr_signature: Optional[str] = None
    qr_path: Optional[str] = None
    image_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate catch data after initialization."""
        if self.weight_g <= 0:
            raise ValueError("Weight must be positive")
        if self.price_per_kg <= 0:
            raise ValueError("Price must be positive")
        if self.freshness_days < 0:
            raise ValueError("Freshness days cannot be negative")
    
    def calculate_current_freshness(self) -> Tuple[float, str]:
        """
        Calculate current freshness based on time elapsed and storage conditions.
        
        Returns:
            Tuple of (remaining_freshness_days, freshness_category)
        """
        current_time = datetime.now()
        time_elapsed = current_time - self.created_at
        hours_elapsed = time_elapsed.total_seconds() / 3600
        
        # Determine decay rate based on storage temperature
        if self.storage_temp <= STORAGE_TEMP_THRESHOLDS["excellent"]:
            decay_rate = DECAY_RATES["excellent_storage"]
        elif self.storage_temp <= STORAGE_TEMP_THRESHOLDS["good"]:
            decay_rate = DECAY_RATES["good_storage"]
        else:
            decay_rate = DECAY_RATES["poor_storage"]
        
        # Calculate remaining freshness
        freshness_lost = (hours_elapsed / 24) * decay_rate * self.freshness_days
        remaining_freshness = max(0, self.freshness_days - freshness_lost)
        
        # Determine category
        category = self._get_freshness_category(remaining_freshness)
        
        return round(remaining_freshness, 2), category
    
    def _get_freshness_category(self, freshness_days: float) -> str:
        """Get freshness category based on remaining days."""
        if freshness_days >= FRESHNESS_THRESHOLDS[FreshnessCategory.EXCELLENT]:
            return FreshnessCategory.EXCELLENT.value
        elif freshness_days >= FRESHNESS_THRESHOLDS[FreshnessCategory.GOOD]:
            return FreshnessCategory.GOOD.value
        elif freshness_days >= FRESHNESS_THRESHOLDS[FreshnessCategory.FAIR]:
            return FreshnessCategory.FAIR.value
        elif freshness_days >= FRESHNESS_THRESHOLDS[FreshnessCategory.POOR]:
            return FreshnessCategory.POOR.value
        else:
            return FreshnessCategory.EXPIRED.value
    
    @property
    def is_available(self) -> bool:
        """Check if catch is available for purchase."""
        return self.status == CatchStatus.AVAILABLE.value
    
    @property
    def is_sold(self) -> bool:
        """Check if catch is sold."""
        return self.status == CatchStatus.SOLD.value
    
    @property
    def is_expired(self) -> bool:
        """Check if catch is expired."""
        current_freshness, _ = self.calculate_current_freshness()
        return current_freshness <= FRESHNESS_THRESHOLDS[FreshnessCategory.POOR]
    
    @property
    def total_value(self) -> float:
        """Calculate total value of the catch."""
        return (self.weight_g / 1000) * self.price_per_kg
    
    @property
    def weight_kg(self) -> float:
        """Get weight in kilograms."""
        return self.weight_g / 1000
    
    def to_dict(self) -> dict:
        """Convert catch to dictionary."""
        current_freshness, current_category = self.calculate_current_freshness()
        
        return {
            "catch_id": self.catch_id,
            "fisher_name": self.fisher_name,
            "user_id": self.user_id,
            "species": self.species,
            "species_tamil": self.species_tamil,
            "weight_g": self.weight_g,
            "weight_kg": self.weight_kg,
            "price_per_kg": self.price_per_kg,
            "total_value": self.total_value,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "freshness_days": self.freshness_days,
            "current_freshness_days": current_freshness,
            "freshness_category": self.freshness_category,
            "current_freshness_category": current_category,
            "storage_temp": self.storage_temp,
            "hours_since_catch": self.hours_since_catch,
            "area_temperature": self.area_temperature,
            "status": self.status,
            "is_available": self.is_available,
            "is_sold": self.is_sold,
            "is_expired": self.is_expired,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "image_path": self.image_path,
            "qr_path": self.qr_path
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Catch':
        """Create Catch from dictionary."""
        # Convert datetime strings to datetime objects
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
