"""
Catch service for catch management and ML integration.

This module handles catch creation, querying, and ML-based
fish species and freshness prediction.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid

from database.repositories import CatchRepository
from models import Catch
from config.constants import CatchStatus, TAMIL_FISH_NAMES
from PIL import Image


logger = logging.getLogger(__name__)


class CatchService:
    """Service for catch management."""
    
    def __init__(self):
        """Initialize the catch service."""
        self.catch_repo = CatchRepository()
    
    def create_catch(
        self,
        fisher_name: str,
        user_id: int,
        species: str,
        weight_g: int,
        price_per_kg: float,
        location: str,
        latitude: float,
        longitude: float,
        freshness_days: float,
        storage_temp: float,
        hours_since_catch: float,
        area_temperature: float,
        species_tamil: Optional[str] = None,
        image_path: Optional[str] = None,
        qr_code: Optional[str] = None,
        qr_signature: Optional[str] = None,
        qr_path: Optional[str] = None
    ) -> Optional[Catch]:
        """
        Create a new catch record.
        
        Args:
            fisher_name: Name of the fisher
            user_id: Fisher's user ID
            species: Fish species
            weight_g: Weight in grams
            price_per_kg: Price per kilogram
            location: Location name
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            freshness_days: Initial freshness in days
            storage_temp: Storage temperature in Celsius
            hours_since_catch: Hours since catch
            area_temperature: Area temperature
            species_tamil: Tamil name (optional)
            image_path: Path to fish image (optional)
            qr_code: QR code data (optional)
            qr_signature: QR signature (optional)
            qr_path: QR code image path (optional)
            
        Returns:
            Catch object if successful, None otherwise
        """
        # Generate catch ID
        catch_id = str(uuid.uuid4())[:8].upper()
        
        # Get Tamil name if not provided
        if not species_tamil:
            species_tamil = TAMIL_FISH_NAMES.get(species, "மீன்")
        
        # Determine freshness category
        catch = Catch(
            catch_id=catch_id,
            fisher_name=fisher_name,
            user_id=user_id,
            species=species,
            species_tamil=species_tamil,
            weight_g=weight_g,
            price_per_kg=price_per_kg,
            location=location,
            latitude=latitude,
            longitude=longitude,
            freshness_days=freshness_days,
            storage_temp=storage_temp,
            hours_since_catch=hours_since_catch,
            area_temperature=area_temperature,
            image_path=image_path,
            qr_code=qr_code,
            qr_signature=qr_signature,
            qr_path=qr_path
        )
        
        # Save to database
        try:
            result = self.catch_repo.create_catch(catch.to_dict())
            if result:
                logger.info(f"Catch created: {catch_id} ({species})")
                return catch
            return None
        except Exception as e:
            logger.error(f"Failed to create catch: {e}")
            return None
    
    def get_catch_by_id(self, catch_id: str) -> Optional[Catch]:
        """
        Get catch by ID.
        
        Args:
            catch_id: Catch ID
            
        Returns:
            Catch object or None
        """
        catch_data = self.catch_repo.find_by_id(catch_id)
        if catch_data:
            return Catch.from_dict(catch_data)
        return None
    
    def get_fisher_catches(
        self,
        fisher_name: str,
        limit: Optional[int] = None
    ) -> List[Catch]:
        """
        Get all catches for a fisher.
        
        Args:
            fisher_name: Fisher's name
            limit: Maximum number of results
            
        Returns:
            List of Catch objects
        """
        catches_data = self.catch_repo.find_by_fisher(fisher_name, limit)
        return [Catch.from_dict(c) for c in catches_data]
    
    def get_available_catches(
        self,
        species: Optional[List[str]] = None,
        min_freshness: float = 0.0,
        max_price: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Catch]:
        """
        Get available catches with filters.
        
        Args:
            species: List of species to filter by
            min_freshness: Minimum freshness in days
            max_price: Maximum price per kg
            limit: Maximum number of results
            
        Returns:
            List of Catch objects
        """
        catches_data = self.catch_repo.find_available_catches(
            species, min_freshness, max_price, limit
        )
        return [Catch.from_dict(c) for c in catches_data]
    
    def update_catch_status(self, catch_id: str, status: str) -> bool:
        """
        Update catch status.
        
        Args:
            catch_id: Catch ID
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        return self.catch_repo.update_status(catch_id, status)
    
    def mark_as_sold(self, catch_id: str) -> bool:
        """Mark catch as sold."""
        return self.catch_repo.mark_as_sold(catch_id)
    
    def mark_as_expired(self, catch_id: str) -> bool:
        """Mark catch as expired."""
        return self.catch_repo.mark_as_expired(catch_id)
    
    def get_catches_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0
    ) -> List[Catch]:
        """
        Get catches near a location.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Search radius in kilometers
            
        Returns:
            List of Catch objects
        """
        catches_data = self.catch_repo.get_catches_by_location(
            latitude, longitude, radius_km, CatchStatus.AVAILABLE.value
        )
        return [Catch.from_dict(c) for c in catches_data]
