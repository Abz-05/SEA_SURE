"""
Catch repository for catch-related database operations.

This module handles all database operations related to fish catches,
including creation, querying, and status management.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from database.repositories.base_repository import BaseRepository
from config.constants import CatchStatus


logger = logging.getLogger(__name__)


class CatchRepository(BaseRepository):
    """Repository for catch database operations."""
    
    @property
    def table_name(self) -> str:
        """Get the table name."""
        return "catches"
    
    @property
    def primary_key(self) -> str:
        """Get the primary key column name."""
        return "catch_id"
    
    def create_catch(self, catch_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new catch record.
        
        Args:
            catch_data: Dictionary containing catch information
            
        Returns:
            Catch ID if successful, None otherwise
        """
        # Ensure created_at is set
        if "created_at" not in catch_data:
            catch_data["created_at"] = datetime.now()
        
        # Ensure status is set
        if "status" not in catch_data:
            catch_data["status"] = CatchStatus.AVAILABLE.value
        
        return self.create(catch_data)
    
    def find_by_fisher(
        self,
        fisher_name: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find all catches by a specific fisher.
        
        Args:
            fisher_name: Name of the fisher
            limit: Maximum number of results
            
        Returns:
            List of catches
        """
        return self.find_by_criteria(
            {"fisher_name": fisher_name},
            limit=limit,
            order_by="created_at"
        )
    
    def find_by_user_id(
        self,
        user_id: int,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find all catches by a specific user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            
        Returns:
            List of catches
        """
        return self.find_by_criteria(
            {"user_id": user_id},
            limit=limit,
            order_by="created_at"
        )
    
    def find_available_catches(
        self,
        species: Optional[List[str]] = None,
        min_freshness: float = 0.0,
        max_price: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find available catches with optional filters.
        
        Args:
            species: List of species to filter by
            min_freshness: Minimum freshness in days
            max_price: Maximum price per kg
            limit: Maximum number of results
            
        Returns:
            List of available catches
        """
        query = "SELECT * FROM catches WHERE status = %s AND freshness_days >= %s"
        params = [CatchStatus.AVAILABLE.value, min_freshness]
        
        if species:
            placeholders = ','.join(['%s'] * len(species))
            query += f" AND species IN ({placeholders})"
            params.extend(species)
        
        if max_price:
            query += " AND price_per_kg <= %s"
            params.append(max_price)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.db.execute_query(query, params=tuple(params), fetch_all=True) or []
    
    def update_status(self, catch_id: str, status: str) -> bool:
        """
        Update catch status.
        
        Args:
            catch_id: Catch ID
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        return self.update(catch_id, {"status": status})
    
    def mark_as_sold(self, catch_id: str) -> bool:
        """
        Mark a catch as sold.
        
        Args:
            catch_id: Catch ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(catch_id, CatchStatus.SOLD.value)
    
    def mark_as_expired(self, catch_id: str) -> bool:
        """
        Mark a catch as expired.
        
        Args:
            catch_id: Catch ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(catch_id, CatchStatus.EXPIRED.value)
    
    def get_catches_by_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 50.0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find catches within a radius of a location.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Radius in kilometers
            status: Optional status filter
            
        Returns:
            List of catches within radius
        """
        # Using Haversine formula for distance calculation
        query = """
            SELECT *,
                (6371 * acos(
                    cos(radians(%s)) * cos(radians(latitude)) *
                    cos(radians(longitude) - radians(%s)) +
                    sin(radians(%s)) * sin(radians(latitude))
                )) AS distance_km
            FROM catches
            WHERE (6371 * acos(
                cos(radians(%s)) * cos(radians(latitude)) *
                cos(radians(longitude) - radians(%s)) +
                sin(radians(%s)) * sin(radians(latitude))
            )) <= %s
        """
        params = [latitude, longitude, latitude, latitude, longitude, latitude, radius_km]
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        query += " ORDER BY distance_km ASC"
        
        return self.db.execute_query(query, params=tuple(params), fetch_all=True) or []
    
    def get_species_statistics(self) -> List[Dict[str, Any]]:
        """
        Get statistics by species.
        
        Returns:
            List of species with counts and average prices
        """
        query = """
            SELECT 
                species,
                COUNT(*) as count,
                AVG(price_per_kg) as avg_price,
                AVG(freshness_days) as avg_freshness,
                SUM(CASE WHEN status = 'available' THEN 1 ELSE 0 END) as available_count
            FROM catches
            GROUP BY species
            ORDER BY count DESC
        """
        
        return self.db.execute_query(query, fetch_all=True) or []
    
    def get_fisher_statistics(self, fisher_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific fisher.
        
        Args:
            fisher_name: Name of the fisher
            
        Returns:
            Dictionary with fisher statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_catches,
                SUM(CASE WHEN status = 'available' THEN 1 ELSE 0 END) as available_catches,
                SUM(CASE WHEN status = 'sold' THEN 1 ELSE 0 END) as sold_catches,
                AVG(freshness_days) as avg_freshness,
                SUM(weight_g) / 1000.0 as total_weight_kg,
                AVG(price_per_kg) as avg_price
            FROM catches
            WHERE fisher_name = %s
        """
        
        return self.db.execute_query(query, params=(fisher_name,), fetch_one=True)
    
    def search_catches(
        self,
        search_term: str,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search catches by species or location.
        
        Args:
            search_term: Search term
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List of matching catches
        """
        query = """
            SELECT * FROM catches
            WHERE (species ILIKE %s OR location ILIKE %s OR catch_id LIKE %s)
        """
        params = [f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"]
        
        if status:
            query += " AND status = %s"
            params.append(status)
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        return self.db.execute_query(query, params=tuple(params), fetch_all=True) or []
