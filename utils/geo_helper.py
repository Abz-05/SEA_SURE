"""
Geographic helper utilities.

This module provides geographic calculations and location services.
"""

import logging
from typing import Optional, Tuple, Dict
import random
import numpy as np

from config.constants import TN_COASTAL_CITIES, DEFAULT_LOCATION, EARTH_RADIUS_KM

# Optional imports
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False


logger = logging.getLogger(__name__)


class GeoHelper:
    """Helper class for geographic operations."""
    
    def __init__(self):
        """Initialize the geo helper."""
        self.geocoder = None
        if GEOPY_AVAILABLE:
            try:
                self.geocoder = Nominatim(user_agent="sea_sure_app")
            except Exception as e:
                logger.warning(f"Geocoder initialization failed: {e}")
    
    def calculate_distance_km(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """
        Calculate distance between two coordinates in kilometers.
        
        Args:
            lat1: First latitude
            lon1: First longitude
            lat2: Second latitude
            lon2: Second longitude
            
        Returns:
            Distance in kilometers
        """
        if GEOPY_AVAILABLE:
            try:
                return geodesic((lat1, lon1), (lat2, lon2)).kilometers
            except:
                pass
        
        # Fallback: Haversine formula
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return EARTH_RADIUS_KM * c
    
    def get_city_name(self, lat: float, lon: float) -> str:
        """
        Get city name from coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            City name
        """
        if self.geocoder:
            try:
                location = self.geocoder.reverse(f"{lat}, {lon}", timeout=5)
                if location:
                    address = location.raw.get('address', {})
                    city = address.get('city') or address.get('town') or address.get('village')
                    if city:
                        return city
            except Exception as e:
                logger.debug(f"Geocoding failed: {e}")
        
        # Fallback: Find nearest coastal city
        min_distance = float('inf')
        nearest_city = "Unknown Location"
        
        for city, (city_lat, city_lon) in TN_COASTAL_CITIES.items():
            distance = self.calculate_distance_km(lat, lon, city_lat, city_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_city = city
        
        return f"Near {nearest_city}"
    
    def get_random_coastal_location(
        self,
        base_city: str = 'Chennai',
        radius_km: float = 50.0
    ) -> Tuple[float, float]:
        """
        Get random location near a coastal city.
        
        Args:
            base_city: Base city name
            radius_km: Radius in kilometers
            
        Returns:
            Tuple of (latitude, longitude)
        """
        base_lat, base_lon = TN_COASTAL_CITIES.get(base_city, DEFAULT_LOCATION)
        
        # Random angle and distance
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0, radius_km)
        
        # Calculate offset
        lat_offset = (distance * np.cos(angle)) / 111.0
        lon_offset = (distance * np.sin(angle)) / (111.0 * np.cos(np.radians(base_lat)))
        
        return base_lat + lat_offset, base_lon + lon_offset
    
    def get_coastal_cities(self) -> Dict[str, Tuple[float, float]]:
        """Get all coastal cities."""
        return TN_COASTAL_CITIES.copy()
