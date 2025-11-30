"""
ML service for fish species and freshness prediction.

This module wraps the ML model with caching and error handling.
"""

import logging
from typing import Optional, Dict, Any, Tuple
from PIL import Image

from config.settings import settings


logger = logging.getLogger(__name__)


class MLService:
    """Service for ML predictions."""
    
    def __init__(self):
        """Initialize the ML service."""
        self.predictor = None
        self._initialized = False
        self._load_model()
    
    def _load_model(self):
        """Load the ML model."""
        try:
            from combined_inference import CombinedFishPredictor
            self.predictor = CombinedFishPredictor()
            self._initialized = True
            logger.info("ML model loaded successfully")
        except Exception as e:
            logger.warning(f"ML model not available: {e}")
            self._initialized = False
    
    @property
    def is_available(self) -> bool:
        """Check if ML model is available."""
        return self._initialized and self.predictor is not None
    
    def predict(
        self,
        image: Image.Image,
        weight: int,
        storage_temp: float,
        hours_since_catch: float,
        area_temp: float
    ) -> Dict[str, Any]:
        """
        Predict fish species and freshness.
        
        Args:
            image: Fish image
            weight: Weight in grams
            storage_temp: Storage temperature
            hours_since_catch: Hours since catch
            area_temp: Area temperature
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_available:
            return {
                "prediction_success": False,
                "error": "ML model not available",
                "species": "Unknown",
                "species_tamil": "மீன்",
                "freshness_days_remaining": 2.0,
                "freshness_category": "Good",
                "species_confidence": 0.0,
                "recommendations": ["ML model not loaded. Using default values."]
            }
        
        try:
            result = self.predictor.predict_complete(
                image,
                weight=weight,
                storage_temp=storage_temp,
                hours_since_catch=hours_since_catch,
                area_temp=area_temp
            )
            return result
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {
                "prediction_success": False,
                "error": str(e),
                "species": "Unknown",
                "species_tamil": "மீன்",
                "freshness_days_remaining": 2.0,
                "freshness_category": "Good",
                "species_confidence": 0.0,
                "recommendations": [f"Prediction failed: {str(e)}"]
            }
