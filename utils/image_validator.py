"""
Image validation utilities.

This module provides image quality validation and preprocessing.
"""

import logging
from typing import Tuple
import numpy as np
from PIL import Image
import cv2

from config.constants import IMAGE_VALIDATION


logger = logging.getLogger(__name__)


class ImageValidator:
    """Image quality validator."""
    
    def __init__(self):
        """Initialize the image validator."""
        self.min_width = IMAGE_VALIDATION["min_width"]
        self.min_height = IMAGE_VALIDATION["min_height"]
        self.max_pixels = IMAGE_VALIDATION["max_pixels"]
        self.blur_threshold = IMAGE_VALIDATION["blur_threshold"]
    
    def validate(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Validate image quality.
        
        Args:
            image: PIL Image object
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check size
            width, height = image.size
            
            if width < self.min_width or height < self.min_height:
                return False, f"Image too small. Minimum size: {self.min_width}x{self.min_height} pixels"
            
            if width * height > self.max_pixels:
                return False, "Image too large. Please use a smaller image"
            
            # Check blur
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < self.blur_threshold:
                return False, "Image appears blurry. Please take a clearer photo"
            
            return True, "Image quality acceptable"
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return True, "Could not validate image quality"
    
    def save_image(self, image: Image.Image, catch_id: str, storage_path: str) -> Optional[str]:
        """
        Save fish image to storage.
        
        Args:
            image: PIL Image object
            catch_id: Catch ID
            storage_path: Storage directory path
            
        Returns:
            File path if successful, None otherwise
        """
        try:
            import time
            filename = f"fish_{catch_id}_{int(time.time())}.png"
            filepath = os.path.join(storage_path, filename)
            image.save(filepath)
            logger.info(f"Fish image saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save fish image: {e}")
            return None


# Add missing import
import os
from typing import Optional
