"""
QR code generation and verification utilities.

This module handles QR code creation with digital signatures.
"""

import logging
import json
import hmac
import hashlib
from typing import Dict, Any, Tuple, Optional
from io import BytesIO
import base64
import time
import os

import qrcode
from PIL import Image

from config.settings import settings


logger = logging.getLogger(__name__)


class QRGenerator:
    """QR code generator with digital signatures."""
    
    def __init__(self):
        """Initialize the QR generator."""
        self.secret_key = settings.qr_secret_key
        self.use_signature = settings.use_qr_signature
        self.storage_path = settings.qr_storage_path
    
    def generate(self, catch_data: Dict[str, Any]) -> Tuple[str, str, Optional[str], str]:
        """
        Generate QR code for catch data.
        
        Args:
            catch_data: Dictionary containing catch information
            
        Returns:
            Tuple of (base64_image, json_data, file_path, signature)
        """
        # Prepare QR data
        qr_data = {
            "catch_id": str(catch_data["catch_id"]),
            "species": str(catch_data["species"]),
            "species_tamil": str(catch_data.get("species_tamil", "மீன்")),
            "freshness_days": float(catch_data["freshness_days"]),
            "freshness_category": str(catch_data["freshness_category"]),
            "price_per_kg": float(catch_data["price_per_kg"]),
            "fisher_name": str(catch_data["fisher_name"]),
            "location": str(catch_data["location"]),
            "timestamp": catch_data.get("timestamp", time.time())
        }
        
        # Convert to JSON
        qr_data_json = json.dumps(qr_data, sort_keys=True)
        
        # Generate signature
        signature = ""
        if self.use_signature:
            signature = hmac.new(
                self.secret_key.encode(),
                qr_data_json.encode(),
                hashlib.sha256
            ).hexdigest()
            qr_data["signature"] = signature
            qr_data_json = json.dumps(qr_data, sort_keys=True)
        
        # Create QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data_json)
        qr.make(fit=True)
        
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Save to file
        qr_filename = f"qr_{catch_data['catch_id']}_{int(time.time())}.png"
        qr_path = os.path.join(self.storage_path, qr_filename)
        
        try:
            qr_img.save(qr_path)
            logger.info(f"QR code saved: {qr_path}")
        except Exception as e:
            logger.error(f"Failed to save QR code: {e}")
            qr_path = None
        
        # Convert to base64
        buffer = BytesIO()
        qr_img.save(buffer, format="PNG")
        qr_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return qr_b64, qr_data_json, qr_path, signature
    
    def verify(self, qr_data_json: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify QR code signature.
        
        Args:
            qr_data_json: QR code JSON data
            
        Returns:
            Tuple of (is_valid, qr_data)
        """
        try:
            qr_data = json.loads(qr_data_json)
            
            if not self.use_signature:
                return True, qr_data
            
            if 'signature' not in qr_data:
                return False, qr_data
            
            # Extract signature
            signature = qr_data.pop('signature')
            
            # Recalculate signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                json.dumps(qr_data, sort_keys=True).encode(),
                hashlib.sha256
            ).hexdigest()
            
            is_valid = (signature == expected_signature)
            qr_data['signature'] = signature  # Restore signature
            
            return is_valid, qr_data
            
        except Exception as e:
            logger.error(f"QR verification error: {e}")
            return False, {}
