"""
Application constants and enumerations.

This module defines all constant values used throughout the application
to avoid magic numbers and strings.
"""

from enum import Enum
from typing import Dict, Tuple


# ============================================================================
# Enumerations
# ============================================================================

class FreshnessCategory(str, Enum):
    """Fish freshness categories."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    EXPIRED = "Expired"


class CatchStatus(str, Enum):
    """Catch status values."""
    AVAILABLE = "available"
    SOLD = "sold"
    EXPIRED = "expired"
    RESERVED = "reserved"


class OrderStatus(str, Enum):
    """Order status values."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class UserRole(str, Enum):
    """User role types."""
    FISHER = "fisher"
    BUYER = "buyer"
    ADMIN = "admin"


class NotificationType(str, Enum):
    """Notification types."""
    ORDER = "order"
    CATCH = "catch"
    SYSTEM = "system"
    ALERT = "alert"


# ============================================================================
# Freshness Constants
# ============================================================================

# Freshness thresholds in days
FRESHNESS_THRESHOLDS = {
    FreshnessCategory.EXCELLENT: 2.0,
    FreshnessCategory.GOOD: 1.5,
    FreshnessCategory.FAIR: 1.0,
    FreshnessCategory.POOR: 0.5,
    FreshnessCategory.EXPIRED: 0.0
}

# Decay rates based on storage temperature (per day)
DECAY_RATES = {
    "excellent_storage": 0.015,  # <= 4°C
    "good_storage": 0.03,        # 4-10°C
    "poor_storage": 0.06         # > 10°C
}

# Temperature thresholds for storage quality
STORAGE_TEMP_THRESHOLDS = {
    "excellent": 4.0,
    "good": 10.0
}


# ============================================================================
# Geographic Constants
# ============================================================================

# Tamil Nadu coastal cities with coordinates (latitude, longitude)
TN_COASTAL_CITIES: Dict[str, Tuple[float, float]] = {
    'Chennai': (13.0827, 80.2707),
    'Kanyakumari': (8.0883, 77.5385),
    'Tuticorin': (8.7642, 78.1348),
    'Nagapattinam': (10.7669, 79.8420),
    'Karaikal': (10.9254, 79.8380),
    'Cuddalore': (11.7529, 79.7714),
    'Mamallapuram': (12.6267, 80.1927),
    'Rameswaram': (9.2876, 79.3129),
    'Mandapam': (9.2806, 79.1378),
    'Pamban': (9.2750, 79.2093)
}

# Default location (Chennai)
DEFAULT_LOCATION = (13.0827, 80.2707)

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


# ============================================================================
# UI Constants
# ============================================================================

UI_CONSTANTS = {
    # Colors
    "primary_color": "#667eea",
    "secondary_color": "#764ba2",
    "success_color": "#28a745",
    "warning_color": "#ffc107",
    "danger_color": "#dc3545",
    "info_color": "#2196f3",
    
    # Freshness indicator colors (RGBA)
    "freshness_high_color": [0, 255, 0, 160],
    "freshness_medium_color": [255, 255, 0, 160],
    "freshness_low_color": [255, 0, 0, 160],
    
    # Icons
    "freshness_high_icon": "🟢",
    "freshness_medium_icon": "🟡",
    "freshness_low_icon": "🔴",
    
    # Sizes
    "max_image_width": 400,
    "thumbnail_width": 150,
    "qr_code_size": 200,
    
    # Limits
    "max_catches_per_page": 50,
    "max_orders_per_page": 50,
    "max_notifications": 20,
    
    # Map settings
    "default_map_zoom": 8,
    "default_3d_map_pitch": 50,
}


# ============================================================================
# Validation Constants
# ============================================================================

# Image validation
IMAGE_VALIDATION = {
    "min_width": 300,
    "min_height": 300,
    "max_pixels": 10_000_000,
    "blur_threshold": 100,
    "allowed_formats": ["jpg", "jpeg", "png"],
    "max_file_size_mb": 10
}

# Phone validation pattern
PHONE_PATTERN = r'^(\+91)?[6-9]\d{9}$'

# Password requirements
PASSWORD_REQUIREMENTS = {
    "min_length": 6,
    "max_length": 128,
    "require_uppercase": False,
    "require_lowercase": False,
    "require_digit": False,
    "require_special": False
}


# ============================================================================
# Business Logic Constants
# ============================================================================

# Price ranges (INR per kg)
PRICE_RANGES = {
    "min_price": 50,
    "max_price": 5000,
    "default_price": 300
}

# Weight ranges (grams)
WEIGHT_RANGES = {
    "min_weight": 1,
    "max_weight": 50000,
    "default_weight": 500
}

# Temperature ranges (Celsius)
TEMPERATURE_RANGES = {
    "min_storage_temp": 0,
    "max_storage_temp": 35,
    "default_storage_temp": 5,
    "min_area_temp": 20,
    "max_area_temp": 40
}

# Time ranges
TIME_RANGES = {
    "max_hours_since_catch": 72,
    "default_hours_since_catch": 6,
    "otp_validity_minutes": 5,
    "session_timeout_minutes": 60
}


# ============================================================================
# Database Constants
# ============================================================================

# Pagination
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100

# Query timeouts (seconds)
QUERY_TIMEOUT = 30
LONG_QUERY_TIMEOUT = 60


# ============================================================================
# Error Messages
# ============================================================================

ERROR_MESSAGES = {
    # Authentication
    "invalid_credentials": "Invalid phone number or password",
    "account_deactivated": "Your account has been deactivated",
    "phone_already_registered": "Phone number already registered",
    "invalid_phone": "Please enter a valid phone number",
    "weak_password": "Password must be at least 6 characters",
    "passwords_mismatch": "Passwords do not match",
    "rate_limit_exceeded": "Too many attempts. Please try again later",
    
    # Catch
    "catch_not_found": "Catch not found",
    "catch_already_sold": "This catch has already been sold",
    "catch_expired": "This catch has expired",
    "invalid_image": "Invalid or poor quality image",
    
    # Order
    "order_not_found": "Order not found",
    "insufficient_quantity": "Insufficient quantity available",
    "order_already_processed": "Order has already been processed",
    
    # System
    "database_error": "Database error occurred",
    "ml_model_error": "ML model prediction failed",
    "api_error": "External API error",
    "unknown_error": "An unexpected error occurred"
}


# ============================================================================
# Success Messages
# ============================================================================

SUCCESS_MESSAGES = {
    "login_success": "Login successful",
    "registration_success": "Account created successfully",
    "catch_created": "Catch recorded successfully",
    "order_placed": "Order placed successfully",
    "order_approved": "Order approved successfully",
    "order_delivered": "Order marked as delivered",
    "qr_verified": "QR code verified successfully"
}


# ============================================================================
# Tamil Fish Names Mapping
# ============================================================================

TAMIL_FISH_NAMES = {
    "Tuna": "சூரை",
    "Mackerel": "கானாங்கெளுத்தி",
    "Sardine": "மத்தி",
    "Pomfret": "வாவல்",
    "Kingfish": "வஞ்சிரம்",
    "Seer Fish": "வஞ்சிரம்",
    "Anchovy": "நெத்திலி",
    "Prawn": "இறால்",
    "Crab": "நண்டு",
    "Squid": "கணவாய்",
    "Catfish": "கெளுத்தி",
    "Snapper": "விலாங்கு",
    "Grouper": "கல்லுமீன்",
    "Barracuda": "ஓலா",
    "Mullet": "கணவாய்"
}


# ============================================================================
# Export all constants
# ============================================================================

__all__ = [
    # Enums
    'FreshnessCategory',
    'CatchStatus',
    'OrderStatus',
    'UserRole',
    'NotificationType',
    
    # Freshness
    'FRESHNESS_THRESHOLDS',
    'DECAY_RATES',
    'STORAGE_TEMP_THRESHOLDS',
    
    # Geographic
    'TN_COASTAL_CITIES',
    'DEFAULT_LOCATION',
    'EARTH_RADIUS_KM',
    
    # UI
    'UI_CONSTANTS',
    
    # Validation
    'IMAGE_VALIDATION',
    'PHONE_PATTERN',
    'PASSWORD_REQUIREMENTS',
    
    # Business Logic
    'PRICE_RANGES',
    'WEIGHT_RANGES',
    'TEMPERATURE_RANGES',
    'TIME_RANGES',
    
    # Database
    'DEFAULT_PAGE_SIZE',
    'MAX_PAGE_SIZE',
    'QUERY_TIMEOUT',
    'LONG_QUERY_TIMEOUT',
    
    # Messages
    'ERROR_MESSAGES',
    'SUCCESS_MESSAGES',
    
    # Tamil Names
    'TAMIL_FISH_NAMES'
]
