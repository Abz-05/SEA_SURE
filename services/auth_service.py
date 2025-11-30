"""
Authentication service for user management and authentication.

This module handles user registration, login, password management,
and OTP verification.
"""

import logging
import re
import random
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import bcrypt

from database.repositories import UserRepository
from models import User
from config.constants import UserRole, ERROR_MESSAGES, SUCCESS_MESSAGES, PHONE_PATTERN
from config.settings import settings

# Twilio imports (optional)
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


logger = logging.getLogger(__name__)


class AuthService:
    """Service for authentication and user management."""
    
    def __init__(self):
        """Initialize the authentication service."""
        self.user_repo = UserRepository()
        self.twilio_client = None
        self._otp_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Twilio if available
        if TWILIO_AVAILABLE and settings.twilio_enabled:
            try:
                self.twilio_client = TwilioClient(
                    settings.twilio_account_sid,
                    settings.twilio_auth_token
                )
                logger.info("Twilio client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio: {e}")
    
    def validate_phone(self, phone: str) -> bool:
        """
        Validate phone number format.
        
        Args:
            phone: Phone number to validate
            
        Returns:
            True if valid, False otherwise
        """
        phone_clean = phone.replace(' ', '').replace('-', '')
        return bool(re.match(PHONE_PATTERN, phone_clean))
    
    def normalize_phone(self, phone: str) -> str:
        """
        Normalize phone number to standard format.
        
        Args:
            phone: Phone number to normalize
            
        Returns:
            Normalized phone number with +91 prefix
        """
        phone = phone.replace(' ', '').replace('-', '')
        if len(phone) == 10:
            phone = '+91' + phone
        elif not phone.startswith('+91'):
            phone = '+91' + phone
        return phone
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed: Hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def register(
        self,
        name: str,
        phone: str,
        password: str,
        role: str = UserRole.BUYER.value
    ) -> tuple[Optional[User], Optional[str]]:
        """
        Register a new user.
        
        Args:
            name: User's full name
            phone: Phone number
            password: Plain text password
            role: User role
            
        Returns:
            Tuple of (User object, error message)
        """
        # Validate phone
        if not self.validate_phone(phone):
            return None, ERROR_MESSAGES["invalid_phone"]
        
        # Normalize phone
        normalized_phone = self.normalize_phone(phone)
        
        # Check if phone already exists
        if self.user_repo.phone_exists(normalized_phone):
            return None, ERROR_MESSAGES["phone_already_registered"]
        
        # Validate password
        if len(password) < 6:
            return None, ERROR_MESSAGES["weak_password"]
        
        # Hash password
        password_hash = self.hash_password(password)
        
        # Create user
        try:
            user_id = self.user_repo.create_user(
                name=name,
                phone=normalized_phone,
                password_hash=password_hash,
                role=role
            )
            
            if user_id:
                user_data = self.user_repo.find_by_id(user_id)
                if user_data:
                    user = User.from_dict(user_data)
                    logger.info(f"User registered: {name} ({normalized_phone})")
                    return user, None
            
            return None, ERROR_MESSAGES["unknown_error"]
            
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return None, str(e)
    
    def login(
        self,
        phone: str,
        password: str
    ) -> tuple[Optional[User], Optional[str]]:
        """
        Authenticate a user.
        
        Args:
            phone: Phone number
            password: Plain text password
            
        Returns:
            Tuple of (User object, error message)
        """
        # Validate phone
        if not self.validate_phone(phone):
            return None, ERROR_MESSAGES["invalid_phone"]
        
        # Normalize phone
        normalized_phone = self.normalize_phone(phone)
        
        # Find user
        user_data = self.user_repo.find_by_phone(normalized_phone)
        
        if not user_data:
            return None, ERROR_MESSAGES["invalid_credentials"]
        
        # Check if account is active
        if not user_data.get('is_active', True):
            return None, ERROR_MESSAGES["account_deactivated"]
        
        # Verify password
        if not self.verify_password(password, user_data['password_hash']):
            return None, ERROR_MESSAGES["invalid_credentials"]
        
        # Update last login
        self.user_repo.update_last_login(user_data['user_id'])
        
        # Create User object
        user = User.from_dict(user_data)
        logger.info(f"User logged in: {user.name} ({user.phone})")
        
        return user, None
    
    def generate_otp(self) -> str:
        """
        Generate a 6-digit OTP.
        
        Returns:
            6-digit OTP string
        """
        return f"{random.randint(100000, 999999):06d}"
    
    def send_otp(self, phone: str) -> tuple[bool, Optional[str]]:
        """
        Send OTP to phone number via SMS.
        
        Args:
            phone: Phone number
            
        Returns:
            Tuple of (success, error message)
        """
        if not self.validate_phone(phone):
            return False, ERROR_MESSAGES["invalid_phone"]
        
        normalized_phone = self.normalize_phone(phone)
        otp = self.generate_otp()
        
        # Store OTP in cache with expiry
        self._otp_cache[normalized_phone] = {
            'otp': otp,
            'expires_at': datetime.now() + timedelta(minutes=5),
            'attempts': 0
        }
        
        # Send SMS if Twilio is available
        if self.twilio_client and settings.twilio_from_number:
            try:
                self.twilio_client.messages.create(
                    body=f"Your SEA_SURE verification code is: {otp}. Valid for 5 minutes.",
                    from_=settings.twilio_from_number,
                    to=normalized_phone
                )
                logger.info(f"OTP sent to {normalized_phone}")
                return True, None
            except Exception as e:
                logger.error(f"Failed to send OTP: {e}")
                return False, str(e)
        else:
            # In development mode, log OTP
            if settings.dev_mode:
                logger.info(f"OTP for {normalized_phone}: {otp}")
                return True, None
            else:
                return False, "SMS service not configured"
    
    def verify_otp(self, phone: str, otp: str) -> bool:
        """
        Verify OTP for a phone number.
        
        Args:
            phone: Phone number
            otp: OTP to verify
            
        Returns:
            True if OTP is valid, False otherwise
        """
        normalized_phone = self.normalize_phone(phone)
        
        if normalized_phone not in self._otp_cache:
            return False
        
        otp_data = self._otp_cache[normalized_phone]
        
        # Check if expired
        if datetime.now() > otp_data['expires_at']:
            del self._otp_cache[normalized_phone]
            return False
        
        # Check attempts
        if otp_data['attempts'] >= 3:
            del self._otp_cache[normalized_phone]
            return False
        
        # Verify OTP
        if otp == otp_data['otp']:
            del self._otp_cache[normalized_phone]
            return True
        else:
            otp_data['attempts'] += 1
            return False
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None
        """
        user_data = self.user_repo.find_by_id(user_id)
        if user_data:
            return User.from_dict(user_data)
        return None
    
    def get_user_by_phone(self, phone: str) -> Optional[User]:
        """
        Get user by phone number.
        
        Args:
            phone: Phone number
            
        Returns:
            User object or None
        """
        normalized_phone = self.normalize_phone(phone)
        user_data = self.user_repo.find_by_phone(normalized_phone)
        if user_data:
            return User.from_dict(user_data)
        return None
