"""
User repository for user-related database operations.

This module handles all database operations related to users,
including authentication, profile management, and user queries.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from database.repositories.base_repository import BaseRepository
from config.constants import UserRole


logger = logging.getLogger(__name__)


class UserRepository(BaseRepository):
    """Repository for user database operations."""
    
    @property
    def table_name(self) -> str:
        """Get the table name."""
        return "users"
    
    @property
    def primary_key(self) -> str:
        """Get the primary key column name."""
        return "user_id"
    
    def find_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """
        Find a user by phone number.
        
        Args:
            phone: Phone number to search for
            
        Returns:
            User record or None if not found
        """
        query = "SELECT * FROM users WHERE phone = %s"
        return self.db.execute_query(query, params=(phone,), fetch_one=True)
    
    def create_user(
        self,
        name: str,
        phone: str,
        password_hash: str,
        role: str = UserRole.BUYER.value
    ) -> Optional[int]:
        """
        Create a new user.
        
        Args:
            name: User's full name
            phone: Phone number
            password_hash: Hashed password
            role: User role (fisher, buyer, admin)
            
        Returns:
            User ID if successful, None otherwise
        """
        data = {
            "name": name,
            "phone": phone,
            "password_hash": password_hash,
            "role": role,
            "verified": False,
            "is_active": True,
            "created_at": datetime.now()
        }
        
        return self.create(data)
    
    def update_last_login(self, user_id: int) -> bool:
        """
        Update user's last login timestamp.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update(user_id, {"last_login": datetime.now()})
    
    def verify_user(self, user_id: int) -> bool:
        """
        Mark user as verified.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update(user_id, {"verified": True})
    
    def deactivate_user(self, user_id: int) -> bool:
        """
        Deactivate a user account.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update(user_id, {"is_active": False})
    
    def activate_user(self, user_id: int) -> bool:
        """
        Activate a user account.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update(user_id, {"is_active": True})
    
    def find_by_role(self, role: str) -> List[Dict[str, Any]]:
        """
        Find all users with a specific role.
        
        Args:
            role: User role to filter by
            
        Returns:
            List of users with the specified role
        """
        return self.find_by_criteria({"role": role}, order_by="created_at")
    
    def find_active_users(self) -> List[Dict[str, Any]]:
        """
        Find all active users.
        
        Returns:
            List of active users
        """
        return self.find_by_criteria({"is_active": True}, order_by="created_at")
    
    def phone_exists(self, phone: str) -> bool:
        """
        Check if a phone number is already registered.
        
        Args:
            phone: Phone number to check
            
        Returns:
            True if phone exists, False otherwise
        """
        query = "SELECT EXISTS(SELECT 1 FROM users WHERE phone = %s) as exists"
        result = self.db.execute_query(query, params=(phone,), fetch_one=True)
        return result['exists'] if result else False
    
    def get_user_stats(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with user statistics
        """
        query = """
            SELECT 
                u.user_id,
                u.name,
                u.role,
                COALESCE(catch_count, 0) as total_catches,
                COALESCE(order_count, 0) as total_orders,
                COALESCE(total_revenue, 0) as total_revenue
            FROM users u
            LEFT JOIN (
                SELECT fisher_name, COUNT(*) as catch_count
                FROM catches
                GROUP BY fisher_name
            ) c ON u.name = c.fisher_name
            LEFT JOIN (
                SELECT buyer_user_id, COUNT(*) as order_count, SUM(total_price) as total_revenue
                FROM orders
                GROUP BY buyer_user_id
            ) o ON u.user_id = o.buyer_user_id
            WHERE u.user_id = %s
        """
        
        return self.db.execute_query(query, params=(user_id,), fetch_one=True)
    
    def search_users(
        self,
        search_term: str,
        role: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search users by name or phone.
        
        Args:
            search_term: Search term for name or phone
            role: Optional role filter
            limit: Maximum number of results
            
        Returns:
            List of matching users
        """
        query = """
            SELECT * FROM users
            WHERE (name ILIKE %s OR phone LIKE %s)
        """
        params = [f"%{search_term}%", f"%{search_term}%"]
        
        if role:
            query += " AND role = %s"
            params.append(role)
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        return self.db.execute_query(query, params=tuple(params), fetch_all=True) or []
