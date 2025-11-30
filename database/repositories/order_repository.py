"""
Order repository for order-related database operations.

This module handles all database operations related to orders,
including creation, status management, and analytics.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from database.repositories.base_repository import BaseRepository
from config.constants import OrderStatus


logger = logging.getLogger(__name__)


class OrderRepository(BaseRepository):
    """Repository for order database operations."""
    
    @property
    def table_name(self) -> str:
        """Get the table name."""
        return "orders"
    
    @property
    def primary_key(self) -> str:
        """Get the primary key column name."""
        return "order_id"
    
    def create_order(
        self,
        order_id: str,
        catch_id: str,
        buyer_user_id: int,
        buyer_name: str,
        quantity_kg: float,
        total_price: float,
        buyer_latitude: Optional[float] = None,
        buyer_longitude: Optional[float] = None
    ) -> Optional[str]:
        """
        Create a new order.
        
        Args:
            order_id: Unique order ID
            catch_id: ID of the catch being ordered
            buyer_user_id: Buyer's user ID
            buyer_name: Buyer's name
            quantity_kg: Quantity in kilograms
            total_price: Total order price
            buyer_latitude: Buyer's latitude
            buyer_longitude: Buyer's longitude
            
        Returns:
            Order ID if successful, None otherwise
        """
        data = {
            "order_id": order_id,
            "catch_id": catch_id,
            "buyer_user_id": buyer_user_id,
            "buyer_name": buyer_name,
            "quantity_kg": quantity_kg,
            "total_price": total_price,
            "status": OrderStatus.PENDING.value,
            "created_at": datetime.now()
        }
        
        if buyer_latitude is not None:
            data["buyer_latitude"] = buyer_latitude
        if buyer_longitude is not None:
            data["buyer_longitude"] = buyer_longitude
        
        return self.create(data)
    
    def find_by_buyer(
        self,
        buyer_user_id: int,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find orders by buyer.
        
        Args:
            buyer_user_id: Buyer's user ID
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List of orders
        """
        criteria = {"buyer_user_id": buyer_user_id}
        if status:
            criteria["status"] = status
        
        return self.find_by_criteria(criteria, limit=limit, order_by="created_at")
    
    def find_by_catch(self, catch_id: str) -> List[Dict[str, Any]]:
        """
        Find all orders for a specific catch.
        
        Args:
            catch_id: Catch ID
            
        Returns:
            List of orders
        """
        return self.find_by_criteria({"catch_id": catch_id}, order_by="created_at")
    
    def find_fisher_orders(
        self,
        fisher_name: str,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find orders for a specific fisher's catches.
        
        Args:
            fisher_name: Fisher's name
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List of orders with catch details
        """
        query = """
            SELECT o.*, c.species, c.fisher_name, c.location as catch_location,
                   c.image_path, c.price_per_kg
            FROM orders o
            JOIN catches c ON o.catch_id = c.catch_id
            WHERE c.fisher_name = %s
        """
        params = [fisher_name]
        
        if status:
            query += " AND o.status = %s"
            params.append(status)
        
        query += " ORDER BY o.created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.db.execute_query(query, params=tuple(params), fetch_all=True) or []
    
    def find_fisher_orders_by_user_id(
        self,
        user_id: int,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find orders for a specific fisher's catches by user ID.
        
        Args:
            user_id: Fisher's user ID
            status: Optional status filter
            limit: Maximum number of results
            
        Returns:
            List of orders with catch details
        """
        query = """
            SELECT o.*, c.species, c.fisher_name, c.location as catch_location,
                   c.image_path, c.price_per_kg
            FROM orders o
            JOIN catches c ON o.catch_id = c.catch_id
            WHERE c.user_id = %s
        """
        params = [user_id]
        
        if status:
            query += " AND o.status = %s"
            params.append(status)
        
        query += " ORDER BY o.created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.db.execute_query(query, params=tuple(params), fetch_all=True) or []
    
    def update_status(self, order_id: str, status: str) -> bool:
        """
        Update order status.
        
        Args:
            order_id: Order ID
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        return self.update(order_id, {"status": status})
    
    def approve_order(self, order_id: str) -> bool:
        """
        Approve an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(order_id, OrderStatus.APPROVED.value)
    
    def reject_order(self, order_id: str) -> bool:
        """
        Reject an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(order_id, OrderStatus.REJECTED.value)
    
    def mark_as_delivered(self, order_id: str) -> bool:
        """
        Mark an order as delivered.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(order_id, OrderStatus.DELIVERED.value)
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(order_id, OrderStatus.CANCELLED.value)
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """
        Get overall order statistics.
        
        Returns:
            Dictionary with order statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_orders,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_orders,
                SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved_orders,
                SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) as delivered_orders,
                SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected_orders,
                SUM(total_price) as total_revenue,
                AVG(total_price) as avg_order_value,
                SUM(quantity_kg) as total_quantity_kg
            FROM orders
        """
        
        return self.db.execute_query(query, fetch_one=True)
    
    def get_buyer_statistics(self, buyer_user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific buyer.
        
        Args:
            buyer_user_id: Buyer's user ID
            
        Returns:
            Dictionary with buyer statistics
        """
        query = """
            SELECT 
                COUNT(*) as total_orders,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_orders,
                SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) as delivered_orders,
                SUM(total_price) as total_spent,
                AVG(total_price) as avg_order_value,
                SUM(quantity_kg) as total_quantity_kg
            FROM orders
            WHERE buyer_user_id = %s
        """
        
        return self.db.execute_query(query, params=(buyer_user_id,), fetch_one=True)
    
    def get_daily_revenue(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily revenue for the past N days.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of daily revenue records
        """
        query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as order_count,
                SUM(total_price) as revenue,
                SUM(quantity_kg) as total_quantity
            FROM orders
            WHERE created_at >= CURRENT_DATE - INTERVAL '%s days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """
        
        return self.db.execute_query(query, params=(days,), fetch_all=True) or []
