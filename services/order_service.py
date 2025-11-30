"""
Order service for order management and notifications.

This module handles order creation, status updates, and notifications.
"""

import logging
from typing import Optional, List
import uuid

from database.repositories import OrderRepository, CatchRepository
from models import Order, Catch
from config.constants import OrderStatus


logger = logging.getLogger(__name__)


class OrderService:
    """Service for order management."""
    
    def __init__(self):
        """Initialize the order service."""
        self.order_repo = OrderRepository()
        self.catch_repo = CatchRepository()
    
    def create_order(
        self,
        catch_id: str,
        buyer_user_id: int,
        buyer_name: str,
        quantity_kg: float,
        total_price: float,
        buyer_latitude: Optional[float] = None,
        buyer_longitude: Optional[float] = None
    ) -> tuple[Optional[Order], Optional[str]]:
        """
        Create a new order.
        
        Args:
            catch_id: ID of the catch
            buyer_user_id: Buyer's user ID
            buyer_name: Buyer's name
            quantity_kg: Quantity in kilograms
            total_price: Total price
            buyer_latitude: Buyer's latitude
            buyer_longitude: Buyer's longitude
            
        Returns:
            Tuple of (Order object, error message)
        """
        # Verify catch exists and is available
        catch_data = self.catch_repo.find_by_id(catch_id)
        if not catch_data:
            return None, "Catch not found"
        
        if catch_data.get('status') != 'available':
            return None, "Catch is not available"
        
        # Generate order ID
        order_id = str(uuid.uuid4())[:8].upper()
        
        # Create order
        try:
            result = self.order_repo.create_order(
                order_id=order_id,
                catch_id=catch_id,
                buyer_user_id=buyer_user_id,
                buyer_name=buyer_name,
                quantity_kg=quantity_kg,
                total_price=total_price,
                buyer_latitude=buyer_latitude,
                buyer_longitude=buyer_longitude
            )
            
            if result:
                # Mark catch as sold
                self.catch_repo.mark_as_sold(catch_id)
                
                # Get order
                order_data = self.order_repo.find_by_id(order_id)
                if order_data:
                    order = Order.from_dict(order_data)
                    logger.info(f"Order created: {order_id}")
                    return order, None
            
            return None, "Failed to create order"
            
        except Exception as e:
            logger.error(f"Order creation error: {e}")
            return None, str(e)
    
    def get_order_by_id(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        order_data = self.order_repo.find_by_id(order_id)
        if order_data:
            return Order.from_dict(order_data)
        return None
    
    def get_buyer_orders(self, buyer_user_id: int) -> List[Order]:
        """Get all orders for a buyer."""
        orders_data = self.order_repo.find_by_buyer(buyer_user_id)
        return [Order.from_dict(o) for o in orders_data]
    
    def get_fisher_orders(self, user_id: int) -> List[dict]:
        """Get all orders for a fisher's catches."""
        return self.order_repo.find_fisher_orders_by_user_id(user_id)
    
    def approve_order(self, order_id: str) -> bool:
        """Approve an order."""
        return self.order_repo.approve_order(order_id)
    
    def reject_order(self, order_id: str) -> bool:
        """
        Reject an order and make catch available again.
        
        Args:
            order_id: Order ID
            
        Returns:
            True if successful
        """
        # Get order to find catch_id
        order_data = self.order_repo.find_by_id(order_id)
        if order_data:
            # Reject order
            if self.order_repo.reject_order(order_id):
                # Make catch available again
                self.catch_repo.update_status(
                    order_data['catch_id'],
                    'available'
                )
                return True
        return False
    
    def mark_as_delivered(self, order_id: str) -> bool:
        """Mark order as delivered."""
        return self.order_repo.mark_as_delivered(order_id)
