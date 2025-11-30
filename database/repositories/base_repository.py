"""
Base repository with common CRUD operations.

This module provides an abstract base class for all repositories,
implementing common database operations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from database.connection import get_db_manager


logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    """Abstract base repository for database operations."""
    
    def __init__(self):
        """Initialize the repository."""
        self.db = get_db_manager()
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Get the table name for this repository."""
        pass
    
    @property
    @abstractmethod
    def primary_key(self) -> str:
        """Get the primary key column name."""
        pass
    
    def find_by_id(self, id_value: Any) -> Optional[Dict[str, Any]]:
        """
        Find a record by its primary key.
        
        Args:
            id_value: Primary key value
            
        Returns:
            Dict containing the record, or None if not found
        """
        query = f"SELECT * FROM {self.table_name} WHERE {self.primary_key} = %s"
        return self.db.execute_query(query, params=(id_value,), fetch_one=True)
    
    def find_all(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: Optional[str] = None,
        order_desc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find all records with optional pagination and sorting.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Column to order by
            order_desc: Whether to order descending
            
        Returns:
            List of records
        """
        query = f"SELECT * FROM {self.table_name}"
        
        if order_by:
            direction = "DESC" if order_desc else "ASC"
            query += f" ORDER BY {order_by} {direction}"
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        return self.db.execute_query(query, fetch_all=True) or []
    
    def find_by_criteria(
        self,
        criteria: Dict[str, Any],
        limit: Optional[int] = None,
        offset: int = 0,
        order_by: Optional[str] = None,
        order_desc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find records matching criteria.
        
        Args:
            criteria: Dictionary of column: value pairs
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Column to order by
            order_desc: Whether to order descending
            
        Returns:
            List of matching records
        """
        if not criteria:
            return self.find_all(limit, offset, order_by, order_desc)
        
        where_clauses = []
        params = []
        
        for column, value in criteria.items():
            where_clauses.append(f"{column} = %s")
            params.append(value)
        
        query = f"SELECT * FROM {self.table_name} WHERE {' AND '.join(where_clauses)}"
        
        if order_by:
            direction = "DESC" if order_desc else "ASC"
            query += f" ORDER BY {order_by} {direction}"
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        return self.db.execute_query(query, params=tuple(params), fetch_all=True) or []
    
    def create(self, data: Dict[str, Any]) -> Optional[Any]:
        """
        Create a new record.
        
        Args:
            data: Dictionary of column: value pairs
            
        Returns:
            Primary key of created record, or None on failure
        """
        columns = list(data.keys())
        placeholders = ["%s"] * len(columns)
        values = [data[col] for col in columns]
        
        query = f"""
            INSERT INTO {self.table_name} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            RETURNING {self.primary_key}
        """
        
        try:
            result = self.db.execute_query(
                query,
                params=tuple(values),
                fetch_one=True,
                commit=True
            )
            return result[self.primary_key] if result else None
        except Exception as e:
            logger.error(f"Failed to create record in {self.table_name}: {e}")
            return None
    
    def update(self, id_value: Any, data: Dict[str, Any]) -> bool:
        """
        Update a record by its primary key.
        
        Args:
            id_value: Primary key value
            data: Dictionary of column: value pairs to update
            
        Returns:
            True if update successful, False otherwise
        """
        if not data:
            return False
        
        set_clauses = []
        params = []
        
        for column, value in data.items():
            set_clauses.append(f"{column} = %s")
            params.append(value)
        
        params.append(id_value)
        
        query = f"""
            UPDATE {self.table_name}
            SET {', '.join(set_clauses)}
            WHERE {self.primary_key} = %s
        """
        
        try:
            self.db.execute_query(query, params=tuple(params), commit=True)
            return True
        except Exception as e:
            logger.error(f"Failed to update record in {self.table_name}: {e}")
            return False
    
    def delete(self, id_value: Any) -> bool:
        """
        Delete a record by its primary key.
        
        Args:
            id_value: Primary key value
            
        Returns:
            True if deletion successful, False otherwise
        """
        query = f"DELETE FROM {self.table_name} WHERE {self.primary_key} = %s"
        
        try:
            self.db.execute_query(query, params=(id_value,), commit=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete record from {self.table_name}: {e}")
            return False
    
    def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records matching criteria.
        
        Args:
            criteria: Optional dictionary of column: value pairs
            
        Returns:
            Number of matching records
        """
        query = f"SELECT COUNT(*) as count FROM {self.table_name}"
        params = None
        
        if criteria:
            where_clauses = []
            param_values = []
            
            for column, value in criteria.items():
                where_clauses.append(f"{column} = %s")
                param_values.append(value)
            
            query += f" WHERE {' AND '.join(where_clauses)}"
            params = tuple(param_values)
        
        result = self.db.execute_query(query, params=params, fetch_one=True)
        return result['count'] if result else 0
    
    def exists(self, id_value: Any) -> bool:
        """
        Check if a record exists by its primary key.
        
        Args:
            id_value: Primary key value
            
        Returns:
            True if record exists, False otherwise
        """
        query = f"SELECT EXISTS(SELECT 1 FROM {self.table_name} WHERE {self.primary_key} = %s) as exists"
        result = self.db.execute_query(query, params=(id_value,), fetch_one=True)
        return result['exists'] if result else False
    
    def execute_custom_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
        commit: bool = False
    ):
        """
        Execute a custom SQL query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results
            commit: Whether to commit the transaction
            
        Returns:
            Query results if fetch_one or fetch_all is True, None otherwise
        """
        return self.db.execute_query(query, params, fetch_one, fetch_all, commit)
