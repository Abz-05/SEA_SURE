"""
Database connection management with connection pooling.

This module provides a robust database connection manager with:
- Connection pooling for performance
- Context managers for automatic cleanup
- Health checks and retry logic
- Transaction support
"""

import logging
from contextlib import contextmanager
from typing import Optional, Generator
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from config.settings import settings


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages PostgreSQL database connections with pooling."""
    
    def __init__(self):
        """Initialize the database manager."""
        self._pool: Optional[pool.SimpleConnectionPool] = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the connection pool.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self._initialized:
            logger.warning("Database pool already initialized")
            return True
        
        try:
            self._pool = psycopg2.pool.SimpleConnectionPool(
                settings.db_pool_min,
                settings.db_pool_max,
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                cursor_factory=RealDictCursor
            )
            
            self._initialized = True
            logger.info(
                f"Database connection pool initialized "
                f"(min={settings.db_pool_min}, max={settings.db_pool_max})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            return False
    
    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._initialized = False
            logger.info("Database connection pool closed")
    
    @contextmanager
    def get_connection(self) -> Generator:
        """
        Get a database connection from the pool.
        
        Yields:
            psycopg2.connection: Database connection
            
        Example:
            >>> with db_manager.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM users")
        """
        if not self._initialized:
            raise RuntimeError("Database pool not initialized. Call initialize() first.")
        
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, commit: bool = False) -> Generator:
        """
        Get a database cursor with automatic connection management.
        
        Args:
            commit: Whether to commit the transaction on success
            
        Yields:
            psycopg2.cursor: Database cursor
            
        Example:
            >>> with db_manager.get_cursor(commit=True) as cursor:
            ...     cursor.execute("INSERT INTO users (...) VALUES (...)")
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                logger.error(f"Database cursor error: {e}")
                raise
            finally:
                cursor.close()
    
    def health_check(self) -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            bool: True if database is accessible, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
        commit: bool = False
    ):
        """
        Execute a database query with automatic connection management.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_one: Whether to fetch one result
            fetch_all: Whether to fetch all results
            commit: Whether to commit the transaction
            
        Returns:
            Query results if fetch_one or fetch_all is True, None otherwise
            
        Example:
            >>> result = db_manager.execute_query(
            ...     "SELECT * FROM users WHERE user_id = %s",
            ...     params=(user_id,),
            ...     fetch_one=True
            ... )
        """
        with self.get_cursor(commit=commit) as cursor:
            cursor.execute(query, params)
            
            if fetch_one:
                return cursor.fetchone()
            elif fetch_all:
                return cursor.fetchall()
            
            return None
    
    @property
    def is_initialized(self) -> bool:
        """Check if the database pool is initialized."""
        return self._initialized


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager: Global database manager
        
    Example:
        >>> db = get_db_manager()
        >>> if not db.is_initialized:
        ...     db.initialize()
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
    
    return _db_manager


def initialize_database() -> bool:
    """
    Initialize the global database manager.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    db = get_db_manager()
    return db.initialize()
