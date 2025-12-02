import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import qrcode
from io import BytesIO
import base64
import json
import uuid
from pathlib import Path
import hashlib
import hmac
import logging
import logging.handlers
from typing import Dict, Any, Tuple, Optional, List
import cv2
import requests
from math import radians, cos, sin, asin, sqrt
import time
import random
import re
from contextlib import contextmanager

# Authentication imports
import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool

# Map libraries
try:
    import pydeck as pdk
    import folium
    from streamlit_folium import st_folium
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    MAPS_AVAILABLE = True
except ImportError:
    MAPS_AVAILABLE = False
    st.warning("⚠️ Map libraries not available. Install: pip install pydeck folium streamlit-folium geopy")

# Twilio SMS
try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioException
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# Environment configuration
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Database configuration with validation
# Database configuration with environment detection
IS_PRODUCTION = os.getenv('ENVIRONMENT', 'production').lower() == 'production'

if IS_PRODUCTION:
    DB_CONFIG = {
        'host': os.getenv('SUPABASE_DB_HOST', 'aws-1-ap-southeast-2.pooler.supabase.com'),
        'port': int(os.getenv('SUPABASE_DB_PORT', '6543')),
        'database': os.getenv('SUPABASE_DB_NAME', 'postgres'),
        'user': os.getenv('SUPABASE_DB_USER', 'postgres.tashfkdhagnlhrfqepoy'),
        'password': os.getenv('SUPABASE_DB_PASSWORD', 'Abzana#20052108'),
        'sslmode': 'require'
    }
else:
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5433')),
        'database': os.getenv('DB_NAME', 'seasure_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'Abzu#2005')
    }

# Environment variables with defaults
QR_STORAGE_PATH = os.getenv('QR_STORAGE_PATH', 'storage/qr/')
QR_SECRET_KEY = os.getenv('QR_SECRET_KEY', 'default-secret-key-change-me')
USE_QR_SIGNATURE = os.getenv('USE_QR_SIGNATURE', 'true').lower() == 'true'
OPENWEATHERMAP_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.getenv('TWILIO_FROM_NUMBER')
DEV_MODE = os.getenv('DEV_MODE', 'false').lower() == 'true'

# Create necessary directories with error handling
REQUIRED_DIRECTORIES = [
    'logs',
    'storage/offline_cache',
    QR_STORAGE_PATH,
    'uploads',
    'storage/fish_images'
]

for directory in REQUIRED_DIRECTORIES:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        st.error(f"Failed to create directory {directory}: {e}")

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Enhanced logging setup with rotation and formatting"""
    logger = logging.getLogger('sea_sure')
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # File handler with rotation
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            'logs/sea_sure_integrated.log',
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler for development
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    except Exception as e:
        st.error(f"Logging setup failed: {e}")
    
    return logger

logger = setup_logging()

# ============================================================================
# DATABASE CONNECTION MANAGEMENT
# ============================================================================

class DatabaseManager:
    """Enhanced database connection manager with pooling and error handling"""
    
    _pool = None
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize_pool(cls):
        """Initialize database connection pool"""
        if cls._pool is not None:
            return True
        
        try:
            cls._pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=20,
                **DB_CONFIG
            )
            logger.info("PostgreSQL connection pool initialized successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"Database pool initialization failed: {e}")
            st.error(f"Database connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing database: {e}")
            return False
    
    @classmethod
    @contextmanager
    def get_connection(cls):
        """Context manager for database connections"""
        if cls._pool is None:
            cls.initialize_pool()
        
        conn = None
        try:
            conn = cls._pool.getconn() if cls._pool else psycopg2.connect(**DB_CONFIG)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn and cls._pool:
                cls._pool.putconn(conn)
            elif conn:
                conn.close()
    
    @classmethod
    def execute_query(cls, query: str, params: tuple = None, fetch_one: bool = False, 
                     fetch_all: bool = False, commit: bool = False) -> Optional[Any]:
        """Execute database query with automatic connection management"""
        try:
            with cls.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params)
                    
                    if commit:
                        conn.commit()
                    
                    if fetch_one:
                        return cursor.fetchone()
                    elif fetch_all:
                        return cursor.fetchall()
                    
                    return None
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

# Initialize database manager
db_manager = DatabaseManager()

# ============================================================================
# ML MODEL LOADING
# ============================================================================

sys.path.append(".")

try:
    from combined_inference import CombinedFishPredictor
    
    @st.cache_resource
    def load_predictor():
        """Load ML predictor with caching and error handling"""
        try:
            predictor = CombinedFishPredictor()
            if predictor.is_available:
                logger.info("ML models loaded successfully")
                return predictor, None
            else:
                logger.warning("ML models partially loaded")
                return predictor, "Some models may not be available"
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            return None, str(e)
except ImportError as e:
    logger.warning(f"ML inference module not available: {e}")
    def load_predictor():
        return None, "ML models not available - install required packages"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SEA_SURE - Smart Fisheries Platform",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Abz-05/SEA-SURE',
        'Report a bug': 'https://github.com/Abz-05/SEA-SURE/issues',
        'About': '# SEA_SURE\nSmart Fisheries Management Platform'
    }
)

# ============================================================================
# ENHANCED CSS STYLES
# ============================================================================

# ============================================================================
# PROFESSIONAL OCEAN-THEMED CSS STYLES - REPLACE ENTIRE SECTION
# ============================================================================

st.markdown("""
<style>
    /* Main Layout - Ocean Blue Gradient */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Headers - Professional White Text */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        color: #ffffff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        letter-spacing: 2px;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 1.5rem 0 1rem 0;
        padding: 1rem 1.5rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.06) 100%);
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Cards - Glass Morphism Effect */
    .card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
        backdrop-filter: blur(15px);
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        color: #ffffff;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,212,255,0.3);
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.12) 100%);
        border-color: rgba(0,212,255,0.4);
    }
    
    /* Fish Card - Ocean Theme */
    .fish-card {
        background: linear-gradient(135deg, rgba(30,60,114,0.6) 0%, rgba(42,82,152,0.6) 100%);
        backdrop-filter: blur(20px);
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 2px solid rgba(0,212,255,0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        color: #ffffff;
    }
    
    .fish-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.2), transparent);
        transition: left 0.6s ease;
    }
    
    .fish-card:hover::before {
        left: 100%;
    }
    
    .fish-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 12px 40px rgba(0,212,255,0.5);
        transform: translateY(-5px);
        background: linear-gradient(135deg, rgba(30,60,114,0.75) 0%, rgba(42,82,152,0.75) 100%);
    }
    
    /* Order Card */
    .order-card {
        background: linear-gradient(135deg, rgba(30,60,114,0.6) 0%, rgba(42,82,152,0.6) 100%);
        backdrop-filter: blur(20px);
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 2px solid rgba(0,212,255,0.3);
        transition: all 0.3s ease;
        color: #ffffff;
    }
    
    .order-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,212,255,0.4);
    }
    
    /* User Info - Aqua Accent */
    .user-info {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #ffffff;
        padding: 1.3rem;
        border-radius: 14px;
        margin: 1rem 0;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,212,255,0.4);
        border: 2px solid rgba(255,255,255,0.3);
    }
    
    /* Stats Cards - Professional Blue Tones */
    .stats-card {
        background: linear-gradient(135deg, #0099cc 0%, #006699 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,153,204,0.4);
        transition: all 0.3s ease;
        cursor: pointer;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    .stats-card:hover {
        transform: scale(1.05) translateY(-8px);
        box-shadow: 0 15px 40px rgba(0,212,255,0.6);
        background: linear-gradient(135deg, #00b8e6 0%, #0099cc 100%);
    }
    
    .stats-value {
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0.5rem 0;
        animation: countUp 0.6s ease;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: translateY(20px) scale(0.9); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    .stats-label {
        font-size: 1.05rem;
        opacity: 0.95;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* QR Card - Light Cyan */
    .qr-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.3) 0%, rgba(0,153,204,0.3) 100%);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 18px;
        border: 3px solid #00d4ff;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,212,255,0.4);
        animation: slideIn 0.5s ease;
        color: #ffffff;
    }
    
    .qr-card h4 {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Freshness Indicators - Ocean Colors */
    .freshness-high {
        color: #00ff9d;
        font-weight: 800;
        font-size: 1.2rem;
        animation: pulse 2s infinite;
        text-shadow: 0 0 12px rgba(0,255,157,0.6);
    }
    
    .freshness-medium {
        color: #ffd93d;
        font-weight: 800;
        font-size: 1.2rem;
        text-shadow: 0 0 12px rgba(255,217,61,0.6);
    }
    
    .freshness-low {
        color: #ff6b9d;
        font-weight: 800;
        font-size: 1.2rem;
        animation: pulse 1.2s infinite;
        text-shadow: 0 0 12px rgba(255,107,157,0.6);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.85; transform: scale(1.03); }
    }
    
    /* Enhanced Buttons - Ocean Blue */
    .stButton>button {
        border-radius: 10px;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        border: 2px solid rgba(0,212,255,0.4);
        background: linear-gradient(135deg, rgba(0,153,204,0.5) 0%, rgba(0,102,153,0.5) 100%);
        color: #ffffff;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0,212,255,0.5);
        border-color: #00d4ff;
        background: linear-gradient(135deg, rgba(0,212,255,0.6) 0%, rgba(0,153,204,0.6) 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Status Badges - Professional Colors */
    .status-badge {
        padding: 0.5rem 1.3rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
        margin: 0.3rem;
        animation: fadeIn 0.4s ease;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .status-available {
        background: linear-gradient(135deg, rgba(0,255,157,0.5) 0%, rgba(0,204,136,0.5) 100%);
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .status-sold {
        background: linear-gradient(135deg, rgba(255,107,157,0.5) 0%, rgba(220,20,60,0.5) 100%);
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .status-pending {
        background: linear-gradient(135deg, rgba(255,217,61,0.5) 0%, rgba(255,193,7,0.5) 100%);
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .status-approved {
        background: linear-gradient(135deg, rgba(0,212,255,0.5) 0%, rgba(0,153,204,0.5) 100%);
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .status-delivered {
        background: linear-gradient(135deg, rgba(0,255,157,0.5) 0%, rgba(0,204,136,0.5) 100%);
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Info Boxes - Ocean Tones */
    .info-box {
        background: linear-gradient(135deg, rgba(0,153,204,0.4) 0%, rgba(0,102,153,0.4) 100%);
        backdrop-filter: blur(15px);
        border-left: 5px solid #00d4ff;
        padding: 1.3rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #ffffff;
        animation: slideIn 0.4s ease;
        box-shadow: 0 6px 20px rgba(0,153,204,0.3);
        border: 1px solid rgba(0,212,255,0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255,193,7,0.4) 0%, rgba(255,152,0,0.4) 100%);
        backdrop-filter: blur(15px);
        border-left: 5px solid #ffd93d;
        padding: 1.3rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 6px 20px rgba(255,193,7,0.3);
        border: 1px solid rgba(255,217,61,0.3);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(0,204,136,0.4) 0%, rgba(0,153,102,0.4) 100%);
        backdrop-filter: blur(15px);
        border-left: 5px solid #00ff9d;
        padding: 1.3rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 6px 20px rgba(0,204,136,0.3);
        border: 1px solid rgba(0,255,157,0.3);
    }
    
    /* Loading Animation - Cyan */
    .loading {
        display: inline-block;
        width: 24px;
        height: 24px;
        border: 4px solid rgba(0,212,255,0.3);
        border-radius: 50%;
        border-top-color: #00d4ff;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Image Containers - Ocean Border */
    .image-container {
        border: 3px solid rgba(0,212,255,0.4);
        border-radius: 15px;
        padding: 0.6rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(0,212,255,0.5);
        border-color: #00d4ff;
    }
    
    /* Streamlit Input Overrides */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0,212,255,0.3);
        color: #ffffff;
        border-radius: 10px;
        padding: 0.7rem;
    }
    
    .stTextInput>div>div>input::placeholder,
    .stTextArea>div>div>textarea::placeholder {
        color: rgba(255,255,255,0.6);
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 15px rgba(0,212,255,0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.06) 100%);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 0.6rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255,255,255,0.8);
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(0,212,255,0.3);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30,60,114,0.95) 0%, rgba(42,82,152,0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 2px solid rgba(0,212,255,0.2);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    
    /* Radio Buttons */
    .stRadio>div {
        background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.06) 100%);
        backdrop-filter: blur(10px);
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid rgba(0,212,255,0.2);
    }
    
    .stRadio label {
        color: #ffffff !important;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.9);
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Footer - Ocean Blue */
    .footer {
        text-align: center;
        padding: 2.5rem;
        color: #ffffff;
        margin-top: 3rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(15px);
        border-top: 2px solid rgba(0,212,255,0.3);
        border-radius: 15px 15px 0 0;
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.08) 100%);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 1rem;
        border: 2px solid rgba(0,212,255,0.3);
    }
    
    /* Scrollbar Styling - Ocean Theme */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30,60,114,0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        border-radius: 10px;
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00e6ff 0%, #00b8e6 100%);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        
        .stats-value {
            font-size: 2rem;
        }
        
        .section-header {
            font-size: 1.5rem;
            padding: 0.8rem 1rem;
        }
        
        .card, .fish-card, .order-card {
            padding: 1.2rem;
        }
    }
    
    /* Additional Streamlit Overrides */
    .stMarkdown, .stText {
        color: #ffffff;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p {
        color: rgba(255,255,255,0.95);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# AUTHENTICATION HELPER
# ============================================================================

class AuthHelper:
    """Enhanced authentication helper with improved security"""
    
    def __init__(self):
        self.twilio_client = None
        if TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            try:
                self.twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                logger.info("Twilio initialized successfully")
            except Exception as e:
                logger.error(f"Twilio initialization failed: {e}")
    
    def validate_phone(self, phone: str) -> bool:
        """Validate Indian phone number format"""
        if not phone:
            return False
        phone_clean = phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
        phone_pattern = re.compile(r'^(\+91)?[6-9]\d{9}$')
        return bool(phone_pattern.match(phone_clean))
    
    def normalize_phone(self, phone: str) -> str:
        """Normalize phone number to +91 format"""
        phone = phone.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
        if len(phone) == 10:
            phone = '+91' + phone
        elif not phone.startswith('+91') and len(phone) == 12:
            phone = '+' + phone
        return phone
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        try:
            salt = bcrypt.gensalt(rounds=12)
            return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def generate_otp(self) -> str:
        """Generate 6-digit OTP"""
        return f"{random.randint(100000, 999999):06d}"
    
    def send_otp_sms(self, phone: str, otp: str) -> bool:
        """Send OTP via SMS"""
        if not self.twilio_client or not TWILIO_FROM_NUMBER:
            logger.warning("SMS service not configured")
            return False
        
        try:
            message = self.twilio_client.messages.create(
                body=f"Your SEA_SURE verification code is: {otp}. Valid for 5 minutes.",
                from_=TWILIO_FROM_NUMBER,
                to=phone
            )
            logger.info(f"OTP sent successfully to {phone}")
            return True
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
            return False

auth_helper = AuthHelper()

# ============================================================================
# GEOGRAPHIC HELPER
# ============================================================================

class GeoHelper:
    """Enhanced geographic helper with caching"""
    
    def __init__(self):
        self.geocoder = Nominatim(user_agent="sea_sure_app") if MAPS_AVAILABLE else None
        
        self.tn_coastal_cities = {
        # Northern Tamil Nadu Coastal Areas
        'Pulicat': (13.4167, 80.3167),
        'Ennore': (13.2333, 80.3167),
        'Chennai': (13.0827, 80.2707),
        'Royapuram': (13.1143, 80.2919),
        'Marina Beach': (13.0499, 80.2824),
        'Santhome': (13.0338, 80.2785),
        'Thiruvanmiyur': (12.9833, 80.2594),
        'Neelankarai': (12.9500, 80.2594),
        'Injambakkam': (12.9167, 80.2500),
    
        # Chengalpattu District Coastal Areas
        'Kovalam': (12.7889, 80.2514),
        'Sadras': (12.5667, 80.0833),
        'Mamallapuram': (12.6267, 80.1927),
        'Alamparai': (12.2500, 79.8333),
    
        # Villupuram District Coastal Areas
        'Marakkanam': (12.1833, 79.9500),
    
        # Cuddalore District Coastal Areas
        'Cuddalore': (11.7529, 79.7714),
        'Parangipettai': (11.4833, 79.7667),
        'Silver Beach': (11.7667, 79.7833),
        'Devanampattinam': (11.4167, 79.7833),
    
        # Mayiladuthurai District Coastal Areas
        'Tarangambadi': (11.0333, 79.8500),
        'Karaikal': (10.9254, 79.8380),
    
        # Nagapattinam District Coastal Areas
        'Nagapattinam': (10.7669, 79.8420),
        'Vedaranyam': (10.3667, 79.8500),
        'Kodiyakarai': (10.2833, 79.8500),
        'Thirumullaivasal': (10.8167, 79.8500),
        'Poompuhar': (11.1500, 79.8500),
    
        # Thanjavur District Coastal Areas
        'Manora': (10.5833, 79.8333),
    
        # Pudukkottai District Coastal Areas
        'Adirampattinam': (10.3333, 79.3833),
    
        # Ramanathapuram District Coastal Areas
        'Rameswaram': (9.2876, 79.3129),
        'Mandapam': (9.2806, 79.1378),
        'Pamban': (9.2750, 79.2093),
        'Dhanushkodi': (9.1833, 79.4333),
        'Thangachimadam': (9.2167, 79.1500),
        'Erwadi': (9.1167, 78.4500),
        'Thondi': (9.7333, 79.0167),
        'Uchipuli': (9.3333, 78.8333),
        'Kilakarai': (9.2333, 78.7833),
        'Sayalkudi': (9.1833, 78.4500),
        'Mimisal': (9.4500, 78.9833),
    
        # Tuticorin District Coastal Areas
        'Tuticorin': (8.7642, 78.1348),
        'Vaipar': (8.7333, 78.0833),
        'Vembar': (8.9833, 78.2500),
        'Kayalpattinam': (8.5667, 78.1167),
        'Punnakayal': (8.6333, 78.1167),
        'Kulasekarapattinam': (8.3833, 78.0500),
        'Thiruchendur': (8.4833, 78.1167),
        'Alantalai': (8.6333, 78.0333),
        'Palayakayal': (8.5167, 78.0833),
    
        # Tirunelveli District Coastal Areas
        'Uvari': (8.4667, 77.9667),
        'Kayal': (8.5833, 77.9833),
    
        # Kanyakumari District Coastal Areas
        'Kanyakumari': (8.0883, 77.5385),
        'Colachel': (8.1833, 77.2500),
        'Muttom': (8.1167, 77.3167),
        'Kovalam Kanyakumari': (8.3833, 77.3333),
        'Manakudy': (8.1500, 77.4333),
        'Thengapattanam': (8.1667, 77.4000),
        'Midalam': (8.2833, 77.4167),
        'Pozhikkarai': (8.0667, 77.5167),
        'Chinnamuttom': (8.1500, 77.3000),
        'Pallam': (8.2167, 77.3833)
        }
    
    @st.cache_data(ttl=3600)
    def get_random_coastal_location(_self, base_city: str = 'Chennai', radius_km: float = 50) -> Tuple[float, float]:
        """Generate random coastal location with caching"""
        base_lat, base_lon = _self.tn_coastal_cities.get(base_city, (13.0827, 80.2707))
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0, radius_km)
        lat_offset = (distance * np.cos(angle)) / 111.0
        lon_offset = (distance * np.sin(angle)) / (111.0 * np.cos(np.radians(base_lat)))
        return base_lat + lat_offset, base_lon + lon_offset
    
    def calculate_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using geodesic or Haversine formula"""
        try:
            if MAPS_AVAILABLE:
                return geodesic((lat1, lon1), (lat2, lon2)).kilometers
            else:
                return self._haversine_distance(lat1, lon1, lat2, lon2)
        except Exception as e:
            logger.error(f"Distance calculation failed: {e}")
            return 0.0
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine formula for distance calculation"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c  # Earth radius in km
    
    def safe_float_multiply(val1, val2):
        """Safely multiply two values, converting Decimal to float"""
        try:
            from decimal import Decimal
            if isinstance(val1, Decimal):
                val1 = float(val1)
            if isinstance(val2, Decimal):
                val2 = float(val2)
            return float(val1) * float(val2)
        except Exception as e:
            logger.error(f"Float multiplication error: {e}")
            return 0.0
        
    @st.cache_data(ttl=86400)
    def get_city_name(_self, lat: float, lon: float) -> str:
        """Get city name from coordinates with caching"""
        if _self.geocoder:
            try:
                location = _self.geocoder.reverse(f"{lat}, {lon}", timeout=5)
                if location:
                    address = location.raw.get('address', {})
                    city = address.get('city') or address.get('town') or address.get('village')
                    if city:
                        return city
            except Exception as e:
                logger.debug(f"Geocoding failed: {e}")
        
        # Fallback to nearest coastal city
        min_distance = float('inf')
        nearest_city = "Unknown Location"
        
        for city, (city_lat, city_lon) in _self.tn_coastal_cities.items():
            distance = _self.calculate_distance_km(lat, lon, city_lat, city_lon)
            if distance < min_distance:
                min_distance = distance
                nearest_city = city
        
        return f"Near {nearest_city}"

geo_helper = GeoHelper()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data(ttl=1800)
def get_area_temperature(lat: float, lon: float, date: datetime = None) -> float:
    """Get area temperature with API fallback to estimation"""
    if date is None:
        date = datetime.now()
    
    # Try Weather API first
    if OPENWEATHERMAP_API_KEY:
        try:
            url = "http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': OPENWEATHERMAP_API_KEY,
                'units': 'metric'
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                temp = float(data['main']['temp'])
                logger.info(f"Temperature from API: {temp}°C")
                return round(temp, 1)
        except Exception as e:
            logger.warning(f"Weather API error: {e}")
    
    # Fallback to estimation
    month = date.month
    if 3 <= month <= 5:  # Summer
        base_temp, variation = 33.0, 3.0
    elif 6 <= month <= 9:  # Monsoon
        base_temp, variation = 28.0, 2.0
    elif month in [10, 11]:  # Post-monsoon
        base_temp, variation = 27.0, 2.0
    else:  # Winter
        base_temp, variation = 25.0, 2.0
    
    temp_adjustment = (lat - 13.0) * -0.3
    temp = base_temp + temp_adjustment + np.random.uniform(-variation, variation)
    return round(np.clip(temp, 20.0, 40.0), 1)

def calculate_dynamic_freshness(catch_timestamp: datetime, initial_freshness_days: float, 
                                storage_temp: float = 5.0) -> Tuple[float, str]:
    """Calculate remaining freshness with improved accuracy"""
    try:
        current_time = datetime.now()
        time_elapsed = current_time - catch_timestamp
        hours_elapsed = time_elapsed.total_seconds() / 3600
        
        # Temperature-based decay rate
        if storage_temp <= 4:
            decay_rate = 0.015  # Excellent storage
        elif storage_temp <= 10:
            decay_rate = 0.03   # Good storage
        elif storage_temp <= 20:
            decay_rate = 0.05   # Fair storage
        else:
            decay_rate = 0.08   # Poor storage
        
        # Calculate freshness loss
        freshness_lost = (hours_elapsed / 24) * decay_rate * initial_freshness_days
        remaining_freshness = max(0, initial_freshness_days - freshness_lost)
        
        # Determine category
        if remaining_freshness >= 2.0:
            category = "Excellent"
        elif remaining_freshness >= 1.5:
            category = "Good"
        elif remaining_freshness >= 1.0:
            category = "Fair"
        elif remaining_freshness >= 0.5:
            category = "Poor"
        else:
            category = "Expired"
        
        return round(remaining_freshness, 2), category
    except Exception as e:
        logger.error(f"Freshness calculation error: {e}")
        return 0.0, "Unknown"

def validate_image_quality(image: Image.Image) -> Tuple[bool, str]:
    """Validate image quality with comprehensive checks"""
    try:
        # Convert to array
        img_array = np.array(image)
        
        # Check dimensions
        if len(img_array.shape) < 2:
            return False, "Invalid image format"
        
        height, width = img_array.shape[:2]
        
        # Size checks
        if width < 300 or height < 300:
            return False, f"Image too small ({width}x{height}). Minimum 300x300 pixels required."
        
        if width * height > 10000000:
            return False, "Image too large. Maximum 10MP supported."
        
        # Convert to grayscale for blur detection
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            return False, f"Image appears blurry (score: {laplacian_var:.0f}). Please capture a clearer photo."
        
        # Brightness check
        brightness = np.mean(gray)
        if brightness < 30:
            return False, "Image too dark. Please improve lighting."
        elif brightness > 225:
            return False, "Image overexposed. Please reduce lighting."
        
        return True, f"Image quality acceptable (sharpness: {laplacian_var:.0f})"
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False, f"Image validation failed: {str(e)}"
def save_fish_image(image: Image.Image, catch_id: str) -> Optional[str]:
    """Save fish image with error handling"""
    try:
        filename = f"fish_{catch_id}_{int(time.time())}.png"
        filepath = os.path.join('storage/fish_images', filename)
        # Resize if too large
        max_size = (1920, 1920)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
        # Save with optimization
        image.save(filepath, format='PNG', optimize=True)
        logger.info(f"Fish image saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save fish image: {e}")
        return None
def generate_qr_code(catch_data: Dict[str, Any]) -> Tuple[str, str, Optional[str], str]:
    """Generate QR code with enhanced security"""
    try:
        qr_data = {
            "catch_id": str(catch_data["catch_id"]),
            "species": str(catch_data["species"]),
            "species_tamil": str(catch_data.get("species_tamil", "மீன்")),
            "freshness_days": float(catch_data["freshness_days"]),
            "freshness_category": str(catch_data["freshness_category"]),
            "price_per_kg": float(catch_data["price_per_kg"]),
            "fisher_name": str(catch_data["fisher_name"]),
            "location": str(catch_data["location"]),
            "timestamp": datetime.now().isoformat()
}
        qr_data_json = json.dumps(qr_data, sort_keys=True)
    
        # Generate signature
        signature = ""
        if USE_QR_SIGNATURE:
            signature = hmac.new(
                QR_SECRET_KEY.encode(),
                qr_data_json.encode(),
            hashlib.sha256
        ).hexdigest()
        qr_data["signature"] = signature
        qr_data_json = json.dumps(qr_data, sort_keys=True)
    
        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=5
    )
        qr.add_data(qr_data_json)
        qr.make(fit=True)
    
        qr_img = qr.make_image(fill_color="black", back_color="white")
    
        # Save to file
        qr_filename = f"qr_{catch_data['catch_id']}_{int(time.time())}.png"
        qr_path = os.path.join(QR_STORAGE_PATH, qr_filename)
    
        try:
            qr_img.save(qr_path)
            logger.info(f"QR code saved: {qr_path}")
        except Exception as e:
            logger.error(f"Failed to save QR image: {e}")
            qr_path = None
    
        # Convert to base64
        buffer = BytesIO()
        qr_img.save(buffer, format="PNG")
        qr_b64 = base64.b64encode(buffer.getvalue()).decode()
    
        return qr_b64, qr_data_json, qr_path, signature
    
    except Exception as e:
        logger.error(f"QR code generation failed: {e}")
        raise
#============================================================================
#DATABASE FUNCTIONS
#============================================================================'''
def init_database() -> bool:
    """Initialize database with health check"""
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        st.error(f"❌ Database connection failed: {e}")
        return False
def create_user(name: str, phone: str, password: str, role: str = 'buyer') -> int:
    """Create new user with validation"""
    try:
        normalized_phone = auth_helper.normalize_phone(phone)
        hashed_password = auth_helper.hash_password(password)
        query = """
        INSERT INTO users (name, phone, password_hash, role, verified, created_at)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING user_id
    """
        result = db_manager.execute_query(
        query,
        (name, normalized_phone, hashed_password, role, False, datetime.now()),
        fetch_one=True,
        commit=True
    )
    
        if result:
            logger.info(f"User created: {name} ({normalized_phone})")
            return result['user_id']
    
        return None
    
    except psycopg2.IntegrityError:
        raise ValueError("Phone number already registered")
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise
def verify_user_credentials(phone: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify user credentials with enhanced security"""
    try:
        normalized_phone = auth_helper.normalize_phone(phone)
        query = """
        SELECT user_id, name, phone, password_hash, role, verified, is_active
        FROM users WHERE phone = %s
    """
        
        user = db_manager.execute_query(query, (normalized_phone,), fetch_one=True)
    
        if not user:
            logger.warning(f"Login attempt for non-existent user: {normalized_phone}")
            return None
    
        if not user['is_active']:
            raise ValueError("Account is deactivated")
    
        if not auth_helper.verify_password(password, user['password_hash']):
            logger.warning(f"Invalid password attempt for: {normalized_phone}")
            return None
    
        # Update last login
        update_query = "UPDATE users SET last_login = %s WHERE user_id = %s"
        db_manager.execute_query(update_query, (datetime.now(), user['user_id']), commit=True)
    
        logger.info(f"Successful login: {user['name']} ({normalized_phone})")
    
        return {
            'user_id': user['user_id'],
            'name': user['name'],
            'phone': user['phone'],
            'role': user['role'],
            'verified': user['verified']
    }
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        return None
def save_catch_to_db(catch_data: Dict[str, Any], qr_code_data: str,
qr_path: Optional[str] = None, qr_signature: str = "") -> bool:
    """Save catch to database with comprehensive error handling"""
    try:
        # Convert NumPy types to native Python types
        def convert_to_native(value):
            """Convert numpy types to native Python types"""
            if value is None:
                return None
            if isinstance(value, (np.integer, np.floating)):
                return float(value) if isinstance(value, np.floating) else int(value)
            if isinstance(value, np.ndarray):
                return value.item()
            return value
        
        query = """
        INSERT INTO catches
        (catch_id, fisher_name, user_id, species, species_tamil,
        freshness_days, freshness_category, weight_g, storage_temp,
        hours_since_catch, price_per_kg, location, latitude, longitude,
        qr_code, area_temperature, qr_signature, qr_path, image_path,
        created_at, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Convert all parameters to native Python types
        params = (
            str(catch_data["catch_id"]),
            str(catch_data["fisher_name"]),
            int(catch_data.get("user_id")) if catch_data.get("user_id") else None,
            str(catch_data["species"]),
            str(catch_data.get("species_tamil", "மீன்")),
            float(convert_to_native(catch_data.get("freshness_days", 2.0))),
            str(catch_data.get("freshness_category", "Good")),
            int(convert_to_native(catch_data["weight_g"])),
            float(convert_to_native(catch_data.get("storage_temp", 5.0))),
            float(convert_to_native(catch_data.get("hours_since_catch", 6.0))),
            int(convert_to_native(catch_data.get("price_per_kg", 300))),
            str(catch_data.get("location", "Unknown")),
            float(convert_to_native(catch_data.get("latitude", 0.0))),
            float(convert_to_native(catch_data.get("longitude", 0.0))),
            str(qr_code_data),
            float(convert_to_native(catch_data.get("area_temperature", 25.0))),
            str(qr_signature) if qr_signature else "",
            str(qr_path) if qr_path else None,
            str(catch_data.get("image_path")) if catch_data.get("image_path") else None,
            datetime.now(),
            'available'
        )
    
        db_manager.execute_query(query, params, commit=True)
        logger.info(f"Catch saved successfully: {catch_data['catch_id']}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save catch: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
@st.cache_data(ttl=60)
def get_all_catches() -> pd.DataFrame:
    """Get all catches with caching"""
    try:
        query = """
        SELECT catch_id, fisher_name, species, species_tamil, freshness_days,
        freshness_category, weight_g, price_per_kg, location,
        created_at, status, qr_code, latitude, longitude,
        area_temperature, qr_signature, image_path, storage_temp,
        hours_since_catch
        FROM catches
        ORDER BY created_at DESC
        """
        results = db_manager.execute_query(query, fetch_all=True)
    
        if results:
            df = pd.DataFrame(results)
            return df
    
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Get catches error: {e}")
        return pd.DataFrame()
def get_catches_with_dynamic_freshness() -> pd.DataFrame:
    """Get catches with real-time freshness calculation"""
    df = get_all_catches()
    
    # Ensure proper DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.warning("get_all_catches did not return a DataFrame")
        return pd.DataFrame()
    
    if len(df) > 0:
        # Convert numeric columns to proper types
        numeric_columns = ['weight_g', 'price_per_kg', 'storage_temp', 
                          'freshness_days', 'latitude', 'longitude', 
                          'area_temperature', 'hours_since_catch']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        for idx, row in df.iterrows():
            try:
                catch_time = pd.to_datetime(row['created_at'])
                storage_temp = float(row.get('storage_temp', 5.0))
                initial_freshness = float(row['freshness_days'])
            
                remaining_freshness, category = calculate_dynamic_freshness(
                    catch_time, initial_freshness, storage_temp
                )
            
                df.at[idx, 'current_freshness_days'] = remaining_freshness
                df.at[idx, 'current_freshness_category'] = category
                df.at[idx, 'is_expired'] = remaining_freshness <= 0.5
            
            except Exception as e:
                logger.error(f"Error calculating freshness for catch {row['catch_id']}: {e}")
                df.at[idx, 'current_freshness_days'] = 0.0
                df.at[idx, 'current_freshness_category'] = 'Unknown'
                df.at[idx, 'is_expired'] = True
    
        # Update status for expired catches
        df.loc[df['is_expired'] == True, 'status'] = 'expired'

    return df
def create_notification(user_id: int, notification_type: str,
title: str, message: str) -> bool:
    """Create user notification"""
    try:
        query = """
        INSERT INTO notifications (user_id, type, title, message, is_read, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        db_manager.execute_query(
            query,
            (user_id, notification_type, title, message, False, datetime.now()),
            commit=True
        )
    
        logger.info(f"Notification created for user {user_id}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create notification: {e}")
        return False
@st.cache_data(ttl=30)
def get_user_notifications(user_id: int) -> pd.DataFrame:
    """Get user notifications with caching"""
    try:
        query = """
        SELECT notification_id, type, title, message, is_read, created_at
        FROM notifications
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 20
        """
        results = db_manager.execute_query(query, (user_id,), fetch_all=True)
    
        if results:
            return pd.DataFrame(results)
    
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Failed to get notifications: {e}")
        return pd.DataFrame()
#============================================================================
#MAP FUNCTIONS
#============================================================================
def create_fisher_map(fisher_name: str, catches_df: pd.DataFrame):
    """Create 3D map visualization for fisher catches"""
    if not MAPS_AVAILABLE:
        return None
    try:
        # Ensure catches_df is a proper DataFrame
        if not isinstance(catches_df, pd.DataFrame):
            logger.error("catches_df is not a DataFrame")
            return None
        
        # Create a copy and filter
        fisher_catches = catches_df[catches_df['fisher_name'] == fisher_name].copy()
        
        if len(fisher_catches) == 0:
            # Default location
            fisher_catches = pd.DataFrame([{
                'latitude': 13.0827, 
                'longitude': 80.2707,
                'species': 'No catches yet', 
                'current_freshness_days': 0.0,
                'price_per_kg': 0.0, 
                'weight_g': 0.0
            }])
        
        # Convert to native types and ensure proper column types
        fisher_catches['current_freshness_days'] = pd.to_numeric(
            fisher_catches['current_freshness_days'], errors='coerce'
        ).fillna(0.0)
        
        fisher_catches['price_per_kg'] = pd.to_numeric(
            fisher_catches['price_per_kg'], errors='coerce'
        ).fillna(0.0)
        
        fisher_catches['latitude'] = pd.to_numeric(
            fisher_catches['latitude'], errors='coerce'
        ).fillna(13.0827)
        
        fisher_catches['longitude'] = pd.to_numeric(
            fisher_catches['longitude'], errors='coerce'
        ).fillna(80.2707)
        
        # Color coding based on freshness
        def get_color(freshness):
            try:
                freshness = float(freshness)
                if freshness >= 2.0:
                    return [0, 255, 0, 160]
                elif freshness >= 1.0:
                    return [255, 255, 0, 160]
                else:
                    return [255, 0, 0, 160]
            except:
                return [128, 128, 128, 160]
        
        fisher_catches['color'] = fisher_catches['current_freshness_days'].apply(get_color)
        
        # Size based on price
        fisher_catches['size'] = fisher_catches['price_per_kg'].apply(
            lambda x: max(float(x) / 10.0, 5.0)
        )
        
        # Create layer
        scatterplot_layer = pdk.Layer(
            'ScatterplotLayer',
            data=fisher_catches.to_dict('records'),
            get_position='[longitude, latitude]',
            get_color='color',
            get_radius='size',
            radius_scale=100,
            radius_min_pixels=5,
            radius_max_pixels=50,
            pickable=True,
            auto_highlight=True,
        )
        
        # Calculate center
        center_lat = float(fisher_catches['latitude'].mean())
        center_lon = float(fisher_catches['longitude'].mean())
        
        # Create deck
        deck = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=10,
                pitch=50,
            ),
            layers=[scatterplot_layer],
            tooltip={
                'html': '<b>{species}</b><br/>Freshness: {current_freshness_days} days<br/>Price: ₹{price_per_kg}/kg',
                'style': {'backgroundColor': 'steelblue', 'color': 'white'}
            }
        )
        
        return deck
    
    except Exception as e:
        logger.error(f"Error creating fisher map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
def create_buyer_map(catches_df: pd.DataFrame, species_filter: List[str] = None,
min_freshness: float = 0) -> Optional[Any]:
    """Create interactive map for buyers"""
    if not MAPS_AVAILABLE:
        return None
    try:
        filtered_catches = catches_df[catches_df['status'] == 'available'].copy()
    
        if species_filter:
            filtered_catches = filtered_catches[filtered_catches['species'].isin(species_filter)]
    
        filtered_catches = filtered_catches[
            filtered_catches['current_freshness_days'] >= min_freshness
        ]
    
        # Calculate center
        if len(filtered_catches) == 0:
            center_lat, center_lon = 13.0827, 80.2707
        else:
            center_lat = filtered_catches['latitude'].mean()
            center_lon = filtered_catches['longitude'].mean()
    
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
    
        # Add markers
        for _, catch in filtered_catches.iterrows():
            # Determine marker color
            if catch['current_freshness_days'] >= 2.0:
                color, icon = 'green', 'leaf'
            elif catch['current_freshness_days'] >= 1.0:
                color, icon = 'orange', 'warning-sign'
            else:
                color, icon = 'red', 'exclamation-sign'
        
        # Create popup
        popup_html = f"""
        <div style="width: 200px;">
            <h4>{catch['species']}</h4>
            <p><b>Fisher:</b> {catch['fisher_name']}</p>
            <p><b>Freshness:</b> {catch['current_freshness_days']:.1f} days</p>
            <p><b>Price:</b> ₹{catch['price_per_kg']}/kg</p>
            <p><b>Weight:</b> {catch['weight_g']}g</p>
        </div>
        """
        
        folium.Marker(
            location=[catch['latitude'], catch['longitude']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=color, icon=icon, prefix='glyphicon'),
            tooltip=f"{catch['species']} - {catch['current_freshness_days']:.1f} days"
        ).add_to(m)
    
        return m
    
    except Exception as e:
        logger.error(f"Error creating buyer map: {e}")
        return None
#============================================================================
#USER INTERFACES
#============================================================================
def show_login_page():
    """Enhanced login page with better UX"""
    st.markdown('<h1 class="main-header">🐟 SEA_SURE</h1>', unsafe_allow_html=True)
    st.markdown(
        '<center><h3>Smart Fisheries Platform - Ensuring Freshness, Building Trust</h3></center>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<center><b><h2><div class="card">Welcome to SEA-SURE</div></h2></b></center>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        st.markdown("### Welcome Back!")
        
        with st.form("login_form"):
            login_phone = st.text_input(
                "📱 Phone Number",
                placeholder="+919876543210",
                help="Enter your registered phone number"
            )
            login_password = st.text_input(
                "🔒 Password",
                type="password",
                help="Enter your password"
            )
            
            submitted = st.form_submit_button(
                "🚀 Login",
                type="primary",
                width="stretch"
            )
        
        if submitted:
            if not login_phone or not login_password:
                st.error("⚠️ Please enter phone number and password")
            elif not auth_helper.validate_phone(login_phone):
                st.error("⚠️ Please enter a valid Indian phone number")
            else:
                with st.spinner("Authenticating..."):
                    try:
                        user = verify_user_credentials(login_phone, login_password)
                        if user:
                            st.session_state['user'] = user
                            st.session_state['authenticated'] = True
                            st.success(f"✅ Welcome back, {user['name']}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials. Please try again.")
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                    except Exception as e:
                        st.error("❌ Login failed. Please try again.")
                        logger.error(f"Login error: {e}")
    
    with tab2:
        st.markdown("### Create New Account")
        
        with st.form("register_form"):
            reg_name = st.text_input(
                "👤 Full Name",
                placeholder="John Doe",
                help="Enter your full name"
            )
            reg_phone = st.text_input(
                "📱 Phone Number",
                placeholder="+919876543210",
                help="Enter a valid Indian mobile number"
            )
            reg_password = st.text_input(
                "🔒 Password",
                type="password",
                help="Minimum 6 characters"
            )
            reg_password_confirm = st.text_input(
                "🔒 Confirm Password",
                type="password"
            )
            
            reg_role = st.selectbox(
                "👥 Account Type",
                ["buyer", "fisher", "admin"],
                format_func=lambda x: {
                    "buyer": "🛒 Buyer - Browse and purchase fish",
                    "fisher": "🎣 Fisher - Sell your catch",
                    "admin": "👨‍💼 Admin - System management"
                }[x]
            )
            
            submitted = st.form_submit_button(
                "📝 Create Account",
                type="primary",
                width="stretch"
            )
        
        if submitted:
            # Validation
            errors = []
            
            if not all([reg_name, reg_phone, reg_password, reg_password_confirm]):
                errors.append("Please fill in all fields")
            
            if len(reg_name) < 2:
                errors.append("Name must be at least 2 characters")
            
            if not auth_helper.validate_phone(reg_phone):
                errors.append("Please enter a valid Indian phone number")
            
            if len(reg_password) < 6:
                errors.append("Password must be at least 6 characters")
            
            if reg_password != reg_password_confirm:
                errors.append("Passwords do not match")
            
            if errors:
                for error in errors:
                    st.error(f"⚠️ {error}")
            else:
                with st.spinner("Creating account..."):
                    try:
                        user_id = create_user(reg_name, reg_phone, reg_password, reg_role)
                        
                        if user_id:
                            st.success("✅ Account created successfully!")
                            
                            user = {
                                'user_id': user_id,
                                'name': reg_name,
                                'phone': auth_helper.normalize_phone(reg_phone),
                                'role': reg_role,
                                'verified': False
                            }
                            st.session_state['user'] = user
                            st.session_state['authenticated'] = True
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Registration failed. Please try again.")
                            
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                    except Exception as e:
                        st.error("❌ Registration failed. Please try again.")
                        logger.error(f"Registration error: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)
def fisher_interface():
    """Enhanced fisher dashboard"""
    user = st.session_state.get('user', {})
    fisher_name = user.get('name', 'Fisher')
    user_id = user.get('user_id')
    # Header with logout
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(
            f'<div class="user-info"><h2>👋 Welcome, {fisher_name} | Role: Fisher</h2></div>',
            unsafe_allow_html=True
        )
    with col2:
        if st.button("🔔 Notifications"):
            st.session_state['show_notifications'] = not st.session_state.get('show_notifications', False)
    with col3:
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Notifications sidebar
    if st.session_state.get('show_notifications', False):
        with st.sidebar:
            st.markdown("### 🔔 Recent Notifications")
            notifications = get_user_notifications(user_id)
            if len(notifications) > 0:
                for _, notif in notifications.head(5).iterrows():
                    icon = "📦" if notif['type'] == 'order' else "ℹ️"
                    st.info(f"{icon} **{notif['title']}**\n{notif['message']}")
            else:
                st.info("No new notifications")

    # Navigation
    with st.sidebar:
        st.markdown("### 📋 Navigation")
        page = st.radio(
            "Select Page:",
            ["📸 Add Catch", "🐟 My Catches", "📦 Orders", "🗺️ Map View", "📊 Analytics"],
            label_visibility="collapsed"
        )

    # Page routing
    if page == "📸 Add Catch":
        show_add_catch_page(fisher_name, user_id)
    elif page == "🐟 My Catches":
        show_my_catches_page(fisher_name)
    elif page == "📦 Orders":
        show_orders_page(fisher_name, user_id)
    elif page == "🗺️ Map View":
        show_map_view_page(fisher_name)
    elif page == "📊 Analytics":
        show_analytics_page(fisher_name)
def show_add_catch_page(fisher_name: str, user_id: int):
    """Add catch page with ML integration"""
    st.markdown('<h3 class="section-header">📸 Add New Catch</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card"><h3>🖼️ Fish Image</h3></div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload fish photo",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of the fish"
        )
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Validate image
                valid, msg = validate_image_quality(image)
                if valid:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>📸 Catch Details</h3></div>', unsafe_allow_html=True)
        
        weight = st.number_input(
            "⚖️ Weight (grams)",
            min_value=1,
            max_value=50000,
            value=500,
            step=50,
            help="Total weight of the catch"
        )
        
        storage_temp = st.slider(
            "🌡️ Storage Temperature (°C)",
            0, 35, 5,
            help="Current storage temperature"
        )
        
        hours_since_catch = st.slider(
            "⏱️ Hours Since Catch",
            0, 72, 6,
            help="Time elapsed since catching"
        )
        
        price_per_kg = st.number_input(
            "💰 Price (₹/kg)",
            min_value=1,
            max_value=5000,
            value=300,
            step=50,
            help="Selling price per kilogram"
        )
        
        st.markdown("#### 📍 Location")
        
        if MAPS_AVAILABLE:
            location_method = st.radio(
                "Location Method",
                ["Select City", "Manual Coordinates"]
            )
            
            if location_method == "Select City":
                selected_city = st.selectbox(
                    "Select Location",
                    list(geo_helper.tn_coastal_cities.keys())
                )
                latitude, longitude = geo_helper.tn_coastal_cities[selected_city]
                location = selected_city
            else:
                latitude = st.number_input("Latitude", value=13.0827, format="%.6f")
                longitude = st.number_input("Longitude", value=80.2707, format="%.6f")
                location = geo_helper.get_city_name(latitude, longitude)
        else:
            location = st.text_input("Location", value="Chennai")
            latitude, longitude = 13.0827, 80.2707
        
        st.info(f"📍 Location: {location}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Analysis button
    if uploaded_file and st.button(
        "🔬 Analyze Fish & Generate QR",
        type="primary",
        width="stretch"
    ):
        with st.spinner("🔄 Analyzing fish quality..."):
            try:
                predictor, error = load_predictor()
                area_temp = get_area_temperature(latitude, longitude, datetime.now())
                
                if predictor:
                    result = predictor.predict_complete(
                        image,
                        weight=weight,
                        storage_temp=storage_temp,
                        hours_since_catch=hours_since_catch,
                        area_temp=area_temp
                    )
                    
                    if result.get("prediction_success"):
                        catch_id = str(uuid.uuid4())[:8].upper()
                        
                        # Save image
                        image_path = save_fish_image(image, catch_id)
                        
                        # Prepare catch data
                        catch_data = {
                            "catch_id": catch_id,
                            "fisher_name": fisher_name,
                            "user_id": user_id,
                            "species": result["species"],
                            "species_tamil": result["species_tamil"],
                            "freshness_days": result["freshness_days_remaining"],
                            "freshness_category": result["freshness_category"],
                            "weight_g": weight,
                            "storage_temp": storage_temp,
                            "hours_since_catch": hours_since_catch,
                            "price_per_kg": price_per_kg,
                            "location": location,
                            "latitude": latitude,
                            "longitude": longitude,
                            "area_temperature": area_temp,
                            "image_path": image_path
                        }
                        
                        # Generate QR code
                        qr_b64, qr_data, qr_path, qr_signature = generate_qr_code(catch_data)
                        
                        # Save to database
                        if save_catch_to_db(catch_data, qr_data, qr_path, qr_signature):
                            st.success("✅ Catch saved successfully!")
                            
                            # Display results
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown('<div class="card"><h3>🐟 Analysis Results</h3></div>', unsafe_allow_html=True)
                                st.write(f"**Species:** {result['species']}")
                                st.write(f"**Tamil Name:** {result['species_tamil']}")
                                
                                freshness_days = result['freshness_days_remaining']
                                if freshness_days >= 2:
                                    freshness_class = "freshness-high"
                                    icon = "🟢"
                                elif freshness_days >= 1:
                                    freshness_class = "freshness-medium"
                                    icon = "🟡"
                                else:
                                    freshness_class = "freshness-low"
                                    icon = "🔴"
                                
                                st.markdown(
                                    f"<p class='{freshness_class}'>{icon} Freshness: {freshness_days:.1f} days</p>",
                                    unsafe_allow_html=True
                                )
                                st.write(f"**Category:** {result['freshness_category']}")
                                st.write(f"**Confidence:** {result['species_confidence']:.1%}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="card"><h3>🌡️ Environmental Data</h3></div>', unsafe_allow_html=True)
                                st.write(f"**Area Temperature:** {area_temp:.1f}°C")
                                st.write(f"**Storage Temp:** {storage_temp}°C")
                                st.write(f"**Location:** {location}")
                                st.write(f"**Coordinates:** ({latitude:.4f}, {longitude:.4f})")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown('<div class="qr-card"><h3>📱 QR Code</h3></div>', unsafe_allow_html=True)
                                st.markdown(f"**Catch ID:** `{catch_id}`")
                                st.image(f"data:image/png;base64,{qr_b64}", width=200)
                                if USE_QR_SIGNATURE:
                                    st.success("🔒 Digitally Signed")
                                st.download_button(
                                    "💾 Download QR",
                                    base64.b64decode(qr_b64),
                                    f"qr_{catch_id}.png",
                                    "image/png",
                                    width="stretch"
                                )
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Recommendations
                            if result.get("recommendations"):
                                st.markdown('<div class="info-box"><h3>💡 Recommendations</h3></div>', unsafe_allow_html=True)
                                for rec in result["recommendations"]:
                                    st.write(f"• {rec}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Price recommendations
                            if result.get("price_range_per_kg"):
                                price_range = result["price_range_per_kg"]
                                st.markdown('<div class="success-box"><h3>💰 Suggested Price Range</h3></div>', unsafe_allow_html=True)
                                st.write(f"**Min:** ₹{price_range['min']}/kg")
                                st.write(f"**Recommended:** ₹{price_range['recommended']}/kg")
                                st.write(f"**Max:** ₹{price_range['max']}/kg")
                                st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to save catch to database")
                    else:
                        st.error(f"❌ Analysis failed: {result.get('error')}")
                else:
                    st.warning(f"⚠️ ML models not available: {error}")
                    st.info("You can still add catch manually with default values")
                    
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                st.error(f"❌ Error during analysis: {str(e)}")

def show_my_catches_page(fisher_name: str):
    """Display fisher's catches with filters"""
    st.markdown('<h3 class="section-header">🐟 My Catches</h3>', unsafe_allow_html=True)
    
    all_catches = get_catches_with_dynamic_freshness()
    my_catches = all_catches[all_catches['fisher_name'] == fisher_name]
    
    if len(my_catches) > 0:
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.selectbox(
                "🎯 Status Filter",
                ["All", "Available", "Sold", "Expired"]
            )
        
        with col2:
            species_filter = st.multiselect(
                "🐟 Species",
                options=my_catches['species'].unique()
            )
        
        with col3:
            sort_by = st.selectbox(
                "📊 Sort By",
                ["created_at", "current_freshness_days", "price_per_kg"],
                format_func=lambda x: {
                    "created_at": "Date Added",
                    "current_freshness_days": "Freshness",
                    "price_per_kg": "Price"
                }[x]
            )
        
        with col4:
            sort_order = st.radio("⬆️⬇️", ["Desc", "Asc"], horizontal=True)
        
        # Apply filters
        filtered_catches = my_catches.copy()
        if status_filter != "All":
            filtered_catches = filtered_catches[
                filtered_catches['status'] == status_filter.lower()
            ]
        if species_filter:
            filtered_catches = filtered_catches[
                filtered_catches['species'].isin(species_filter)
            ]
        
        filtered_catches = filtered_catches.sort_values(
            sort_by,
            ascending=(sort_order == "Asc")
        )
        
        st.write(f"**Showing {len(filtered_catches)} catches**")
        
        # Display catches
        for _, catch in filtered_catches.iterrows():
            st.markdown(f'<div class="fish-card"><h4>{catch["species"]}</h4></div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
            
            with col1:
                if catch.get('image_path') and os.path.exists(catch['image_path']):
                    try:
                        img = Image.open(catch['image_path'])
                        st.image(img, width=150)
                    except:
                        st.write("🐟")
                else:
                    st.markdown("### 🐟")
                st.write(f"*{catch['species_tamil']}*")
            
            with col2:
                freshness_days = catch['current_freshness_days']
                if freshness_days >= 2:
                    freshness_class = "freshness-high"
                    icon = "🟢"
                elif freshness_days >= 1:
                    freshness_class = "freshness-medium"
                    icon = "🟡"
                else:
                    freshness_class = "freshness-low"
                    icon = "🔴"
                
                st.markdown(
                    f"<p class='{freshness_class}'>{icon} Freshness: {freshness_days:.1f} days</p>",
                    unsafe_allow_html=True
                )
                st.write(f"**Category:** {catch['current_freshness_category']}")
                st.write(f"**Weight:** {catch['weight_g']}g")
            
            with col3:
                st.write(f"**Price:** ₹{catch['price_per_kg']}/kg")
                total_value = (catch['weight_g'] / 1000) * catch['price_per_kg']
                st.write(f"**Value:** ₹{total_value:.2f}")
                st.write(f"**Location:** {catch['location']}")
            
            with col4:
                status_class = f"status-{catch['status']}"
                st.markdown(
                    f"<span class='status-badge {status_class}'>{catch['status'].title()}</span>",
                    unsafe_allow_html=True
                )
                st.write(f"**ID:** {catch['catch_id']}")
                
                if st.button("📱 QR", key=f"qr_{catch['catch_id']}"):
                    st.session_state[f"show_qr_{catch['catch_id']}"] = \
                        not st.session_state.get(f"show_qr_{catch['catch_id']}", False)
            
            if st.session_state.get(f"show_qr_{catch['catch_id']}", False):
                try:
                    qr_data = json.loads(catch['qr_code'])
                    st.json(qr_data)
                except:
                    st.error("Could not display QR data")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🔭 No catches recorded yet. Add your first catch!")

def show_orders_page(fisher_name: str, user_id: int):
    """Display incoming orders for fisher"""
    st.markdown('<h3 class="section-header">📦 Incoming Orders</h3>', unsafe_allow_html=True)
    
    try:
        query = """
            SELECT o.order_id, o.buyer_name, o.quantity_kg, o.total_price,
                   o.status, o.created_at, o.buyer_latitude, o.buyer_longitude,
                   c.species, c.catch_id, c.location as catch_location,
                   c.image_path, c.price_per_kg
            FROM orders o
            JOIN catches c ON o.catch_id = c.catch_id
            WHERE c.fisher_name = %s OR c.user_id = %s
            ORDER BY o.created_at DESC
        """
        
        orders_df = pd.read_sql_query(
            query,
            db_manager.get_connection().__enter__(),
            params=(fisher_name, user_id)
        )
        
        if len(orders_df) > 0:
            # Order statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pending_orders = len(orders_df[orders_df['status'] == 'pending'])
                st.markdown(f'<div class="stats-card"><div class="stats-value">{pending_orders}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Pending Orders</div>', unsafe_allow_html=True)
            
            with col2:
                approved_orders = len(orders_df[orders_df['status'] == 'approved'])
                st.markdown(f'<div class="stats-card"><div class="stats-value">{approved_orders}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Approved Orders</div>', unsafe_allow_html=True)
            
            with col3:
                total_revenue = orders_df['total_price'].sum()
                st.markdown(f'<div class="stats-card"><div class="stats-value">₹{total_revenue:.0f}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Total Revenue</div>', unsafe_allow_html=True)
            
            with col4:
                total_orders = len(orders_df)
                st.markdown(f'<div class="stats-card"><div class="stats-value">{total_orders}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Total Orders</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display orders
            for _, order in orders_df.iterrows():
                st.markdown(f'<div class="card"><h4>{order["species"]}</h4></div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    if order.get('image_path') and os.path.exists(order['image_path']):
                        try:
                            img = Image.open(order['image_path'])
                            st.image(img, width=150)
                        except:
                            st.write("🐟")
                    else:
                        st.markdown("### 🐟")
                
                with col2:
                    st.write(f"**Order ID:** {order['order_id']}")
                    st.write(f"**Catch ID:** {order['catch_id']}")
                    st.write(f"**Buyer:** {order['buyer_name']}")
                    st.write(f"**Ordered:** {order['created_at']}")
                
                with col3:
                    st.write(f"**Quantity:** {order['quantity_kg']:.2f} kg")
                    st.write(f"**Price:** ₹{order['price_per_kg']}/kg")
                    st.write(f"**Total:** ₹{order['total_price']:.2f}")
                
                with col4:
                    status_class = f"status-{order['status']}"
                    st.markdown(
                        f"<span class='status-badge {status_class}'>{order['status'].title()}</span>",
                        unsafe_allow_html=True
                    )
                    
                    if order['status'] == 'pending':
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("✅ Approve", key=f"approve_{order['order_id']}", type="primary"):
                                try:
                                    update_query = """
                                        UPDATE orders SET status = 'approved', approved_at = %s 
                                        WHERE order_id = %s
                                    """
                                    db_manager.execute_query(
                                        update_query,
                                        (datetime.now(), order['order_id']),
                                        commit=True
                                    )
                                    
                                    # Notify buyer
                                    buyer_query = "SELECT buyer_user_id FROM orders WHERE order_id = %s"
                                    buyer_result = db_manager.execute_query(
                                        buyer_query,
                                        (order['order_id'],),
                                        fetch_one=True
                                    )
                                    
                                    if buyer_result and buyer_result['buyer_user_id']:
                                        create_notification(
                                            buyer_result['buyer_user_id'],
                                            'order',
                                            'Order Approved!',
                                            f'Your order {order["order_id"]} has been approved and will arrive soon!'
                                        )
                                    
                                    st.success("✅ Order approved!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to approve order: {e}")
                        
                        with col_b:
                            if st.button("❌ Reject", key=f"reject_{order['order_id']}"):
                                try:
                                    update_query = """
                                        UPDATE orders SET status = 'rejected', rejected_at = %s 
                                        WHERE order_id = %s
                                    """
                                    db_manager.execute_query(
                                        update_query,
                                        (datetime.now(), order['order_id']),
                                        commit=True
                                    )
                                    
                                    # Make catch available again
                                    catch_update = """
                                        UPDATE catches SET status = 'available' WHERE catch_id = %s
                                    """
                                    db_manager.execute_query(
                                        catch_update,
                                        (order['catch_id'],),
                                        commit=True
                                    )
                                    
                                    st.success("Order rejected")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to reject order: {e}")
                    
                    elif order['status'] == 'approved':
                        if st.button("📦 Mark Delivered", key=f"deliver_{order['order_id']}"):
                            try:
                                update_query = """
                                    UPDATE orders SET status = 'delivered', delivered_at = %s 
                                    WHERE order_id = %s
                                """
                                db_manager.execute_query(
                                    update_query,
                                    (datetime.now(), order['order_id']),
                                    commit=True
                                )
                                
                                st.success("✅ Marked as delivered!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to update order: {e}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("🔭 No orders received yet")
    
    except Exception as e:
        st.error(f"Failed to load orders: {e}")
        logger.error(f"Orders page error: {e}")

def show_map_view_page(fisher_name: str):
    """Display map visualization"""
    st.markdown('<h3 class="section-header">🗺️ Catch Locations Map</h3>', unsafe_allow_html=True)
    
    if not MAPS_AVAILABLE:
        st.warning("⚠️ Map visualization requires additional libraries")
    else:
        all_catches = get_catches_with_dynamic_freshness()
        
        # Ensure we have a proper DataFrame
        if not isinstance(all_catches, pd.DataFrame) or len(all_catches) == 0:
            st.info("🔭 No catches to display on map")
            return
        
        my_catches = all_catches[all_catches['fisher_name'] == fisher_name].copy()
        
        if len(my_catches) > 0:
            # 3D Deck map
            st.markdown("### 🌍 3D Catch Distribution")
            try:
                deck = create_fisher_map(fisher_name, my_catches)
                if deck:
                    st.pydeck_chart(deck)
                else:
                    st.warning("Unable to generate map visualization")
            except Exception as e:
                logger.error(f"Map display error: {e}")
                st.error("Map visualization unavailable")
            
            st.markdown("---")
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📍 Total Locations", len(my_catches))
            with col2:
                avg_freshness = my_catches['current_freshness_days'].mean()
                st.metric("⏱️ Avg Freshness", f"{avg_freshness:.1f} days")
            with col3:
                available = len(my_catches[my_catches['status'] == 'available'])
                st.metric("✅ Available", available)
        else:
            st.info("🔭 No catches to display on map")

def show_analytics_page(fisher_name: str):
    """Display fisher analytics"""
    st.markdown('<p class="section-header"><h3>📊 Fisher Analytics</h3></p>', unsafe_allow_html=True)
    
    all_catches = get_catches_with_dynamic_freshness()
    my_catches = all_catches[all_catches['fisher_name'] == fisher_name]
    
    if len(my_catches) > 0:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="stats-card"><div class="stats-value">{len(my_catches)}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label"> Total Catches </div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            avg_freshness = my_catches['current_freshness_days'].mean()
            st.markdown(f'<div class="stats-card"><div class="stats-value">{avg_freshness:.1f}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Avg Freshness (days)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            total_weight = my_catches['weight_g'].sum() / 1000
            st.markdown(f'<div class="stats-card"><div class="stats-value">{total_weight:.1f}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Total Weight (kg)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            total_value = (my_catches['weight_g'] * my_catches['price_per_kg'] / 1000).sum()
            st.markdown(f'<div class="stats-card"><div class="stats-value">₹{total_value:.0f}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Total Value</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🐟 Species Distribution")
            species_counts = my_catches['species'].value_counts()
            fig = px.pie(
                values=species_counts.values,
                names=species_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("### 📊 Status Distribution")
            status_counts = my_catches['status'].value_counts()
            fig = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                color=status_counts.values,
                color_continuous_scale='Viridis',
                labels={'x': 'Status', 'y': 'Count'}
            )
            st.plotly_chart(fig, width="stretch")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Daily Catch Trend")
            my_catches['date'] = pd.to_datetime(my_catches['created_at']).dt.date
            daily_catches = my_catches.groupby('date').size().reset_index(name='count')
            fig = px.line(
                daily_catches,
                x='date',
                y='count',
                markers=True,
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("### 💰 Price by Species")
            avg_price = my_catches.groupby('species')['price_per_kg'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=avg_price.values,
                y=avg_price.index,
                orientation='h',
                color=avg_price.values,
                color_continuous_scale='RdYlGn',
                labels={'x': 'Price (₹/kg)', 'y': 'Species'}
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("🔭 No data available for analytics")

def buyer_interface():
    """Buyer marketplace interface"""
    user = st.session_state.get('user', {})
    buyer_name = user.get('name', 'Buyer')
    user_id = user.get('user_id')
    
    # Header
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(
            f'<div class="user-info"><h2>👋 Welcome, {buyer_name} | Role: Buyer</h2></div>',
            unsafe_allow_html=True
        )
    with col2:
        if st.button("🔔 Notifications"):
            st.session_state['show_notifications'] = not st.session_state.get('show_notifications', False)
    with col3:
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Notifications
    if st.session_state.get('show_notifications', False):
        with st.sidebar:
            st.markdown("### 🔔 Recent Notifications")
            notifications = get_user_notifications(user_id)
            if len(notifications) > 0:
                for _, notif in notifications.head(5).iterrows():
                    icon = "📦" if notif['type'] == 'order' else "ℹ️"
                    st.info(f"{icon} **{notif['title']}**\n{notif['message']}")
            else:
                st.info("No new notifications")
    
    st.markdown("## 🛒 Buyer Marketplace")
    
    # Sidebar navigation
    with st.sidebar:
        page = st.radio(
            "📋 Navigation",
            ["🐟 Browse Fish", "🗺️ Map View", "📱 QR Verify", "📦 My Orders"]
        )
    
    # Page routing
    if page == "🐟 Browse Fish":
        show_browse_fish_page(buyer_name, user_id)
    elif page == "🗺️ Map View":
        show_buyer_map_page()
    elif page == "📱 QR Verify":
        show_qr_verify_page()
    elif page == "📦 My Orders":
        show_my_orders_page(buyer_name, user_id)

def show_browse_fish_page(buyer_name: str, user_id: int):
    """Browse available fish"""
    st.markdown("### 🐟 Available Fresh Fish")
    
    all_catches = get_catches_with_dynamic_freshness()
    available = all_catches[all_catches['status'] == 'available']
    available = available[available['is_expired'] == False]
    
    # Filters
    st.markdown("### 🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        species_filter = st.multiselect("🐟 Species", options=available['species'].unique())
    
    with col2:
        min_freshness = st.slider("⏱️ Min Freshness (days)", 0.0, 4.0, 0.5, 0.1)
    
    with col3:
        max_price = st.number_input("💰 Max Price (₹/kg)", value=1000, step=50)
    
    with col4:
        sort_by = st.selectbox(
            "📊 Sort By",
            ["current_freshness_days", "price_per_kg", "created_at"],
            format_func=lambda x: {
                "current_freshness_days": "Freshness",
                "price_per_kg": "Price",
                "created_at": "Recent"
            }[x]
        )
    
    # Apply filters
    filtered = available.copy()
    if species_filter:
        filtered = filtered[filtered['species'].isin(species_filter)]
    filtered = filtered[filtered['current_freshness_days'] >= min_freshness]
    filtered = filtered[filtered['price_per_kg'] <= max_price]
    filtered = filtered.sort_values(sort_by, ascending=False)
    
    st.write(f"**Found {len(filtered)} fresh fish**")
    st.markdown("---")
    
    # Display fish
    if len(filtered) > 0:
        for _, catch in filtered.iterrows():
            st.markdown(f'<div class="fish-card"><h4>{catch["species"]}</h4></div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                if catch.get('image_path') and os.path.exists(catch['image_path']):
                    try:
                        img = Image.open(catch['image_path'])
                        st.image(img, width=180)
                    except:
                        st.markdown("### 🐟")
                else:
                    st.markdown("### 🐟")
                st.markdown(f"### {catch['species_tamil']}")

            with col2:
                st.write(f"**Fisher:** {catch['fisher_name']}")
                st.write(f"**Location:** {catch['location']}")
                st.write(f"**Weight:** {catch['weight_g']}g")
                st.write(f"**Catch ID:** `{catch['catch_id']}`")
            
            with col3:
                freshness_days = catch['current_freshness_days']
                if freshness_days >= 2:
                    freshness_class = "freshness-high"
                    icon = "🟢"
                elif freshness_days >= 1:
                    freshness_class = "freshness-medium"
                    icon = "🟡"
                else:
                    freshness_class = "freshness-low"
                    icon = "🔴"
                
                st.markdown(
                    f"<p class='{freshness_class}'>{icon} Freshness: {freshness_days:.1f} days</p>",
                    unsafe_allow_html=True
                )
                st.write(f"**Category:** {catch['current_freshness_category']}")
                st.write(f"**Price:** ₹{catch['price_per_kg']}/kg")
                total = (catch['weight_g'] / 1000) * catch['price_per_kg']
                st.write(f"**Total Value:** ₹{total:.2f}")
            
            with col4:
                st.markdown("#### 🛒 Purchase")
                # Convert ALL numeric values to float
                max_weight_kg = float(catch['weight_g']) / 1000.0
                price_per_kg = float(catch['price_per_kg'])  # Convert Decimal to float
    
                quantity = st.number_input(
                    "Quantity (kg)",
                    min_value=0.1,
                    max_value=max_weight_kg,
                    value=0.5,
                    step=0.1,
                    key=f"qty_{catch['catch_id']}"
                )
    
                total_price = float(quantity) * price_per_kg  # Ensure both are float
                st.write(f"**Total:** ₹{total_price:.2f}")
                
                if st.button("🛒 Buy Now", key=f"buy_{catch['catch_id']}", type="primary", width="stretch"):
                    try:
                        order_id = str(uuid.uuid4())[:8].upper()
                        
                        # Convert all values to native Python types
                        insert_query = """
                            INSERT INTO orders (order_id, catch_id, buyer_user_id, buyer_name,
                                              quantity_kg, total_price, status, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        db_manager.execute_query(
                            insert_query,
                            (order_id, str(catch['catch_id']), int(user_id), str(buyer_name),
                             float(quantity), float(total_price), 'pending', datetime.now()),
                            commit=True
                        )
                        
                        # Mark catch as sold
                        update_query = "UPDATE catches SET status = 'sold' WHERE catch_id = %s"
                        db_manager.execute_query(update_query, (catch['catch_id'],), commit=True)
                        
                        st.success(f"✅ Order {order_id} placed successfully!")
                        st.balloons()
                        
                        # Clear cache to refresh data
                        st.cache_data.clear()
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Order failed: {e}")
                        logger.error(f"Order creation error: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🔍 No fish found matching your criteria")

def show_buyer_map_page():
    """Display buyer map view"""
    st.markdown('<h3 class="section-header">🗺️ Fish Locations</h3>', unsafe_allow_html=True)
    
    if not MAPS_AVAILABLE:
        st.warning("⚠️ Map visualization requires: pip install pydeck folium streamlit-folium geopy")
        return
    
    all_catches = get_catches_with_dynamic_freshness()
    available = all_catches[all_catches['status'] == 'available']
    available = available[available['is_expired'] == False]
    
    if len(available) > 0:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            species_options = available['species'].unique().tolist()
            selected_species = st.multiselect(
                "🐟 Species:",
                species_options,
                default=species_options[:3] if len(species_options) > 3 else species_options
            )
        
        with col2:
            map_min_freshness = st.slider("⏱️ Min Freshness (days):", 0.0, 4.0, 1.0, 0.1)
        
        with col3:
            max_distance = st.slider("📍 Max Distance (km):", 0, 200, 100, 10)
        
        # Apply filters
        filtered_catches = available[
            (available['species'].isin(selected_species)) &
            (available['current_freshness_days'] >= map_min_freshness)
        ].copy()
        
        st.write(f"**Found {len(filtered_catches)} fish on map**")
        
        # Create map
        try:
            buyer_map = create_buyer_map(filtered_catches, selected_species, map_min_freshness)
            
            if buyer_map:
                st_folium(buyer_map, width=700, height=500)
        except Exception as e:
            st.error(f"Map rendering failed: {e}")
            logger.error(f"Map error: {e}")
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🐟 Available Fish", len(filtered_catches))
        with col2:
            avg_price = filtered_catches['price_per_kg'].mean()
            st.metric("💰 Avg Price", f"₹{avg_price:.0f}/kg")
        with col3:
            avg_freshness = filtered_catches['current_freshness_days'].mean()
            st.metric("⏱️ Avg Freshness", f"{avg_freshness:.1f} days")
    else:
        st.info("🔭 No fish available on map")

def show_qr_verify_page():
    """QR code verification page"""
    st.markdown('<h3 class="section-header">📱 QR Code Verification</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card"><h3>🔍 Verify Fish Authenticity</h3></div>', unsafe_allow_html=True)
        
        qr_input = st.text_area(
            "Paste QR Code Data:",
            height=200,
            help="Scan QR code and paste JSON data here"
        )
        
        if qr_input and st.button("🔍 Verify", type="primary", width="stretch"):
            try:
                qr_data = json.loads(qr_input)
                
                # Verify signature if enabled
                is_authentic = True
                if USE_QR_SIGNATURE and 'signature' in qr_data:
                    qr_data_copy = dict(qr_data)
                    signature = qr_data_copy.pop('signature')
                    expected_signature = hmac.new(
                        QR_SECRET_KEY.encode(),
                        json.dumps(qr_data_copy, sort_keys=True).encode(),
                        hashlib.sha256
                    ).hexdigest()
                    
                    is_authentic = (signature == expected_signature)
                
                if is_authentic:
                    st.success("✅ QR Code Authenticated!")
                else:
                    st.error("⚠️ QR Code Signature Invalid!")
                
                st.markdown("---")
                
                # Display fish details
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### 🐟 Fish Details")
                    st.write(f"**Catch ID:** `{qr_data.get('catch_id')}`")
                    st.write(f"**Species:** {qr_data.get('species')}")
                    st.write(f"**Tamil Name:** {qr_data.get('species_tamil')}")
                    st.write(f"**Freshness:** {qr_data.get('freshness_days', 0):.1f} days")
                    st.write(f"**Category:** {qr_data.get('freshness_category')}")
                
                with col_b:
                    st.markdown("#### 📍 Source Info")
                    st.write(f"**Fisher:** {qr_data.get('fisher_name')}")
                    st.write(f"**Location:** {qr_data.get('location')}")
                    st.write(f"**Price:** ₹{qr_data.get('price_per_kg')}/kg")
                    st.write(f"**Timestamp:** {qr_data.get('timestamp')}")
                
                # Look up in database
                try:
                    query = "SELECT * FROM catches WHERE catch_id = %s"
                    catch_result = db_manager.execute_query(
                        query,
                        (qr_data.get('catch_id'),),
                        fetch_one=True
                    )
                    
                    if catch_result:
                        st.markdown("---")
                        st.markdown("### 📊 Current Status")
                        
                        # Calculate current freshness
                        catch_time = pd.to_datetime(catch_result['created_at'])
                        storage_temp = float(catch_result.get('storage_temp', 5.0))
                        initial_freshness = float(catch_result['freshness_days'])
                        
                        current_freshness, category = calculate_dynamic_freshness(
                            catch_time, initial_freshness, storage_temp
                        )
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Current Freshness", f"{current_freshness:.1f} days")
                        with col_b:
                            st.metric("Status", catch_result['status'].title())
                        with col_c:
                            st.metric("Category", category)
                        
                        # Show image if available
                        if catch_result.get('image_path') and os.path.exists(catch_result['image_path']):
                            st.markdown("### 📸 Fish Image")
                            img = Image.open(catch_result['image_path'])
                            st.image(img, width=400)
                    else:
                        st.warning("⚠️ Catch not found in database")
                
                except Exception as e:
                    logger.error(f"Database lookup error: {e}")
                    st.error("❌ Could not verify catch in database")
            
            except json.JSONDecodeError:
                st.error("❌ Invalid QR code data. Please check the format.")
            except Exception as e:
                st.error(f"❌ Verification failed: {e}")
                logger.error(f"QR verification error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box"><h3>ℹ️ How to Verify</h3>', unsafe_allow_html=True)
        st.write("1. Scan QR code on fish packaging")
        st.write("2. Copy the JSON data")
        st.write("3. Paste in the text area")
        st.write("4. Click Verify button")
        st.write("")
        st.write("**Benefits:**")
        st.write("✅ Verify authenticity")
        st.write("✅ Check current freshness")
        st.write("✅ View source details")
        st.write("✅ Make informed decisions")
        st.markdown('</div>', unsafe_allow_html=True)

def show_my_orders_page(buyer_name: str, user_id: int):
    """Display buyer's orders"""
    st.markdown('<h3 class="section-header">📦 My Orders</h3>', unsafe_allow_html=True)
    
    try:
        query = """
            SELECT o.order_id, o.quantity_kg, o.total_price, o.status, o.created_at,
                   c.species, c.fisher_name, c.location, c.image_path, c.catch_id
            FROM orders o
            JOIN catches c ON o.catch_id = c.catch_id
            WHERE o.buyer_user_id = %s OR o.buyer_name = %s
            ORDER BY o.created_at DESC
        """
        
        results = db_manager.execute_query(query, (user_id, buyer_name), fetch_all=True)
        
        if results:
            orders_df = pd.DataFrame(results)
            
            # Order statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pending = len(orders_df[orders_df['status'] == 'pending'])
                st.markdown(f'<div class="stats-card"><div class="stats-value">{pending}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Pending</div>', unsafe_allow_html=True)
            
            with col2:
                approved = len(orders_df[orders_df['status'] == 'approved'])
                st.markdown(f'<div class="stats-card"><div class="stats-value">{approved}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Approved</div>', unsafe_allow_html=True)
            
            with col3:
                total_spent = orders_df['total_price'].sum()
                st.markdown(f'<div class="stats-card"><div class="stats-value">₹{total_spent:.0f}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Total Spent</div>', unsafe_allow_html=True)
            
            with col4:
                total_orders = len(orders_df)
                st.markdown(f'<div class="stats-card"><div class="stats-value">{total_orders}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Total Orders</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display orders
            for _, order in orders_df.iterrows():
                st.markdown(f'<div class="order-card"><h4>{order["species"]}</h4></div>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
                
                with col1:
                    if order.get('image_path') and os.path.exists(order['image_path']):
                        try:
                            img = Image.open(order['image_path'])
                            st.image(img, width=150)
                        except:
                            st.markdown("### 🐟")
                    else:
                        st.markdown("### 🐟")
                
                with col2:
                    st.write(f"**Order ID:** {order['order_id']}")
                    st.write(f"**Catch ID:** {order['catch_id']}")
                    st.write(f"**Fisher:** {order['fisher_name']}")
                    st.write(f"**Location:** {order['location']}")
                
                with col3:
                    st.write(f"**Quantity:** {order['quantity_kg']:.2f} kg")
                    st.write(f"**Total:** ₹{order['total_price']:.2f}")
                    st.write(f"**Ordered:** {order['created_at']}")
                
                with col4:
                    status = order['status']
                    status_class = f"status-{status}"
                    
                    if status == 'pending':
                        st.markdown(f"<span class='status-badge {status_class}'>⏳ Pending</span>", unsafe_allow_html=True)
                        st.info("Waiting for fisher approval")
                    elif status == 'approved':
                        st.markdown(f"<span class='status-badge {status_class}'>✅ Approved</span>", unsafe_allow_html=True)
                        st.success("Fish will arrive soon!")
                    elif status == 'delivered':
                        st.markdown(f"<span class='status-badge {status_class}'>📦 Delivered</span>", unsafe_allow_html=True)
                        st.success("Order completed")
                    elif status == 'rejected':
                        st.markdown(f"<span class='status-badge status-sold'>❌ Rejected</span>", unsafe_allow_html=True)
                        st.error("Order was rejected")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("🔭 No orders placed yet. Browse fish to start shopping!")
    
    except Exception as e:
        st.error(f"Failed to load orders: {e}")
        logger.error(f"Orders page error: {e}")

def admin_interface():
    """Admin dashboard interface"""
    st.markdown("## 👨‍💼 Admin Dashboard")
    
    user = st.session_state.get('user', {})
    admin_name = user.get('name', 'Admin')
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            f'<div class="user-info"><h2>👋 Welcome, {admin_name} | Role: Administrator</h2></div>',
            unsafe_allow_html=True
        )
    with col2:
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📋 Navigation")
        page = st.radio(
            "Select Page:",
            ["📊 Analytics", "👥 Users", "🐟 Catches", "📦 Orders", "⚙️ System"],
            label_visibility="collapsed"
        )
    
    if page == "📊 Analytics":
        show_admin_analytics()
    elif page == "👥 Users":
        show_admin_users()
    elif page == "🐟 Catches":
        show_admin_catches()
    elif page == "📦 Orders":
        show_admin_orders()
    elif page == "⚙️ System":
        show_admin_system()

def show_admin_analytics():
    """Admin analytics page"""
    st.markdown('<h4 class="section-header">📊 System Analytics</h4>', unsafe_allow_html=True)
    
    try:
        # Get metrics
        user_count_query = "SELECT COUNT(*) as count FROM users"
        catch_count_query = "SELECT COUNT(*) as count FROM catches"
        order_count_query = "SELECT COUNT(*) as count FROM orders"
        revenue_query = "SELECT COALESCE(SUM(total_price), 0) as total FROM orders"
        
        user_count = db_manager.execute_query(user_count_query, fetch_one=True)['count']
        catch_count = db_manager.execute_query(catch_count_query, fetch_one=True)['count']
        order_count = db_manager.execute_query(order_count_query, fetch_one=True)['count']
        total_revenue = db_manager.execute_query(revenue_query, fetch_one=True)['total']
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="stats-card"><div class="stats-value">{user_count}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Total Users</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="stats-card"><div class="stats-value">{catch_count}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Total Catches</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="stats-card"><div class="stats-value">{order_count}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Total Orders</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="stats-card"><div class="stats-value">₹{total_revenue:.0f}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="stats-label">Total Revenue</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        all_catches = get_catches_with_dynamic_freshness()
        
        if len(all_catches) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🐟 Species Distribution")
                species_counts = all_catches['species'].value_counts().head(10)
                fig = px.bar(
                    x=species_counts.values,
                    y=species_counts.index,
                    orientation='h',
                    color=species_counts.values,
                    color_continuous_scale='Viridis',
                    labels={'x': 'Count', 'y': 'Species'}
                )
                st.plotly_chart(fig, width="stretch")
            
            with col2:
                st.markdown("### 📈 Daily Catch Trend")
                all_catches_copy = all_catches.copy()
                all_catches_copy['date'] = pd.to_datetime(all_catches_copy['created_at']).dt.date
                daily_catches = all_catches_copy.groupby('date').size().reset_index(name='count')
                fig = px.line(
                    daily_catches,
                    x='date',
                    y='count',
                    markers=True,
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, width="stretch")
        
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")
        logger.error(f"Admin analytics error: {e}")

def show_admin_users():
    """Admin users management"""
    st.markdown('<h4 class="section-header">👥 User Management</h4>', unsafe_allow_html=True)
    
    try:
        query = """
            SELECT user_id, name, phone, role, verified, is_active, created_at, last_login
            FROM users
            ORDER BY created_at DESC
        """
        results = db_manager.execute_query(query, fetch_all=True)
        
        if results:
            users_df = pd.DataFrame(results)
            
            # User statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fishers = len(users_df[users_df['role'] == 'fisher'])
                st.markdown(f'<div class="stats-card"><div class="stats-value">{fishers}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Fishers</div>', unsafe_allow_html=True)
            
            with col2:
                buyers = len(users_df[users_df['role'] == 'buyer'])
                st.markdown(f'<div class="stats-card"><div class="stats-value">{buyers}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Buyers</div>', unsafe_allow_html=True)
            
            with col3:
                admins = len(users_df[users_df['role'] == 'admin'])
                st.markdown(f'<div class="stats-card"><div class="stats-value">{admins}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Admins</div>', unsafe_allow_html=True)
            
            with col4:
                active = len(users_df[users_df['is_active'] == True])
                st.markdown(f'<div class="stats-card"><div class="stats-value">{active}</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="stats-label">Active Users</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # User table
            st.markdown("### 📋 User List")
            display_df = users_df.copy()
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d')
            display_df['last_login'] = pd.to_datetime(display_df['last_login']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(display_df, width="stretch", hide_index=True)
            
            # Export button
            csv = users_df.to_csv(index=False)
            st.download_button(
                "📥 Download Users CSV",
                csv,
                "users.csv",
                "text/csv",
                width="stretch"
            )
        else:
            st.info("🔭 No users registered yet")
    
    except Exception as e:
        st.error(f"Failed to load users: {e}")
        logger.error(f"Admin users error: {e}")

def show_admin_catches():
    """Admin catches management"""
    st.markdown('<h4 class="section-header">🐟 All Catches</h4>', unsafe_allow_html=True)
    
    all_catches = get_catches_with_dynamic_freshness()
    
    if len(all_catches) > 0:
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            available = len(all_catches[all_catches['status'] == 'available'])
            st.metric("Available", available)
        
        with col2:
            sold = len(all_catches[all_catches['status'] == 'sold'])
            st.metric("Sold", sold)
        
        with col3:
            expired = len(all_catches[all_catches['is_expired'] == True])
            st.metric("Expired", expired)
        
        with col4:
            avg_freshness = all_catches['current_freshness_days'].mean()
            st.metric("Avg Freshness", f"{avg_freshness:.1f} days")
        
        st.markdown("---")
        
        # Display table
        st.markdown("### 📋 Catch List")
        display_cols = ['catch_id', 'fisher_name', 'species', 'current_freshness_days',
                       'current_freshness_category', 'weight_g', 'price_per_kg',
                       'location', 'status', 'created_at']
        
        display_df = all_catches[display_cols].copy()
        display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_df, width="stretch", hide_index=True)
        
        # Export button
        csv = all_catches.to_csv(index=False)
        st.download_button(
            "📥 Download Catches CSV",
            csv,
            "catches.csv",
            "text/csv",
            width="stretch"
        )
    else:
        st.info("🔭 No catches recorded yet")

def show_admin_orders():
    """Admin orders management"""
    st.markdown('<h4 class="section-header">📦 All Orders</h4>', unsafe_allow_html=True)
    
    try:
        query = """
            SELECT o.order_id, o.buyer_name, o.quantity_kg, o.total_price,
                   o.status, o.created_at, c.species, c.fisher_name, c.location
            FROM orders o
            JOIN catches c ON o.catch_id = c.catch_id
            ORDER BY o.created_at DESC
        """
        results = db_manager.execute_query(query, fetch_all=True)
        
        if results:
            orders_df = pd.DataFrame(results)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                pending = len(orders_df[orders_df['status'] == 'pending'])
                st.metric("Pending", pending)
            
            with col2:
                approved = len(orders_df[orders_df['status'] == 'approved'])
                st.metric("Approved", approved)
            
            with col3:
                total_value = orders_df['total_price'].sum()
                st.metric("Total Value", f"₹{total_value:.0f}")
            
            with col4:
                avg_order = orders_df['total_price'].mean()
                st.metric("Avg Order Value", f"₹{avg_order:.0f}")
            
            st.markdown("---")
            
            # Display table
            display_df = orders_df.copy()
            display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(display_df, width="stretch", hide_index=True)
            
            # Export button
            csv = orders_df.to_csv(index=False)
            st.download_button(
                "📥 Download Orders CSV",
                csv,
                "orders.csv",
                "text/csv",
                width="stretch"
            )
        else:
            st.info("🔭 No orders placed yet")
    
    except Exception as e:
        st.error(f"Failed to load orders: {e}")
        logger.error(f"Admin orders error: {e}")

def show_admin_system():
    """Admin system management"""
    st.markdown('<h4 class="section-header">⚙️ System Management</h4>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><h3>🔧 System Status</h3></div>', unsafe_allow_html=True)
        
        # Database status
        try:
            with db_manager.get_connection() as conn:
                st.success("✅ Database: Connected")
        except:
            st.error("❌ Database: Disconnected")
        
        # ML Models status
        predictor, error = load_predictor()
        if predictor and predictor.is_available:
            st.success("✅ ML Models: Loaded")
        else:
            st.warning(f"⚠️ ML Models: {error}")
        
        # Maps status
        if MAPS_AVAILABLE:
            st.success("✅ Maps: Available")
        else:
            st.warning("⚠️ Maps: Not Available")
        
        # Twilio status
        if TWILIO_AVAILABLE and auth_helper.twilio_client:
            st.success("✅ SMS Service: Configured")
        else:
            st.warning("⚠️ SMS Service: Not Configured")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><h3>📊 System Statistics</h3></div>', unsafe_allow_html=True)
        
        try:
            # Database size
            size_query = """
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            """
            db_size = db_manager.execute_query(size_query, fetch_one=True)
            if db_size:
                st.info(f"Database Size: {db_size['size']}")
            
            # Table counts
            tables_query = """
                SELECT 
                    'Users' as table_name, COUNT(*) as count FROM users
                UNION ALL
                SELECT 'Catches', COUNT(*) FROM catches
                UNION ALL
                SELECT 'Orders', COUNT(*) FROM orders
                UNION ALL
                SELECT 'Notifications', COUNT(*) FROM notifications
            """
            table_counts = db_manager.execute_query(tables_query, fetch_all=True)
            
            if table_counts:
                st.markdown("**Record Counts:**")
                for row in table_counts:
                    st.write(f"- {row['table_name']}: {row['count']:,}")
        
        except Exception as e:
            st.error(f"Failed to get statistics: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # System actions
    st.markdown("---")
    st.markdown("### 🔧 System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Clear Cache", width="stretch"):
            st.cache_data.clear()
            st.success("✅ Cache cleared successfully!")
    
    with col2:
        if st.button("🧹 Cleanup Old Data", width="stretch"):
            try:
                # Clean old notifications
                cleanup_query = """
                    DELETE FROM notifications 
                    WHERE created_at < NOW() - INTERVAL '30 days'
                    AND is_read = TRUE
                """
                db_manager.execute_query(cleanup_query, commit=True)
                
                # Clean expired catches
                expired_query = """
                    DELETE FROM catches 
                    WHERE status = 'expired' 
                    AND created_at < NOW() - INTERVAL '7 days'
                """
                db_manager.execute_query(expired_query, commit=True)
                
                st.success("✅ Old data cleaned successfully!")
            except Exception as e:
                st.error(f"Cleanup failed: {e}")
    
    with col3:
        if st.button("📊 Export All Data", width="stretch"):
            try:
                # Export all data as JSON
                export_data = {
                    'users': db_manager.execute_query("SELECT * FROM users", fetch_all=True),
                    'catches': db_manager.execute_query("SELECT * FROM catches", fetch_all=True),
                    'orders': db_manager.execute_query("SELECT * FROM orders", fetch_all=True),
                    'export_date': datetime.now().isoformat()
                }
                
                # Convert to JSON string
                json_data = json.dumps(export_data, default=str, indent=2)
                
                st.download_button(
                    "📥 Download Export",
                    json_data,
                    f"sea_sure_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    width="stretch"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    # Logs section
    st.markdown("---")
    st.markdown("### 📋 Recent Logs")
    
    try:
        log_file = Path('logs/sea_sure_integrated.log')
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = f.readlines()
                recent_logs = logs[-50:]  # Last 50 lines
            
            log_text = ''.join(recent_logs)
            st.text_area("Recent Log Entries", log_text, height=300)
        else:
            st.info("No log file found")
    except Exception as e:
        st.error(f"Failed to load logs: {e}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point with enhanced error handling"""
    
    # Initialize database
    if not init_database():
        st.error("❌ Database initialization failed. Please check your configuration.")
        st.info("💡 Ensure PostgreSQL is running and database credentials are correct in .env file")
        st.stop()
    
    # Initialize connection pool
    if not db_manager.initialize_pool():
        logger.warning("Connection pool initialization failed, using direct connections")
    
    # Check if user is authenticated
    if not st.session_state.get('authenticated', False):
        show_login_page()
    else:
        # Get user info
        user = st.session_state.get('user', {})
        role = user.get('role', 'buyer')
        
        # Route to appropriate interface based on role
        try:
            if role == 'fisher':
                fisher_interface()
            elif role == 'buyer':
                buyer_interface()
            elif role == 'admin':
                admin_interface()
            else:
                st.error(f"❌ Unknown role: {role}")
                logger.error(f"Invalid user role: {role}")
        except Exception as e:
            st.error(f"❌ Application error: {e}")
            logger.error(f"Interface error for role {role}: {e}")
            
            # Offer logout option
            if st.button("🔄 Restart Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
            **🐟 SEA_SURE Platform v3.0**
            
            Smart Fisheries Management
        """)
    
    with footer_col2:
        st.markdown("""
            **Features**
            - Multi-role Authentication
            - Dynamic Freshness Tracking
            - QR Code Security
            - Real-time Notifications
            - Order Management
        """)
    
    with footer_col3:
        st.markdown("""
            **Technologies**
            - Streamlit
            - PostgreSQL
            - ML/AI (PyTorch, XGBoost)
            - Real-time Analytics
        """)
    
    st.markdown(f"""
        <p style='text-align: center; color: #666;'>
            Built with ❤️ for Tamil Nadu Fisheries | 
            Last Updated: {datetime.now().strftime('%Y-%m-%d')} |
            Version 3.0.0
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        # Log application start
        logger.info("=" * 80)
        logger.info("SEA_SURE Application Started")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Dev Mode: {DEV_MODE}")
        logger.info(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
        logger.info("=" * 80)
        
        # Run main application
        main()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.critical(f"Critical application error: {e}")
        st.error(f"""
            ❌ **Critical Error**
            
            The application encountered a critical error:
            {str(e)}
            
            Please check the logs for more details.
        """)
        
        # Show traceback in dev mode
        if DEV_MODE:
            import traceback
            st.code(traceback.format_exc())
    finally:
        # Cleanup
        logger.info("Application shutting down...")
