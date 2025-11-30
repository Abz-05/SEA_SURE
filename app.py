"""
SEA_SURE - Smart Fisheries Platform
Main Application Entry Point

This is the refactored version using modular architecture.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image
import time

# Import configuration
from config.settings import settings
from config.constants import UserRole, CatchStatus, OrderStatus

# Import database
from database import initialize_database, get_db_manager

# Import services
from services import AuthService, CatchService, OrderService, MLService

# Import utilities
from utils import GeoHelper, QRGenerator, ImageValidator, setup_logging
from ui import get_styles

# Setup logging
logger = setup_logging()

# Page configuration
st.set_page_config(
    page_title="SEA_SURE - Smart Fisheries Platform",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply styles
st.markdown(get_styles(), unsafe_allow_html=True)

# Initialize services (cached)
@st.cache_resource
def get_services():
    """Initialize and cache all services."""
    return {
        'auth': AuthService(),
        'catch': CatchService(),
        'order': OrderService(),
        'ml': MLService(),
        'geo': GeoHelper(),
        'qr': QRGenerator(),
        'image': ImageValidator()
    }

# Initialize database
@st.cache_resource
def init_app():
    """Initialize the application."""
    if not initialize_database():
        st.error("❌ Failed to connect to database. Please check configuration.")
        st.stop()
    logger.info("Application initialized")
    return True

# Initialize
init_app()
services = get_services()

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'user' not in st.session_state:
    st.session_state['user'] = None


def show_login_page():
    """Display login/registration page."""
    st.markdown('<h1 class="main-header">🐟 SEA_SURE</h1>', unsafe_allow_html=True)
    st.markdown('<center><h3>Smart Fisheries Platform - Ensuring Freshness, Building Trust</h3></center>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            
            login_phone = st.text_input("📱 Phone Number", placeholder="+919876543210", key="login_phone")
            login_password = st.text_input("🔒 Password", type="password", key="login_pass")
            
            if st.button("🚀 Login", type="primary", use_container_width=True):
                if not login_phone or not login_password:
                    st.error("⚠️ Please enter phone number and password")
                else:
                    user, error = services['auth'].login(login_phone, login_password)
                    if user:
                        st.session_state['user'] = user.to_dict()
                        st.session_state['authenticated'] = True
                        st.success(f"✅ Welcome back, {user.name}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
        
        with tab2:
            st.markdown("### Create New Account")
            
            reg_name = st.text_input("👤 Full Name", placeholder="John Doe", key="reg_name")
            reg_phone = st.text_input("📱 Phone Number", placeholder="+919876543210", key="reg_phone")
            reg_password = st.text_input("🔒 Password", type="password", key="reg_pass")
            reg_password_confirm = st.text_input("🔒 Confirm Password", type="password", key="reg_pass_confirm")
            
            reg_role = st.selectbox(
                "👥 Account Type", 
                [UserRole.BUYER.value, UserRole.FISHER.value, UserRole.ADMIN.value],
                format_func=lambda x: {
                    UserRole.BUYER.value: "🛒 Buyer - Browse and purchase fish",
                    UserRole.FISHER.value: "🎣 Fisher - Sell your catch",
                    UserRole.ADMIN.value: "👨‍💼 Admin - System management"
                }[x]
            )
            
            if st.button("📝 Create Account", type="primary", use_container_width=True):
                if not all([reg_name, reg_phone, reg_password, reg_password_confirm]):
                    st.error("⚠️ Please fill in all fields")
                elif reg_password != reg_password_confirm:
                    st.error("⚠️ Passwords do not match")
                else:
                    user, error = services['auth'].register(reg_name, reg_phone, reg_password, reg_role)
                    if user:
                        st.success(f"✅ Account created successfully!")
                        st.session_state['user'] = user.to_dict()
                        st.session_state['authenticated'] = True
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
        
        st.markdown("</div>", unsafe_allow_html=True)


def fisher_interface():
    """Fisher dashboard interface."""
    user = st.session_state.get('user', {})
    fisher_name = user.get('name', 'Fisher')
    user_id = user.get('user_id')
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f'<div class="user-info">👋 Welcome, {fisher_name} | Role: Fisher</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.markdown("## 🎣 Fisher Dashboard")
    
    # Sidebar navigation
    with st.sidebar:
        page = st.radio(
            "📋 Navigation",
            ["📸 Add Catch", "🐟 My Catches", "📦 Orders"]
        )
    
    if page == "📸 Add Catch":
        st.markdown("### 📸 Add New Catch")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload fish photo", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                valid, msg = services['image'].validate(image)
                if valid:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")
        
        with col2:
            weight = st.number_input("⚖️ Weight (grams)", min_value=1, value=500, step=50)
            storage_temp = st.slider("🌡️ Storage Temperature (°C)", 0, 35, 5)
            hours_since_catch = st.slider("⏱️ Hours Since Catch", 0, 72, 6)
            price_per_kg = st.number_input("💰 Price (₹/kg)", min_value=1, value=300, step=50)
            
            # Location
            cities = list(services['geo'].get_coastal_cities().keys())
            selected_city = st.selectbox("📍 Location", cities)
            latitude, longitude = services['geo'].get_coastal_cities()[selected_city]
            
        if uploaded_file and st.button("🔬 Analyze Fish & Generate QR", type="primary"):
            with st.spinner("🔄 Analyzing..."):
                # Get area temperature (mock for now)
                area_temp = 28.0
                
                # ML Prediction
                result = services['ml'].predict(
                    image, weight, storage_temp, hours_since_catch, area_temp
                )
                
                if result.get("prediction_success"):
                    # Save image
                    image_path = services['image'].save_image(image, "TEMP", str(settings.fish_images_path))
                    
                    # Generate QR
                    qr_data = {
                        "catch_id": "TEMP",
                        "species": result["species"],
                        "species_tamil": result["species_tamil"],
                        "freshness_days": result["freshness_days_remaining"],
                        "freshness_category": result["freshness_category"],
                        "price_per_kg": price_per_kg,
                        "fisher_name": fisher_name,
                        "location": selected_city
                    }
                    qr_b64, qr_json, qr_path, qr_sig = services['qr'].generate(qr_data)
                    
                    # Create catch
                    catch = services['catch'].create_catch(
                        fisher_name=fisher_name,
                        user_id=user_id,
                        species=result["species"],
                        weight_g=weight,
                        price_per_kg=price_per_kg,
                        location=selected_city,
                        latitude=latitude,
                        longitude=longitude,
                        freshness_days=result["freshness_days_remaining"],
                        storage_temp=storage_temp,
                        hours_since_catch=hours_since_catch,
                        area_temperature=area_temp,
                        species_tamil=result["species_tamil"],
                        image_path=image_path,
                        qr_code=qr_json,
                        qr_signature=qr_sig,
                        qr_path=qr_path
                    )
                    
                    if catch:
                        st.success("✅ Catch saved successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Species:** {result['species']}")
                            st.write(f"**Freshness:** {result['freshness_days_remaining']:.1f} days")
                        with col2:
                            st.write(f"**Category:** {result['freshness_category']}")
                            st.write(f"**Confidence:** {result['species_confidence']:.1%}")
                        with col3:
                            st.image(f"data:image/png;base64,{qr_b64}", width=200)
                    else:
                        st.error("❌ Failed to save catch")
                else:
                    st.error(f"❌ Analysis failed: {result.get('error')}")
    
    elif page == "🐟 My Catches":
        st.markdown("### 🐟 My Catches")
        
        catches = services['catch'].get_fisher_catches(fisher_name)
        
        if catches:
            for catch in catches:
                st.markdown('<div class="fish-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**{catch.species}** ({catch.species_tamil})")
                    st.write(f"**ID:** {catch.catch_id}")
                
                with col2:
                    current_fresh, category = catch.calculate_current_freshness()
                    st.write(f"**Freshness:** {current_fresh:.1f} days ({category})")
                    st.write(f"**Price:** ₹{catch.price_per_kg}/kg")
                
                with col3:
                    st.write(f"**Status:** {catch.status}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📭 No catches recorded yet")
    
    elif page == "📦 Orders":
        st.markdown("### 📦 Incoming Orders")
        
        orders = services['order'].get_fisher_orders(user_id)
        
        if orders:
            for order in orders:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([2, 2, 2])
                
                with col1:
                    st.write(f"**Order:** {order['order_id']}")
                    st.write(f"**Buyer:** {order['buyer_name']}")
                
                with col2:
                    st.write(f"**Species:** {order['species']}")
                    st.write(f"**Quantity:** {order['quantity_kg']} kg")
                
                with col3:
                    st.write(f"**Total:** ₹{order['total_price']:.2f}")
                    st.write(f"**Status:** {order['status']}")
                    
                    if order['status'] == 'pending':
                        if st.button("✅ Approve", key=f"approve_{order['order_id']}"):
                            if services['order'].approve_order(order['order_id']):
                                st.success("Order approved!")
                                st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📭 No orders yet")


def buyer_interface():
    """Buyer marketplace interface."""
    user = st.session_state.get('user', {})
    buyer_name = user.get('name', 'Buyer')
    user_id = user.get('user_id')
    
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f'<div class="user-info">👋 Welcome, {buyer_name} | Role: Buyer</div>', unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.markdown("## 🛒 Buyer Marketplace")
    
    # Get available catches
    catches = services['catch'].get_available_catches()
    
    if catches:
        st.write(f"**Found {len(catches)} fresh fish**")
        
        for catch in catches:
            current_fresh, category = catch.calculate_current_freshness()
            
            # Skip expired
            if current_fresh < 0.5:
                continue
            
            st.markdown('<div class="fish-card">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
            
            with col1:
                st.markdown(f"### {catch.species}")
                st.write(f"*{catch.species_tamil}*")
                st.write(f"**Fisher:** {catch.fisher_name}")
            
            with col2:
                icon = "🟢" if current_fresh >= 2 else "🟡" if current_fresh >= 1 else "🔴"
                st.write(f"{icon} **Freshness:** {current_fresh:.1f} days")
                st.write(f"**Category:** {category}")
            
            with col3:
                st.write(f"**Price:** ₹{catch.price_per_kg}/kg")
                st.write(f"**Weight:** {catch.weight_kg:.2f} kg")
                st.write(f"**Total:** ₹{catch.total_value:.2f}")
            
            with col4:
                quantity = st.number_input(
                    "Quantity (kg)",
                    min_value=0.1,
                    max_value=catch.weight_kg,
                    value=catch.weight_kg,
                    step=0.1,
                    key=f"qty_{catch.catch_id}"
                )
                
                total_price = quantity * catch.price_per_kg
                st.write(f"**Total:** ₹{total_price:.2f}")
                
                if st.button("🛒 Buy Now", key=f"buy_{catch.catch_id}", type="primary"):
                    order, error = services['order'].create_order(
                        catch_id=catch.catch_id,
                        buyer_user_id=user_id,
                        buyer_name=buyer_name,
                        quantity_kg=quantity,
                        total_price=total_price
                    )
                    
                    if order:
                        st.success(f"✅ Order {order.order_id} placed successfully!")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🔍 No fish available at the moment")


def admin_interface():
    """Admin dashboard interface."""
    st.markdown("## 👨‍💼 Admin Dashboard")
    st.info("Admin interface - Coming soon!")


# Main application logic
def main():
    """Main application entry point."""
    
    if not st.session_state.get('authenticated', False):
        show_login_page()
    else:
        user = st.session_state.get('user', {})
        role = user.get('role', UserRole.BUYER.value)
        
        if role == UserRole.FISHER.value:
            fisher_interface()
        elif role == UserRole.BUYER.value:
            buyer_interface()
        elif role == UserRole.ADMIN.value:
            admin_interface()
        else:
            st.error(f"❌ Unknown role: {role}")


if __name__ == "__main__":
    main()
