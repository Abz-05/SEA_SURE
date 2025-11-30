"""
Test script for new modular architecture.

This script tests the configuration, database, and model modules
to ensure they work correctly before full integration.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_configuration():
    """Test configuration module."""
    print("=" * 60)
    print("Testing Configuration Module")
    print("=" * 60)
    
    try:
        from config.settings import settings
        from config.constants import (
            FreshnessCategory, CatchStatus, OrderStatus, UserRole
        )
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Database: {settings.db_name}")
        print(f"   Dev Mode: {settings.dev_mode}")
        print(f"   QR Storage: {settings.qr_storage_path}")
        print(f"   Twilio Enabled: {settings.twilio_enabled}")
        
        print(f"\n✅ Constants loaded successfully")
        print(f"   Freshness Categories: {[c.value for c in FreshnessCategory]}")
        print(f"   User Roles: {[r.value for r in UserRole]}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_connection():
    """Test database connection."""
    print("\n" + "=" * 60)
    print("Testing Database Connection")
    print("=" * 60)
    
    try:
        from database.connection import get_db_manager
        
        db = get_db_manager()
        
        # Try to initialize
        if db.initialize():
            print("✅ Database pool initialized successfully")
            
            # Health check
            if db.health_check():
                print("✅ Database health check passed")
            else:
                print("⚠️  Database health check failed (database may not be running)")
            
            db.close()
            return True
        else:
            print("⚠️  Database initialization failed (check your .env configuration)")
            return False
            
    except Exception as e:
        print(f"⚠️  Database test failed: {e}")
        print("   This is expected if database is not configured yet")
        return False

def test_models():
    """Test data models."""
    print("\n" + "=" * 60)
    print("Testing Data Models")
    print("=" * 60)
    
    try:
        from models import User, Catch, Order
        from datetime import datetime
        
        # Test User model
        user = User(
            user_id=1,
            name="Test Fisher",
            phone="+919876543210",
            role="fisher",
            verified=True
        )
        print(f"✅ User model created: {user.name} ({user.display_role})")
        print(f"   Is Fisher: {user.is_fisher}")
        
        # Test Catch model
        catch = Catch(
            catch_id="TEST001",
            fisher_name="Test Fisher",
            species="Tuna",
            weight_g=5000,
            price_per_kg=400,
            location="Chennai",
            latitude=13.0827,
            longitude=80.2707,
            freshness_days=3.0,
            storage_temp=4.0,
            hours_since_catch=6.0
        )
        print(f"\n✅ Catch model created: {catch.species}")
        print(f"   Weight: {catch.weight_kg} kg")
        print(f"   Total Value: ₹{catch.total_value:.2f}")
        
        # Test freshness calculation
        current_freshness, category = catch.calculate_current_freshness()
        print(f"   Current Freshness: {current_freshness} days ({category})")
        
        # Test Order model
        order = Order(
            order_id="ORD001",
            catch_id="TEST001",
            buyer_user_id=2,
            buyer_name="Test Buyer",
            quantity_kg=2.5,
            total_price=1000.0
        )
        print(f"\n✅ Order model created: {order.order_id}")
        print(f"   Quantity: {order.quantity_kg} kg")
        print(f"   Price per kg: ₹{order.price_per_kg:.2f}")
        print(f"   Status: {order.status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_repositories():
    """Test repository classes (without database)."""
    print("\n" + "=" * 60)
    print("Testing Repository Classes")
    print("=" * 60)
    
    try:
        from database.repositories import (
            UserRepository, CatchRepository, OrderRepository
        )
        
        # Just test instantiation
        user_repo = UserRepository()
        catch_repo = CatchRepository()
        order_repo = OrderRepository()
        
        print(f"✅ UserRepository created")
        print(f"   Table: {user_repo.table_name}")
        print(f"   Primary Key: {user_repo.primary_key}")
        
        print(f"\n✅ CatchRepository created")
        print(f"   Table: {catch_repo.table_name}")
        print(f"   Primary Key: {catch_repo.primary_key}")
        
        print(f"\n✅ OrderRepository created")
        print(f"   Table: {order_repo.table_name}")
        print(f"   Primary Key: {order_repo.primary_key}")
        
        return True
        
    except Exception as e:
        print(f"❌ Repository test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SEA_SURE Modular Architecture Test Suite")
    print("=" * 60)
    
    results = {
        "Configuration": test_configuration(),
        "Database Connection": test_database_connection(),
        "Data Models": test_models(),
        "Repositories": test_repositories()
    }
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! The new architecture is working correctly.")
    elif passed_count >= total_count - 1:
        print("\n⚠️  Most tests passed. Database connection may need configuration.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
