-- SEA_SURE Enhanced Database Schema
-- Database: seasure_db
-- Version: 2.0 - Enhanced with Order Management and Fisher-Buyer Linkage

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis"; -- For geographic data

-- ============================================================================
-- USERS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'buyer' CHECK (role IN ('fisher', 'buyer', 'admin')),
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    -- Geographic data
    default_latitude DECIMAL(10, 8),
    default_longitude DECIMAL(11, 8),
    default_location VARCHAR(255),
    -- User preferences
    language_preference VARCHAR(10) DEFAULT 'en',
    notification_enabled BOOLEAN DEFAULT TRUE
);

-- ============================================================================
-- OTPs TABLE (for phone verification)
-- ============================================================================
CREATE TABLE IF NOT EXISTS otps (
    otp_id SERIAL PRIMARY KEY,
    phone VARCHAR(20) NOT NULL,
    otp_code VARCHAR(10) NOT NULL,
    purpose VARCHAR(50) DEFAULT 'verification',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '5 minutes'),
    used_at TIMESTAMP,
    is_used BOOLEAN DEFAULT FALSE
);

-- ============================================================================
-- CATCHES TABLE (Enhanced with image storage and dynamic freshness tracking)
-- ============================================================================
CREATE TABLE IF NOT EXISTS catches (
    catch_id VARCHAR(50) PRIMARY KEY,
    fisher_name VARCHAR(255),
    user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    species VARCHAR(255) NOT NULL,
    species_tamil VARCHAR(255) DEFAULT 'மீன்',
    -- Freshness data
    freshness_days DECIMAL(5, 2) NOT NULL,
    freshness_category VARCHAR(50),
    -- Physical properties
    weight_g DECIMAL(10, 2) NOT NULL,
    storage_temp DECIMAL(5, 2) DEFAULT 5.0,
    hours_since_catch INTEGER DEFAULT 0,
    -- Pricing
    price_per_kg DECIMAL(10, 2) NOT NULL,
    -- Location data
    location VARCHAR(255),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    location_point GEOGRAPHY(POINT, 4326),
    area_temperature DECIMAL(5, 2),
    -- QR and security
    qr_code TEXT,
    qr_signature VARCHAR(255),
    qr_path VARCHAR(500),
    -- Image storage
    image_path VARCHAR(500),
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'available' CHECK (status IN ('available', 'sold', 'expired', 'reserved')),
    offline_cached BOOLEAN DEFAULT FALSE,
    -- Additional tracking
    view_count INTEGER DEFAULT 0,
    last_viewed_at TIMESTAMP
);

-- ============================================================================
-- ORDERS TABLE (Enhanced with Fisher-Buyer Linkage and Status Tracking)
-- ============================================================================
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(50) PRIMARY KEY,
    -- Catch reference
    catch_id VARCHAR(50) REFERENCES catches(catch_id) ON DELETE CASCADE,
    -- Buyer information
    buyer_user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    buyer_name VARCHAR(255) NOT NULL,
    buyer_phone VARCHAR(20),
    buyer_latitude DECIMAL(10, 8),
    buyer_longitude DECIMAL(11, 8),
    -- Order details
    quantity_kg DECIMAL(10, 2) NOT NULL,
    total_price DECIMAL(10, 2) NOT NULL,
    -- Status tracking with timestamps
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'in_transit', 'delivered', 'cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    approved_at TIMESTAMP,
    rejected_at TIMESTAMP,
    delivered_at TIMESTAMP,
    cancelled_at TIMESTAMP,
    -- Delivery information
    distance_km DECIMAL(10, 2),
    delivery_address TEXT,
    estimated_delivery_time TIMESTAMP,
    tracking_number VARCHAR(100),
    -- Payment information
    payment_status VARCHAR(50) DEFAULT 'pending' CHECK (payment_status IN ('pending', 'paid', 'refunded', 'failed')),
    payment_method VARCHAR(50),
    payment_reference VARCHAR(255),
    -- Fisher notes and buyer notes
    fisher_notes TEXT,
    buyer_notes TEXT,
    -- Rating and feedback (after delivery)
    buyer_rating INTEGER CHECK (buyer_rating >= 1 AND buyer_rating <= 5),
    buyer_feedback TEXT,
    fisher_rating INTEGER CHECK (fisher_rating >= 1 AND fisher_rating <= 5),
    fisher_feedback TEXT
);

-- ============================================================================
-- NOTIFICATIONS TABLE (For real-time updates to users)
-- ============================================================================
CREATE TABLE IF NOT EXISTS notifications (
    notification_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL CHECK (type IN ('order', 'payment', 'delivery', 'system', 'freshness_alert', 'promotion')),
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP,
    -- Action link (optional)
    action_url VARCHAR(500),
    action_label VARCHAR(100),
    -- Priority
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent'))
);

-- ============================================================================
-- ANALYTICS TABLE (For system metrics and business intelligence)
-- ============================================================================
CREATE TABLE IF NOT EXISTS analytics (
    analytics_id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    user_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
    data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Session tracking
    session_id VARCHAR(100),
    ip_address VARCHAR(50),
    user_agent TEXT
);

-- ============================================================================
-- FRESHNESS_LOGS TABLE (Track freshness decay over time)
-- ============================================================================
CREATE TABLE IF NOT EXISTS freshness_logs (
    log_id SERIAL PRIMARY KEY,
    catch_id VARCHAR(50) REFERENCES catches(catch_id) ON DELETE CASCADE,
    calculated_freshness DECIMAL(5, 2) NOT NULL,
    freshness_category VARCHAR(50),
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    storage_temp DECIMAL(5, 2),
    hours_elapsed INTEGER,
    notes TEXT
);

-- ============================================================================
-- FISHER_BUYER_INTERACTIONS TABLE (Track fisher-buyer relationships)
-- ============================================================================
CREATE TABLE IF NOT EXISTS fisher_buyer_interactions (
    interaction_id SERIAL PRIMARY KEY,
    fisher_user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    buyer_user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    interaction_type VARCHAR(50) CHECK (interaction_type IN ('order_placed', 'order_approved', 'order_delivered', 'message_sent', 'rating_given')),
    order_id VARCHAR(50) REFERENCES orders(order_id) ON DELETE SET NULL,
    interaction_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- PROMOTIONS TABLE (For special offers and combo deals)
-- ============================================================================
CREATE TABLE IF NOT EXISTS promotions (
    promotion_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    promotion_type VARCHAR(50) CHECK (promotion_type IN ('discount', 'combo_deal', 'free_delivery', 'flash_sale')),
    discount_percentage DECIMAL(5, 2),
    min_purchase_amount DECIMAL(10, 2),
    max_discount_amount DECIMAL(10, 2),
    applicable_species VARCHAR(255)[],
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    usage_count INTEGER DEFAULT 0,
    max_usage INTEGER,
    created_by INTEGER REFERENCES users(user_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES for Performance Optimization
-- ============================================================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

-- Catches indexes
CREATE INDEX IF NOT EXISTS idx_catches_user_id ON catches(user_id);
CREATE INDEX IF NOT EXISTS idx_catches_status ON catches(status);
CREATE INDEX IF NOT EXISTS idx_catches_species ON catches(species);
CREATE INDEX IF NOT EXISTS idx_catches_created_at ON catches(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_catches_freshness ON catches(freshness_days DESC);
CREATE INDEX IF NOT EXISTS idx_catches_location ON catches USING GIST(location_point);
CREATE INDEX IF NOT EXISTS idx_catches_fisher_status ON catches(fisher_name, status);

-- Orders indexes
CREATE INDEX IF NOT EXISTS idx_orders_catch_id ON orders(catch_id);
CREATE INDEX IF NOT EXISTS idx_orders_buyer_user_id ON orders(buyer_user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_payment_status ON orders(payment_status);

-- Notifications indexes
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON notifications(is_read);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON notifications(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_user_unread ON notifications(user_id, is_read) WHERE is_read = FALSE;

-- Analytics indexes
CREATE INDEX IF NOT EXISTS idx_analytics_event_type ON analytics(event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_created_at ON analytics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id);

-- Freshness logs indexes
CREATE INDEX IF NOT EXISTS idx_freshness_logs_catch_id ON freshness_logs(catch_id);
CREATE INDEX IF NOT EXISTS idx_freshness_logs_calculated_at ON freshness_logs(calculated_at DESC);

-- Fisher-buyer interactions indexes
CREATE INDEX IF NOT EXISTS idx_interactions_fisher ON fisher_buyer_interactions(fisher_user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_buyer ON fisher_buyer_interactions(buyer_user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_type ON fisher_buyer_interactions(interaction_type);

-- ============================================================================
-- VIEWS for Common Queries
-- ============================================================================

-- Active catches with fisher details
CREATE OR REPLACE VIEW active_catches_view AS
SELECT 
    c.*,
    u.name as fisher_full_name,
    u.phone as fisher_phone,
    u.verified as fisher_verified,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - c.created_at)) / 3600 as hours_since_created
FROM catches c
LEFT JOIN users u ON c.user_id = u.user_id
WHERE c.status = 'available';

-- Order details with complete information
CREATE OR REPLACE VIEW order_details_view AS
SELECT 
    o.*,
    c.species,
    c.fisher_name,
    c.location as catch_location,
    c.image_path as catch_image,
    c.latitude as catch_latitude,
    c.longitude as catch_longitude,
    u.name as buyer_full_name,
    u.phone as buyer_full_phone,
    fu.name as fisher_full_name,
    fu.phone as fisher_full_phone
FROM orders o
JOIN catches c ON o.catch_id = c.catch_id
LEFT JOIN users u ON o.buyer_user_id = u.user_id
LEFT JOIN users fu ON c.user_id = fu.user_id;

-- Fisher statistics view
CREATE OR REPLACE VIEW fisher_statistics AS
SELECT 
    u.user_id,
    u.name as fisher_name,
    COUNT(c.catch_id) as total_catches,
    COUNT(CASE WHEN c.status = 'available' THEN 1 END) as available_catches,
    COUNT(CASE WHEN c.status = 'sold' THEN 1 END) as sold_catches,
    AVG(c.freshness_days) as avg_freshness,
    SUM(c.weight_g) / 1000 as total_weight_kg,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_price) as total_revenue,
    AVG(o.buyer_rating) as avg_buyer_rating
FROM users u
LEFT JOIN catches c ON u.user_id = c.user_id
LEFT JOIN orders o ON c.catch_id = o.catch_id
WHERE u.role = 'fisher'
GROUP BY u.user_id, u.name;

-- Buyer statistics view
CREATE OR REPLACE VIEW buyer_statistics AS
SELECT 
    u.user_id,
    u.name as buyer_name,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_price) as total_spent,
    AVG(o.total_price) as avg_order_value,
    COUNT(CASE WHEN o.status = 'delivered' THEN 1 END) as completed_orders,
    AVG(o.fisher_rating) as avg_fisher_rating
FROM users u
LEFT JOIN orders o ON u.user_id = o.buyer_user_id
WHERE u.role = 'buyer'
GROUP BY u.user_id, u.name;

-- Daily sales summary view
CREATE OR REPLACE VIEW daily_sales_summary AS
SELECT 
    DATE(o.created_at) as sale_date,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_price) as total_revenue,
    AVG(o.total_price) as avg_order_value,
    COUNT(DISTINCT o.buyer_user_id) as unique_buyers,
    COUNT(DISTINCT c.user_id) as unique_fishers,
    SUM(o.quantity_kg) as total_quantity_kg
FROM orders o
JOIN catches c ON o.catch_id = c.catch_id
GROUP BY DATE(o.created_at)
ORDER BY sale_date DESC;

-- ============================================================================
-- FUNCTIONS for Business Logic
-- ============================================================================

-- Function to update location_point when coordinates change
CREATE OR REPLACE FUNCTION update_location_point()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.latitude IS NOT NULL AND NEW.longitude IS NOT NULL THEN
        NEW.location_point = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to automatically update catch status based on freshness
CREATE OR REPLACE FUNCTION check_freshness_expiry()
RETURNS TRIGGER AS $$
BEGIN
    -- Calculate hours since catch
    IF EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - NEW.created_at)) / 3600 > (NEW.freshness_days * 24) THEN
        NEW.status = 'expired';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to create notification when order status changes
CREATE OR REPLACE FUNCTION notify_order_status_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status != NEW.status THEN
        -- Notify buyer
        IF NEW.buyer_user_id IS NOT NULL THEN
            INSERT INTO notifications (user_id, type, title, message, priority)
            VALUES (
                NEW.buyer_user_id,
                'order',
                'Order Status Updated',
                CONCAT('Your order ', NEW.order_id, ' status changed to: ', NEW.status),
                CASE 
                    WHEN NEW.status = 'approved' THEN 'high'
                    WHEN NEW.status = 'delivered' THEN 'high'
                    ELSE 'normal'
                END
            );
        END IF;
        
        -- Notify fisher when order is placed
        IF NEW.status = 'pending' THEN
            INSERT INTO notifications (user_id, type, title, message, priority)
            SELECT 
                c.user_id,
                'order',
                'New Order Received',
                CONCAT('You have received a new order (', NEW.order_id, ') for ', c.species),
                'high'
            FROM catches c
            WHERE c.catch_id = NEW.catch_id AND c.user_id IS NOT NULL;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to log freshness calculations
CREATE OR REPLACE FUNCTION log_freshness_calculation(
    p_catch_id VARCHAR(50),
    p_calculated_freshness DECIMAL(5, 2),
    p_category VARCHAR(50),
    p_storage_temp DECIMAL(5, 2)
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO freshness_logs (catch_id, calculated_freshness, freshness_category, storage_temp)
    VALUES (p_catch_id, p_calculated_freshness, p_category, p_storage_temp);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger for catches location point update
DROP TRIGGER IF EXISTS update_catches_location_point ON catches;
CREATE TRIGGER update_catches_location_point
BEFORE INSERT OR UPDATE ON catches
FOR EACH ROW
EXECUTE FUNCTION update_location_point();

-- Trigger for order status change notifications
DROP TRIGGER IF EXISTS notify_on_order_status_change ON orders;
CREATE TRIGGER notify_on_order_status_change
AFTER UPDATE ON orders
FOR EACH ROW
EXECUTE FUNCTION notify_order_status_change();

-- Trigger to update catches.updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_catches_updated_at ON catches;
CREATE TRIGGER update_catches_updated_at
BEFORE UPDATE ON catches
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SAMPLE DATA INSERTION (Optional - for testing)
-- ============================================================================

-- Insert admin user (password: admin123)
INSERT INTO users (name, phone, password_hash, role, verified, is_active)
VALUES (
    'System Admin',
    '+919999999999',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpLaEg7u2',
    'admin',
    TRUE,
    TRUE
) ON CONFLICT (phone) DO NOTHING;

-- ============================================================================
-- PERMISSIONS (Configure as needed)
-- ============================================================================

-- Grant permissions to application user
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO seasure_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO seasure_app_user;

-- ============================================================================
-- MAINTENANCE FUNCTIONS
-- ============================================================================

-- Function to clean up old notifications
CREATE OR REPLACE FUNCTION cleanup_old_notifications(days_old INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM notifications
    WHERE created_at < CURRENT_TIMESTAMP - (days_old || ' days')::INTERVAL
    AND is_read = TRUE;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup expired catches
CREATE OR REPLACE FUNCTION cleanup_expired_catches(days_old INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM catches
    WHERE status = 'expired'
    AND created_at < CURRENT_TIMESTAMP - (days_old || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS for Documentation
-- ============================================================================

COMMENT ON TABLE users IS 'Stores user information for fishers, buyers, and admins';
COMMENT ON TABLE catches IS 'Stores fish catch information with dynamic freshness tracking';
COMMENT ON TABLE orders IS 'Stores order information with fisher-buyer linkage and status tracking';
COMMENT ON TABLE notifications IS 'Stores user notifications for real-time updates';
COMMENT ON TABLE analytics IS 'Stores system events and user actions for business intelligence';
COMMENT ON TABLE freshness_logs IS 'Tracks freshness decay calculations over time';
COMMENT ON TABLE fisher_buyer_interactions IS 'Tracks all interactions between fishers and buyers';
COMMENT ON TABLE promotions IS 'Stores promotional offers and deals';

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- To use this schema:
-- 1. Create database: CREATE DATABASE seasure_db;
-- 2. Connect to database: \c seasure_db
-- 3. Run this script: \i schema.sql
-- 4. Verify tables: \dt