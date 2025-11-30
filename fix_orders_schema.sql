-- Fix orders table schema - Add missing timestamp columns
-- These columns are needed for order approval/rejection/delivery tracking

-- Add approved_at column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'orders' AND column_name = 'approved_at'
    ) THEN
        ALTER TABLE orders ADD COLUMN approved_at TIMESTAMP;
        RAISE NOTICE 'Added approved_at column to orders table';
    ELSE
        RAISE NOTICE 'approved_at column already exists';
    END IF;
END $$;

-- Add rejected_at column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'orders' AND column_name = 'rejected_at'
    ) THEN
        ALTER TABLE orders ADD COLUMN rejected_at TIMESTAMP;
        RAISE NOTICE 'Added rejected_at column to orders table';
    ELSE
        RAISE NOTICE 'rejected_at column already exists';
    END IF;
END $$;

-- Add delivered_at column if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'orders' AND column_name = 'delivered_at'
    ) THEN
        ALTER TABLE orders ADD COLUMN delivered_at TIMESTAMP;
        RAISE NOTICE 'Added delivered_at column to orders table';
    ELSE
        RAISE NOTICE 'delivered_at column already exists';
    END IF;
END $$;

-- Add cancelled_at column if it doesn't exist (for future use)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'orders' AND column_name = 'cancelled_at'
    ) THEN
        ALTER TABLE orders ADD COLUMN cancelled_at TIMESTAMP;
        RAISE NOTICE 'Added cancelled_at column to orders table';
    ELSE
        RAISE NOTICE 'cancelled_at column already exists';
    END IF;
END $$;

-- Verify the changes
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'orders' 
  AND column_name IN ('approved_at', 'rejected_at', 'delivered_at', 'cancelled_at')
ORDER BY column_name;
