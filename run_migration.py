"""
Database Migration Script - Add Missing Timestamp Columns to Orders Table
Fixes the issue where fishers cannot approve/reject/deliver orders
"""
import psycopg2
import os

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'seasure_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'Abzu#2005')
}

def run_migration():
    """Run the database migration to add missing timestamp columns"""
    try:
        # Connect to the database
        print(f"Connecting to database: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        print("\n" + "="*60)
        print("Starting Migration: Add Timestamp Columns to Orders Table")
        print("="*60 + "\n")
        
        # Array of columns to add
        columns = [
            ('approved_at', 'TIMESTAMP'),
            ('rejected_at', 'TIMESTAMP'),
            ('delivered_at', 'TIMESTAMP'),
            ('cancelled_at', 'TIMESTAMP')
        ]
        
        for column_name, data_type in columns:
            # Check if column exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'orders' AND column_name = %s
                )
            """, (column_name,))
            
            exists = cur.fetchone()[0]
            
            if not exists:
                # Add the column
                cur.execute(f"ALTER TABLE orders ADD COLUMN {column_name} {data_type}")
                conn.commit()
                print(f"✅ Added column: {column_name} ({data_type})")
            else:
                print(f"ℹ️  Column already exists: {column_name}")
        
        print("\n" + "="*60)
        print("Verifying Changes")
        print("="*60 + "\n")
        
        # Verify the changes
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'orders' 
              AND column_name IN ('approved_at', 'rejected_at', 'delivered_at', 'cancelled_at')
            ORDER BY column_name
        """)
        
        results = cur.fetchall()
        print("Current timestamp columns in orders table:")
        for row in results:
            print(f"  - {row[0]}: {row[1]}")
        
        print("\n" + "="*60)
        print("✅ Migration completed successfully!")
        print("="*60)
        print("\nYou can now approve/reject/deliver orders as a fisher.")
        
        # Close connection
        cur.close()
        conn.close()
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error occurred: {e}")
        print(f"Error code: {e.pgcode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = run_migration()
    if success:
        print("\n✨ Database schema is now up to date!")
    else:
        print("\n⚠️  Migration failed. Please check the error messages above.")
