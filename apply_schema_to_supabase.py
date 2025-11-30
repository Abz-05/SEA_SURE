"""
Apply SEA_SURE Database Schema to Supabase
This script applies the schema.sql to your Supabase production database
"""

import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database connection details
DB_CONFIG = {
    'host': os.getenv('SUPABASE_DB_HOST', 'aws-1-ap-southeast-2.pooler.supabase.com'),
    'port': int(os.getenv('SUPABASE_DB_PORT', '5432')),  # Use direct port 5432, not pooler
    'database': os.getenv('SUPABASE_DB_NAME', 'postgres'),
    'user': os.getenv('SUPABASE_DB_USER', 'postgres.tashfkdhagnlhrfqepoy'),
    'password': os.getenv('SUPABASE_DB_PASSWORD', 'Abzana#20052108'),
    'sslmode': 'require'
}

def apply_schema():
    """Apply schema.sql to Supabase database"""
    try:
        print("🔌 Connecting to Supabase database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        print("📄 Reading schema.sql file...")
        with open('schema.sql', 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        print("🚀 Applying schema to database...")
        cursor.execute(schema_sql)
        conn.commit()
        
        print("✅ Schema applied successfully!")
        
        # Verify tables created
        print("\n📊 Verifying tables...")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"\n✅ Found {len(tables)} tables:")
        for table in tables:
            print(f"   • {table[0]}")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 Database setup complete!")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        print(f"\nError details: {e.pgerror}")
    except FileNotFoundError:
        print("❌ schema.sql file not found in current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("  SEA_SURE - Supabase Schema Setup")
    print("=" * 60)
    print()
    
    # Confirm before proceeding
    print(f"Database: {DB_CONFIG['host']}")
    print(f"User: {DB_CONFIG['user']}")
    print()
    
    response = input("Apply schema to Supabase? (yes/no): ").strip().lower()
    
    if response == 'yes':
        apply_schema()
    else:
        print("❌ Schema application cancelled.")
