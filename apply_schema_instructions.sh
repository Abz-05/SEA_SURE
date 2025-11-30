# Apply Schema to Supabase Database
# Run this from your local terminal

# Method 1: Using psql command
psql "postgresql://postgres.tashfkdhagnlhrfqepoy:Abzana%2320052108@aws-1-ap-southeast-2.pooler.supabase.com:5432/postgres?sslmode=require" -f schema.sql

# Method 2: Using Python
python apply_schema_to_supabase.py
