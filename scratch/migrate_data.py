import sqlite3
import os
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras

def migrate():
    # Load env variables from .env
    load_dotenv()
    
    # 1. Connect to SQLite
    sqlite_path = 'data/forest_fire.db'
    if not os.path.exists(sqlite_path):
        print(f"Error: SQLite database file '{sqlite_path}' does not exist.")
        return
        
    print(f"Connecting to SQLite database: {sqlite_path}")
    sq_conn = sqlite3.connect(sqlite_path)
    sq_conn.row_factory = sqlite3.Row
    sq_cursor = sq_conn.cursor()
    
    # 2. Connect to PostgreSQL
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5434")
    db_name = os.getenv("DB_NAME", "forest_fire")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "postgres")
    
    print(f"Connecting to PostgreSQL database '{db_name}' on {db_host}:{db_port}...")
    try:
        pg_conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )
        pg_cursor = pg_conn.cursor()
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        return
        
    # Tables in migration order (users first to satisfy foreign key constraint on alerts)
    tables = ['users', 'sensor_readings', 'image_predictions', 'ensemble_predictions', 'alerts']
    
    try:
        for table in tables:
            print(f"\nMigrating table '{table}'...")
            
            # Fetch all rows from SQLite
            sq_cursor.execute(f"SELECT * FROM {table}")
            rows = sq_cursor.fetchall()
            
            if not rows:
                print(f"No data found in SQLite table '{table}'. Skipping.")
                continue
                
            print(f"Found {len(rows)} rows in SQLite.")
            
            # Get columns
            columns = rows[0].keys()
            col_list = ", ".join(columns)
            val_placeholders = ", ".join(["%s"] * len(columns))
            
            # Clear PostgreSQL table before migration to prevent duplicates
            pg_cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
            
            # Insert into PostgreSQL
            insert_query = f"INSERT INTO {table} ({col_list}) VALUES ({val_placeholders})"
            
            for row in rows:
                values = [row[col] for col in columns]
                pg_cursor.execute(insert_query, values)
                
            print(f"Successfully migrated {len(rows)} rows to PostgreSQL table '{table}'.")
            
            # Reset SERIAL sequence for Postgres
            pg_cursor.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), coalesce(max(id), 1), max(id) IS NOT NULL) FROM {table}")
            
        # Commit all changes to PostgreSQL
        pg_conn.commit()
        print("\nMigration completed successfully!")
        
    except Exception as e:
        pg_conn.rollback()
        print(f"Error during migration: {e}")
        
    finally:
        sq_conn.close()
        pg_conn.close()

if __name__ == '__main__':
    migrate()
