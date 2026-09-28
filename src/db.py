#!/usr/bin/env python3
"""
Database abstraction layer for Neon PostgreSQL
"""

import os
import csv
import pg8000
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in environment")


def _parse_db_url(url: str) -> dict:
    """Parse DATABASE_URL into connection parameters."""
    # postgresql://user:pass@host:port/dbname?sslmode=require&channel_binding=require
    assert url.startswith("postgresql://")
    url = url[len("postgresql://"):]
    
    # Split credentials from rest
    creds, rest = url.split("@", 1)
    user, password = creds.split(":", 1)
    
    # Split host:port/db from params
    if "?" in rest:
        host_port_db, params = rest.split("?", 1)
    else:
        host_port_db = rest
    host_port, dbname = host_port_db.split("/", 1)
    
    # Handle optional port
    if ":" in host_port:
        host, port = host_port.split(":", 1)
        port = int(port)
    else:
        host = host_port
        port = 5432
    
    return {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "database": dbname,
        "ssl_context": True,
    }


def get_connection():
    """Get a new database connection."""
    params = _parse_db_url(DATABASE_URL)
    return pg8000.connect(**params)


def init_db():
    """Create the draws table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS draws (
            id SERIAL PRIMARY KEY,
            seq_id INT UNIQUE NOT NULL,
            game_name VARCHAR(50) NOT NULL,
            day_of_week VARCHAR(15) NOT NULL,
            draw_date DATE NOT NULL,
            n1 INT NOT NULL,
            n2 INT NOT NULL,
            n3 INT NOT NULL,
            n4 INT NOT NULL,
            n5 INT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_draws_game ON draws(game_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_draws_seq ON draws(seq_id)")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Database initialized: draws table ready")


def seed_from_csv(csv_path="data/draws.csv"):
    """Bulk insert records from CSV if table is empty."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if table has data
    cursor.execute("SELECT COUNT(*) FROM draws")
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"Database already seeded with {count} records, skipping")
        cursor.close()
        conn.close()
        return count
    
    # Read CSV and insert
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            draw_date = r.get("draw_date")
            if not draw_date:
                # Backfill: compute date from seq_id and day_of_week
                # This is a fallback for old CSV without dates
                draw_date = None
            rows.append((int(r["seq_id"]), r["game_name"], r["day_of_week"], draw_date,
                         int(r["n1"]), int(r["n2"]), int(r["n3"]), int(r["n4"]), int(r["n5"])))
    
    # Bulk insert
    cursor.executemany("""
        INSERT INTO draws (seq_id, game_name, day_of_week, draw_date, n1, n2, n3, n4, n5)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, rows)
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Seeded {len(rows)} records into Neon")
    return len(rows)


def get_draws(game_name=None):
    """Query draws ordered by seq_id ASC. Returns list of dicts with draw_date."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if game_name:
        cursor.execute("""
            SELECT seq_id, game_name, day_of_week, draw_date, n1, n2, n3, n4, n5
            FROM draws WHERE game_name = %s ORDER BY seq_id ASC
        """, (game_name,))
    else:
        cursor.execute("""
            SELECT seq_id, game_name, day_of_week, draw_date, n1, n2, n3, n4, n5
            FROM draws ORDER BY seq_id ASC
        """)
    
    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    # Convert date objects to string for JSON serialization
    for row in rows:
        if row.get("draw_date") and hasattr(row["draw_date"], "strftime"):
            row["draw_date"] = row["draw_date"].strftime("%Y-%m-%d")
    
    cursor.close()
    conn.close()
    return rows


def get_draws_df(game_name=None):
    """Query draws as numpy structured array (pandas-like)."""
    rows = get_draws(game_name)
    if not rows:
        return []
    return rows


if __name__ == "__main__":
    init_db()
    seed_from_csv()
    draws = get_draws("National Weekly")
    print(f"Loaded {len(draws)} draws for National Weekly")
    for d in draws[:2]:
        print(d)