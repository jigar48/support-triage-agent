import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    pg = conn.execute("SELECT version()").fetchone()[0]
    vec = conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()[0]
    print("Postgres:", pg.split(",")[0])
    print("pgvector:", vec)
