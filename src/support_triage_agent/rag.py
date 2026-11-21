"""Chunk help-center articles, embed them, store them in pgvector, search them."""

import os
from dataclasses import dataclass
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pgvector.psycopg import register_vector

load_dotenv()

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
MAX_CHARS = 600
DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "help_center.md"

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
    return _model


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # lets psycopg send/receive numpy arrays as vectors
    return conn


@dataclass
class Chunk:
    article: str
    content: str


def load_articles(path: Path = DATA_FILE) -> list[tuple[str, str]]:
    """Split the markdown file into (title, body) pairs on '# ' headings."""
    articles, title, lines = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            if title:
                articles.append((title, "\n".join(lines).strip()))
            title, lines = line[2:].strip(), []
        else:
            lines.append(line)
    if title:
        articles.append((title, "\n".join(lines).strip()))
    return articles


def chunk_article(title: str, body: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Group paragraphs into chunks of up to max_chars. Each chunk starts with
    the article title so it still makes sense when retrieved on its own."""
    chunks, current = [], ""
    for para in (p.strip() for p in body.split("\n\n")):
        if not para:
            continue
        if current and len(current) + len(para) > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return [Chunk(title, f"{title}\n\n{c}") for c in chunks]


def ingest() -> int:
    chunks = [c for title, body in load_articles() for c in chunk_article(title, body)]
    vectors = list(get_model().passage_embed([c.content for c in chunks]))
    with connect() as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(
            f"""CREATE TABLE chunks (
                id bigserial PRIMARY KEY,
                article text NOT NULL,
                content text NOT NULL,
                embedding vector({EMBED_DIM}) NOT NULL
            )"""
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (article, content, embedding) VALUES (%s, %s, %s)",
                [(c.article, c.content, v) for c, v in zip(chunks, vectors)],
            )
        conn.execute("CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)")
    return len(chunks)


def search(query: str, k: int = 3) -> list[tuple[str, str, float]]:
    """Return the k closest chunks as (article, content, similarity)."""
    qvec = next(iter(get_model().query_embed([query])))
    with connect() as conn:
        return conn.execute(
            """SELECT article, content, 1 - (embedding <=> %s) AS similarity
               FROM chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (qvec, qvec, k),
        ).fetchall()
