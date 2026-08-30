"""
Project : InsightCart
File : connection.py
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# Fallback to local SQLite during tests if DATABASE_URL is not set
database_url = settings.DATABASE_URL or "sqlite:///./test.db"

# Neon fix: Ensure proper dialect for psycopg2
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif database_url.startswith("postgresql://") and not database_url.startswith("postgresql+"):
    database_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)

# Connection engine parameters
if "sqlite" in database_url:
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
else:
    # Neon Serverless Postgres configuration
    engine = create_engine(
        database_url,
        pool_pre_ping=True,  # Automatically reconnects if Neon drops idle serverless connection
        pool_recycle=300,   # Recycles connections every 5 minutes
    )

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
