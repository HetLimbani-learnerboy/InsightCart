"""
Project: InsightCart
File: connection.py
"""

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

database_url = settings.DATABASE_URL

if not database_url:
    raise ValueError(
        "DATABASE_URL is missing. Configure your Neon PostgreSQL "
        "connection string in the .env file."
    )

database_url = database_url.strip().strip('"').strip("'")
parsed_url = make_url(database_url)

# Use the installed psycopg2 driver for standard PostgreSQL URLs. Neon requires
# TLS; add sslmode=require when its URL does not specify an SSL mode already.
if parsed_url.get_backend_name() in {"postgres", "postgresql"}:
    if parsed_url.drivername in {"postgres", "postgresql"}:
        parsed_url = parsed_url.set(drivername="postgresql+psycopg2")
    if (
        parsed_url.host
        and parsed_url.host.endswith(".neon.tech")
        and not parsed_url.query.get("sslmode")
    ):
        parsed_url = parsed_url.update_query_dict({"sslmode": "require"})
    database_url = parsed_url.render_as_string(hide_password=False)

# Configure SQLAlchemy engine.
if parsed_url.get_backend_name() == "sqlite":
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={
            "connect_timeout": 10,
        },
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
