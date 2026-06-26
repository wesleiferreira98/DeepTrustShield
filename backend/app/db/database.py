from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def run_migrations():
    """Add new columns to existing tables without breaking current data."""
    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(jobs)"))
        columns = {row[1] for row in result}
        if "mode" not in columns:
            conn.execute(text("ALTER TABLE jobs ADD COLUMN mode VARCHAR(10) DEFAULT 'image'"))
            conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
