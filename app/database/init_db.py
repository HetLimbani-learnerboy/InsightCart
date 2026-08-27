from app.database.connection import engine, Base
from app.database.models import ReviewPrediction


def init_db():

    Base.metadata.create_all(
        bind=engine
    )

    print("Database tables created successfully.")


if __name__ == "__main__":
    init_db()