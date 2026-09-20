from app.config import Settings
from app.worker import create_engine


if __name__ == "__main__":
    create_engine(Settings())
