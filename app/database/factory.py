from app.database.base import Repository


def create_repository(url: str) -> Repository:
    if url.startswith("memory"):
        from app.database.memory_repo import InMemoryRepository      # tests / demos, data lost on restart
        return InMemoryRepository()
    if url.startswith("mongodb"):
        from app.database.mongo_repo import MongoRepository          # imported lazily so tests run without pymongo
        return MongoRepository(url)
    raise ValueError(f"Unsupported DATABASE_URL {url!r}. Use mongodb://localhost:27017/wingo (see backend/.env.example).")
