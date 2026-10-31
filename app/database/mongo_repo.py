"""In-memory repository for tests / demos (DATABASE_URL=memory://). Data is lost on restart.
It is the normal MongoRepository running on a tiny in-process stand-in for pymongo (app/database/fake_mongo.py),
so the logic is identical to production; it is NOT a real database."""
from app.database.fake_mongo import FakeClient
from app.database.mongo_repo import MongoRepository


class InMemoryRepository(MongoRepository):
    def __init__(self, url: str = "memory://wingo", client=None):
        super().__init__(url, client if client is not None else FakeClient())

    # read-only views used by tests to check what was (and was not) written
    @property
    def preds(self) -> list:
        return list(self.db.predictions.docs)

    @property
    def results(self) -> list:
        return list(self.db.results.docs)
