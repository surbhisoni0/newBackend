"""A tiny in-memory stand-in for the parts of pymongo that MongoRepository uses, so the repository logic can be tested
without a MongoDB server. It is NOT MongoDB: keep a real-database check in your own setup (see README)."""
import copy


def _match(doc, flt):
    for k, v in flt.items():
        val = doc.get(k)
        if isinstance(v, dict):
            for op, arg in v.items():
                ok = {"$ne": lambda: val != arg, "$in": lambda: val in arg, "$gte": lambda: val is not None and val >= arg,
                      "$lt": lambda: val is not None and val < arg, "$type": lambda: isinstance(val, str)}[op]()
                if not ok:
                    return False
        elif val != v:
            return False
    return True


class Cursor:
    def __init__(self, docs): self.docs = docs
    def sort(self, key, direction=None):
        keys = key if isinstance(key, list) else [(key, direction)]
        for k, d in reversed(keys):
            self.docs.sort(key=lambda x: (x.get(k) is None, x.get(k)), reverse=(d == -1))
        return self
    def skip(self, n): self.docs = self.docs[n:]; return self
    def limit(self, n): self.docs = self.docs[:n] if n else self.docs; return self
    def __iter__(self): return iter(copy.deepcopy(self.docs))


def _project(doc, proj):
    if not proj: return copy.deepcopy(doc)
    if any(v == 0 for v in proj.values()): return {k: copy.deepcopy(v) for k, v in doc.items() if proj.get(k, 1) != 0}
    return {k: copy.deepcopy(v) for k, v in doc.items() if k == "_id" or proj.get(k)}


class Collection:
    def __init__(self): self.docs, self.unique = [], []
    def create_index(self, keys, unique=False, **kw):
        if unique: self.unique.append(keys)
    def _check(self, new):
        for f in self.unique:
            if new.get(f) is not None and any(d.get(f) == new[f] for d in self.docs):
                raise RuntimeError(f"duplicate key on {f}")
        if any(d["_id"] == new["_id"] for d in self.docs): raise RuntimeError("duplicate _id")
    def insert_one(self, d): self._check(d); self.docs.append(copy.deepcopy(d))
    def insert_many(self, ds):
        for d in ds: self.insert_one(d)
    def find(self, flt=None, proj=None): return Cursor([_project(d, proj) for d in self.docs if _match(d, flt or {})])
    def find_one(self, flt=None, proj=None, sort=None):
        c = self.find(flt, proj)
        if sort: c.sort(sort)
        return next(iter(c.limit(1)), None)
    def count_documents(self, flt, limit=0):
        n = sum(1 for d in self.docs if _match(d, flt)); return min(n, limit) if limit else n
    def delete_many(self, flt): self.docs = [d for d in self.docs if not _match(d, flt)]
    def update_many(self, flt, upd):
        for d in self.docs:
            if _match(d, flt): d.update(upd["$set"])
    def update_one(self, flt, upd): 
        for d in self.docs:
            if _match(d, flt): d.update(upd["$set"]); return
    def replace_one(self, flt, doc, upsert=False):
        for i, d in enumerate(self.docs):
            if _match(d, flt): self.docs[i] = copy.deepcopy(doc); return
        if upsert: self.docs.append(copy.deepcopy(doc))
    def find_one_and_update(self, flt, upd, upsert=False, return_document=None):
        for d in self.docs:
            if _match(d, flt):
                for k, v in upd["$inc"].items(): d[k] = d.get(k, 0) + v
                return copy.deepcopy(d)
        d = {**flt, **upd["$inc"]}; self.docs.append(d); return copy.deepcopy(d)


class Database:
    def __init__(self): self._c = {}
    def __getattr__(self, name): return self._c.setdefault(name, Collection())


class FakeClient:
    def __init__(self):
        self.db = Database()
        self.admin = type("A", (), {"command": staticmethod(lambda *_: {"ok": 1})})()
    def get_default_database(self, default=None): return self.db
