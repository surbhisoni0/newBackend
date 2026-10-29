"""Lets the Mongo repository tests run on a PC where pymongo is not installed (the repo only needs ReturnDocument)."""
import sys, types
try:
    import pymongo  # noqa: F401
except ImportError:
    m = types.ModuleType("pymongo"); m.ReturnDocument = types.SimpleNamespace(AFTER=True); m.MongoClient = None
    sys.modules["pymongo"] = m
