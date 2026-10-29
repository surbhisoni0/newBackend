"""Import a CSV of results:  python -m app.scripts.import_csv path/to/file.csv [--newest-first]"""
import sys
from app.config import get_settings
from app.database.factory import create_repository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m app.scripts.import_csv FILE.csv [--newest-first]")
    s = get_settings(); repo = create_repository(s.database_url); repo.init()
    svc = PredictorService(repo, PredictionEngine(refit_every=s.refit_every), s.min_history, s.min_validation_n)
    out = svc.import_csv(open(sys.argv[1], encoding="utf-8-sig").read(), "--newest-first" in sys.argv)
    print(f"imported {out['imported']} rows; total now {out['total']}")


if __name__ == "__main__":
    main()
