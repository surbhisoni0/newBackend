"""Re-evaluate all models on the stored history:  python -m app.scripts.run_backtest   (run from backend/)"""
from app.config import get_settings
from app.database.factory import create_repository
from app.prediction.engine import PredictionEngine
from app.services.predictor_service import PredictorService


def main():
    s = get_settings(); repo = create_repository(s.database_url); repo.init()
    svc = PredictorService(repo, PredictionEngine(refit_every=s.refit_every), s.min_history, s.min_validation_n)
    summ = svc.run_backtest()
    print(f"walk-forward on {summ['n_predictions']} predictions (data n={summ['data_n']}, K={summ['K']} tests)")
    print(f"{'model':11s} {'top1':>7s} {'top3':>7s} {'B/S':>7s} {'color':>7s}   baselines 10.0 / 30.0 / 50.0 / 40.0")
    for k, m in summ["models"].items():
        print(f"{k:11s} " + " ".join(f"{100 * m[x]['rate']:6.1f}%" for x in ("top1", "top3", "bs", "color")))
    print("signal:", svc.current_signal()["label"])


if __name__ == "__main__":
    main()
