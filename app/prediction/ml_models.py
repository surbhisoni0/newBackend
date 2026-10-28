"""scikit-learn models exactly as validated in the Step-8 walk-forward study (strong regularisation)."""
import warnings
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def make_models() -> dict:
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=400)),
        "RandForest": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, n_jobs=-1, random_state=0),
        "HistGB": HistGradientBoostingClassifier(max_iter=40, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                                 l2_regularization=1.0, random_state=0),
    }
