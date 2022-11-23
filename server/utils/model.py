import joblib
from surprise import SVD

def load_model(path: str) -> SVD:
    with open(path, "rb") as fin:
        return joblib.load(fin)