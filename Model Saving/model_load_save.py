import dill
import pandas as pd

def save_model(model):
    with open("model.pkl", "wb") as f:
        dill.dump(model, f)


def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        model = dill.load(f)

    return model