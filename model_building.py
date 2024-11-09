import xgboost as xgb
from data_cleaning import main
from sklearn.metrics import classification_report
import pandas as pd
import dill

def load_data():
    with open("transformed_data.pkl", "rb") as f:
        X_train, X_test, y_train, y_test = dill.load(f)

    return X_train, y_train, X_test, y_test


def build_model(X_train, y_train, X_test, y_test):
    params = {
        "objective": "binary:logistic",
        "n_estimators": 500,
        'learning_rate': 0.0010812936756470217,
        'max_depth': 6,
        'subsample': 0.36482338465400405,
        'colsample_bytree': 0.17190210997311706,
        'min_child_weight': 15
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)
    return model


def main():
    X_train, y_train, X_test, y_test = load_data() # reading data
    model = build_model(X_train, y_train, X_test, y_test) # building the model

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)


if __name__=="__main__":
    main()