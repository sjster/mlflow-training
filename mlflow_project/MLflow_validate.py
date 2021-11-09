import mlflow
import sklearn
from xgboost import XGBClassifier
import pandas as pd
import sys

if __name__ == "__main__":

    mlflow.set_tracking_uri("http://localhost:5000")
    try:
        mlflow.create_experiment('XGBoost_mlflow_validate')
        mlflow.set_experiment('XGBoost_mlflow_validate')
    except:
        mlflow.set_experiment('XGBoost_mlflow_validate')

    X_val = pd.read_csv("/Users/srijith.rajamohan/Documents/Development/Mlflow/mlflow_training/data/X_val.csv")
    y_val = pd.read_csv("/Users/srijith.rajamohan/Documents/Development/Mlflow/mlflow_training/data/y_val.csv")

    print(X_val.columns)
    print(len(X_val.columns))
    print(len(y_val.columns))

    with mlflow.start_run(run_name="xgboost_validate") as mlflow_run:
        my_model_reload = mlflow.sklearn.load_model('my_local_model')
        xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(my_model_reload, X_val, y_val,
                                                                    prefix="val_")
        print(pd.DataFrame(xgbc_val_metrics, index=[0]))
