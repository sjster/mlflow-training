import pyspark
from pyspark.sql import *
import mlflow
import pandas as pd
import sys
import os

home_folder = "/Users/srijith.rajamohan/Documents/Development/Mlflow/mlflow_training2/"

if __name__ == "__main__":

    mlflow.set_tracking_uri("http://localhost:5001")
    #mlflow.set_tracking_uri("sqlite:///myflow.db")
    # Create a new experiment if the experiment does not exist. If it does, make it the current experiment
    try:
        mlflow.create_experiment('XGBoost_mlflow_training')
        mlflow.set_experiment('XGBoost_mlflow_training')
    except:
        mlflow.set_experiment('XGBoost_mlflow_training')

    spark = (SparkSession.builder
        #    .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0")
            .master("local[*]")
            .getOrCreate())

    mlflow.autolog(log_input_examples=True, log_models=True, exclusive=False)
    print("Arguments received", str(sys.argv[1]))
    data_file = str(sys.argv[1]) if len(sys.argv) > 1 else home_folder + "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


    input_df = spark.read.format("csv").option('header', "true").load(data_file)
    train_df, test_df = input_df.randomSplit([0.90, 0.1], seed=42)
    #use later for additional inference testing
    new_df = test_df
    print(train_df)

    # COMMAND ----------

    from pyspark.sql.functions import when, col
    test_df = test_df.withColumn("churn", when(test_df.Churn == 'Yes' ,1).otherwise(0))
    train_df = train_df.withColumn("churn", when(train_df.Churn == 'Yes' ,1).otherwise(0))

    # COMMAND ----------

    import sklearn.metrics
    import numpy as np
    test_pdf = test_df.toPandas()
    y_test = test_pdf["churn"]
    X_test = test_pdf.drop("churn", axis=1)
    X_test.to_csv(home_folder + "data/X_val.csv", index=False)
    y_test.to_csv(home_folder + "data/y_val.csv", index=False)

    # COMMAND ----------

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    transformers = []
    numerical_pipeline = Pipeline(steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    transformers.append(("numerical", numerical_pipeline, ["MonthlyCharges", "TotalCharges"]))

    # COMMAND ----------

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    one_hot_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers.append(("onehot", one_hot_pipeline, ["Contract", "Dependents", "DeviceProtection", "InternetService", "MultipleLines", "OnlineBackup", "OnlineSecurity", "PaperlessBilling", "Partner", "PaymentMethod", "PhoneService", "SeniorCitizen", "StreamingMovies", "StreamingTV", "TechSupport", "gender", "tenure"]))

    # COMMAND ----------

    from sklearn.feature_extraction import FeatureHasher
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    for feature in ["customerID"]:
        hash_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
            (f"{feature}_hasher", FeatureHasher(n_features=1024, input_type="string"))])
        transformers.append((f"{feature}_hasher", hash_transformer, [feature]))

    # COMMAND ----------

    from sklearn.compose import ColumnTransformer

    preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

    # COMMAND ----------

    from sklearn.preprocessing import StandardScaler

    standardizer = StandardScaler()

    # COMMAND ----------

    from sklearn.model_selection import train_test_split

    df_loaded = train_df.toPandas()
    target_col = "churn"
    split_X = df_loaded.drop([target_col], axis=1)
    split_y = df_loaded[target_col]

    X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, random_state=398741429, stratify=split_y)

    # COMMAND ----------

    import mlflow
    import sklearn
    from xgboost import XGBClassifier
    from sklearn import set_config
    from sklearn.pipeline import Pipeline

    set_config(display="diagram")

    xgbc_classifier = XGBClassifier(
      learning_rate=0.05064395325585373,
      max_depth=3,
      min_child_weight=8,
      subsample=0.9602619596041891,
      random_state=398741429,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("standardizer", standardizer),
        ("classifier", xgbc_classifier),
    ])


    # COMMAND ----------

    with mlflow.start_run(run_name="xgboost") as mlflow_run:
        model.fit(X_train, y_train)

        # Training metrics are logged by MLflow autologging
        # Log metrics for the validation set
        xgbc_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val,
                                                                    prefix="val_")
        print(pd.DataFrame(xgbc_val_metrics, index=[0]))

    os.system("rm -rf my_local_model")
    mlflow.sklearn.save_model(model, "my_local_model")
    mlflow.sklearn.log_model(model,
    artifact_path="artifacts",
    registered_model_name="xgboost_model"
  )
