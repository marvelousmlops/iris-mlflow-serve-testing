# Databricks notebook source
# MAGIC %pip install -e .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from typing import Union

import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

iris = datasets.load_iris(as_frame=True)
X = iris.data
y = iris.target
logger.info("The dataset is loaded.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)
preprocessor = ColumnTransformer(transformers=[("std_scaler", StandardScaler(), iris.feature_names)])

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)
logger.info("🚀 Starting training...")
pipeline.fit(X_train, y_train)

# COMMAND ----------


class ModelWrapper(mlflow.pyfunc.PythonModel):
    """A wrapper class for machine learning models to be used with MLflow.

    This class encapsulates a model and provides a standardized predict method.
    """

    def __init__(self, model: object) -> None:
        self.model = model
        self.class_names = iris.target_names

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Union[pd.DataFrame, np.array],  # noqa
    ) -> Union[pd.DataFrame, np.ndarray]:  # noqa
        """Perform predictions using the wrapped model.

        :param context: The MLflow PythonModelContext, which provides runtime information.
        :param model_input: The input data for prediction, either as a pandas DataFrame or a NumPy array.
        :return: redictions mapped to class names in original input format.
        """
        raw_predictions = self.model.predict(model_input)
        mapped_predictions = [self.class_names[int(pred)] for pred in raw_predictions]
        return mapped_predictions


# COMMAND ----------

mlflow.autolog(disable=True)
mlflow.set_tracking_uri("databricks://dbc-c2e8445d-159d")
mlflow.set_experiment("/Shared/iris-demo")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    y_proba = pipeline.predict_proba(X_test)

    # Evaluation metrics
    auc_test = roc_auc_score(y_test, y_proba, multi_class="ovr")
    logger.info(f"AUC Report: {auc_test}")

    # Log parameters and metrics
    mlflow.log_param("model_type", "LogisticRegression Classifier with preprocessing")
    mlflow.log_metric("auc", auc_test)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=["setosa"])
    dataset = mlflow.data.from_pandas(iris.frame, name="train_set")
    mlflow.log_input(dataset, context="training")

    mlflow.pyfunc.log_model(
        python_model=ModelWrapper(pipeline),
        artifact_path="pyfunc-lg-pipeline-model",
        signature=signature,
    )

# COMMAND ----------
