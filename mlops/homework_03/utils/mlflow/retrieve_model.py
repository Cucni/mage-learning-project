import os
import pickle
from pathlib import Path

from mlflow.runs import ViewType

import mlflow
from mlflow import MlflowClient


def load_best_model(experiment_name):
    mlflow.set_tracking_uri("http://mlflow:5000")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=["metrics.rmse DESC"],
        max_results=1,
        run_view_type=ViewType.ACTIVE_ONLY,
    )[0]

    print(f"runs:/{best_run.info.run_id}/model")
    model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")

    dst_artifact_path = "./tmp_best_vectorizer"
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{best_run.info.run_id}/dict_vectorizer/dict_vectorizer.pkl",
        dst_path=dst_artifact_path,
    )
    with open(Path(dst_artifact_path) / "dict_vectorizer.pkl", "rb") as pfile:
        dict_vectorizer = pickle.load(pfile)
    os.remove(Path(dst_artifact_path) / "dict_vectorizer.pkl")
    os.rmdir(Path(dst_artifact_path))

    return model, dict_vectorizer
