import mlflow
import pickle
from sklearn.metrics import root_mean_squared_error

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("linearregression-yellow-taxi")

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    lm, lm_params, dv, dv_params, X, y = data

    rmse = root_mean_squared_error(y, lm.predict(X))

    with mlflow.start_run():
        mlflow.sklearn.log_model(lm, 'model')
        mlflow.log_params(lm.get_params(deep=True))
        mlflow.log_param("intercept", lm.intercept_)
        mlflow.log_metric("rmse", rmse)
        dv_path = "dict_vectorizer.pkl"
        with open(dv_path,"wb") as pfile:
            pickle.dump(dv, pfile)
        
        mlflow.log_artifact(dv_path, "dict_vectorizer")

