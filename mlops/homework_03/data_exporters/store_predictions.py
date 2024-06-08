from pathlib import Path

import pandas as pd
from mage_ai.io.file import FileIO

if "data_exporter" not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data_to_file(data, data_2, **kwargs) -> None:
    """
    Template for exporting data to filesystem.

    Docs: https://docs.mage.ai/design/data-loading#fileio
    """
    X, preds = data
    df = pd.DataFrame(X.toarray()).assign(predictions=preds)

    filepath = Path("mlops/homework_03/data/predictions.csv")
    FileIO().export(df, filepath, mode="a", header=filepath.exists())
