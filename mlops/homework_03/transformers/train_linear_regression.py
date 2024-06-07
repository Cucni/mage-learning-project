import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    dv = DictVectorizer()
    categorical = ["PULocationID", "DOLocationID"]

    df_dicts = data[categorical].to_dict(orient="records")

    X = dv.fit_transform(df_dicts)
    y = data["duration"].values
    lm = LinearRegression()
    lm.fit(X, y)

    lm_params = {'First 10 Coefficients': lm.coef_.flatten()[:10].tolist(), 'Intercept': lm.intercept_}

    return lm, lm_params, dv, dv.get_params(), X, y.tolist()


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'