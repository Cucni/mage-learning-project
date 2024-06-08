if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

# Payload format
# {
#   "pipeline_run": {
#     "variables": {
#       "key1": "value1",
#       "key2": "value2"
#     }
#   }
# }

FEATURES = ["PULocationID", "DOLocationID"]


@custom
def transform_custom(data, *args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    model, dict_vectorizer = data
    inputs_dict = {feat: str(kwargs.get(feat)) for feat in FEATURES}
    X = dict_vectorizer.transform(inputs_dict)

    preds = model.predict(X)

    print(f"Predicted trip duration: {preds}")

    return X, preds.tolist()


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, "The output is undefined"
