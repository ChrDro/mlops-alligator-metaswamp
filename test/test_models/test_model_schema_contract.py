"""
Contract tests between the registered MLflow models and the API request schemas.

Both failure modes below were real production bugs on 2026-07-27, and neither was
visible until predictions silently came back as NULL for three of four models:

1. **Set drift.** `ForeignKey` was missing four features the model expected and sent
   five it did not know. The fk training script picks features by *dropping* columns,
   so extending the feature set elsewhere left this schema behind.

2. **Order drift.** `CompositePrimaryKey` and `CompositeForeignKey` declared the right
   features in the wrong order. `model.predict()` survives that because pyfunc reorders
   by name, but `predict_proba` on the raw sklearn estimator compares positionally and
   raises "The feature names should match those that were passed during fit".
   `pk_model` happened to be in the right order, which is why it alone kept working.

Order therefore matters as much as membership, and both are asserted separately so a
failure says which kind of drift occurred.
"""

import pytest

from .conftest import MODEL_CONTRACTS


MODEL_NAMES = [contract[0] for contract in MODEL_CONTRACTS]


# Tests if model signature is present in mlflow
def _signature_or_fail(model_signatures: dict, model_name: str) -> list[str]:
    """Return the signature, or fail with the reason it could not be read."""
    signature = model_signatures[model_name]
    if isinstance(signature, Exception):
        pytest.fail(
            f"Could not read the signature of '{model_name}@dev': {signature}\n"
            f"The API serves this model, so a missing or unreadable version means "
            f"every request to its endpoint fails with a 400.",
        )
    return signature


# Tests set of features for model in mflow if every feature needed is present
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_feature_set_matches(model_name, model_signatures, pydantic_fields):
    """The API schema must offer exactly the features the model was trained on."""
    expected = _signature_or_fail(model_signatures, model_name)
    declared = pydantic_fields[model_name]

    missing = sorted(set(expected) - set(declared))
    unexpected = sorted(set(declared) - set(expected))
    drift = {"missing_from_schema": missing, "unknown_to_model": unexpected}

    assert drift == {"missing_from_schema": [], "unknown_to_model": []}, (
        f"Feature set drift for '{model_name}':\n"
        f"  model expects but schema lacks : {missing or 'none'}\n"
        f"  schema sends but model lacks   : {unexpected or 'none'}\n"
        f"Either retrain the model or update the Pydantic schema so both agree."
    )


# Tests order of features for model in mflow
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_feature_order_matches(model_name, model_signatures, pydantic_fields):
    """
    Declaration order must match the training order.

    predict.py aligns columns to the signature before predicting, so a mismatch is no
    longer fatal at runtime - but a schema that drifts out of order is a sign the two
    definitions are maintained independently, which is how the set drift crept in.
    """
    expected = _signature_or_fail(model_signatures, model_name)
    declared = pydantic_fields[model_name]

    if set(expected) != set(declared):
        pytest.skip("Feature sets differ; test_feature_set_matches reports the details.")

    first_difference = next(
        (i for i, (a, b) in enumerate(zip(expected, declared, strict=True)) if a != b),
        None,
    )

    assert first_difference is None, (
        f"Feature order drift for '{model_name}' at position {first_difference}:\n"
        f"  model    : {expected[first_difference]}\n"
        f"  schema   : {declared[first_difference]}\n"
        f"Reorder the Pydantic fields to match the model signature."
    )
