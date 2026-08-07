"""
Tests for the offline-quality gauge in webservice/predict.py + metrics.py.

Why this metric exists: the only F1 Grafana had came from the Prefect backtest, and that
one replays rows the model was trained on, so it reads far too high (~0.99 accuracy for
fk_model against an honest 0.81). `model_offline_f1` publishes the cross-validated score
of the version actually being served, next to it.

Two properties carry the honesty of the thing and are asserted directly:

- **The estimator label must be accurate.** A single-fold holdout score and a 5-fold
  grouped CV mean are not interchangeable - the fold-to-fold spread was measured at up to
  0.21 F1 - so a model still on the old metric must not be published as if it were
  cross-validated.
- **It must never break startup.** This is dashboard provenance, not serving. An
  unreachable registry, an unregistered model or a run with no F1 metric must all yield
  "no entry", not an exception.
"""

import pytest
from prometheus_client import REGISTRY


@pytest.fixture
def predict_module(monkeypatch):
    """webservice.predict with MLflow stubbed out at the client boundary."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.invalid:5000")
    import predict

    return predict


class FakeVersion:
    def __init__(self, version: str, run_id: str) -> None:
        self.version = version
        self.run_id = run_id


class FakeRun:
    def __init__(self, metrics: dict) -> None:
        self.data = type("Data", (), {"metrics": metrics})()


class FakeClient:
    """MlflowClient stand-in driven by a {model_name: metrics} mapping."""

    def __init__(self, per_model: dict, missing: tuple = ()) -> None:
        self.per_model = per_model
        self.missing = missing

    def get_model_version_by_alias(self, name, _alias):
        if name in self.missing or name not in self.per_model:
            from mlflow.exceptions import MlflowException

            message = f"alias not found for {name}"
            raise MlflowException(message)
        return FakeVersion(version="7", run_id=f"run-{name}")

    def get_run(self, run_id):
        name = run_id.removeprefix("run-")
        return FakeRun(self.per_model[name])


def install(monkeypatch, predict_module, per_model, *, reachable=True, missing=()):
    monkeypatch.setattr(predict_module, "_registry_reachable", lambda _uri: reachable)
    monkeypatch.setattr(
        predict_module,
        "MlflowClient",
        lambda *_a, **_k: FakeClient(per_model, missing),
    )


class TestEstimatorLabelling:
    def test_cross_validated_models_are_labelled_as_such(self, monkeypatch, predict_module):
        install(
            monkeypatch,
            predict_module,
            {"fk_model": {"cv_f1_mean": 0.8113, "cv_f1_std": 0.0556}},
        )
        quality = predict_module.read_offline_quality()

        assert quality["fk_model"]["estimator"] == "cv_mean_5fold_grouped"
        assert quality["fk_model"]["f1"] == pytest.approx(0.8113)
        assert quality["fk_model"]["f1_std"] == pytest.approx(0.0556)

    def test_single_fold_models_are_not_passed_off_as_cross_validated(
        self,
        monkeypatch,
        predict_module,
    ):
        """pk/cfk/normalform still log test_f1_score; the label has to say so."""
        install(monkeypatch, predict_module, {"pk_model": {"test_f1_score": 0.95}})
        quality = predict_module.read_offline_quality()

        assert quality["pk_model"]["estimator"] == "single_fold_holdout"
        assert quality["pk_model"]["f1_std"] is None, "a single holdout has no spread"

    def test_the_cross_validated_metric_wins_when_a_run_has_both(
        self,
        monkeypatch,
        predict_module,
    ):
        """A rerun can leave both metrics on one run; the better estimator must win."""
        install(
            monkeypatch,
            predict_module,
            {"fk_model": {"test_f1_score": 0.70, "cv_f1_mean": 0.81, "cv_f1_std": 0.05}},
        )
        quality = predict_module.read_offline_quality()

        assert quality["fk_model"]["estimator"] == "cv_mean_5fold_grouped"
        assert quality["fk_model"]["f1"] == pytest.approx(0.81)


class TestNeverBreaksStartup:
    def test_an_unreachable_registry_yields_nothing(self, monkeypatch, predict_module):
        install(monkeypatch, predict_module, {"fk_model": {"cv_f1_mean": 0.8}}, reachable=False)

        assert predict_module.read_offline_quality() == {}

    def test_an_unregistered_model_is_skipped_not_raised(self, monkeypatch, predict_module):
        install(
            monkeypatch,
            predict_module,
            {"fk_model": {"cv_f1_mean": 0.81}},
            missing=("pk_model",),
        )
        quality = predict_module.read_offline_quality()

        assert "pk_model" not in quality
        assert "fk_model" in quality

    def test_a_run_without_any_f1_metric_is_skipped(self, monkeypatch, predict_module):
        """An old or interrupted run may carry no F1 at all."""
        install(monkeypatch, predict_module, {"fk_model": {"oof_precision": 0.8}})

        assert predict_module.read_offline_quality() == {}

    def test_publish_survives_a_registry_that_raises(self, monkeypatch):
        """The startup hook must swallow failures - it is provenance, not serving."""
        import app

        def boom():
            message = "registry exploded"
            raise RuntimeError(message)

        monkeypatch.setattr(app, "read_offline_quality", boom)
        app.publish_offline_quality()  # must not raise


class TestGaugeExport:
    def test_values_and_labels_reach_the_prometheus_registry(self):
        from metrics import record_offline_quality

        record_offline_quality(
            model="test_gauge_model",
            f1=0.8113,
            estimator="cv_mean_5fold_grouped",
            version="7",
            f1_std=0.0556,
        )

        assert REGISTRY.get_sample_value(
            "model_offline_f1",
            {"model": "test_gauge_model", "estimator": "cv_mean_5fold_grouped"},
        ) == pytest.approx(0.8113)
        assert REGISTRY.get_sample_value(
            "model_offline_f1_std",
            {"model": "test_gauge_model"},
        ) == pytest.approx(0.0556)
        assert REGISTRY.get_sample_value(
            "model_served_version",
            {"model": "test_gauge_model"},
        ) == pytest.approx(7.0)

    def test_a_non_numeric_version_does_not_raise(self):
        """Registry versions are numeric strings today; do not die if that ever changes."""
        from metrics import record_offline_quality

        record_offline_quality(
            model="odd_version_model",
            f1=0.5,
            estimator="single_fold_holdout",
            version="not-a-number",
        )

        assert REGISTRY.get_sample_value(
            "model_offline_f1",
            {"model": "odd_version_model", "estimator": "single_fold_holdout"},
        ) == pytest.approx(0.5)
