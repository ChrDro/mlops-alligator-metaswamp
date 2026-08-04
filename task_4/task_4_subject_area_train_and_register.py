"""
Discover subject areas in `raw_metadata.csv`, then train, log and register a
servable classifier that assigns a table to one of them.

Two stages, on purpose:

1. **Discovery (unsupervised, offline).** Sentence-transformer embeddings -> UMAP ->
   HDBSCAN -> c-TF-IDF keywords -> one subject-area name per cluster from the local
   Ollama model. This produces the label set; no labels exist beforehand.

2. **Distillation (supervised, servable).** The cluster assignments become training
   labels for a TF-IDF + linear text classifier over `table_name` and `columns`.
   That model is what gets registered and served.

Why distil instead of registering the BERTopic model itself: serving BERTopic means
shipping sentence-transformers, torch, UMAP and HDBSCAN inside the prediction image
(several GB, slow cold start) just to route a table name to a cluster. The distilled
pipeline needs only scikit-learn, which the image already has, and it exposes
`predict_proba` - so it behaves exactly like the other five models, including the
confidence value the API returns. The cost is fidelity to the clustering, which is
measured here: `test_accuracy` is agreement with the cluster label on held-out
tables, and it is logged per candidate rather than assumed.

Candidates (all with balanced class weights - cluster sizes are very uneven):
- word+char TF-IDF -> LogisticRegression
- word+char TF-IDF -> LinearSVC wrapped for probabilities
- word TF-IDF      -> ComplementNB (a strong, cheap text baseline)

Prerequisites: the `ollama` service healthy (`docker compose up -d ollama`) and an
MLflow server reachable at MLFLOW_TRACKING_URI.
"""

import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from bertopic import BERTopic
from dotenv import dotenv_values
from hdbscan import HDBSCAN
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# task_4/ is not a package; the labeling helper sits next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subject_area_labeling import (
    KEYWORDS_PER_TOPIC,
    LABEL_CACHE_PATH,
    LLM_BASE_URL,
    LLM_MODEL,
    cache_key_for,
    generate_label,
    load_label_cache,
    save_label_cache,
)


RSEED = 42

_env = dotenv_values()

DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

_minio_user = _env.get("MINIO_ROOT_USER")
_minio_password = _env.get("MINIO_ROOT_PASSWORD")
if _minio_user and _minio_password:
    os.environ["AWS_ACCESS_KEY_ID"] = _minio_user
    os.environ["AWS_SECRET_ACCESS_KEY"] = _minio_password
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-central-1")

MODEL_ARTIFACT_NAME = "subject_area_model"
MODEL_NAME = "subject_area_model"
MODEL_ALIAS = "dev"

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# The two request fields, in the order the Pydantic schema declares them. The
# contract test in test/test_models compares this order against the schema.
FEATURE_COLUMNS = ["table_name", "columns"]

# A cluster this small is not a subject area worth serving, and it cannot be split
# across a train/test holdout either.
MIN_TABLES_PER_SUBJECT_AREA = 4


def wait_for_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int,
) -> ModelVersion:
    """Wait until the registered model version is ready to serve."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(model_name, version)
        status = model_version.status
        if status == "READY":
            return model_version
        if status == "FAILED_REGISTRATION":
            msg_error_model_version = f"Model version {model_name} v{version} failed registration."
            raise RuntimeError(msg_error_model_version)
        time.sleep(1)

    msg_waiting_model = f"Timed out waiting for {model_name} v{version} to become READY."
    raise RuntimeError(msg_waiting_model)


# --- stage 1: discovery -------------------------------------------------------


def load_data() -> tuple[Path, pd.DataFrame]:
    print("\n------Data Loading------")
    input_path = Path.cwd().resolve() / "data/raw_metadata.csv"
    df = pd.read_csv(input_path)
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    return input_path, df


def build_table_descriptions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per table: its name, its column names, and a text description.

    The table name carries most of the signal; the column names disambiguate tables
    whose name is generic (`record`, `status`, `part`).
    """
    print("\n------One row per table------")
    tables = (
        df_raw.groupby(["table_name"], sort=False)["column_name"]
        .apply(lambda cols: ", ".join(cols.tolist()))
        .reset_index()
        .rename(columns={"column_name": "columns"})
    )
    tables["description"] = tables["table_name"].str.replace("_", " ") + ": " + tables["columns"]
    print(f"Unique tables: {len(tables)}")
    print(tables[["table_name", "description"]].head(5))
    return tables


def embed_descriptions(tables: pd.DataFrame) -> tuple[np.ndarray, SentenceTransformer]:
    print(f"\n------Embedding with {EMBED_MODEL_NAME}------")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embed_model.encode(
        tables["description"].tolist(),
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    print(f"Embedding matrix: {embeddings.shape}")
    return embeddings, embed_model


def cluster_tables(tables: pd.DataFrame, embeddings: np.ndarray) -> tuple[BERTopic, pd.DataFrame]:
    """Cluster the table embeddings and attach the topic id to each table."""
    print("\n------Clustering (UMAP + HDBSCAN + c-TF-IDF)------")

    # UMAP is imported lazily: it pulls in numba, which is slow to import and only
    # needed here, not on the serving path.
    from umap import UMAP  # noqa: PLC0415 - deliberate, see above

    umap_model = UMAP(
        n_neighbors=6,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RSEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TABLES_PER_SUBJECT_AREA,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        # nr_topics=None, i.e. no post-hoc topic reduction. "auto" merged HDBSCAN's
        # 41 clusters down to 16 and produced a single 271-table bucket - 70% of the
        # corpus under one meaningless name, with the silhouette going from +0.70 to
        # -0.17. Measured on this dataset; do not re-enable without re-measuring.
        nr_topics=None,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, _probs = topic_model.fit_transform(tables["description"].tolist(), embeddings)
    tables["topic_id"] = topics

    n_topics = len({t for t in topics if t >= 0})
    n_outliers = int(sum(1 for t in topics if t < 0))
    print(f"Topics found: {n_topics} | outliers: {n_outliers}/{len(tables)}")
    return topic_model, tables


def cluster_quality(topic_model: BERTopic, tables: pd.DataFrame) -> dict[str, float]:
    """Silhouette and coverage of the clustering, measured in UMAP space.

    UMAP space is what HDBSCAN actually clustered; scoring the 384-D embeddings
    instead deflates the silhouette because UMAP distorts global distances.
    """
    assigned = tables["topic_id"] >= 0
    coverage = float(assigned.sum() / len(tables))

    umap_coords = topic_model.umap_model.embedding_
    n_clusters = tables.loc[assigned, "topic_id"].nunique()
    if n_clusters < 2:
        # silhouette_score is undefined for a single cluster.
        return {"cluster_silhouette": float("nan"), "cluster_coverage": coverage}

    silhouette = float(
        silhouette_score(umap_coords[assigned.to_numpy()], tables.loc[assigned, "topic_id"]),
    )
    print(f"Cluster silhouette (UMAP space): {silhouette:+.3f} | coverage: {coverage:.1%}")
    return {"cluster_silhouette": silhouette, "cluster_coverage": coverage}


def name_subject_areas(topic_model: BERTopic, tables: pd.DataFrame) -> pd.DataFrame:
    """Give every cluster a subject-area name from the local LLM."""
    print(f"\n------Naming subject areas ({LLM_MODEL} at {LLM_BASE_URL})------")
    cache = load_label_cache()
    hits = 0

    labels: dict[int, str] = {}
    for topic_id in sorted(topic_model.get_topics()):
        if topic_id < 0:
            continue
        keywords = [word for word, _ in topic_model.get_topic(topic_id)][:KEYWORDS_PER_TOPIC]
        rep_docs = topic_model.get_representative_docs(topic_id)

        hits += int(cache_key_for(keywords) in cache)
        labels[topic_id] = generate_label(keywords, rep_docs, cache) or f"topic_{topic_id}"
        print(f"  Topic {topic_id:2d}: {labels[topic_id]:<32s} <- {rep_docs[0][:60]}")

    save_label_cache(cache)
    print(f"{len(labels)} subject areas named, {hits} from cache -> {LABEL_CACHE_PATH}")

    tables["subject_area"] = tables["topic_id"].map(labels).fillna("outlier")
    return tables


def build_supervised_dataset(tables: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the clustering into (X, y) for the distilled classifier.

    Outliers are dropped rather than kept as a class: HDBSCAN's -1 bucket is
    everything that fit nowhere, so it has no shared vocabulary to learn. A table the
    served model cannot place shows up as a low `probability` instead.

    Two different LLM runs can also land on the same name for two clusters; those are
    merged here, because the served label is the name, not the topic id.
    """
    print("\n------Supervised dataset------")
    labelled = tables[tables["topic_id"] >= 0].copy()

    counts = labelled["subject_area"].value_counts()
    too_small = counts[counts < MIN_TABLES_PER_SUBJECT_AREA]
    if not too_small.empty:
        print(
            f"Dropping {len(too_small)} subject area(s) with "
            f"< {MIN_TABLES_PER_SUBJECT_AREA} tables:",
        )
        for area, n in too_small.items():
            print(f"  {area} ({n})")
        labelled = labelled[~labelled["subject_area"].isin(too_small.index)]

    X = labelled[FEATURE_COLUMNS].reset_index(drop=True)
    y = labelled["subject_area"].reset_index(drop=True)

    print(f"Tables: {len(X)} | subject areas: {y.nunique()}")
    print(y.value_counts().to_string())
    return X, y


# --- stage 2: distillation ----------------------------------------------------


def make_text_features(*, use_char_ngrams: bool) -> ColumnTransformer:
    """Vectorise the two text fields separately.

    `table_name` and `columns` are different signals - one short and decisive, one
    long and noisy - so each gets its own vocabulary rather than being concatenated
    into a single bag of words.

    Character n-grams matter here because the corpus is bilingual and full of
    fragments: they connect `bestellung` to `bestellungen`, and `lieferadresse` to
    `lieferdienst`, which word-level tokens treat as unrelated.
    """
    name_word = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-zA-Z]+",  # noqa: S106 - a regex, not a credential
        lowercase=True,
        sublinear_tf=True,
    )
    cols_word = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-zA-Z]+",  # noqa: S106 - a regex, not a credential
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
    )

    transformers = [
        ("name_word", name_word, "table_name"),
        ("cols_word", cols_word, "columns"),
    ]
    if use_char_ngrams:
        transformers.append(
            (
                "name_char",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
                "table_name",
            ),
        )
    return ColumnTransformer(transformers)


def build_candidates() -> list[dict]:
    """The candidate pipelines, each ending in an estimator with predict_proba."""
    return [
        {
            "name": "tfidf_word_char_logreg",
            "model_type": "logistic_regression",
            "pipeline": Pipeline(
                [
                    ("features", make_text_features(use_char_ngrams=True)),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RSEED,
                        ),
                    ),
                ],
            ),
        },
        {
            "name": "tfidf_word_char_linearsvc",
            "model_type": "linear_svc_calibrated",
            "pipeline": Pipeline(
                [
                    ("features", make_text_features(use_char_ngrams=True)),
                    (
                        "classifier",
                        # LinearSVC has no predict_proba, and the API returns a
                        # confidence for every prediction. cv="prefit" is not an
                        # option (nothing is fitted yet) and k-fold calibration needs
                        # k members of the smallest class, which a 3-table training
                        # split does not have - so the sigmoid is fitted on the
                        # training data itself. That makes the probability usable for
                        # ranking, not a calibrated frequency.
                        CalibratedClassifierCV(
                            LinearSVC(class_weight="balanced", random_state=RSEED),
                            method="sigmoid",
                            cv=2,
                        ),
                    ),
                ],
            ),
        },
        {
            "name": "tfidf_word_complement_nb",
            "model_type": "complement_naive_bayes",
            "pipeline": Pipeline(
                [
                    ("features", make_text_features(use_char_ngrams=False)),
                    ("classifier", ComplementNB()),
                ],
            ),
        },
    ]


def split_train_test(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Stratified holdout, so every subject area appears in both halves.

    With clusters as small as MIN_TABLES_PER_SUBJECT_AREA this leaves one test table
    for the rarest areas. That is a genuinely thin holdout and the reason the metrics
    below are logged per candidate rather than quoted as accuracy in isolation.
    """
    print("\n------Train Test Split------")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RSEED,
        stratify=y,
    )
    print(f"Train: {len(X_train)} tables | Test: {len(X_test)} tables")
    return X_train, X_test, y_train, y_test


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Weighted multiclass metrics.

    zero_division=0 because a thin holdout can leave a class with no predictions,
    which would otherwise warn and yield nan instead of the 0 it means.
    """
    return {
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def train_all_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    print("\n------Training candidate models------")
    candidates = []
    for candidate in build_candidates():
        print(f"  fitting {candidate['name']} ...")
        pipeline = candidate["pipeline"]
        pipeline.fit(X_train, y_train)
        candidate["train_metrics"] = compute_metrics(y_train, pipeline.predict(X_train))
        candidate["test_metrics"] = compute_metrics(y_test, pipeline.predict(X_test))
        candidates.append(candidate)
    return candidates


def print_candidate_summary(candidates: list[dict]) -> None:
    print("\n------Evaluation Table Test Set------\n")
    print(pd.DataFrame({c["name"]: c["test_metrics"] for c in candidates}).T)


def setup_experiment(client: MlflowClient, experiment_name: str) -> str:
    """Create the experiment if needed and return its id."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_candidate_run(
    candidate: dict,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    input_path: Path,
    discovery_metrics: dict[str, float],
) -> str:
    """Log one candidate as its own MLflow run and return the run id.

    Logged with the sklearn flavor so the serving side can reach the fitted estimator
    and call predict_proba on it, exactly as for the other five models.
    """
    pipeline = candidate["pipeline"]
    input_example = X_train.head(5)
    signature = infer_signature(input_example, pipeline.predict(input_example))

    with mlflow.start_run(run_name=candidate["name"]) as run:
        mlflow.log_params(
            {
                key: value
                for key, value in pipeline.get_params(deep=False).items()
                if key != "steps"
            },
        )
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("holdout_rows", len(X_test))
        mlflow.log_param("input_path", str(input_path))
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("alias", MODEL_ALIAS)
        mlflow.log_param("candidate", candidate["name"])
        mlflow.log_param("embedding_model", EMBED_MODEL_NAME)
        mlflow.log_param("labeling_model", LLM_MODEL)

        for metric_name, value in candidate["train_metrics"].items():
            mlflow.log_metric(f"train_{metric_name}", value)
        for metric_name, value in candidate["test_metrics"].items():
            mlflow.log_metric(f"test_{metric_name}", value)
        # Quality of the label source, identical for every candidate: a high
        # test_accuracy against an incoherent clustering means nothing.
        for metric_name, value in discovery_metrics.items():
            if not np.isnan(value):
                mlflow.log_metric(metric_name, value)

        mlflow.set_tags(
            {
                "model_type": candidate["model_type"],
                "candidate": candidate["name"],
                "developer": "test",
                "dataset": "raw-metadata-table-names",
                "target_column": "subject_area",
                "n_features": len(FEATURE_COLUMNS),
                "n_samples": len(X),
                "n_classes": y.nunique(),
                "labels_from": "bertopic-clusters-named-by-local-llm",
            },
        )

        mlflow.log_dict({"features": FEATURE_COLUMNS}, "features.json")
        mlflow.log_dict({"subject_areas": sorted(y.unique().tolist())}, "subject_areas.json")

        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_NAME,
            serialization_format="pickle",
            signature=signature,
            input_example=input_example,
        )

    test_f1 = candidate["test_metrics"]["f1_score"]
    name = candidate["name"]
    print(f"Logged run {run.info.run_id} for '{name}' (test weighted-F1={test_f1:.4f})")
    return run.info.run_id


def register_best_candidate(
    candidates: list[dict],
    run_ids: dict[str, str],
    model_name: str,
    alias: str,
) -> ModelVersion:
    """Register the highest test weighted-F1 candidate and point the alias at it."""
    print("\n------MLflow Model Registration------")

    best = max(candidates, key=lambda c: c["test_metrics"]["f1_score"])
    best_run_id = run_ids[best["name"]]
    best_f1 = best["test_metrics"]["f1_score"]
    model_uri = f"runs:/{best_run_id}/{MODEL_ARTIFACT_NAME}"

    print(f"Best candidate: '{best['name']}' with test weighted-F1={best_f1:.4f}")
    print(f"Registering {model_uri} as {model_name}")

    client = MlflowClient()
    registration = mlflow.register_model(model_uri=model_uri, name=model_name)
    model_version = wait_for_model_version(
        client=client,
        model_name=model_name,
        version=registration.version,
        timeout_seconds=60,
    )
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version.version,
    )

    print(
        f"Registered {model_name} version {model_version.version} and assigned alias '{alias}'.",
    )
    print(
        "The FastAPI service can now resolve models:/"
        f"{model_name}@{alias} from your local MLflow server.",
    )
    return model_version


def export_subject_areas(tables: pd.DataFrame) -> None:
    """Write the per-table cluster assignment next to the other task outputs."""
    out_path = Path.cwd().resolve() / "data/subject_area_labels.csv"
    tables[["table_name", "subject_area", "topic_id"]].to_csv(out_path, index=False)
    print(f"\nSaved {len(tables)} rows -> {out_path}")


def main() -> None:
    # Stage 1 - discover and name the subject areas.
    input_path, df_raw = load_data()
    tables = build_table_descriptions(df_raw)
    embeddings, _embed_model = embed_descriptions(tables)
    topic_model, tables = cluster_tables(tables, embeddings)
    discovery_metrics = cluster_quality(topic_model, tables)
    tables = name_subject_areas(topic_model, tables)
    export_subject_areas(tables)

    # Stage 2 - distil the clustering into a servable classifier.
    X, y = build_supervised_dataset(tables)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    candidates = train_all_candidates(X_train, y_train, X_test, y_test)
    print_candidate_summary(candidates)

    client = MlflowClient()
    setup_experiment(client, "subject_area_model_training")

    run_ids: dict[str, str] = {}
    for candidate in candidates:
        run_ids[candidate["name"]] = log_candidate_run(
            candidate=candidate,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            input_path=input_path,
            discovery_metrics=discovery_metrics,
        )

    register_best_candidate(
        candidates=candidates,
        run_ids=run_ids,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )


if __name__ == "__main__":
    main()
