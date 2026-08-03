"""
Explain *why* prediction calls failed, instead of only reporting that they did.

Both prediction flows call the model API once per row and swallow individual
failures, so one unlucky row cannot abort a whole run. The price of that is a run
summary naming the symptom and nothing else - "model calls failing" - which is
what you are left staring at after a run that predicted nothing.

The API does say what went wrong: FastAPI puts it in the response body's `detail`
field, and `requests` keeps that body on the exception it raises. The helpers here
hold on to it and turn a run's collected failures into a report that names the
cause and the command that fixes it.
"""

import re
from collections.abc import Iterable


# Which script registers which model, so the report can name the fix rather than
# leaving the reader to guess. Keys are the registry names used in webservice/app.py.
MODEL_TRAINING_SCRIPTS = {
    "pk_model": "task_1/task_1_pk_train_and_register.py",
    "composite_pk_model": "task_1/task_1_cpk_train_and_register.py",
    "fk_model": "task_2/task_2_fk_train_and_register.py",
    "composite_fk_model": "task_2/task_2_cfk_train_and_register.py",
    "denormalization_model": "task_3/task_3_denormalization_train_and_register.py",
}

# MLflow's wording for a name that is not in the registry at all, as in
# "RESOURCE_DOES_NOT_EXIST: Registered Model with name=pk_model not found".
_MISSING_MODEL = re.compile(r"Registered Model with name=(\S+?) not found")

# The name exists but carries no `dev` alias - a different fix (move the alias)
# from registering the model in the first place.
_MISSING_ALIAS = re.compile(r"alias (?:'|\")?(\w+)(?:'|\")? not found", re.IGNORECASE)

# The service never answered. `requests` raises these without a response body, so
# the detail is the exception class name rather than anything the server said.
_UNREACHABLE = re.compile(
    r"ConnectionError|ConnectTimeout|ReadTimeout|Timeout|NewConnectionError|"
    r"Max retries exceeded|Connection refused",
    re.IGNORECASE,
)


def failure_detail(exc: Exception) -> str:
    """
    The server's own explanation for a failed call, or the transport error when it
    never answered.

    Without this the flows log `400 Client Error: Bad Request for url: ...`, which
    is the one part of the response carrying no information: every distinct cause
    (model not registered, alias missing, feature schema drift) looks identical.
    """
    response = getattr(exc, "response", None)
    if response is None:
        # Connection refused, DNS failure, timeout - nothing came back to read.
        return f"{type(exc).__name__}: {exc}"

    try:
        payload = response.json()
    except ValueError:
        # An HTML error page or a proxy response rather than the API's JSON.
        text = response.text.strip()
        return text[:300] if text else f"HTTP {response.status_code} with an empty body"

    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload

    if isinstance(detail, list):
        # FastAPI request validation (422): one entry per offending field.
        return "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', []))}: {item.get('msg', '')}"
            for item in detail
            if isinstance(item, dict)
        )

    return str(detail)


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _shorten(reason: str, limit: int = 200) -> str:
    """
    Keep a reason readable in a log line. A single rejected request can list every
    missing field, which buries the point when several are printed together.
    """
    collapsed = " ".join(reason.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit].rstrip()}... ({len(collapsed)} chars total)"


def diagnose(reasons: Iterable[str], model_api_url: str) -> str:
    """
    Turn the failure details collected during a run into a cause-and-fix report.

    `reasons` are the strings returned by `failure_detail`. Callers add their own
    headline and consequence around this block, because how many rows were left
    behind and what happens to them next is flow-specific.
    """
    unique = sorted(set(reasons))

    if not unique:
        return (
            f"  Cause: unknown - no call reported an error, yet nothing was predicted.\n"
            f"  Check the model service at {model_api_url} and the queue contents."
        )

    missing_models = sorted({name for reason in unique for name in _MISSING_MODEL.findall(reason)})
    if missing_models:
        known = [name for name in missing_models if name in MODEL_TRAINING_SCRIPTS]
        unknown = [name for name in missing_models if name not in MODEL_TRAINING_SCRIPTS]

        subject = _plural(len(missing_models), "this model is", "these models are")
        cause = (
            f"  Cause: {subject} not registered in MLflow, so the service has "
            f"nothing to predict with:"
        )
        lines = [cause, f"           {', '.join(missing_models)}"]
        if known:
            lines.append("  Fix:   train and register, then retrigger this flow:")
            lines.extend(f"           python {MODEL_TRAINING_SCRIPTS[name]}" for name in known)
        if unknown:
            lines.append(
                f"  Fix:   register {', '.join(unknown)} in MLflow - no training script "
                f"in this repo declares that name.",
            )
        return "\n".join(lines)

    missing_aliases = sorted(
        {alias for reason in unique for alias in _MISSING_ALIAS.findall(reason)},
    )
    if missing_aliases:
        return (
            f"  Cause: the models are registered but carry no "
            f"{', '.join(repr(a) for a in missing_aliases)} alias, which is the version "
            f"the service serves.\n"
            f"  Fix:   point the alias at a model version in the MLflow UI, or rerun a "
            f"training script - each one sets the alias after registering."
        )

    if any(_UNREACHABLE.search(reason) for reason in unique):
        return (
            f"  Cause: the model service at {model_api_url} did not answer:\n"
            f"           {_shorten(unique[0])}\n"
            f"  Fix:   check that the model-service container is up "
            f"(docker compose ps model-service) and that MODEL_API_URL is reachable "
            f"from this container."
        )

    # Anything else - schema drift between the Pydantic models and the logged
    # signature, a bug in the predict path - is reported verbatim. Capped because
    # a whole batch can fail the same way and the point is the distinct reasons.
    shown = unique[:3]
    lines = [f"  Cause: the model service rejected the calls ({len(unique)} distinct):"]
    lines.extend(f"           {_shorten(reason)}" for reason in shown)
    if len(unique) > len(shown):
        lines.append(f"           ... and {len(unique) - len(shown)} more")
    return "\n".join(lines)
