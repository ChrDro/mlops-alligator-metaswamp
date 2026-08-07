"""Turn c-TF-IDF topic keywords into a readable subject-area name.

Shared by the exploratory clustering script and the train-and-register script, so
both name a topic identically - a label that changed between the two would make the
exported CSV and the registered model disagree about what a cluster is called.

The model is an open-weights LLM served locally by Ollama (`docker compose up -d
ollama`), which keeps label generation free of API keys and outbound network.
Labels are cached by keyword set, so reruns reuse a name instead of regenerating
(and possibly renaming) a subject area that has not changed.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Anchored to this file, not the working directory: the training script runs from the
# repo root (`python task_4/...`) while ad-hoc use often runs from task_4/.
REPO_ROOT = Path(__file__).resolve().parent.parent
LABEL_CACHE_PATH = REPO_ROOT / "models" / "task_4_label_cache.json"

# Explicit path, and before the getenv calls below: OLLAMA_HOST_PORT in .env exists
# precisely because a host may already run Ollama natively on 11434. Falling through
# to that default would reach a different server with a different model and 404.
load_dotenv(REPO_ROOT / ".env")

LLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
LLM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# How many keywords describe a topic. Also the cache key, so changing this
# invalidates every cached label.
KEYWORDS_PER_TOPIC = 8

llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key="ollama",  # ignored by Ollama, but the SDK requires a non-empty value
)


def clean_label(raw: str) -> str:
    """Reduce a chatty completion to a bare noun phrase.

    Small local models tend to wrap the answer in quotes, prefix it with
    'Subject area:', or append a sentence of justification. Keep the first line,
    drop that scaffolding, and cap the length so a runaway answer cannot end up as a
    subject area.
    """
    label = raw.strip().splitlines()[0] if raw.strip() else ""
    label = re.sub(r"^(subject area|answer|label)\s*[:\-]\s*", "", label, flags=re.I)
    label = label.strip(" \t\"'`*.")
    return " ".join(label.split()[:5])


def cache_key_for(keywords: list[str]) -> str:
    """Cache key for a topic.

    Keyed on the keywords rather than the topic id: HDBSCAN renumbers topics between
    runs, so a topic that keeps its keywords keeps its label.
    """
    return "|".join(keywords[:KEYWORDS_PER_TOPIC])


def load_label_cache(path: Path = LABEL_CACHE_PATH) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_label_cache(cache: dict[str, str], path: Path = LABEL_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def generate_label(
    keywords: list[str],
    rep_docs: list[str],
    cache: dict[str, str] | None = None,
) -> str:
    """Name one subject area from its keywords and a few representative tables.

    Returns an empty string when the model answers with nothing usable; callers
    decide on the fallback (the scripts here fall back to `topic_<id>`).
    """
    keywords = keywords[:KEYWORDS_PER_TOPIC]
    key = cache_key_for(keywords)
    if cache is not None and key in cache:
        return cache[key]

    docs_ctx = " | ".join(doc[:60] for doc in rep_docs[:3])
    prompt = (
        "These database tables belong to the same subject area.\n"
        f"Keywords: {', '.join(keywords)}\n"
        f"Example tables: {docs_ctx}\n"
        "Some names are German. Reply with exactly one short English noun phrase "
        "(2-4 words) naming the subject area. No explanation, no quotes."
    )
    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=24,
        temperature=0.0,
        seed=0,  # Ollama honours this; makes reruns byte-identical
    )
    label = clean_label(response.choices[0].message.content or "")

    if cache is not None and label:
        cache[key] = label
    return label
