# Two stages so the runtime image carries no package installer: pip, setuptools
# and wheel live in the builder only. Reasoning and the measured effect:
# documentation/DOCKER_GHCR_DESIGN.md
FROM python:3.11.13-slim-bookworm AS builder

COPY requirements.txt .
# --prefix rather than a venv: the runtime stage copies /install to /usr/local,
# where site-packages and bin already live. No PATH juggling, and the installer
# stays behind in this stage.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11.13-slim-bookworm AS runtime

WORKDIR /app

# libgomp1 is required by LightGBM, and apt-get upgrade patches the base image's
# OS packages (libgnutls30/libssl3/openssl CVEs) to whatever Debian security has
# shipped as of build time - both unchanged from the previous single-stage build.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /install /usr/local

# AFTER the COPY, on purpose. A serving container installs nothing at run time,
# and removing the installer here makes it irrelevant whether a copy came from
# the base image or from requirements.txt via /install. Removing it *before* the
# COPY would let the builder's copy quietly reinstate it - not hypothetical: the
# CI Trivy scan reports two setuptools versions in the evidently image, one from
# each source. setuptools 65.5.1 is what carries CVE-2024-6345 and
# CVE-2025-47273, so this is worth -2 HIGH findings. pip and wheel carry none
# today (wheel only because requirements.txt pins it above the fixed version) -
# for those two this is hygiene, and it retires the pin.
# Order wheel -> setuptools -> pip is deliberate: pip removes itself last.
#
# Then a sweep, because the two mechanisms cover different gaps. pip uninstall
# also removes distutils-precedence.pth and _distutils_hack, which would
# otherwise raise on every interpreter start - a plain rm cannot know that. But
# --prefix plus COPY can leave a SECOND dist-info in place (the base image's copy
# and a copy pulled in by a floating transitive dependency merge into the same
# site-packages), and pip uninstall only ever handles one of them. Trivy reads
# dist-info, so a leftover is still a reported package. The globs are written out
# rather than as setuptools* so they cannot swallow setuptools_scm.
RUN python -m pip uninstall -y wheel setuptools pip \
    && cd /usr/local/lib/python3.11/site-packages \
    && rm -rf pip pip-*.dist-info setuptools setuptools-*.dist-info \
              wheel wheel-*.dist-info

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8080

# Liveness, not readiness: an image-level default should mean "the process
# answers". docker-compose.yaml overrides this with a stricter /health/ready
# probe, which additionally requires the MLflow registry to resolve the models.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health/live')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
