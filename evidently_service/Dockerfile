# Two stages so the runtime image carries no package installer, same shape as
# webservice/Dockerfile. Reasoning and the measured effect:
# documentation/DOCKER_GHCR_DESIGN.md
FROM python:3.11.13-slim-bookworm AS builder

COPY requirements.txt .
# --prefix rather than a venv: the runtime stage copies /install to /usr/local,
# where site-packages and bin already live. The installer stays in this stage.
# The previous single-stage build also upgraded pip first; that step is gone -
# with every direct dependency pinned, the base image's pip resolves the same
# versions.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11.13-slim-bookworm AS runtime

WORKDIR /app

# apt-get upgrade patches the base image's OS packages (libgnutls30/libssl3/
# openssl CVEs) to whatever Debian security has shipped as of build time.
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /install /usr/local

# After the COPY, and this image is the reason it has to be: Trivy against the
# unhardened build reports TWO setuptools versions here - 65.5.1 from the base
# image and 70.3.0 pulled in by a floating transitive dependency. Removing only
# the base image's copy would leave CVE-2025-47273 in place. The rm sweep exists
# because pip uninstall handles just one dist-info while Trivy reads all of them;
# pip uninstall stays because it is what removes distutils-precedence.pth and
# _distutils_hack, which would otherwise raise on every interpreter start.
RUN python -m pip uninstall -y wheel setuptools pip \
    && cd /usr/local/lib/python3.11/site-packages \
    && rm -rf pip pip-*.dist-info setuptools setuptools-*.dist-info \
              wheel wheel-*.dist-info

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# After the install, not before. It used to be first, so every code change
# invalidated the complete dependency layer.
COPY --chown=appuser:appuser . /app

USER appuser

EXPOSE 8085

# This service has no dedicated health route; a served /tracks proves Flask is
# answering. docker-compose.yaml sets its own timings.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8085/tracks')"

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8085", "app:app"]
