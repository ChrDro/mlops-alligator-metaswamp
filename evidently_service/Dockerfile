FROM python:3.11.13-slim-bookworm

WORKDIR /app

# Patch the base image's OS packages (libgnutls30/libssl3/openssl CVEs) to
# whatever Debian security has shipped as of build time.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

COPY . /app

# One layer, and --no-cache-dir so pip's download cache does not end up in the
# image: that cache plus the separate pip-upgrade layer were dead weight.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8085

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8085", "app:app"]
