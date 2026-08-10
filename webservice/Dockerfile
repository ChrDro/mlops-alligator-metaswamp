FROM python:3.11.13-slim-bookworm

WORKDIR /app

# Install system dependencies required by LightGBM, and patch the base
# image's OS packages (libgnutls30/libssl3/openssl CVEs) to whatever Debian
# security has shipped as of build time.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
