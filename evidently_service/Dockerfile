FROM python:3.11.13-slim-bookworm

WORKDIR /app

COPY . /app

# One layer, and --no-cache-dir so pip's download cache does not end up in the
# image: that cache plus the separate pip-upgrade layer were dead weight.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8085

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8085", "app:app"]
