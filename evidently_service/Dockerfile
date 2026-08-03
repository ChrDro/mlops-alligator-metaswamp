FROM python:3.11.13-slim-bookworm

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8085

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8085", "app:app"]
