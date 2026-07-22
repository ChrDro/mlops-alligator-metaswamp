FROM python:3.11.13-slim-bookworm

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8085

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8085", "app:app"]
