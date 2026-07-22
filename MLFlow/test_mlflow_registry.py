import os
import mlflow

# 1. Umgebungsvariablen für MinIO (S3-Schnittstelle) definieren
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadminpassword"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

# 2. Verbindung zum MLflow Tracking Server herstellen
mlflow.set_tracking_uri("http://localhost:5000")

# 3. Experiment anlegen/auswählen
experiment_name = "Postgres_MinIO_Prod_Test"
mlflow.set_experiment(experiment_name)

print("Starte MLflow Test-Run auf PostgreSQL + MinIO...")

with mlflow.start_run() as run:
    # Metadaten loggen (landet vollautomatisch in PostgreSQL)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_metric("val_loss", 0.12)
    
    # Artefakt erstellen und hochladen (landet vollautomatisch in MinIO)
    filename = "model_summary.txt"
    with open(filename, "w") as f:
        f.write("PostgreSQL speichert die Metriken, MinIO speichert diese Datei!")
        
    mlflow.log_artifact(filename)
    os.remove(filename)

print(f"Run erfolgreich! ID: {run.info.run_id}")
