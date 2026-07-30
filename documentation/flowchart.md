# Flowchart

**Stand: 27. Juli 2026** — abgeglichen mit dem tatsächlichen Repo- und Container-Zustand.

Gestrichelte, graue Knoten existieren als Container oder Datei, sind aber **noch nicht
funktional angeschlossen**. Sie stehen bewusst im Diagramm, damit die Lücke sichtbar
bleibt statt zu suggerieren, das Monitoring sei fertig.

---

## 1. Gesamtsystem

```mermaid
graph TD
    subgraph CI["CI/CD - GitHub Actions"]
        PUSH["Push / Pull Request"] --> LINT["Ruff check + format"]
        LINT --> PYTEST["pytest"]
    end

    subgraph TRAIN["Modelltraining und Registrierung - manuell angestoßen"]
        TRAINDATA["Trainingsdaten<br/>data/*.csv"] --> DQ["Datenqualitätstests<br/>test/test_data"]
        DQ --> SCRIPTS["5 Trainingsskripte<br/>task_1 pk + cpk<br/>task_2 fk + cfk<br/>task_3 normalform"]
        SCRIPTS --> CAND["je 4 Kandidaten<br/>RandomForest und XGBoost,<br/>Baseline + RandomizedSearchCV"]
        CAND --> TRACK["MLflow Tracking<br/>Params, Metriken, Confusion Matrix"]
        TRACK --> BEST{"bester nach test_f1"}
        BEST --> REG["Registry: register_model<br/>+ Alias 'dev'"]
    end

    subgraph PERSIST["Persistenz"]
        PGDB[("Postgres<br/>mlflow_db + prefect")]
        MINIO[("MinIO / S3<br/>Modellartefakte")]
        BACKUP["Stündliche S3-Backups<br/>postgres_backup<br/>prefect_postgres_backup"]
        PGDB --> BACKUP
        MINIO --> BACKUP
    end

    TRACK --> PGDB
    REG --> MINIO

    subgraph SERVE["Model Service - FastAPI"]
        LOAD["Modelle laden<br/>models:/name@dev<br/>lru_cache"]
        ALIGN["_align_to_signature<br/>Spalten an Signatur ausrichten"]
        EP["POST /predict_pk /predict_cpk<br/>/predict_fk /predict_cfk<br/>/predict_normalform"]
        HOOK["POST /events/new-data<br/>Push-Trigger"]
        METRICS["GET /metrics"]
        LOAD --> ALIGN --> EP
    end

    REG -.->|"Alias dev"| LOAD
    PGDB --- REG
    MINIO --- LOAD

    subgraph STREAM["Streaming-Trigger - Prefect"]
        DETECT["change-detection-poller"]
        PKFK["key-prediction-pipeline<br/>Track keys"]
        NF["normalform-prediction-pipeline<br/>Track nf"]
    end

    SRC[("Trino / DuckDB<br/>new_predict_data")] --> DETECT
    HOOK --> DETECT
    DETECT --> PKFK
    DETECT --> NF
    PKFK -->|"HTTP"| EP
    NF -->|"HTTP"| EP
    PKFK --> RESULTS[("prediction_results<br/>key_results + nf_results")]
    NF --> RESULTS

    subgraph MON["Monitoring"]
        PROM["Prometheus<br/>scraped model-service,<br/>evidently, prometheus"]
        ALERTS["7 Alert-Regeln<br/>Fehlerrate, Latenz p95,<br/>Traffic, Speicher, Down"]
        AM["Alertmanager<br/>UI auf :9093"]
        NOTIFY["Notification-Kanal<br/>Slack / E-Mail"]
        GRAF["Grafana<br/>Dashboards"]
        EVID["Evidently Service<br/>Drift + Data Quality"]
        PROM --> ALERTS --> AM --> NOTIFY
        PROM --> GRAF
    end

    METRICS --> PROM
    EP -.->|"Aufruf in app.py<br/>auskommentiert"| EVID
    EVID -.-> GRAF

    classDef offen fill:#f4f4f4,stroke:#999,stroke-width:1px,color:#666
    class NOTIFY,GRAF,EVID offen
    style SRC fill:#f9f,stroke:#333,stroke-width:2px
    style RESULTS fill:#cfc,stroke:#333,stroke-width:2px
```

**Noch nicht angeschlossen:**

- **Grafana** läuft, aber `grafana/provisioning/` und `grafana/dashboards/` sind leer.
- **Evidently** läuft noch auf den `green_taxi_data`-Tutorialdaten; der Monitoring-Aufruf
  in [app.py](../webservice/app.py) ist auskommentiert.
- **Alertmanager** empfängt Alerts, hat aber einen Null-Receiver — es geht nichts raus.

---

## 2. Streaming-Trigger im Detail

Zwei Wege lösen denselben Detector aus. Der Push ist schnell, der Cron fängt alles ab,
was der Push verpasst. Details und Begründungen in [STREAMING_PIPELINE.md](STREAMING_PIPELINE.md).

```mermaid
graph TD
    LOADER["Loader schreibt nach<br/>duckdb.new_predict_data"]
    LOADER -->|"optional, sofort"| WH["POST /events/new-data"]
    WH --> EV1(["Event<br/>alligator.new-data.arrived"])
    CRON["Cron alle 15 Minuten<br/>Sicherheitsnetz"] --> DET

    EV1 -->|"Automation"| DET["change-detection-poller"]

    DET --> R1["1 - verwaiste Claims freigeben<br/>älter als 120 Minuten"]
    R1 --> R2["2 - Snapshot: row_count<br/>und column_count je Tabelle"]
    R2 --> R3{"3 - Diff gegen<br/>source_watermarks"}
    R3 -->|"neu oder geändert"| R4[("4 - pending_changes<br/>durable Arbeitsliste")]
    R3 -->|"unverändert"| STOP["kein Signal"]
    R4 --> R5["5 - Watermarks schreiben<br/>erst NACH der Arbeitsliste"]
    R5 --> R6{"offene Arbeit vorhanden?"}
    R6 -->|"ja"| EV2(["Event<br/>alligator.changes.recorded"])
    R6 -->|"nein"| STOP

    EV2 -->|"Automation"| P1["key-prediction-pipeline"]
    EV2 -->|"Automation"| P2["normalform-prediction-pipeline"]

    P1 --> Q1["nur geänderte Tabellen profilieren"]
    Q1 --> Q2["Queue füllen,<br/>bereits prognostizierte<br/>Spalten erneut öffnen"]
    Q2 --> Q3["Queue leeren:<br/>pk, cpk, fk, cfk je Zeile"]
    Q3 --> Q4[("key_results<br/>append")]

    P2 --> N1["nur geänderte Tabellen profilieren"]
    N1 --> N2["Prediction je Spalte"]
    N2 --> N3["Mehrheitsentscheid<br/>je Tabelle"]
    N3 --> N4[("nf_results<br/>append")]

    style EV1 fill:#ffe6b3,stroke:#333
    style EV2 fill:#ffe6b3,stroke:#333
    style R4 fill:#cce5ff,stroke:#333,stroke-width:2px
    style Q4 fill:#cfc,stroke:#333
    style N4 fill:#cfc,stroke:#333
```

Beide Pipelines hängen am **selben** Event und laufen parallel — keine kann die andere
aushungern. Die Ergebnisse werden angehängt, nicht überschrieben: eine neu geladene
Tabelle behält ihre alten Vorhersagen als Historie.

---

## 3. Lebenszyklus einer erkannten Änderung

Der subtilste Teil und der, an dem beim ersten Ende-zu-Ende-Test tatsächlich etwas
schiefging: eine Pipeline darf ihre Änderung erst abschließen, wenn die Vorhersagen
wirklich vorliegen. Sonst gilt die Arbeit als erledigt, obwohl nichts entstanden ist,
und der Retry-Pfad sieht sie nie wieder.

Jede Zeile durchläuft diesen Zyklus **pro Track getrennt** — `keys` und `nf` haben
eigene Claim- und Completion-Spalten.

```mermaid
stateDiagram-v2
    [*] --> Offen: Detector schreibt Zeile
    Offen --> Beansprucht: Pipeline claimt<br/>claimed_by = run_id
    Beansprucht --> Fertig: alle Vorhersagen<br/>gespeichert
    Beansprucht --> Offen: Lauf schlägt fehl<br/>release, Flow FAILED
    Beansprucht --> Offen: Prozess hart beendet<br/>Sweep nach 120 Minuten
    Fertig --> [*]

    note right of Offen
        Der nächste Detector-Lauf
        signalisiert, solange
        irgendetwas offen ist -
        nicht nur bei neuen Funden.
    end note
```

Der Detector sendet sein Signal auf Basis der **offenen Arbeit**, nicht auf Basis
dessen, was er gerade gefunden hat. Andernfalls wäre freigegebene Arbeit für immer
liegengeblieben: die Watermarks sind zu diesem Zeitpunkt schon geschrieben, kein
späterer Lauf würde die Tabelle noch einmal als geändert erkennen.

---

## 4. Was der Chart nicht abbildet

- **Batch-Betrieb ohne Streaming.** Beide Pipelines laufen weiterhin per
  `python prefect/<pipeline>.py` als vollständiger Schema-Scan. Streaming ist per
  `use_pending_changes=True` opt-in und wird nur von den Deployments gesetzt.
- **Retraining.** Nicht implementiert; die Trainingsskripte werden von Hand gestartet.
- **Docker-Build in CI.** Images werden lokal gebaut, es gibt keinen GHCR-Push.
