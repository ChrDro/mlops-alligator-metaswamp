# Training Data Quality Tests

## Übersicht

Diese Test-Suite validiert die Qualität der Training-CSV **bevor** das Model trainiert wird. Sie verhindert häufige Fehler durch schlechte Datenqualität, fehlende Spalten oder inkonsistente Features.

**Datei:** `data/summary_output_task_1_2_training.csv`  
**Anzahl Tests:** 37 Tests in 9 Kategorien

---

## 🎯 Warum diese Tests?

Da deine Daten aus einer Datenbank kommen, können folgende Probleme auftreten:
- ✅ **CSV-Import** kann Datentypen ändern
- ✅ **Query-Logik** kann fehlerhafte Features produzieren
- ✅ **Neue Daten** können andere Verteilungen haben
- ✅ **Schema-Änderungen** können Spalten entfernen

Diese Tests fangen solche Probleme **früh** ab, bevor das Training startet.

---

## 📊 Test-Kategorien

### **Kategorie 1: Data Quality Tests** (6 Tests) ⭐⭐⭐

#### `TestSchemaValidation`
- ✅ CSV existiert
- ✅ CSV ist nicht leer (mindestens 100 Zeilen)
- ✅ Alle erforderlichen Spalten vorhanden
- ✅ Keine unerwarteten Extra-Spalten
- ✅ Identifier-Spalten vorhanden
- ✅ Target-Spalten vorhanden

**Verhindert:** Training-Crashes durch fehlende Spalten

---

#### `TestDataTypeValidation` (4 Tests)
- ✅ Numerische Features sind numerisch
- ✅ Target-Spalten sind binär (0 oder 1)
- ✅ Boolean-ähnliche Spalten sind 0 oder 1
- ✅ `column_type` ist String

**Verhindert:** Type Errors während Training

---

#### `TestNoMissingValuesInCriticalColumns` (3 Tests)
- ✅ Identifier haben keine NULLs
- ✅ Targets haben keine NULLs
- ✅ Kritische Features haben <50% NULLs

**Verhindert:** Training-Fehler durch fehlende Werte

---

### **Kategorie 2: Data Distribution Tests** (9 Tests) ⭐⭐

#### `TestTargetDistribution`
- ✅ `pk_target` ist nicht extrem unbalanced (<1% oder >99%)
- ✅ `pk_target` hat beide Klassen (0 und 1)
- ✅ Alle Targets haben beide Klassen
- ✅ Print Target Distribution Summary

**Verhindert:** Schlechte Models durch extreme Class Imbalance

---

#### `TestUniqueRatioBounds` (5 Tests)
- ✅ `unique_ratio` zwischen 0 und 1
- ✅ `unique_ratio_relative_to_max` zwischen 0 und 1
- ✅ `table_max_unique_ratio` zwischen 0 und 1
- ✅ `null_ratio` zwischen 0 und 1
- ✅ `relative_ordinal_position` zwischen 0 und 1

**Verhindert:** Fehlerhafte Feature-Berechnungen

---

### **Kategorie 3: Business Logic Tests** (9 Tests) ⭐⭐

#### `TestConsistencyChecks` (5 Tests)
- ✅ `is_unique=1` → `unique_ratio=1.0`
- ✅ `unique_ratio=1.0` → `is_unique=1`
- ✅ `is_non_null=1` → `null_ratio=0.0`
- ✅ `ordinal_position` >= 1
- ✅ `count` >= 0

**Verhindert:** Inkonsistente Features (Data Leakage)

---

#### `TestGroupConsistency` (4 Tests)
- ✅ `table_column_count` ist gleich für alle Spalten derselben Tabelle
- ✅ `table_row_count` ist gleich für alle Spalten derselben Tabelle
- ✅ `table_has_unique_column` ist gleich für alle Spalten
- ✅ `table_max_unique_ratio` ist gleich für alle Spalten

**Verhindert:** Fehlerhafte Aggregationen (Data Leakage)

---

### **Kategorie 4: Pipeline Integration Tests** (6 Tests) ⭐

#### `TestOneHotEncoding` (2 Tests)
- ✅ `column_type` hat erwartete Werte
- ✅ `pd.get_dummies()` funktioniert

**Verhindert:** Crashes bei One-Hot Encoding

---

#### `TestTrainTestSplit` (4 Tests)
- ✅ Mindestens 5 Datenbanken (für 5-fold split)
- ✅ Jede Datenbank hat mehrere Samples
- ✅ Stratifikation ist möglich (beide Klassen pro DB)
- ✅ `StratifiedGroupKFold` funktioniert (keine DB-Überlappung)

**Verhindert:** Crashes bei Train-Test-Split

---

## 🚀 Tests ausführen

### Alle Data Quality Tests
```bash
pytest tests/test_data/test_training_data_quality.py -v
```

### Nur Schema Tests (schnell)
```bash
pytest tests/test_data/test_training_data_quality.py::TestSchemaValidation -v
```

### Nur Distribution Tests
```bash
pytest tests/test_data/test_training_data_quality.py::TestTargetDistribution -v
```

### Mit detailliertem Output
```bash
pytest tests/test_data/test_training_data_quality.py -v -s
```
(Der `-s` Flag zeigt Print-Statements wie Target Distribution Summary)

---

## 📈 Beispiel-Output

### Erfolgreicher Testlauf
```
tests/test_data/test_training_data_quality.py::TestSchemaValidation::test_training_csv_exists PASSED
tests/test_data/test_training_data_quality.py::TestSchemaValidation::test_training_csv_is_not_empty PASSED
...
tests/test_data/test_training_data_quality.py::TestTargetDistribution::test_target_distribution_summary PASSED

=== Target Distribution Summary ===

pk_target:
  Positive (1): 1234 (11.9%)
  Negative (0): 9136 (88.1%)

composite_pk_target:
  Positive (1): 456 (4.4%)
  Negative (0): 9914 (95.6%)

...

============================== 37 passed in 2.45s ===============================
```

### Fehlgeschlagener Test (Beispiel)
```
FAILED tests/test_data/test_training_data_quality.py::TestSchemaValidation::test_training_csv_has_all_required_columns

AssertionError: Missing required columns: ['ordinal_position', 'pk_target']
```

---

## 🐛 Häufige Fehler und Lösungen

### "Training data not found"
**Problem:** CSV existiert nicht  
**Lösung:** Daten-Pipeline ausführen
```bash
python src/get_trino_summaries_task_1_2_training.py
```

### "pk_target is extremely imbalanced: 0.2% positive"
**Problem:** Zu wenige positive Beispiele  
**Lösung:** Mehr Daten sammeln oder Sampling-Strategie anpassen

### "Missing required columns: ['unique_ratio']"
**Problem:** Spalte fehlt in CSV  
**Lösung:** Query in Data Pipeline überprüfen und neu ausführen

### "is_unique=1 but unique_ratio<1.0"
**Problem:** Inkonsistente Feature-Berechnung  
**Lösung:** Logik in `get_trino_summaries` überprüfen

### "StratifiedGroupKFold split failed"
**Problem:** Nicht genug Datenbanken oder zu unbalanced  
**Lösung:** Mehr Datenbanken hinzufügen oder Split-Strategie ändern

---

## 🔄 Integration in CI/CD

Diese Tests sollten in deiner CI/CD Pipeline laufen **vor** dem Model Training:

```yaml
# .github/workflows/train-model.yml
jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - name: Validate Training Data
        run: pytest tests/test_data/test_training_data_quality.py -v
  
  train-model:
    needs: validate-data  # Nur wenn Data Quality OK
    runs-on: ubuntu-latest
    steps:
      - name: Train Model
        run: python task_1/task_1_pk_train_and_register.py
```

---

## 🔄 Integration in Prefect Pipeline

Füge einen Data Quality Check als ersten Task hinzu:

```python
from prefect import flow, task
import pytest

@task
def validate_training_data():
    """Run data quality tests before training."""
    exit_code = pytest.main([
        "tests/test_data/test_training_data_quality.py",
        "-v"
    ])
    if exit_code != 0:
        raise ValueError("Data quality tests failed!")

@task
def train_model():
    # ... existing training code
    pass

@flow
def training_pipeline():
    validate_training_data()  # Runs first
    train_model()             # Only runs if validation passes
```

---

## 📝 Erweitern der Tests

### Neue Features hinzufügen
Wenn du neue Features zur CSV hinzufügst:

1. **Aktualisiere `EXPECTED_FEATURE_COLUMNS`** (Zeile 27)
```python
EXPECTED_FEATURE_COLUMNS = [
    # ... existing columns
    "new_feature_name",  # Add here
]
```

2. **Optional: Füge spezifische Tests hinzu**
```python
def test_new_feature_is_valid(self, training_df: pd.DataFrame):
    """Test that new_feature has valid range."""
    assert training_df["new_feature"].min() >= 0
    assert training_df["new_feature"].max() <= 100
```

### Neue Targets hinzufügen
Für neue Target-Spalten (z.B. Task 3):

1. **Aktualisiere `EXPECTED_TARGET_COLUMNS`** (Zeile 67)
```python
EXPECTED_TARGET_COLUMNS = [
    "pk_target",
    "composite_pk_target",
    "fk_target",
    "composite_fk_target",
    "normalization_target",  # Add new target
]
```

Tests laufen automatisch für alle Targets!

---

## 🎯 Best Practices

### Wann diese Tests laufen sollten:
1. ✅ **Vor jedem Training** - Lokale Checks
2. ✅ **In CI/CD** - Automatische Validierung
3. ✅ **Nach Data Pipeline Updates** - Regression Tests
4. ✅ **Bei neuen Datenquellen** - Integration Tests

### Was tun wenn Tests fehlschlagen:
1. **Nicht überspringen!** - Fehler früh beheben spart Zeit
2. **Root Cause finden** - Ist es die DB-Query oder CSV-Import?
3. **Test anpassen** - Nur wenn die Anforderung sich geändert hat
4. **Daten fixen** - Meist ist es ein Data Quality Issue

---

## 📚 Zusätzliche Ressourcen

- **Great Expectations:** Für Production-Grade Data Testing
- **Pandera:** Schema Validation für Pandas DataFrames
- **dbt Tests:** Für Tests direkt in der Datenbank

---

## 🔗 Verwandte Tests

- `tests/test_api/` - API Endpoint Tests
- `tests/test_models/` - Model Loading Tests
- `tests/README.md` - Allgemeine Test-Übersicht

---

**Viel Erfolg mit Data Quality! 🚀**
