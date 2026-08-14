# Task 3 — Feature-Audit und Umbauplan

Ergebnis der Feature-Analyse vom 2026-08-04 auf `data/nf_test_analyse.csv`
(13.822 Zeilen, 1.560 Tabellen, 503 Datenbanken) plus dem daraus abgeleiteten Plan.

Ergänzt die Ideensammlung in [TASK_3.md](TASK_3.md). Kann nach
`documentation/NORMALFORM_PREDICTION.md` verschoben werden — dieser Pfad wird in
[prefect/normalform_pipeline.py:5](prefect/normalform_pipeline.py#L5) referenziert, existiert
aber nicht.

## Stand

| Phase | Status |
| :--- | :--- |
| **0** — Redundanz aufräumen | Code erledigt (50 → 29 Features). Offen: Retraining + Evidently-Artefakte neu bauen |
| **0.5** — 1NF-Detektor | Implementiert. Offen: Schwellen kalibrieren, geteiltes Modul, Verifikation an echten Tabellen |
| **1** — Messung ehrlich machen | Teilweise: README korrigiert, Baseline dokumentiert. Offen: Guard, Split-Diskrepanz, Skew-Test |
| **2** — Neue Trainingsdaten | **2a + 2b erledigt** — 132 generierte Tabellen, 32 gemessene Features, `data/nf_training.csv`. Offen: 2c Harness, 2d Skalierung — **blockiert Phase 3** |
| **3** — Spaltenpaar-Features | Offen |
| **4** — Modell und Serving | Offen |

Abschnitt 1 dokumentiert den Audit-Stand vom 2026-08-04 auf dem **50-Feature-Satz**.
Die Zahlen bleiben als Begründung stehen, auch wo der Code inzwischen anders aussieht;
nachgemessene Werte für den 29er-Satz stehen jeweils dabei.

## Entscheidungen

### E1 — Schlüsselkandidat kommt aus einer UCC-Suche (2026-08-04)

Die 2NF/3NF-Features brauchen einen Schlüsselkandidaten, um eine *partielle* von einer
*transitiven* Abhängigkeit zu unterscheiden. Heute ist er geraten
(`table_has_composite_pk`, 77,5 % Übereinstimmung). Verworfen wurden: die Heuristik
beizubehalten, und `pk_model` / `composite_pk_model` vorzuschalten.

**Festgelegt:** level-weise UCC-Suche auf den Daten, Apriori-Pruning (jede Obermenge einer
UCC ist nicht minimal).

| Parameter | Wert |
| :--- | :--- |
| Maximales Level | 3 — Composite Keys mit mehr als 3 Spalten sind selten, darüber wird es exponentiell |
| Scan | vollständig, kein Sampling |
| Kein Treffer bis Level 3 | melden, mit der nächsten Tabelle weitermachen; Gemeldetes wird manuell geprüft |

Kein Sampling, weil eine Stichprobe **zusätzliche falsche** UCCs findet (unique im Sample,
nicht in der Vollmenge — umgekehrt nie). Sie taugt nur als Kandidatenfilter, bestätigen
müsste man ohnehin auf den vollen Daten.

Nicht selbst implementieren: **Desbordante** (C++ mit Python-Interface, enthält HyUCC)
oder Metanome decken das ab.

**Drei Konsequenzen für die Umsetzung:**

- [ ] **Batchen statt schleifen.** Worst Case sind `n + C(n,2) + C(n,3)` Tests — bei
      40 Spalten ~10.700. Als Einzelqueries dauert das Stunden, als Aggregat-Ausdrücke in
      wenigen Queries gebündelt (wie bei den Paar-Features in 3.1) ist es beherrschbar.
      Ausgerechnet der Report-Fall „keine UCC" ist der, in dem das Pruning nichts kappt und
      wirklich alles gerechnet wird.
- [ ] **Der `unknown`-Zustand muss in den Trainingsdaten vorkommen.** Tabellen ohne UCC
      nicht aus dem Training werfen, sondern ihre FD-Features als `unknown` markieren und
      drinlassen — sonst trifft das Modell in Produktion auf einen Eingabezustand, den es
      nie gesehen hat.
- [ ] **„Keine UCC" ist selbst ein Feature**, nicht nur ein Fehlerfall: keine unique
      Spaltenkombination bis Level 3 heißt sehr wahrscheinlich Duplikatzeilen, also eine
      Verletzung des Relationenmodells. Als `table_has_no_ucc_le3` mitnehmen.
- [ ] Spalte im Manifest für den Prüfgrund, damit die manuelle Kontrolle eine Warteschlange
      hat statt Logzeilen.

**Bedingung:** UCC-Suche braucht die realistischen Tabellengrößen aus Phase 2. Bei den
heutigen 6–12 Zeilen sind 61,4 % aller Spalten zufällig unique — die Suche fände fast nur
Scheinschlüssel.

**Offen bleibt** die Messung: der Schlüssel aus dem Manifest ist Ground Truth und darf vom
Extractor nicht gesehen werden, taugt aber zum Gegenprüfen. Damit lässt sich trennen,
wie viel des FD-Feature-Fehlers aus der Schlüsselerkennung stammt und wie viel aus der
FD-Berechnung — was heute niemand auseinanderhalten kann.

---

## 1. Befunde

### 1.1 Redundanz

50 Features, aber **numerischer Rang 37**; 26 Hauptkomponenten decken 99 % der Varianz.

| Befund | Details |
| :--- | :--- |
| 7 Features konstant | `null_count`, `null_ratio`, `is_non_null`, `table_avg_null_ratio`, `name_contains_key`, `name_contains_table_name`, `name_is_singular_table_id` |
| 5 Gruppen bitgleicher Spalten | `count`≡`table_row_count`, `ordinal_position`≡`null_ratio_rank`, `is_first_column`≡`is_least_null_in_table`, `table_column_count`≡`table_non_null_column_count` |
| 15 Features aus einer Basisgröße | alles was von `number_unique_values / count` abhängt, Korrelationen bis 0,99 |

**Wichtige Einschränkung:** Tabellen-Aggregate wie `table_ratio_1nf_violations` sind
**nicht** redundant zu ihrem Spalten-Pendant. Das Modell sieht pro Aufruf genau eine
Zeile und kann kein `groupby` rechnen — der Aggregatwert ist echte Zusatzinformation.
Streicht man sie, fällt F1 von 0,9935 auf 0,7710.

### 1.2 Der Leakage-Guard ist wirkungslos

`table_contains_1nf_violation` wird in
[task_3_denormalization_train_and_register.py:134](task_3/task_3_denormalization_train_and_register.py#L134)
als Leckage gedroppt — `table_ratio_1nf_violations > 0` ist aber bitgleich dasselbe (100 %).

| Feature > 0 | → Klasse | Trefferquote |
| :--- | ---: | ---: |
| `table_ratio_1nf_violations` | 0 | 100,0 % |
| `table_has_partial_dependency` | 1 | 100,0 % |

**46,2 % aller Zeilen** sind durch zwei Bits vollständig determiniert. Gain-Importance der
7 definitionsnahen Features: **88,5 %**.

### 1.3 2NF vs. 3NF wird über ein Generator-Artefakt entschieden

Für die verbleibende Unterscheidung gibt es kein Definitions-Flag — das Modell weicht auf
eine Signatur des Datengenerators aus:

```
                        Klasse 2        Klasse 3
table_avg_unique_ratio  max = 0,8778    min = 0,8485    ← ein Schwellwert bei ~0,86
table_std_unique_ratio  min = 0,0907    max = 0,2624
```

Dieser Zusammenhang existiert in echten Datenbanken nicht.

### 1.4 Zerlegung des gemessenen F1

| Feature-Satz | n | test-F1 (weighted) |
| :--- | ---: | ---: |
| alle Features | 50 | 0,9935 |
| redundanzbereinigt | 29 | **0,9939** |
| nur die 7 definitionsnahen | 7 | 0,7341 |
| nur die 43 Statistik-Features | 43 | 0,7413 |
| nur 3 Tabellen-Definitionsflags | 3 | 0,7341 |
| bereinigt, ohne die 2 Generator-Trenner | 27 | 0,9862 |

Zwei Definitions-Flags plus zwei Generator-Artefakte erklären das gesamte Ergebnis.
Split-Strategie ist nicht die Ursache: `group=database` liefert 0,9931 statt 0,9935.

### 1.5 Kein einziger NULL in den Trainingsdaten

Drei Features sind dadurch nicht bloß tot, sondern **in Produktion falsch belegt** — das
Modell hat auf ihnen *Spaltenposition* gelernt, bekommt beim Predict aber echte
Null-Statistiken:

| Feature | im Training | in Produktion |
| :--- | :--- | :--- |
| `null_ratio_rank` | ≡ `ordinal_position` | echter Null-Rang |
| `is_least_null_in_table` | ≡ `is_first_column` | echtes Null-Minimum |
| `table_non_null_column_count` | ≡ `table_column_count` | echte Zählung |

Nebenwirkung: das Evidently-Referenzset hat `null_ratio ≡ 0` — der erste echte NULL löst
Drift-Alarm auf sieben Features gleichzeitig aus.

### 1.6 Train/Serve-Skew

Die Pipeline berechnet Heuristiken, wo die Trainingsdaten Ground Truth enthalten:

| Feature | Übereinstimmung Pipeline-Formel ↔ Trainingsdaten |
| :--- | ---: |
| `table_has_composite_pk` | **77,5 %** |
| `is_composite_key_part` | 95,3 % |
| `table_id_named_column_count` | 98,4 % |

Betroffen sind genau die Features mit 88,5 % Gain-Importance. Ursache ist auch die
Code-Duplikation: [`recompute_table_features`](prefect/normalform_pipeline.py#L455-L544)
ist ein wörtlicher Port des Trainings-Extractors, kein geteiltes Modul.

### 1.7 Trainingsdaten tragen Spaltenpaar-Features nicht

- Nur die aggregierte CSV liegt im Repo, **nicht** die 503 `synthetic_db_*` Quelldatenbanken.
  `get_trino_summaries_task_3_training.py` (in der Pipeline-Docstring referenziert) fehlt.
  `COUNT(DISTINCT a, b)` ist aus Spalten-Aggregaten prinzipiell nicht rekonstruierbar
  → **kein Backfill möglich**.
- Tabellen haben **6–12 Zeilen** (Median 9), **61,4 % aller Spalten sind vollständig unique**.
  Der FD-Test `COUNT(DISTINCT a,b) == COUNT(DISTINCT a)` ist bei uniquem `a` immer erfüllt
  → trivial wahr. Bei 9 Zeilen ist auch sonst jede Zufalls-FD hochwahrscheinlich.

**Konsequenz: neuer Datensatz, nicht neuer Extractor.**

### 1.8 Der 1NF-Detektor schlug bei Freitextspalten fehl

> **Erledigt am 2026-08-04** (Phase 0.5). Der Befund bleibt als Begründung stehen; der Code
> sieht heute anders aus — siehe
> [`build_1nf_violation_expr()`](prefect/normalform_pipeline.py#L219).

Bis dahin galt:

```sql
CASE WHEN COUNT_IF(CAST("col" AS VARCHAR) LIKE '%,%') > 0 THEN 1 ELSE 0 END
```

Das `> 0` war der Hauptschaden: **eine einzige Zeile** mit einem Komma setzt das Flag. Eine
Kommentar-, Beschreibungs- oder Notizspalte reicht — und weil
`table_ratio_1nf_violations > 0` in 100 % der Fälle auf Klasse 0 führt (siehe 1.2), kippt
damit die **gesamte Tabellenvorhersage** auf 0NF. Die Precision-Anforderung an dieses eine
Flag ist entsprechend extrem: jeder False Positive kostet eine ganze Tabelle.

Eine Listenspalte unterscheidet sich von Freitext nicht durch "hat ein Komma", sondern
durch drei Eigenschaften: der Wert ist **vollständig** eine Trennzeichen-Liste, die Tokens
sind **kurz**, und das gilt für **fast alle** Zeilen.

**Spalte droppen ist der falsche Weg.** Die Spalte fällt sonst aus allen Tabellen-Aggregaten
der übrigen Spalten heraus: `table_column_count` schrumpft, `relative_ordinal_position`
verschiebt sich, `table_avg_unique_ratio` / `table_std_unique_ratio` /
`table_ratio_of_pk_candidates` ändern sich. Das beschreibt dem Modell eine Tabelle, die es
nicht gibt — neuer Train/Serve-Skew statt weniger. Fachlich ist eine Beschreibungsspalte
zudem ein normales Nicht-Schlüssel-Attribut und für 2NF/3NF relevant; nur für die
*1NF-Frage* ist sie ein Sonderfall.

---

## 2. TODO-Liste

### Phase 0 — Redundanz aufräumen (Code erledigt 2026-08-04, Artefakte offen)

**50 → 29 Features.** Alle Codeänderungen sind drin; offen ist nur, was einen laufenden
Stack braucht.

- [x] 21 Features in `redundant_columns_to_drop` in
      [task_3_denormalization_train_and_register.py:145-176](task_3/task_3_denormalization_train_and_register.py#L145-L176)
      gestrichen — bewusst als eigene Liste neben `columns_to_drop`, das sind zwei
      verschiedene Dinge (Identifier/Target/Leakage-Guard vs. Redundanz):
  - [x] konstant (7): `null_count`, `null_ratio`, `is_non_null`, `table_avg_null_ratio`,
        `name_contains_key`, `name_contains_table_name`, `name_is_singular_table_id`
  - [x] exakte Duplikate (3): `count`, `null_ratio_rank`, `table_non_null_column_count`
  - [x] Schwellwert-Ableitungen derselben Zeile (5): `is_first_column`,
        `is_least_null_in_table`, `is_unique`, `table_has_unique_column`,
        `table_has_no_single_pk_candidate`
  - [x] Differenzen (2): `other_unique_columns_in_table`, `other_near_unique_columns_in_table`
  - [x] Rohwert/Quotient-Paare (4): `ordinal_position`, `number_unique_values`,
        `table_unique_column_count`, `unique_ratio_relative_to_max`
- [x] **Tabellen-Aggregate NICHT gestrichen** — sie sind für ein zeilenweises Modell
      keine Redundanz (siehe 1.1). In allen drei Dateien als Kommentarblock markiert,
      damit es beim nächsten Aufräumen nicht doch jemand tut.
- [x] Feature-Liste synchron:
      [MODEL_FEATURES:83-136](prefect/normalform_pipeline.py#L83-L136) ·
      [Pydantic-Schema:21-77](webservice/data_model_denormalization.py#L21-L77) ·
      Trainings-Drop-Block · [curl_tests/test_curl_predict_normalform.sh](curl_tests/test_curl_predict_normalform.sh)
      — in allen dreien sind die 21 **auskommentiert statt gelöscht**, mit der Identität
      als Kommentar (`"count",  # == table_row_count`), damit der Streichgrund am Code
      klebt. `evidently_service/build_monitoring_references.py` brauchte **keine
      Änderung**: es liest die Liste über
      [`request_features()`:127-132](evidently_service/build_monitoring_references.py#L127-L132)
      per `list(model.model_fields)` direkt aus dem Pydantic-Modell.
- [x] Gegengeprüft: README-`T3`-Zeilen == Pydantic-Felder == `MODEL_FEATURES`, alle 29,
      Mengen identisch; Feldreihenfolge == Trainings-`X` (wichtig für die MLflow-Signatur);
      Typen Pydantic ↔ Trainingsdaten ohne Abweichung
- [x] Feature-Tabelle in [README.md](README.md) aktualisiert — `T3`-Markierung bei allen 21
      entfernt (14 bleiben `T1, T2`, 7 werden `— (dropped, redundant)`), plus eine Tabelle
      der vier Streichgründe
- [ ] **Neu trainieren**, F1 gegenchecken (Erwartung 0,9939 — auf dem 29er-Satz bereits
      offline verifiziert). Bis dahin schlägt
      `test_model_schema_contract.py::test_feature_set_matches[denormalization_model]`
      fehl und listet exakt die 21 unter "model expects but schema lacks" — das ist der
      erwartete Zustand, kein Defekt.
- [ ] **Danach** `data/holdouts/nf_columns.csv` und
      `evidently_service/references/nf_columns.csv` neu bauen
      (`python evidently_service/build_monitoring_references.py --only nf_columns`,
      Model-Service muss laufen), dann
      `docker compose up -d --build evidently_service` — die Referenzen sind ins Image
      gebacken. Beide CSVs tragen derzeit noch das 50-Spalten-Layout und Scores des alten
      Modells.

### Phase 0.5 — 1NF-Detektor reparieren (implementiert 2026-08-04)

Das Feature behält Name und Typ, nur der berechnete Wert wird genauer. Da das Modell auf
Ground-Truth-Flags trainiert wurde, bewegt jede Verbesserung der Heuristik die Eingabe
**näher** an die Trainingssemantik — die Registry-Version bleibt unangetastet.
`table_ratio_1nf_violations` zieht automatisch nach, es ist der Mittelwert des Flags
([recompute_table_features](prefect/normalform_pipeline.py#L538)).

- [x] **Schwelle statt `> 0`**: Flag erst ab einem *Anteil* listenförmiger nicht-leerer
      Werte (`LIST_VALUE_MIN_SHARE = 0.8`). Das allein behebt den Einzelfall-Kipper.
- [x] **Verankertes Muster statt `LIKE '%,%'`**: der ganze Wert muss eine Liste kurzer
      Tokens sein. Satzzeichen (`.!?`) sind aus der Token-Klasse ausgeschlossen, das
      erledigt den Großteil der Prosa.
- [x] **Längenschranke**: `AVG(LENGTH(...)) <= 120`.
- [x] Implementiert als [`build_1nf_violation_expr()`:219](prefect/normalform_pipeline.py#L219),
      eingesetzt in `union_parts`. Konstanten:
      [`LIST_VALUE_PATTERN`:170](prefect/normalform_pipeline.py#L170) und die zwei Schwellen.
- [x] **f-String-Falle umgangen** — bestätigt: in einem `f"""..."""` wird `{1,40}`
      *stillschweigend* zum Tupel `(1, 40)` ausgewertet, das Regex wäre kaputt und Python
      meldet nichts. Gelöst durch Hochziehen des Musters in eine Modulkonstante (sauberer
      als `{{1,40}}` zu doppeln). Backslashes braucht das Regex bewusst keine.
- [x] **Namens-Veto als zweite Sicherung** —
      [`apply_freetext_veto()`:253](prefect/normalform_pipeline.py#L253), aufgerufen
      [vor `recompute_table_features`](prefect/normalform_pipeline.py#L677); danach wäre es
      zu spät, dort wird `table_ratio_1nf_violations` gemittelt. Zwingt nur das Flag auf 0,
      die Spalte bleibt in jeder anderen Statistik.
  - [x] Als Config-Datei: [prefect/nf_freetext_columns.json](prefect/nf_freetext_columns.json),
        mit `DEFAULT_FREETEXT_NAME_TOKENS` als Fallback, falls sie fehlt
  - [x] Substring-Match auf Kleinschreibung — verifiziert für `bemerkung_kunde`,
        `notes_intern`, `product_description`
  - [x] `exclude_column_names.txt` **nicht** repariert, sondern die Freitext-Gruppe sauber
        daraus extrahiert. Die Originaldatei ist weiterhin beim Kopieren beschädigt
        (`NotesKommentar`, `CreatedByGeaendertAm`, `Migration_FlagTemp`, `ImportIDBatchID`)
        — falls sie bleiben soll, gehört sie einmal aufgeräumt oder gelöscht.
  - [x] Nur die Gruppe **Freitexte** übernommen. Warum die Audit- und Temp-Gruppen draußen
        bleiben, steht als `_comment` in der JSON-Datei.
  - [x] Kurze Tokens (`text`, `info`, `version`) nicht aufgenommen — gegengeprüft, dass
        `context_text`, `info_id`, `version_number` **nicht** greifen
- [x] Regex gegen 9 Beispielwerte geprüft: Komma-/Semikolon-/Pipe-Listen greifen, Zahl,
      Einzelwert, Prosa mit Satzzeichen und langes erstes Token greifen nicht.
- [ ] **Bekannte Lücke, bewusst offen:** kurze Prosa ohne Satzzeichen (`"Danke, bis
      morgen"`) erfüllt das Muster. Abgefangen wird sie nur durch die Anteilsschwelle und
      das Namens-Veto. Nächster billiger Diskriminator wäre die **Trennzeichen-Dichte**
      (`rot,grün,blau` = 0,154 Trenner/Zeichen gegen 0,059 bei dem Beispiel).
- [ ] Schwellen (0,8 / 40 / 120) an den **Freitext-Ködern** aus Phase 2 kalibrieren —
      derzeit geratene Startwerte. Zweiter Fall zum Nachjustieren: eine `tags`-Spalte, bei
      der viele Zeilen nur *ein* Tag haben, erreicht die 0,8 nie (Einzelwerte matchen
      nicht). Falls das auftritt, den Nenner auf "Werte, die überhaupt ein Trennzeichen
      enthalten" umstellen.
- [ ] Detektor + Veto in ein **geteiltes Modul** (`nf_features.py`) ziehen. Derzeit liegen
      sie in `normalform_pipeline.py`; der Trainings-Extractor, mit dem geteilt werden
      müsste, fehlt ohnehin im Repo (siehe 1.7). Spätestens wenn er aus Phase 2 zurückkommt,
      muss er dieselbe Funktion importieren statt sie zu kopieren (siehe 1.6).
- [ ] Gegen echte Tabellen mit Kommentarspalten verifizieren, bevor es live geht — bislang
      nur gegen konstruierte Beispielwerte geprüft.

### Phase 1 — Messung ehrlich machen

- [ ] Leakage-Guard reparieren: entweder `table_ratio_1nf_violations` und
      `table_has_partial_dependency` mit droppen, oder die Warnung in der README
      korrigieren (aktuell suggeriert sie eine Entschärfung, die nicht stattfindet)
- [ ] Ehrliche Baseline dokumentieren: **0,7117** ohne die 7 definitionsnahen Features
      (auf dem reduzierten 29er-Satz nachgemessen; auf dem alten 50er-Satz waren es 0,7413,
      weil dort mehr Statistik-Features den Ausfall teilweise auffingen)
- [ ] Generator-Artefakt in 1.3 in der README als Limitierung benennen
- [ ] Split-Diskrepanz klären: Code gruppiert nach `table_name`
      ([Zeile 183](task_3/task_3_denormalization_train_and_register.py#L183)),
      README behauptet `database`
- [ ] Train/Serve-Skew aus 1.6 als Test absichern (Formel-Parität Training ↔ Pipeline)

### Phase 2 — Neue Trainingsdaten (Voraussetzung für alles Weitere)

Die drei offenen Baustellen — gemessenes `table_ratio_1nf_violations` statt Label,
Spaltenpaar-Features, Atomaritäts-Features — scheitern alle an derselben Ursache: es
wurden nur Aggregate persistiert, keine Rohwerte (Befund 1.7). Sie werden deshalb
gemeinsam von **einem** Extractor auf denselben generierten Daten bedient.

**Jedes abgeschriebene Feature bekommt einen gemessenen Nachfolger:**

| heute — vom Label abgeschrieben | künftig — aus den Daten gemessen |
| :--- | :--- |
| `is_this_col_violating_1nf` | `col_list_like_ratio`, `col_separator_density`, `col_distinct_token_ratio` |
| `table_ratio_1nf_violations` | `table_max_list_like_ratio`, `table_ratio_list_like_columns` |
| `is_composite_key_part`, `table_has_composite_pk` | Schlüsselkandidat aus UCC-Suche (**E1**) |
| `is_this_col_partial_dependency`, `table_has_partial_dependency` | `table_partial_fd_count` / `_ratio` / `_max_strength` |
| **— existiert nicht —** | `table_transitive_fd_count` / `_ratio` / `_max_strength` ← **2NF vs. 3NF** |

Die letzte Zeile ist der Kern: für 2NF-vs-3NF gibt es heute *kein* Feature, deshalb ist das
Generator-Artefakt aus 1.3 in die Lücke gesprungen.

**Die Architekturregel, die alles trägt:**

```
Generator ──▶ Iceberg-Tabellen (ROHDATEN)  ──▶  nf_features.py  ──▶  Features
     │                                              (sieht NUR die Tabelle)
     └──▶ Manifest (FDs, Label, Parameter)  ──────────────────────▶  y
              nur für Label + Validierung
```

Der Extractor darf das Manifest **nie** sehen — er bekommt ausschließlich die
materialisierte Tabelle, dieselbe Sicht wie in Produktion. Damit ist „Label abschreiben"
strukturell unmöglich statt nur verboten, und derselbe Code läuft in Training und
Pipeline, was den Skew aus 1.6 an der Wurzel erledigt.

#### 2a — Fundament (blockiert alles andere) · ≈ 0,5 d · **erledigt**

Drei Module, 52 Tests, keine Trino- oder MLflow-Abhängigkeit — das Label ist außerhalb
des laufenden Stacks reproduzierbar.

- [x] **NF-Label-Funktion** ([task_3/nf_labeling.py](task_3/nf_labeling.py)) — das Stück,
      dessen Fehlen die aktuellen Daten „frozen" macht. Deterministisch, ~80 Zeilen. Der
      Aufwand liegt in der Korrektheit, nicht im Umfang.

      ```python
      normal_form(attributes: set[str],
                  fds: list[tuple[frozenset[str], frozenset[str]]],
                  violates_1nf: bool) -> int   # 0..3
      ```

  - [x] `closure(X, fds)` — Attributhülle. Superschlüssel ⇔ `closure(X) == attributes`.
  - [x] `candidate_keys(...)` — **alle minimalen Superschlüssel, nicht einer.** Pruning:
        Attribute, die nur links vorkommen, sind in jedem Schlüssel; solche, die nur rechts
        vorkommen, in keinem. Nur der Rest wird kombinatorisch durchsucht. Ist dieser Rest
        größer als `MAX_SEARCHED_ATTRIBUTES` (20), bricht die Suche mit einer Meldung ab,
        statt 2²⁰ Teilmengen aufzuzählen — eine unpräzise FD-Deklaration soll auffallen,
        nicht hängen.
  - [x] `prime` = Vereinigung **aller** Kandidatenschlüssel. Der häufigste Fehler ist, hier
        nur einen Schlüssel zu betrachten.
  - [x] **1NF kommt nicht aus den FDs.** Atomarität ist eine Eigenschaft der Werte, keine
        Abhängigkeit — `violates_1nf` wird von außen gemessen hereingereicht und schlägt
        alles andere: `True` → 0, unabhängig von den FDs.
  - [x] **2NF**: kein Nicht-Prime-Attribut hängt von einer *echten Teilmenge* eines
        Kandidatenschlüssels ab. Geprüft über die Hülle, nicht über die deklarierten FDs —
        eine partielle Abhängigkeit kann transitiv entstehen (`A → C`, `C → D` unter
        Schlüssel `{A,B}`). Es genügen die **maximalen** echten Teilmengen `K \ {b}`: die
        Hülle ist monoton, jede kleinere Teilmenge wird von einer davon dominiert. Das macht
        aus `2^|K|` Teilmengen genau `|K|`.
  - [x] **3NF**: für jedes nichttriviale `X → A` gilt — `X` ist Superschlüssel **oder** `A`
        ist prime. Die zweite Hälfte wird regelmäßig vergessen; ohne sie labelt man
        3NF-Tabellen als 2NF.
  - [x] **Test-Orakel** ([test/test_task_3/test_nf_labeling.py](test/test_task_3/test_nf_labeling.py),
        9 Fälle, jeder Erwartungswert von Hand gerechnet): einfacher Schlüssel ·
        zusammengesetzter Schlüssel mit partieller Abhängigkeit · transitive Kette ·
        mehrere Kandidatenschlüssel · prime Attribut auf der rechten Seite
        (`{city,street} → zip`, `zip → city`: 3NF trotz Nicht-Superschlüssel links) ·
        gar keine FDs (alles prime → 3NF) · partiell **und** transitiv zugleich (→ 1, nicht
        2) · Ein-Spalten-Relation · `violates_1nf=True` auf demselben 3NF-Schema (→ 0)
  - [x] Wird von Generator (Label) **und** Validierungs-Harness (2c) importiert, nie vom
        Extractor.
  - [ ] **Bewusst festgelegte Konvention, kein Lehrsatz:** eine FD mit leerer linker Seite
        (`∅ → A`, konstante Spalte) zählt als partielle Abhängigkeit, weil ∅ echte Teilmenge
        jedes nichtleeren Schlüssels ist → Label 1. Steht als Test da, damit es sichtbar und
        änderbar ist. Falls der Generator absichtlich konstante Spalten erzeugt (Mandanten-ID
        in einer Ein-Mandanten-Tabelle), hier nachjustieren.
- [x] **Manifest-Tabelle** ([task_3/nf_manifest.py](task_3/nf_manifest.py)):
      `iceberg.nf_training.manifest`, `(database, schema, table_name, target_normal_form,
      violates_1nf, declared_fds, candidate_keys, generation_params, review_reason)` —
      letzte Spalte für die Prüf-Warteschlange aus **E1**, `load_manifest(only_review=True)`
      liest sie.
  - [x] **Zwei Spalten mehr als hier ursprünglich geplant**, beide für 2c: `violates_1nf`
        ist nicht aus den FDs rekonstruierbar, muss also gespeichert werden, sonst kann das
        Harness das Label nicht nachrechnen. `candidate_keys` ist zwar abgeleitet, aber
        genau das, was die UCC-Suche aus **E1** auf den Rohdaten reproduzieren muss — ohne
        die Spalte gäbe es nichts, wogegen verglichen wird.
  - [x] `encode_fds`/`decode_fds` kanonisch (sortiert), damit die deklarierten FDs exakt
        round-trippen — sonst rechnet 2c ein anderes Label aus als der Generator schrieb
        und gibt den Daten die Schuld.
  - [x] `build_manifest_row` nimmt **kein** `target_normal_form` entgegen. Das Label wird
        dort aus den FDs abgeleitet; ein Rezept kann keine Normalform behaupten, die seine
        Abhängigkeiten nicht hergeben.
- [x] **Generator-Skelett** ([task_3/nf_generator.py](task_3/nf_generator.py)): schreibt
      echte Iceberg-Tabellen nach `iceberg.nf_training.*`, Rohdaten persistiert.
      `python task_3/nf_generator.py --dry-run` rechnet die Labels ohne Trino aus.
  - [x] `materialise` droppt vor dem Schreiben statt anzuhängen — eine halb überschriebene
        Tabelle verletzt die FDs, die ihr Label behauptet, und niemand stromabwärts könnte
        das bemerken.
  - [x] `_verify_columns` vergleicht die materialisierten Spalten gegen `attributes`. Der
        Fehler, gegen den das schützt, ist lautlos: eine umbenannte Spalte lässt ihre FD nie
        feuern, die Tabelle wirkt dadurch **besser** normalisiert — das Label ist in genau
        der Richtung falsch, die niemand hinterfragt.
  - [x] Labels werden für **alle** Specs berechnet, bevor die erste Tabelle geschrieben
        wird. Eine kaputte FD-Deklaration scheitert am ersten Spec, nicht auf halbem Weg
        durch einen Rebuild.
  - [x] `RECIPES` gefüllt — siehe 2b. `nf_generator.load_specs()` lädt sie verzögert
        (Zyklus: die Rezepte brauchen `TableSpec` aus dem Generator).

#### 2b — Dünner vertikaler Schnitt · **erledigt** (statt 2–3 d veranschlagt)

**132 Tabellen in `iceberg.nf_training`**, alle vier Klassen, alle drei Feature-Familien,
einmal end-to-end gemessen. Ziel war die Mechanik, nicht das Modell — und die Mechanik hat
drei Fehler zutage gefördert, die auf 5.000 Tabellen teuer geworden wären.

| Datei | Inhalt |
| :--- | :--- |
| [task_3/nf_recipes.py](task_3/nf_recipes.py) | Basiskatalog, FD-Algebra, 21 Join-Rezepte, 5 Listen- und 1 Wiederholgruppen-Rezept, Variationsachsen |
| [prefect/nf_features.py](prefect/nf_features.py) | Die drei Feature-Familien, 32 Features, geteilt von Trainingsbau **und** Pipeline |
| [task_3/nf_build_training_set.py](task_3/nf_build_training_set.py) | Der einzige Ort, an dem X und y zusammenkommen → `data/nf_training.csv` |

- [x] TPC-H über den vorhandenen `tpch`-Katalog. Zwei Stufen (`tiny` mit 2.000,
      `sf1` mit 20.000 Zeilen) × zwei Namensvarianten (echt / obfuskiert) × 33 Rezepte.
- [x] **FD-Herleitung durchs Join** — `FDs(fakt) ∪ FDs(dim) ∪ {fk → dim-Attribute}`,
      programmatisch statt per Hand. Kandidatenschlüssel werden **nirgends deklariert**,
      sie fallen aus der FD-Menge heraus — genau das Raten, das zu 1.4 geführt hat.
  - [x] Projektion über die Hülle, nicht über Filtern der FD-Liste: bei `A → B`, `B → C`
        bleibt nach dem Streichen von `B` die Abhängigkeit `A → C` bestehen. Beide FDs
        wegzuwerfen ließe die Tabelle besser normalisiert aussehen als sie ist.
  - [x] **Die FDs wurden gemessen, nicht angenommen** — vollständiger Ein-Attribut-Scan auf
        `tpch.tiny`, jeder Fund gegen `sf1` gegengeprüft (Tabelle unten).
  - [x] Nicht projiziert: `orders.shippriority` (konstant → Abhängigkeit von der leeren
        Menge → zöge jede Tabelle auf 1NF), `customer/supplier.name|address|phone` und
        `part.name` (zufällig eindeutig; `part.name` ist auf `tiny` eindeutig, auf `sf1`
        nicht — das Label hinge am Scale Factor).
- [x] 0NF durch Injektion, Komma / Semikolon / Pipe **und** Wiederholgruppen — jeweils mit
      **abgeglichener Kontrolltabelle**: gleiche Query, gleiche Granularität, gleiche
      Zeilen- und Spaltenzahl, nur die letzte Spalte atomar statt Liste. Damit unterscheidet
      sich das Paar in **genau einer** Eigenschaft, und das ist die, die die
      Atomaritäts-Features tragen müssen. Eine Injektion sitzt bewusst auf einer 2NF-Basis,
      damit 0NF nicht mit der Normalform der Elterntabelle korreliert.
- [x] **Freitext-Köder**: die echten TPC-H-`comment`-Spalten, verteilt über 3NF-, 2NF- und
      1NF-Tabellen. Ehrlicher als synthetische: 16 % der Bestellkommentare enthalten ein
      Komma, aber nur 3 % erfüllen das Listenmuster.
- [x] **Tabellennamen ohne Label-Hinweis** (`nf_017_tiny_real`). Der Extractor sieht
      Tabellen- und Spaltennamen — `tbl_dirty_1nf_7` wäre ein Label, das er direkt ablesen
      kann. Das Rezept steht stattdessen in `generation_params`.
- [x] `nf_features.py` mit allen drei Familien. Zwei gebündelte Queries pro Tabelle statt
      einer Schleife über Spaltenpaare: **eine** paarweise Distinct-Matrix (`C(n,2)`)
      beantwortet UCC-Suche bis Level 2 *und* jede Ein-Attribut-Abhängigkeit.
- [x] Der Kürzungsvorschlag wurde teilweise genutzt: `col_distinct_token_ratio` weggelassen
      (bräuchte `split`+`unnest`), dafür `col_mean_token_length` — siehe Befund unten.

**Die gemessenen TPC-H-Abhängigkeiten:**

| Kandidat | Urteil | Grund |
| :--- | :--- | :--- |
| `part.brand → mfgr` | **echt** | brand kodiert den Hersteller |
| `lineitem.shipdate → linestatus` | **echt** | linestatus ist Funktion des Versanddatums |
| `customer.acctbal → mktsegment` | Artefakt | 1.499 verschiedene Werte auf 1.500 Zeilen |
| `orders.totalprice → orderstatus` | Artefakt | dito, bricht auf sf1 |

Folge: `lineitem` allein ist **2NF**, nicht 3NF, und `part` mit `mfgr` ebenfalls. Beide
werden erst 3NF, wenn das abhängige Attribut wegprojiziert wird — daher kommen die
3NF-Tabellen mit zusammengesetztem Schlüssel, die verhindern, dass „zusammengesetzter
Schlüssel" das Label vorhersagt.

**Drei Fehler, die die Messung gefunden hat:**

- [x] **Entartete Stichprobe.** Der `LIMIT` sortierte nach dem *alphabetisch* sortierten
      Schlüssel. Bei `lineitem` heißt das `ORDER BY linenumber, orderkey` — die ersten 2.000
      Zeilen haben alle `linenumber = 1`. Die Spalte wird konstant, `orderkey` eindeutig,
      und der zusammengesetzte Schlüssel, für den das Rezept existiert, ist aus den Daten
      verschwunden. Die Stichprobe muss die Schlüsselstruktur erhalten, nicht nur die
      Zeilenzahl.
- [x] **UCC-Suche hörte beim ersten Treffer auf.** Eine Freitext-Kommentarspalte ist
      eindeutig und damit eine völlig gültige Level-1-UCC. Dort aufzuhören verdeckt den
      echten Schlüssel `{orderkey, linenumber}` auf Level 2 — und damit die partielle
      Abhängigkeit, die 2NF definiert. Jede 1NF-Tabelle sah dann aus wie eine 2NF-Tabelle.
- [x] **Längenschwelle am falschen Maß.** `nf_026` ist 0NF, wurde aber nicht erkannt: 80
      Teilenummern pro Lieferant ergeben 355 Zeichen, das überschreitet die
      Prosa-Schwelle von 120. Was Prosa von einer Liste trennt, ist die **Länge der Stücke
      zwischen den Trennzeichen**, nicht die Gesamtlänge — eine echte Liste bleibt pro
      Element kurz, egal wie viele Elemente sie hat. `nf_features` misst jetzt
      `LIST_TOKEN_MAX_MEAN_LENGTH = 20` pro Token.

**Ergebnisse (gemessen, nicht geschätzt):**

| Prüfung | Ergebnis |
| :--- | :--- |
| Klassenverteilung (Tabellen) | 0NF 24 · 1NF 20 · 2NF 40 · 3NF 48 |
| **Abnahmekriterium 1** — kein Feature allein > ~0,6 F1 | **erfüllt**, Maximum 0,53 (`table_candidate_key_count`) |
| Atomaritäts-Regel `Liste ∨ Wiederholgruppe` | **22/24 0NF erkannt, 0 Fehlalarme auf 108 Nicht-0NF-Tabellen** |
| Freitext-Köder lösen den Detektor aus | **nein** (Kriterium 3 erfüllt) |
| F1, Split nach `table_name` | 0,9259 (Spalten) · 0,9321 (Tabellen, Mehrheitsentscheid) |
| F1, Split nach `recipe_id` | 0,4328 (Spalten) · 0,5459 (Tabellen) |

Die zwei nicht erkannten 0NF-Tabellen sind die **obfuskierten** Wiederholgruppen: ohne
sprechende Spaltennamen ist eine Wiederholgruppe grundsätzlich nicht sichtbar. Kein Fehler,
sondern eine Grenze — und der Beleg dafür, dass die Atomaritätsfamilie nicht aus
Wertstatistik allein bestehen kann.

Die Lücke zwischen 0,93 und 0,55 ist die ehrliche Zahl: nach `recipe_id` gruppiert testet
man auf Tabellenformen, die im Training nie vorkamen — und 27 Formen sind dafür schlicht zu
wenig. Genau das ist die Anforderung aus **2d** („300+ Quellschemata, die Zahl der Schemata
ist die härtere Anforderung"). Zum Vergleich: das alte Modell erreichte unter dem
`table_name`-Split 0,9939, aber mit Features, von denen eines allein 90,6 % Gain trug.

**Offen aus 2b:**

- [ ] **Die Lehrbuch-Definition von 2NF/3NF überlebt entdeckte Schlüssel nicht.** Sie
      verlangt, dass das *abhängige* Attribut nicht-prim ist. Das ist korrekt, wenn die
      Kandidatenschlüssel bekannt sind, und unbrauchbar, wenn sie entdeckt werden: eine
      near-unique Spalte (ein Preis, ein Kommentar) bildet mit fast jeder anderen Spalte
      eine gültige UCC, die Vereinigung aller UCCs überdeckt die ganze Tabelle, und jedes
      Attribut kommt als prim heraus. Gemessen: `table_prime_ratio` = 1,00 bei 20.000
      Zeilen, beide strengen Zähler identisch null. `nf_features` liefert deshalb **beide**
      Varianten (`table_partial_fd_count` ohne, `table_strict_partial_fd_count` mit
      Prim-Bedingung). Welche trägt, muss das Modell auf mehr Daten entscheiden — nicht
      wir per Definition.
- [ ] Schwellen an den Ködern feinjustieren: die echten TPC-H-Kommentare liegen mit 3–5 %
      Listenanteil weit unter der 0,8-Schwelle. Ein **harter** Köder (zwei kurze
      Prosafragmente mit „, " verbunden) fehlt noch und ist der Fall, an dem
      `col_mean_token_length` sich beweisen müsste.
- [ ] `LIST_VALUE_MAX_MEAN_LENGTH = 120` steht noch in
      [normalform_pipeline.py](prefect/normalform_pipeline.py) — bewusst nicht angefasst,
      weil das ändern würde, was das **aktuell registrierte** Modell bekommt. Gehört zu
      Phase 4.

#### 2c — Validierungs-Harness (**bevor** skaliert wird) · ≈ 1 d

Läuft ab hier bei jedem Generierungslauf mit.

- [ ] **Label-Verifikation**: FD-Discovery auf der materialisierten Tabelle; die entdeckte
      Hülle muss zu den deklarierten FDs äquivalent sein. Sonst erfüllen die Daten *mehr*
      FDs als deklariert und das Label ist falsch.
- [ ] **Signatur-Test**: Decision Stump auf jedes Feature einzeln. Erreicht eins > ~0,6 F1,
      steckt eine Generator-Signatur drin. Der heutige Satz fällt hier krachend durch
      (`table_avg_unique_ratio`).
- [ ] **Köder-Test**: Precision/Recall des 1NF-Detektors **getrennt** berichten, nicht im
      4-Klassen-F1 verschmieren.
- [ ] **Label-Rauschen**: keine Tabelle darf 0NF gelabelt sein ohne verletzende Spalte —
      die 6 `tbl_dirty_1nf_*` im aktuellen Satz sind genau dieser Fehler.

#### 2d — Skalieren

- [ ] Weitere Quellen; Zielgröße **3.000–5.000 Tabellen aus 300+ Quellschemata** — die Zahl
      der *Schemata* ist die härtere Anforderung, danach gruppiert der Split
- [ ] Variationsachsen: Zeilen 500/5k/50k · Spalten 5/10/20/40 · NULL-Rate 0/5/30 % ·
      Namen real/Konvention/obfuskiert je ⅓ · Schlüsseltyp Surrogat/natürlich/Composite-2/-3 ·
      Dirty Data (Tippfehler, Case, Whitespace)
- [ ] Northwind/Sakila/Chinook als **externes Holdout**, nie im Training

#### Abnahmekriterien für Phase 2

1. kein einzelnes Feature erreicht allein > ~0,6 F1
2. `table_avg_unique_ratio` / `table_std_unique_ratio` trennen 2NF und 3NF **nicht**
3. die Freitext-Köder lösen den 1NF-Detektor nicht aus
4. entdeckte FDs ≡ deklarierte FDs auf allen generierten Tabellen
5. das **externe** Holdout liegt in vertretbarem Abstand zum internen Test

Punkt 5 ist der eigentliche Test. Alles davor kann man sich schönmachen.

### Phase 3 — Spaltenpaar-Features

- [ ] Grundsatzentscheidung: exakte FDs auf vollständigen Daten **determinieren** die
      Normalform — dann braucht es kein ML. Der ML-Nutzen liegt bei approximativen /
      gesampelten FDs auf großen Tabellen. Vor der Implementierung klären, welchen der
      beiden Fälle ihr baut (passt zu "unsupervised statt supervised" und
      "ranking with probabilities" in [TASK_3.md](TASK_3.md))
- [ ] Feature-Definitionen festlegen (Vorschlag siehe Abschnitt 3)
- [ ] **Einmal** implementieren, von Trainings-Extractor und Prefect-Pipeline geteilt
      (heute dupliziert → Ursache des Skews aus 1.6)
- [ ] Kostenbudget: `approx_distinct` ab N Zeilen, Obergrenze für Spaltenzahl,
      Timeout-Verhalten
- [ ] Triviale FDs herausfiltern (Determinante unique → FD immer erfüllt)

### Phase 4 — Modell und Serving

- [ ] Neue Signatur → neue Registry-Version; ein registriertes Modell nimmt keine
      nachgeschobenen Features an
- [ ] Prüfen, ob das Modell auf **Tabellenebene** trainiert wird statt spaltenweise +
      Majority Vote ([aggregate_to_table](prefect/normalform_pipeline.py#L769-L800)
      würde entfallen)
- [ ] Evidently-Referenzen und Grafana-Drift-Dashboards neu bauen
- [ ] CI-Qualitätsgate gegen die **ehrliche** Baseline aus Phase 1, nicht gegen 0,99

---

## 3. Vorschlag: Spaltenpaar-Features

### 3.1 Basis-Messgrößen

Pro Tabelle **eine** Query mit `n·(n−1)/2` Aggregat-Ausdrücken — nicht eine Query pro
Paar. Bei 15 Spalten sind das 105 Ausdrücke, das ist unkritisch.

```sql
SELECT
    COUNT(*)                                    AS n,
    COUNT(DISTINCT a)                           AS d_a,
    COUNT(DISTINCT b)                           AS d_b,
    COUNT(DISTINCT CONCAT(
        CAST(a AS VARCHAR), CHR(31), CAST(b AS VARCHAR)))  AS d_ab
FROM t
```

`CHR(31)` (Unit Separator) als Trennzeichen, weil er in Nutzdaten praktisch nie vorkommt —
`COUNT(DISTINCT ROW(a,b))` scheitert in Trino an nicht-vergleichbaren Typen. Für große
Tabellen `approx_distinct` statt `COUNT(DISTINCT ...)`. Die Session-Property
`distinct_aggregations_strategy = single_step` ist in
[get_trino_engine](prefect/normalform_pipeline.py#L277-L287) bereits gesetzt.

### 3.2 Paar-Ebene (Zwischenergebnis, geht nicht direkt ins Modell)

| Feature | Formel | Bedeutung |
| :--- | :--- | :--- |
| `fd_a_to_b` | `d_ab == d_a` | b ist funktional abhängig von a |
| `fd_strength_a_to_b` | `d_a / d_ab` ∈ (0,1] | weiches Maß, robuster als das Boolean |
| `g3_error_a_to_b` | `1 − d_a / d_ab` | Standardmaß für approximative FDs |
| `is_trivial_fd` | `d_a == n` | Determinante unique → **Filterkriterium, kein Feature** |
| `is_equivalent_pair` | `fd_a_to_b AND fd_b_to_a` | 1:1-Beziehung, klassischer Redundanz-Indikator |
| `pair_cardinality_ratio` | `d_a / d_b` | Richtungsindikator |

**Wichtig:** FDs sind gerichtet — für jedes ungeordnete Paar beide Richtungen prüfen.

### 3.3 Tabellen-Ebene (das, was das Modell bekommt)

Aggregiert über alle **nicht-trivialen** Paare. Die ersten beiden Blöcke bilden direkt die
2NF- und 3NF-Definition ab und würden die heutigen Heuristiken
`is_this_col_partial_dependency` / `table_has_partial_dependency` ersetzen:

| Feature | Formel | Zielt auf |
| :--- | :--- | :--- |
| `table_partial_fd_count` | FDs von einer **echten Teilmenge** des Composite Key auf eine Nicht-Schlüsselspalte | **2NF** |
| `table_partial_fd_ratio` | / Anzahl Nicht-Schlüsselspalten | 2NF |
| `table_max_partial_fd_strength` | stärkste solche FD | 2NF, weich |
| `table_transitive_fd_count` | FDs zwischen zwei **Nicht-Schlüsselspalten** | **3NF** |
| `table_transitive_fd_ratio` | / Anzahl solcher Paare | 3NF |
| `table_max_transitive_fd_strength` | stärkste Nicht-Schlüssel→Nicht-Schlüssel-FD | 3NF, weich |
| `table_ratio_determined_columns` | Anteil Spalten, die von mind. einer Nicht-Schlüsselspalte bestimmt werden | 3NF |
| `table_fd_out_degree_max` | max. Anzahl Spalten, die eine einzelne Spalte determiniert | Determinanten-Stärke |
| `table_fd_out_degree_mean` | Mittelwert davon | Determinanten-Dichte |
| `table_equivalent_pair_count` | Anzahl 1:1-Paare | Denormalisierung allgemein |
| `table_fd_ratio_nontrivial` | nicht-triviale FDs / alle geordneten Paare | Gesamtdichte |
| `table_avg_fd_strength` | Mittelwert `fd_strength` über alle Paare | Gesamtdichte, weich |
| `table_std_fd_strength` | Streuung | trennt "wenige starke" von "viele schwache" FDs |
| `table_has_no_ucc_le3` | keine unique Spaltenkombination bis Level 3 (siehe **E1**) | Duplikatzeilen, Verletzung des Relationenmodells — und der Marker, ab dem die FD-Features `unknown` sind |

### 3.4 Spalten-Ebene (nur falls das Modell zeilenweise bleibt)

| Feature | Bedeutung |
| :--- | :--- |
| `col_fd_out_degree` | wie viele andere Spalten determiniert diese Spalte |
| `col_fd_in_degree` | von wie vielen anderen wird sie determiniert |
| `col_is_determined_by_non_key` | 0/1 → direkte 3NF-Verletzung an dieser Spalte |
| `col_is_determined_by_key_subset` | 0/1 → direkte 2NF-Verletzung an dieser Spalte |
| `col_max_fd_strength_in` / `_out` | weiche Varianten |

### 3.5 Fallstricke

- **Triviale FDs dominieren kleine Tabellen.** Determinante unique ⇒ FD immer erfüllt.
  Bei den aktuellen Daten trifft das auf 61,4 % der Spalten zu. Ohne den
  `is_trivial_fd`-Filter misst ihr nur Uniqueness in neuer Verpackung.
- **Kleine Tabellen erzeugen Zufalls-FDs.** Bei ~9 Zeilen ist fast jede FD Zufall. Eine
  Mindestzeilenzahl gehört ins Feature (`table_row_count` bleibt als Kontext wichtig) —
  oder die FD-Features werden unterhalb einer Schwelle als `NULL`/`unknown` markiert.
- **2NF-Features brauchen einen Schlüsselkandidaten.** Entschieden in **E1**: UCC-Suche
  bis Level 3, voller Scan, kein Treffer → melden und weiter. Wichtig dabei: der Schlüssel
  steht im Training zwar im Manifest, der Extractor darf ihn aber nicht sehen — sonst ist
  er wieder vom Label abgeschrieben. Training und Inferenz müssen denselben Weg gehen.
- **Kosten.** `COUNT(DISTINCT)` über viele Paare ist teuer. Bei n Spalten wächst die
  Anzahl der Aggregate quadratisch — ab ~30 Spalten (465 Paare) auf `approx_distinct`
  und/oder Sampling ausweichen.
- **Dieselbe Leakage-Falle wie heute.** Diese Features sind wieder nah an der Definition.
  Der Unterschied: sie werden **aus den Daten berechnet** statt vom Label kopiert — das ist
  legitim. Aber die Konsequenz aus Phase 3, Punkt 1 ernst nehmen: bei exakten FDs auf
  vollständigen Daten ist das Ergebnis ein Regelwerk, kein Klassifikator.
