#!/bin/bash

# Two text fields rather than column statistics: the subject-area model reads the
# table name and its column names. `columns` is the comma-separated column list, the
# same shape the training script builds per table.
curl -X POST http://localhost:8080/predict_subject_area \
     -H "Content-Type: application/json" \
     -d '{
        "table_name": "customer_rating",
        "columns": "rating_id, customer_id, order_id, score, comment, created_at"
    }'
