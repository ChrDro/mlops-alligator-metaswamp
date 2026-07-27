#!/bin/bash
# Call this right after loading data into the source schema. The service returns
# 202 immediately and starts change detection in the background
# 503 means the Prefect API was unreachable.

curl -X POST http://localhost:8080/events/new-data \
     -H "Content-Type: application/json" \
     -d '{
        "schema": "new_predict_data",
        "note": "manual smoke test"
     }'
