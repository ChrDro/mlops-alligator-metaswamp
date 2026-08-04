#!/bin/bash

# 29 features (Phase 0 of TASK_3_PLAN.md cut 21 redundant ones from the original 50).
# Values unchanged from the previous payload, so the two are directly comparable.
curl -X POST http://localhost:8080/predict_normalform \
     -H "Content-Type: application/json" \
     -d '{
        "unique_ratio": 1.0,
        "relative_ordinal_position": 0.14285714285714285,
        "is_first_unique_column": 1,
        "table_column_count": 7,
        "table_row_count": 5,
        "table_near_unique_column_count": 6,
        "table_id_named_column_count": 1,
        "table_max_unique_ratio": 1.0,
        "table_integer_column_count": 1,
        "unique_ratio_rank": 1,
        "name_ends_with_id": 1,
        "name_length": 8,
        "table_avg_unique_ratio": 0.9714,
        "table_ratio_of_pk_candidates": 0.8571,
        "is_this_col_violating_1nf": 0,
        "is_composite_key_part": 0,
        "table_has_composite_pk": 0,
        "is_this_col_partial_dependency": 0,
        "table_ratio_composite_key_cols": 0.0,
        "table_ratio_1nf_violations": 0.4286,
        "table_std_unique_ratio": 0.0756,
        "table_has_partial_dependency": 0,
        "column_type_char": false,
        "column_type_date": false,
        "column_type_decimal": false,
        "column_type_double": false,
        "column_type_integer": true,
        "column_type_timestamp": false,
        "column_type_varchar": false
    }'
