#!/bin/bash

# 53 fields: 49 measured features + 4 column_type dummies. The set was replaced in
# Phase 2 of TASK_3_PLAN.md (the previous payload sent 29 metadata heuristics, two of
# which were copies of the label itself) and extended 2026-08-07 with the anykey/pair
# features (table_*_anykey, table_pair_*) after the Willibald period-1 mispredictions.
#
# GENERATED from a real row of data/nf_training.csv - the first column of the 3NF table
# "syn_0003". Not hand-written values: an invented payload can be internally
# inconsistent (a key size that contradicts the candidate-key count), and the model would
# then be asked about a table that cannot exist. Expected answer: "prediction": 3.
curl -X POST http://localhost:8080/predict_normalform \
     -H "Content-Type: application/json" \
     -d '{

        "ordinal_position": 1,
        "col_unique_ratio": 1.0,
        "col_null_ratio": 0.0,
        "col_list_like_ratio": 0.0,
        "col_mean_token_length": 0.0,
        "col_mean_token_count": 1.0,
        "col_separator_density": 0.0,
        "col_is_freetext_name": 0,
        "col_violates_1nf": 0,
        "col_in_repeating_group": 0,
        "col_in_candidate_key": 1,
        "col_is_prime": 1,
        "col_determines_count": 0,
        "col_depends_on_count": 0,
        "table_row_count": 60,
        "table_sampled": 0,
        "table_sample_ratio": 1.0,
        "table_search_truncated": 0,
        "table_pair_coverage": 1.0,
        "table_ucc_search_coverage": 1.0,
        "table_column_count": 5,
        "table_max_list_like_ratio": 0.0,
        "table_ratio_list_like_columns": 0.0,
        "table_repeating_group_ratio": 0.0,
        "table_constant_column_ratio": 0.0,
        "table_has_no_ucc_le3": 0,
        "table_candidate_key_count": 1,
        "table_key_size": 1,
        "table_composite_key_count": 0,
        "table_prime_ratio": 0.2,
        "table_fd_count": 0,
        "table_fd_ratio": 0.0,
        "table_partial_fd_count": 0,
        "table_partial_fd_ratio": 0.0,
        "table_transitive_fd_count": 0,
        "table_transitive_fd_ratio": 0.0,
        "table_strict_partial_fd_count": 0,
        "table_strict_transitive_fd_count": 0,
        "table_near_fd_count": 0,
        "table_near_fd_ratio": 0.0,
        "table_max_near_fd_strength": 0.0,
        "table_near_ucc_count": 0,
        "table_has_near_key_but_no_key": 0,
        "table_partial_fd_count_anykey": 0,
        "table_transitive_fd_count_anykey": 0,
        "table_pair_fd_count": 0,
        "table_pair_partial_fd_count": 0,
        "table_pair_transitive_fd_count": 0,
        "table_pair_fd_search_coverage": 1.0,
        "column_type_date": false,
        "column_type_double": false,
        "column_type_integer": false,
        "column_type_varchar": false
     }'
