#!/bin/bash
#
# Payload matches the 30 features of fk_model, in signature order.
#
# Note this differs from the pk/cpk/cfk fixtures: the fk training script selects
# features by DROPPING columns, so count, table_has_no_single_pk_candidate,
# table_id_named_column_count, is_least_null_in_table and unique_ratio_relative_to_max
# are deliberately absent here. See webservice/data_model_fk.py.

curl -X POST http://localhost:8080/predict_fk \
     -H "Content-Type: application/json" \
     -d '{
       "number_unique_values": 2,
       "null_count": 0,
       "null_ratio": 0.0,
       "is_unique": 1,
       "ordinal_position": 3,
       "unique_ratio": 1.0,
       "is_first_column": 0,
       "relative_ordinal_position": 1.0,
       "is_first_unique_column": 0,
       "table_column_count": 3,
       "table_unique_column_count": 3,
       "table_row_count": 2,
       "table_has_unique_column": 1,
       "table_near_unique_column_count": 3,
       "table_non_null_column_count": 3,
       "table_max_unique_ratio": 1.0,
       "unique_ratio_rank": 3,
       "null_ratio_rank": 3,
       "name_ends_with_id": 1,
       "name_contains_key": 0,
       "name_contains_table_name": 1,
       "name_is_singular_table_id": 0,
       "name_length": 18,
       "table_integer_column_count": 1,
       "column_type_boolean": false,
       "column_type_date": false,
       "column_type_decimal": false,
       "column_type_double": false,
       "column_type_integer": true,
       "column_type_varchar": false
     }'
