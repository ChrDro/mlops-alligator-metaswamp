#!/bin/bash
#
# Payload matches the 37 features of fk_model, in signature order.
#
# Note this differs from the pk/cpk/cfk fixtures in two ways:
#
# 1. The fk training script selects features by DROPPING columns, so count,
#    table_has_no_single_pk_candidate, table_id_named_column_count,
#    is_least_null_in_table and unique_ratio_relative_to_max are deliberately absent.
#
# 2. The seven cross-table features (n_other_tables_with_same_column_name onwards) exist
#    only for this model. They describe the OTHER tables in the same schema, so unlike
#    every other field here they cannot be read off the column being predicted -
#    prefect/pk_fk_pipeline.py computes them against the whole schema.
#
# The values below describe a plausible child column: a `student_id` in some table that is
# not `student`, non-unique here, while a sibling table has a unique column of the same
# name. That is the shape the model learnt to call a foreign key.
#
# See webservice/data_model_fk.py and src/cross_table_features.py.

curl -X POST http://localhost:8080/predict_fk \
     -H "Content-Type: application/json" \
     -d '{
       "number_unique_values": 2,
       "null_count": 0,
       "null_ratio": 0.0,
       "is_unique": 0,
       "ordinal_position": 3,
       "unique_ratio": 0.4,
       "is_first_column": 0,
       "relative_ordinal_position": 1.0,
       "is_first_unique_column": 0,
       "table_column_count": 3,
       "table_unique_column_count": 3,
       "table_row_count": 5,
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
       "n_other_tables_with_same_column_name": 2,
       "name_unique_in_other_table": 1,
       "is_non_unique_and_name_unique_elsewhere": 1,
       "name_references_other_table_exact": 1,
       "name_references_other_table_fuzzy": 1,
       "name_ends_with_id_no_underscore": 0,
       "name_ends_with_code_or_num": 0,
       "column_type_boolean": false,
       "column_type_date": false,
       "column_type_decimal": false,
       "column_type_double": false,
       "column_type_integer": true,
       "column_type_varchar": false
     }'
