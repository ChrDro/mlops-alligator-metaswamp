# mlops-alligator-metaswamp

## Project Overview

This project aims at predicting and suggesting PK and FK candidates as well as possible domains of database objects by using database metadata and statistics. The data used is publicly available data from various sources like kaggle, tpc, microsoft and mysql.

The goal is to give business users a suggestion of possible primary and foreign keys to decide on. This can be used for future datavault implementations to determine the business keys for Hub, Sat and Link.

## Environment

### **`macOS`** type the following commands :

- Install the virtual environment and the required packages by following commands:

  ```BASH
  pyenv local 3.11.3
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

### **`WindowsOS`** type the following commands :

- Install the virtual environment and the required packages by following commands.

  For `PowerShell` CLI :

  ```PowerShell
  pyenv local 3.11.3
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```
  For `Git-Bash` CLI :

  ```
  pyenv local 3.11.3
  python -m venv .venv
  source .venv/Scripts/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Metadata and statistics overview

`—` = present in the data but not used as a training feature

| Variable | Description | Example values | Used in |
| :--- | :--- | :--- | :--- |
| `database` | Name of the given database the table is saved in | "Formula 1" | identifier |
| `schema` | Name of the given schema the table is saved in | "Season 2025" | identifier |
| `table_name` | Name of the database table | "Drivers" | identifier |
| `column_name` | Name of the column in database table | "Age" | identifier |
| `column_type` | Raw data type of the column (one-hot encoded for training, see below) | "int" | — |
| `min` | Minimum value of column | "18" | — |
| `max` | Maximum value of column | "44" | — |
| `number_of_unique_values` | Number of rows that have unique values | "23" | T1, T2, T3 |
| `count` | Number of rows in the table | "25" | T1, T2, T3 |
| `null_count` | Raw count of missing (null) values in the column | "0" | T1, T2, T3 |
| `null_ratio` | Ratio of null values (null_count / count) | "0.0" | T1, T2, T3 |
| `is_unique` | Column values are fully unique | 0 or 1 | T1, T2, T3 |
| `ordinal_position` | Position of column in the table (1-based) | "2" | T1, T2, T3 |
| `unique_ratio` | Ratio of unique values (number_unique_values / count) | "0.177" | T1, T2, T3 |
| `is_unique_non_null` | Column is unique when excluding nulls | 0 or 1 | — |
| `is_non_null` | Column has no null values | 0 or 1 | T1, T2, T3 |
| `is_first_column` | Column is the first column in the table | 0 or 1 | T1, T2, T3 |
| `relative_ordinal_position` | Ordinal position divided by table_column_count | "0.222" | T1, T2, T3 |
| `is_first_unique_column` | Column is the first unique column in the table | 0 or 1 | T1, T2, T3 |
| `table_column_count` | Total number of columns in the table | "9" | T1, T2, T3 |
| `table_unique_column_count` | Number of fully unique columns in the table | "0" | T1, T2, T3 |
| `table_row_count` | Number of rows in the table | "768" | T1, T2, T3 |
| `other_unique_columns_in_table` | Count of other unique columns in the table | "0" | T1, T2, T3 |
| `table_has_unique_column` | Table has at least one unique column | 0 or 1 | T1, T2, T3 |
| `table_has_no_single_pk_candidate` | No single-column PK candidate exists in the table | 0 or 1 | T1, T2, T3 |
| `table_near_unique_column_count` | Number of near-unique columns in the table | "0" | T1, T2, T3 |
| `table_id_named_column_count` | Number of columns with an ID-like name in the table | "0" | T1, T2, T3 |
| `table_non_null_column_count` | Number of columns with no nulls in the table | "9" | T1, T2, T3 |
| `table_max_unique_ratio` | Highest unique_ratio among all columns in the table | "0.671" | T1, T2, T3 |
| `table_integer_column_count` | Number of integer-typed columns in the table | "7" | T1, T2, T3 |
| `unique_ratio_rank` | Rank of this column's unique_ratio within the table | "4" | T1, T2, T3 |
| `null_ratio_rank` | Rank of this column's null_ratio within the table | "2" | T1, T2, T3 |
| `is_most_unique_in_table` | Column has the highest unique_ratio in the table | 0 or 1 | — |
| `is_least_null_in_table` | Column has the lowest null_ratio in the table | 0 or 1 | T1, T2, T3 |
| `unique_ratio_relative_to_max` | unique_ratio divided by table_max_unique_ratio | "0.264" | T1, T2, T3 |
| `other_near_unique_columns_in_table` | Count of other near-unique columns in the table | "0" | T1, T2, T3 |
| `name_is_id` | Column name is exactly "id" | 0 or 1 | — |
| `name_ends_with_id` | Column name ends with "id" | 0 or 1 | T1, T2, T3 |
| `name_starts_with_id` | Column name starts with "id" | 0 or 1 | — |
| `name_contains_key` | Column name contains "key" | 0 or 1 | T1, T2, T3 |
| `name_contains_uuid` | Column name contains "uuid" | 0 or 1 | — |
| `name_contains_table_name` | Column name contains the table name | 0 or 1 | T1, T2, T3 |
| `name_is_singular_table_id` | Column name matches singular table name + "id" | 0 or 1 | T1, T2, T3 |
| `name_length` | Length of the column name in characters | "7" | T1, T2, T3 |
| `type_is_integer` | Column type is an integer type | 0 or 1 | — |
| `type_is_varchar` | Column type is varchar | 0 or 1 | — |
| `column_type_boolean` | Column type is boolean (one-hot encoded) | 0 or 1 | T1, T2 |
| `column_type_char` | Column type is CHAR (one-hot encoded) | 0 or 1 | T3 |
| `column_type_date` | Column type is date (one-hot encoded) | 0 or 1 | T1, T2, T3 |
| `column_type_decimal` | Column type is decimal (one-hot encoded) | 0 or 1 | T1, T2, T3 |
| `column_type_double` | Column type is double (one-hot encoded) | 0 or 1 | T1, T2, T3 |
| `column_type_integer` | Column type is integer (one-hot encoded) | 0 or 1 | T1, T2, T3 |
| `column_type_timestamp` | Column type is TIMESTAMP (one-hot encoded) | 0 or 1 | T3 |
| `column_type_varchar` | Column type is varchar (one-hot encoded) | 0 or 1 | T1, T2, T3 |
| `is_this_col_violating_1nf` | Column contains non-atomic or multi-valued entries | 0 or 1 | T3 |
| `is_composite_key_part` | Column is part of a composite primary key | 0 or 1 | T3 |
| `is_this_col_partial_dependency` | Column has a partial dependency on the composite PK | 0 or 1 | T3 |
| `table_avg_unique_ratio` | Average unique_ratio across all columns in the table | "0.312" | T3 |
| `table_avg_null_ratio` | Average null_ratio across all columns in the table | "0.05" | T3 |
| `table_std_unique_ratio` | Standard deviation of unique_ratio across all columns in the table | "0.21" | T3 |
| `table_ratio_of_pk_candidates` | Ratio of PK candidate columns to total columns | "0.11" | T3 |
| `table_has_composite_pk` | Table uses a composite primary key | 0 or 1 | T3 |
| `table_ratio_composite_key_cols` | Ratio of composite-key-part columns to total columns | "0.22" | T3 |
| `table_ratio_1nf_violations` | Ratio of 1NF-violating columns to total columns | "0.0" | T3 |
| `table_has_partial_dependency` | Any column in the table is a partial dependency | 0 or 1 | T3 |
| `pk_target` | Is the column a single primary key? | 0 or 1 | target: T1 |
| `composite_pk_target` | Is the column part of a composite primary key? | 0 or 1 | target: T1 |
| `fk_target` | Is the column a foreign key? | 0 or 1 | target: T2 |
| `composite_fk_target` | Is the column part of a composite foreign key? | 0 or 1 | target: T2 |
| `target_normal_form` | Highest normal form the table satisfies (0 = violates 1NF, 1 = 1NF, 2 = 2NF, 3 = 3NF) | 0, 1, 2, or 3 | target: T3 |


### Data Types

In the future more data types like blob should be added while intentionally
leaving out semi-structured data like json and xml.

## Datasets

| Dataset name | Where to find? | Number of tables | Topic |
| :--- | :--- | :--- | :--- |
| Willibald DWA Challenge | https://dwa-compare.info/en/start-2/ | 10 | E-Commerce |
| Synthetic E-Commerce Dataset — Free Sample Preview | https://www.kaggle.com/datasets/oreomonsta123/synthetic-e-commerce-dataset10-tables-8-countries | 10 | E-Commerce |
| Multitable Ecommerce European Fashion | https://www.kaggle.com/datasets/joycemara/european-fashion-store-multitable-dataset | 7 | E-Commerce |
| TPC-H | https://www.tpc.org/tpch/ | 8 | E-Commerce |
| Formula 1 World Championship Dataset 2000–2026 | https://www.kaggle.com/datasets/mkaur1141/formula-1-world-championship-dataset-20002026 | 5 | Sports |
| USDA Nutrition Data: Flattened (SR Legacy) | https://www.kaggle.com/datasets/ericfornow/usda-nutrition-data-flattened-sr-legacy | 7 | Science |
| Solstice Residential Energy Pack (Sample) | https://www.kaggle.com/datasets/justinsolstice/solstice-residential-energy-pack | 16 | Energy Sector |
| Northwind | https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/northwind-pubs | 11 | E-Commerce |
| House Sales in King County, USA | https://www.kaggle.com/datasets/harlfoxem/housesalesprediction | 2 | Finance |
| Take me home | neue-fische database | 1 | unknown |
| Petsowners | neue-fische database | 4 | Veterinary |
| Monalisa | neue-fische database | 6 | Aviation |
| Heart | neue-fische database | 5 | Medicine |
| Gapminder | neue-fische database | 5 | Medicine |
| Bike Store relational database | https://www.kaggle.com/datasets/dillonmyrick/bike-store-sample-database | 9 | E-Commerce |
| LifeAlly(Carrer, health, relationship, Finance) | https://www.kaggle.com/datasets/karanshelar6/lifeallycarrer-health-relationship-finance | 7 | Various |
| Sales, customers, products, and products | https://www.kaggle.com/datasets/arkhepacis/sales-customers-products-and-customers | 5 | E-Commerce |
| F1 Grand Prix Dataset | https://www.kaggle.com/datasets/harshitstark/f1-grandprix-datavault | 14 | Sports |
| A Comprehensive Database on the FIFA World Cup | https://www.kaggle.com/datasets/joshfjelstul/world-cup-database | 27 | Sports |
| Northwind Datavault | Microsoft Northwind Database | 120 | E-Commerce |
| Euroleague & Eurocup Datasets | https://www.kaggle.com/datasets/babissamothrakis/euroleague-datasets | 14 | Sports |
| mimic-iv-clinical-database-demo-2.2 | https://www.kaggle.com/datasets/montassarba/mimic-iv-clinical-database-demo-2-2 | 31 | Medicine |
| Academic Records | Own fake data created by python | 10 | Education |
| Aviation Operations | Own fake data created by python | 6 | Aviation |
| Banking System | Own fake data created by python | 9 | Finance |
| Calculated Field Examples | Own fake data created by python | 4 | Various |
| Clinical Trials | Own fake data created by python | 10 | Medicine |
| Content Management Platform | Own fake data created by python | 10 | Entertainment |
| Diabetes Dataset | Own fake data created by python | 4 | Medicine |
| E-Commerce Operations | Own fake data created by python | 9 | E-Commerce |
| Event Management | Own fake data created by python | 11 | Entertainment |
| Fleet Management | Own fake data created by python | 10 | Transportation |
| Food Delivery | Own fake data created by python | 10 | Food & Beverage |
| Government Database | Own fake data created by python | 6 | Government |
| Healthcare Database | Own fake data created by python | 7 | Healthcare |
| Hospitality Database | Own fake data created by python | 10 | Hospitality |
| HR System | Own fake data created by python | 9 | Human Resources |
| Insurance Database | Own fake data created by python | 10 | Finance |
| Introduction Database | Own fake data created by python | 3 | Various |
| Inventory Management | Own fake data created by python | 9 | Logistics |
| IoT Platform | Own fake data created by python | 10 | Technology |
| Library Database | Own fake data created by python | 9 | Education |
| Manufacturing Database | Own fake data created by python | 10 | Manufacturing |
| Media Streaming Platform | Own fake data created by python | 10 | Entertainment |
| Project Management | Own fake data created by python | 9 | Various |
| Railway Database | Own fake data created by python | 6 | Transportation |
| Real Estate Database | Own fake data created by python | 10 | Finance |
| Retail Data Warehouse | Own fake data created by python | 6 | Retail |
| Retail Point of Sale | Own fake data created by python | 10 | Retail |
| Shipping Database | Own fake data created by python | 6 | Logistics |
| Social Platform | Own fake data created by python | 8 | Technology |
| Sports League Database | Own fake data created by python | 10 | Sports |
| Supply Chain | Own fake data created by python | 10 | Logistics |
| Telecom Billing | Own fake data created by python | 9 | Telecommunications |
| Telehealth Platform | Own fake data created by python | 10 | Healthcare |
| Traffic Management | Own fake data created by python | 6 | Transportation |
| University Database | Own fake data created by python | 6 | Education |
| Synthetic Multi-Domain Collection | Own fake data created by python | 87 | Various |
| FreeSQLcom Sample Databases | Example/Test data from [freesql.com](https://freesql.com) | 62 | Various |
| Spider Text-to-SQL Benchmark | https://yale-lily.github.io/spider | ~1,020 (~190 databases) | Various (138 domains) |
| Chinook Sample Database | https://github.com/lerocha/chinook-database | 11 | Music / Entertainment |
| Sakila Sample Database | https://github.com/jOOQ/sakila | 16 | Entertainment (Video Rental) |

## Tech Stack

- Python and Juypiter Notebook
- DuckDB as Database
- python DuckDB for sql execution
- Pandas and Numpy
- scikit-learn

## Tasks

### 1. PK detection:

Predict a primary key candidate (either single primary key or composite primary key) for a given database table and columns based on approx_unique, count and null_percentage. 

### 2. FK detection:

Predict a foreign key candidate for a given database table and columns based on ...

### 3. Normalization check:

Predict if a table is normalized or not. If normalized is table normalized in normalform 1 to 3 (1nf to 3nf).

### 4. Group building:

Predict and group database schemas and tables based on similarity in naming and structure.

## Contributors

Christian and Niklas