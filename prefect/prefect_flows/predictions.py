# prefect_flows/polling_predictions.py
from prefect import flow, task
from prefect.cache_policies import NONE as NO_CACHE
from sqlalchemy import create_engine, text
from sqlalchemy.sql.expression import select, text
import pandas as pd
import requests
from typing import List, Dict
from datetime import datetime
import time
import os
from dotenv import load_dotenv
from itertools import groupby

load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get('TRINO_IP_ADDRESS'))
TRINO_USERNAME = str(os.environ.get('TRINO_USERNAME'))
TRINO_PASSWORD = str(os.environ.get('TRINO_PASSWORD'))

def get_trino_engine():
    """
    Create and return a Trino engine instance.
    """
    return create_engine(
        f'trino://{TRINO_USERNAME}:{TRINO_PASSWORD}@{TRINO_IP_ADDRESS}:8443/duckdb',
        connect_args={
            "http_scheme": "https",
            "verify": False
        }
    )

@task(cache_policy=NO_CACHE)
def fetch_new_rows(batch_size: int = 100) -> pd.DataFrame:
    """
    Fetch unprocessed rows from Trino.
    """
    trino_engine = get_trino_engine()
    
    query = text("""
        SELECT 
            id,
            database,
            schema,
            table_name,
            column_name,
            number_unique_values,
            count,
            is_unique,
            ordinal_position,
            unique_ratio,
            is_first_column,
            relative_ordinal_position,
            is_first_unique_column,
            table_column_count,
            table_unique_column_count,
            table_row_count,
            table_has_unique_column,
            table_has_no_single_pk_candidate,
            table_near_unique_column_count,
            table_id_named_column_count,
            table_non_null_column_count,
            table_max_unique_ratio,
            unique_ratio_rank,
            null_ratio_rank,
            is_least_null_in_table,
            unique_ratio_relative_to_max,
            name_ends_with_id,
            name_contains_table_name,
            name_is_singular_table_id,
            name_length,
            column_type_boolean,
            column_type_date,
            column_type_decimal,
            column_type_double,
            column_type_integer,
            column_type_varchar
        FROM duckdb.predictions.queue
        WHERE processed = FALSE
        --ORDER BY created_at ASC
        LIMIT :batch_size
    """)
    
    with trino_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"batch_size": batch_size})
    
    return df

@task(retries=2, retry_delay_seconds=10)
def predict_batch(rows: pd.DataFrame) -> List[Dict]:
    """
    Call prediction endpoints for a batch of rows.
    """
    predictions = []
    
    for _, row in rows.iterrows():
        row_data = row.to_dict()
        row_predictions = {"queue_id": row['id']}
        
        # Call all 4 prediction endpoints
        # Use localhost when running outside Docker, model-service when inside Docker
        # Since port 8080 is exposed, we can use localhost from outside
        for model_type in ["pk", "fk", "cpk", "cfk"]:
            try:
                response = requests.post(
                    f"http://localhost:8080/predict_{model_type}",
                    json=row_data,
                    timeout=10
                )
                
                print(response.text)
                response.raise_for_status()
                result = response.json()
                
                # Debug: print the response
                print(f"API Response for {model_type}: {result}")
                
                # Store prediction and probability (API returns 'probability' not 'confidence')
                row_predictions[f"{model_type}_prediction"] = result.get('prediction')
                row_predictions[f"{model_type}_confidence"] = result.get('probability', 0.0)
                
            except Exception as e:
                print(f"Error predicting {model_type} for row {row['id']}: {e}")
                row_predictions[f"{model_type}_prediction"] = None
                row_predictions[f"{model_type}_confidence"] = 0.0
        
        # Add metadata
        row_predictions.update({
            "database": row['database'],
            "table_name": row['table_name'],
            "column_name": row['column_name'],
            #"predicted_at": datetime.now()
        })
        
        predictions.append(row_predictions)
    
    return predictions

@task(cache_policy=NO_CACHE)
def store_predictions_to_trino(predictions: List[Dict]):
    """
    Store predictions and mark rows as processed.
    """
    if not predictions:
        return
    
    trino_engine = get_trino_engine()
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    
    # Convert types to match table schema
    # queue_id should be varchar (string)
    df['queue_id'] = df['queue_id'].astype(str)
    
    # Prediction columns should be varchar (convert None to empty string or 'NULL')
    prediction_cols = ['pk_prediction', 'fk_prediction', 'composite_pk_prediction', 'composite_fk_prediction']
    for col in prediction_cols:
        if col in df.columns:
            df[col] = df[col].fillna('NULL').astype(str)
    
    with trino_engine.connect() as conn:
        # Create schema for storing predictions
        create_schema = text("""
            CREATE SCHEMA IF NOT EXISTS duckdb.prediction_results
        """)
        
        conn.execute(create_schema)
        
        # Insert predictions - use schema parameter as just 'prediction_results' since catalog is in connection string
        df.to_sql(
            'results',
            conn,
            schema='prediction_results',
            if_exists='append',
            index=False
        )
        
        # Mark queue items as processed
        queue_ids = [p['queue_id'] for p in predictions]
        
        # Build the IN clause dynamically
        ids_list = ', '.join([f"{qid}" for qid in queue_ids])
        current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        update_query = text(f"""
            UPDATE duckdb.predictions.queue
            SET processed = TRUE,
                processed_at = '{current_timestamp}'
            WHERE id IN ({ids_list})
        """)
        
        conn.execute(update_query)

@task
def log_metrics(num_processed: int, duration_seconds: float):
    """
    Log metrics to monitoring system (Prometheus/MLflow).
    """
    print(f"Processed {num_processed} predictions in {duration_seconds:.2f}s")
    print(f"Throughput: {num_processed/duration_seconds:.2f} predictions/second")
    
    # Optional: Send to Prometheus pushgateway
    # requests.post(
    #     "http://pushgateway:9091/metrics/job/prediction_batch",
    #     data=f"predictions_processed {num_processed}\n"
    # )

@flow(name="polling-predictions")
def prediction_polling_flow(batch_size: int = 100):
    """
    Main polling flow: fetch, predict, store.
    """
    start_time = time.time()
    
    print(TRINO_IP_ADDRESS)
    
    # Test connection
    trino_engine = get_trino_engine()
    
    try:
        with trino_engine.connect() as connection:
            rows = connection.execute(text("Select 1"))
            print("Connection over HTTPS successful!")
    except Exception as e:
        print(f"Error: {e}")
    
    # Fetch new rows
    rows = fetch_new_rows(batch_size)
    
    if rows.empty:
         print("No new rows to process")
         return
    
    print(f"Fetched {len(rows)} rows to process")
    
    # Make predictions
    predictions = predict_batch(rows)
    
    # Store results
    #store_predictions_to_trino(predictions)
    
    # # Log metrics
    # duration = time.time() - start_time
    # log_metrics(len(predictions), duration)
    
    #print(f"Successfully processed {len(predictions)} predictions")
    
if __name__ == "__main__":
    prediction_polling_flow()