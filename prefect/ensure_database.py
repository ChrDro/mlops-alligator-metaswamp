"""
Create the Prefect database in the shared Postgres instance if it does not exist.

Why this runs at container start instead of an init script
----------------------------------------------------------
The usual place for this is ``/docker-entrypoint-initdb.d`` in the postgres image,
but those scripts only execute when the data directory is *empty*. The
``postgres_data`` volume already holds the MLflow database, so an init script added
now would silently never run. This script is idempotent and works on both a fresh
volume and an existing one.

Prefect gets its **own database** rather than sharing MLflow's. Both tools manage
their own migrations and table sets; sharing one database means a botched migration
in either can take the other down with it.

Run before ``prefect server start`` - the server expects its database to exist.
"""

import asyncio
import os
import sys

import asyncpg


# Name of the database Prefect owns. Kept separate from POSTGRES_DB (MLflow's).
PREFECT_DB = os.environ.get("PREFECT_POSTGRES_DB", "prefect")

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))

CONNECT_ATTEMPTS = 30
CONNECT_RETRY_SECONDS = 2


async def ensure_database() -> None:
    """Connect to the maintenance database and create PREFECT_DB if it is missing."""
    user = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    maintenance_db = os.environ["POSTGRES_DB"]

    connection = None
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            connection = await asyncpg.connect(
                user=user,
                password=password,
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                database=maintenance_db,
            )
            break
        except (OSError, asyncpg.PostgresError) as error:
            if attempt == CONNECT_ATTEMPTS:
                print(f"Postgres unreachable after {attempt} attempts: {error}", file=sys.stderr)
                raise
            print(f"Waiting for Postgres ({attempt}/{CONNECT_ATTEMPTS})...")
            await asyncio.sleep(CONNECT_RETRY_SECONDS)

    try:
        exists = await connection.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            PREFECT_DB,
        )
        if exists:
            print(f"Database '{PREFECT_DB}' already exists.")
            return

        # CREATE DATABASE cannot run inside a transaction and cannot take the name as
        # a bind parameter, so it is interpolated. PREFECT_DB comes from our own
        # compose config, not from user input, and is quoted to be safe.
        await connection.execute(f'CREATE DATABASE "{PREFECT_DB}"')
        print(f"Created database '{PREFECT_DB}'.")
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(ensure_database())
