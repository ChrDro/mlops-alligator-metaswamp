"""Manual connectivity check against Trino over HTTPS.

Run it directly when a pipeline fails with a connection error, to tell "Trino is
unreachable / credentials are wrong" apart from "the query is wrong":

    python src/check_trino_connection.py

This is a script, not a test. It used to be named `test_trino_connection.py`, so
pytest collected it and opened a real Trino connection during collection - and because
the failure is swallowed by the `except` below, nothing about that looked wrong.
Everything now sits behind `main()`, so importing this module has no side effects.
"""

import os

import urllib3
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError


# Supress warning for verify false in sqlalchemy engine
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def main() -> None:
    """Open one connection and run `SELECT 1`."""
    load_dotenv()

    trino_ip_address = str(os.environ.get("TRINO_IP_ADDRESS"))
    trino_username = str(os.environ.get("TRINO_USERNAME"))
    trino_password = str(os.environ.get("TRINO_PASSWORD"))

    engine = create_engine(
        f"trino://{trino_username}:{trino_password}@{trino_ip_address}:8443/iceberg",
        connect_args={
            # Actives SSL/TLS
            "http_scheme": "https",
            # Ignores the Error of self-signed Certificats
            "verify": False,
        },
    )

    try:
        with engine.connect() as connection:
            connection.execute(text("Select 1"))
            print("Connection over HTTPS successful!")
    except DBAPIError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
