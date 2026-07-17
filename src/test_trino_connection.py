import os

import urllib3
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError


# Supress warning for verify false in sqlalchemy engine
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get("TRINO_IP_ADDRESS"))
TRINO_USERNAME = str(os.environ.get("TRINO_USERNAME"))
TRINO_PASSWORD = str(os.environ.get("TRINO_PASSWORD"))

engine = create_engine(
    f"trino://{TRINO_USERNAME}:{TRINO_PASSWORD}@{TRINO_IP_ADDRESS}:8443/iceberg",
    connect_args={
        # Actives SSL/TLS
        "http_scheme": "https",
        # Ignores the Error of self-signed Certificats
        "verify": False,
    },
)

try:
    with engine.connect() as connection:
        rows = connection.execute(text("Select 1"))
        print("Connection over HTTPS successful!")
except DBAPIError as e:
    print(f"Error: {e}")
