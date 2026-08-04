# Prefect server plus the extras the flows import.
#
# These used to be pip-installed by the compose command on every container
# start: ~230 MB of writable layer per recreate, a 745 MiB memory peak while
# installing, and a boot slow enough that the healthcheck needed a 300s
# start_period. Baked into an image they cost one cached layer instead.
FROM prefecthq/prefect:3.6.25-python3.11

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# The flows stay a bind mount (see docker-compose.yaml), so editing
# prefect/*.py takes effect on restart without a rebuild.
WORKDIR /opt/flows
