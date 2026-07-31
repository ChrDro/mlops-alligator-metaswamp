# trino-iceberg

Configuration for the `trino` and `nessie` services. Only config lives here - the
containers are started from the root `docker-compose.yaml` together with the rest of
the stack, replacing the `docker run` script this directory used to hold.

```text
etc/config.properties                    coordinator + HTTPS/password auth (mounted over the image's own)
etc/password-authenticator.properties    file-based authenticator
etc/keystore.jks                         self-signed dev certificate, CN=trino
etc/trino/password.db                    users; add one with htpasswd -B -C 10 <file> <user>
etc/trino/catalog/iceberg.properties     Nessie catalog, warehouse in the stack's MinIO
etc/trino/catalog/duckdb.properties      data/capstone.db
etc/trino/catalog/tpch.properties        generated demo data
data/                                    mounted to /duckdb, holds capstone.db (git-ignored,
                                         DuckDB creates it on first use)
```

Two things that used to be separate are now shared with the rest of the stack:

- **MinIO** - the `storage` container from the old script is gone. Iceberg writes to
  the `warehouse` bucket of the `minio` service that also holds MLflow's artifacts.
- **Bucket creation** - handled by MinIO's entrypoint in the compose file, so the `mc`
  bootstrap container is gone too.

Nessie keeps its version store in RocksDB on the `nessie_data` volume, so Iceberg tables
stay queryable across `docker compose down` / `up`. The store path is deliberately
`/deployments/data/rocksdb` rather than Nessie's default under `/tmp` - see the comment in
`docker-compose.yaml` for the uid-10000 ownership reason. Removing that volume orphans
whatever is already in the `warehouse` bucket, since the pointers to it live here.

If the store is ever reset (volume deleted, or switching back to `IN_MEMORY`), Trino
starts failing with `ref 'main' is no longer valid` - it caches the ref it saw at
startup. `docker compose restart trino` clears that. Recreating Nessie on an intact
volume does not need it.

Nessie runs with authentication and authorization disabled, like the rest of the local
stack: anything that can reach port 19120 can rewrite the catalog.

Editing any file here needs a `docker compose restart trino` (or `nessie`) to take effect.
