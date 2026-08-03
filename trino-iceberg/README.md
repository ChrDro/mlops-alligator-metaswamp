# trino-iceberg

Configuration for the `trino` and `nessie` services. Only config lives here - the
containers are started from the root `docker-compose.yaml` together with the rest of
the stack, replacing the `docker run` script this directory used to hold.

```text
etc/config.properties                    coordinator + HTTPS/password auth (mounted over the image's own)
etc/password-authenticator.properties    file-based authenticator
etc/keystore.jks                         GIT-IGNORED, generated - self-signed dev certificate, CN=trino
etc/trino/password.db                    GIT-IGNORED, generated - bcrypt user entries
etc/trino/catalog/iceberg.properties     Nessie catalog, warehouse in the stack's MinIO
etc/trino/catalog/duckdb.properties      data/capstone.db
etc/trino/catalog/tpch.properties        generated demo data
data/                                    mounted to /duckdb, holds capstone.db (git-ignored,
                                         DuckDB creates it on first use)
```

## Credentials on a fresh clone

The two git-ignored files above hold a private key and a password hash, so they are
not committed. A clone has neither - generate them from `.env` before starting Trino:

```bash
bash trino-iceberg/generate-dev-credentials.sh
```

It is idempotent (existing files are kept; `--force` rotates them) and
`scripts/setup_stack.sh` calls it in preflight, so the full setup path needs no extra
step. Add a second user with
`htpasswd -B -C 10 trino-iceberg/etc/trino/password.db <user>`.

**Generate before the first `docker compose up`.** Compose bind-mounts both paths, and
Docker creates a *directory* for a bind-mount source that does not exist - Trino then
fails to start with a confusing keystore error. The generator recovers from that state
(it removes the stray directories), but avoiding it is cheaper: run the script first,
or use `scripts/setup_stack.sh`.

Replacing the keystore needs `docker compose restart trino`; the coordinator loads it
at startup and does not notice the file changing underneath.

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
