#!/bin/bash
# Generate the local Trino credentials that are deliberately NOT in git: the TLS
# keystore holding a private key, and the password file holding a password hash.
# Everything else under etc/ is committed, so this is all a fresh clone needs.
#
#   bash trino-iceberg/generate-dev-credentials.sh
#
# Reads TRINO_USERNAME, TRINO_PASSWORD and TRINO_KEYSTORE_PASSWORD from .env and
# writes etc/keystore.jks and etc/trino/password.db. Existing files are left alone
# unless --force is passed, so re-running it will not silently invalidate a setup
# that already works.
set -euo pipefail

cd "$(dirname "$0")"
repo_root="$(cd .. && pwd)"
force=false
[[ "${1:-}" == "--force" ]] && force=true

if [[ ! -f "$repo_root/.env" ]]; then
    echo "No .env in $repo_root - copy .env.example first." >&2
    exit 1
fi

# Only the variables this script needs, so a stray line in .env cannot shadow
# something in the caller's shell.
set -a
# shellcheck disable=SC1091  # path is computed above, not a literal for shellcheck to follow
source "$repo_root/.env"
set +a

: "${TRINO_USERNAME:?set TRINO_USERNAME in .env}"
: "${TRINO_PASSWORD:?set TRINO_PASSWORD in .env}"
: "${TRINO_KEYSTORE_PASSWORD:?set TRINO_KEYSTORE_PASSWORD in .env}"

keystore="etc/keystore.jks"
passwd_file="etc/trino/password.db"

# `docker compose up` run BEFORE this script bind-mounts two paths that do not exist
# yet, and Docker creates a *directory* for each. Trino then cannot start, and the
# leftover directory also defeats the plain `rm -f` below - so clear a stale path
# whichever kind it turned out to be.
clear_stale() {
    local path=$1
    if [[ -d "$path" ]]; then
        echo "note   $path is a directory (a bind mount created it) - removing"
        rm -rf "$path"
    else
        rm -f "$path"
    fi
}

# 1. TLS keystore. CN=trino matches the compose service name; the certificate is
# self-signed, which is why every client in the repo connects with verify=False.
if [[ -f "$keystore" && "$force" == false ]]; then
    echo "keep   $keystore (exists - pass --force to replace)"
else
    clear_stale "$keystore"
    keytool -genkeypair \
        -alias trino \
        -keyalg RSA \
        -keysize 2048 \
        -validity 3650 \
        -dname "CN=trino" \
        -ext "SAN=DNS:trino,DNS:localhost,IP:127.0.0.1" \
        -keystore "$keystore" \
        -storetype PKCS12 \
        -storepass "$TRINO_KEYSTORE_PASSWORD" \
        -keypass "$TRINO_KEYSTORE_PASSWORD" >/dev/null
    echo "wrote  $keystore (CN=trino, self-signed, 10 years)"
fi

# 2. Password file for the file-based authenticator. Trino requires bcrypt with a
# minimum cost of 10. htpasswd ships with Apache tools; the container fallback
# keeps this working on machines without it.
if [[ -f "$passwd_file" && "$force" == false ]]; then
    echo "keep   $passwd_file (exists - pass --force to replace)"
else
    clear_stale "$passwd_file"
    mkdir -p "$(dirname "$passwd_file")"
    if command -v htpasswd >/dev/null 2>&1; then
        htpasswd -bBC 10 -c "$passwd_file" "$TRINO_USERNAME" "$TRINO_PASSWORD" 2>/dev/null
    else
        echo "htpasswd not found - hashing in a container instead"
        docker run --rm httpd:2.4-alpine \
            htpasswd -bBnC 10 "$TRINO_USERNAME" "$TRINO_PASSWORD" > "$passwd_file"
    fi
    # htpasswd -n writes a trailing newline that Trino tolerates, but strip the
    # blank line so the file holds exactly one entry per user.
    sed -i.bak '/^$/d' "$passwd_file" && rm -f "$passwd_file.bak"
    echo "wrote  $passwd_file (user: $TRINO_USERNAME)"
fi

echo
echo "Done. Start Trino with: docker compose up -d trino"
