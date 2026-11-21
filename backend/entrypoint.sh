#!/bin/sh
# Hosted volumes are often mounted root-owned, so fix ownership and drop to the
# unprivileged user. Platforms that already run as non-root skip this step.
set -eu

if [ "$(id -u)" = "0" ]; then
    mkdir -p /data
    chown -R app:app /data
    exec setpriv --reuid=app --regid=app --init-groups "$@"
fi

exec "$@"
