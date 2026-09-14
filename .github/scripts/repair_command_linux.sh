#!/usr/bin/env bash
set -euv

# FIXME: if the libraries are already inside the wheel, why do we have
# to provide them to auditwheel again? It would be ideal if we could
# avoid this script entirely.

# We have to include the stempy library in Linux's LD_LIBRARY_PATH,
# or auditwheel won't work. These libraries are already in the wheel.
WHEEL_PATH=$(realpath "$1")
DEST_DIR=$(realpath "$2")
REPAIR_DIR=$(mktemp -d)
trap 'rm -rf -- "$REPAIR_DIR"' EXIT

unzip -q "$WHEEL_PATH" -d "$REPAIR_DIR"

LIBRARY_PATH=$(find "$REPAIR_DIR" -name "libstem.so" -print -quit)
test -n "$LIBRARY_PATH"
LIBRARY_DIR=$(dirname "$LIBRARY_PATH")
export LD_LIBRARY_PATH="$LIBRARY_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

auditwheel repair -w "$DEST_DIR" "$WHEEL_PATH"
