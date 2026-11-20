#!/usr/bin/env bash
# Run `uv sync --frozen <args>` in $AM_ROOT with the lockfile's download URLs pointed at
# the internal Artifactory mirrors (same rewrite as nemo-rl-3976/env/nrl-uv-sync.sh).
# Versions and sha256 hashes are unchanged. The original uv.lock is restored on exit; the
# backup lives in $AM_WS/uv.lock.upstream while the sync runs, and `cmdev push` refuses
# to run while it exists. Git sources (github.com) go through the Kiro split proxy.
#
# Examples:
#   ./am-uv-sync.sh                    # base deps + default groups (build, docs, test)
#   ./am-uv-sync.sh --extra cuda       # adds TE, mamba-ssm, ... (source builds, slow)
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$(readlink -f "$0")")/env.sh"
if [[ -f "$HOME/.config/kiro-tunnel/proxy.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$HOME/.config/kiro-tunnel/proxy.env"
    set +a
    export NO_PROXY="$NO_PROXY,.mi-fds.net" no_proxy="$NO_PROXY,.mi-fds.net"
fi

cd "$AM_ROOT"
backup="$AM_WS/uv.lock.upstream"
if [[ -e "$backup" ]]; then
    echo "error: $backup exists; another sync is running or a previous one was killed." >&2
    echo "Restore it with: cp $backup $AM_ROOT/uv.lock && rm $backup" >&2
    exit 1
fi
cp uv.lock "$backup"
restore() { cp "$backup" "$AM_ROOT/uv.lock" && rm -f "$backup"; }
trap restore EXIT

A=<redacted-url>
sed -i \
    -e "s#url = \"https://files.pythonhosted.org/#url = \"$A/pypi-remote-org/#g" \
    -e "s#url = \"https://download-r2.pytorch.org/#url = \"$A/pytorch-remote/#g" \
    -e "s#url = \"https://download.pytorch.org/#url = \"$A/pytorch-generic-remote/#g" \
    -e "s#url = \"https://github.com/\([^\"]*/releases/download/\)#url = \"$A/github/\1#g" \
    uv.lock

# Build-isolation requirements are resolved outside uv.lock and need a reachable index.
export UV_DEFAULT_INDEX=$A/api/pypi/pypi-virtual/simple

uv sync --frozen "$@"
