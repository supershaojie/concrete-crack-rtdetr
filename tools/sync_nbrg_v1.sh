#!/usr/bin/env bash
# Use the existing protected C19/LIF worktree workflow with an explicit profile.
set -Eeuo pipefail
export C19_LIF_V1_PROFILE=nbrg_v1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/sync_c19_lif_v1.sh" "$@"
