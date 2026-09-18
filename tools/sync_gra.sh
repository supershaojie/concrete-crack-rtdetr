#!/usr/bin/env bash
# Run in the existing main repository. No clone, reset, deletion or URL mutation.
set -Eeuo pipefail
SHA=${1:?Usage: bash tools/sync_gra.sh FULL_DELIVERY_SHA}
MAIN=${GRA_MAIN:-/root/autodl-tmp/projects/Crack_RTDETR}
WT=/root/autodl-tmp/projects/Crack_RTDETR-gra-v1
BRANCH=exp-rtdetr-r18-lite-gra-v1
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]]
origin=$(git -C "$MAIN" remote get-url origin)
case "$origin" in
    https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
    *) echo 'Origin identity requires review (URL not printed or modified).' >&2; exit 2;;
esac
if ! git -C "$MAIN" cat-file -e "${SHA}^{commit}" 2>/dev/null; then
    for attempt in 1 2 3 4 5; do
        printf 'Fetch attempt %s/5; timeout 120 seconds\n' "$attempt"
        if GIT_TERMINAL_PROMPT=0 timeout 120 git -C "$MAIN" -c http.version=HTTP/1.1 \
            -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 fetch --progress --no-tags origin "$BRANCH"; then
            if git -C "$MAIN" cat-file -e "${SHA}^{commit}" 2>/dev/null; then break; fi
            echo 'Fetch finished but target SHA is still missing.' >&2
        else
            rc=$?; printf 'Fetch failed/timed out with exit %s\n' "$rc" >&2
        fi
    done
fi
git -C "$MAIN" cat-file -e "${SHA}^{commit}"
if [[ -e "$WT" ]]; then
    [[ -f "$WT/.git" && $(git -C "$WT" rev-parse --show-toplevel) == "$WT" ]]
    [[ $(git -C "$WT" rev-parse HEAD) == "$SHA" ]]
    [[ -z $(git -C "$WT" status --porcelain --untracked-files=no) ]]
else
    git -C "$MAIN" worktree add --detach "$WT" "$SHA"
fi
git -C "$WT" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec HEAD
bash "$WT/tools/gra_server.sh" environment "$SHA"
