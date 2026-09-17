#!/usr/bin/env bash
# Pin an existing repository's independent worktree without touching its checkout.
set -Eeuo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: sync_psdb_p3.sh FULL_SHA [MAIN_REPO] [WORKTREE] [VERIFIED_BUNDLE]'
  exit 0
fi
[[ $# -ge 1 && $# -le 4 ]] || { echo 'Use --help'; exit 2; }
PSDB_SHA="$1"
PSDB_MAIN="$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")"
PSDB_WORKTREE="$(realpath -m -- "${3:-/root/autodl-tmp/projects/Crack_RTDETR-psdb-p3-v1}")"
PSDB_BRANCH=exp-rtdetr-r18-lite-psdb-p3-v1
[[ "$PSDB_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full 40-character delivered SHA required'; exit 2; }
[[ "$PSDB_MAIN" != "$PSDB_WORKTREE" ]] || { echo 'Independent worktree required'; exit 1; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
case "$(git -C "$PSDB_MAIN" remote get-url origin)" in
 https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
 *) echo 'Unexpected origin; refusing another project'; exit 1 ;;
esac
PSDB_OLD_HEAD="$(git -C "$PSDB_MAIN" rev-parse HEAD)"
PSDB_OLD_STATUS="$(git -C "$PSDB_MAIN" status --porcelain)"
if ! git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA^{commit}" 2>/dev/null; then
  for PSDB_TRY in 1 2 3 4 5; do
    GIT_TERMINAL_PROMPT=0 timeout 120 git -C "$PSDB_MAIN" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --no-write-fetch-head origin "refs/heads/$PSDB_BRANCH:refs/remotes/origin/$PSDB_BRANCH" || true
    git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA^{commit}" 2>/dev/null && break
    echo "Bounded fetch attempt $PSDB_TRY/5 did not provide delivered SHA"
  done
fi
if ! git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA^{commit}" 2>/dev/null; then
  [[ $# == 4 ]] || { echo 'Object still absent. Supply verified delivery bundle as fourth argument; no checkout changed.'; exit 1; }
  PSDB_BUNDLE="$(realpath -- "$4")"
  [[ -f "$PSDB_BUNDLE.sha256" ]] || { echo 'Bundle SHA256 sidecar required'; exit 1; }
  (cd "$(dirname -- "$PSDB_BUNDLE")" && sha256sum --check "$(basename -- "$PSDB_BUNDLE").sha256")
  git -C "$PSDB_MAIN" bundle verify "$PSDB_BUNDLE"
  git -C "$PSDB_MAIN" fetch --no-tags --no-write-fetch-head "$PSDB_BUNDLE" "refs/heads/$PSDB_BRANCH:refs/psdb-p3-delivery/$PSDB_SHA"
fi
git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA^{commit}"
git -C "$PSDB_MAIN" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$PSDB_SHA"
for PSDB_FILE in tools/init_psdb_p3.py tools/check_psdb_p3.py tools/train_psdb_p3.py ultralytics-main/ultralytics/nn/modules/psdb_p3.py; do
  git -C "$PSDB_MAIN" cat-file -e "$PSDB_SHA:$PSDB_FILE"
done
if [[ -e "$PSDB_WORKTREE" ]]; then
  [[ -f "$PSDB_WORKTREE/.git" ]] || { echo 'Existing directory is not linked worktree; preserved'; exit 1; }
  [[ "$(git -C "$PSDB_WORKTREE" rev-parse --show-toplevel)" == "$PSDB_WORKTREE" ]]
  [[ "$(git -C "$PSDB_WORKTREE" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$PSDB_MAIN" rev-parse --path-format=absolute --git-common-dir)" ]]
  [[ "$(git -C "$PSDB_WORKTREE" rev-parse HEAD)" == "$PSDB_SHA" ]] || { echo 'Existing SHA differs; preserved. Choose a new worktree path.'; exit 1; }
  [[ -z "$(git -C "$PSDB_WORKTREE" status --porcelain --untracked-files=no)" ]] || { echo 'Existing edits preserved'; exit 1; }
else
  git -C "$PSDB_MAIN" worktree add --detach "$PSDB_WORKTREE" "$PSDB_SHA"
fi
[[ "$(git -C "$PSDB_MAIN" rev-parse HEAD)" == "$PSDB_OLD_HEAD" ]]
[[ "$(git -C "$PSDB_MAIN" status --porcelain)" == "$PSDB_OLD_STATUS" ]]
python - "$PSDB_WORKTREE" "$PSDB_MAIN" "$PSDB_SHA" <<'PY'
import pathlib, shlex, sys
root, main, sha = sys.argv[1:]
text = 'source /root/miniconda3/etc/profile.d/conda.sh\nconda activate rtdetr\n'
for key, value in {'PSDB_P3_WORKTREE': root, 'PSDB_P3_MAIN': main, 'PSDB_P3_SHA': sha,
                   'PYTHONPATH': str(pathlib.Path(root)/'ultralytics-main'), 'YOLO_AUTOINSTALL': 'false',
                   'PYTHONUNBUFFERED': '1'}.items():
    text += 'export ' + key + '=' + shlex.quote(value) + '\n'
text += 'cd ' + shlex.quote(root) + '\n'
text += '[[ "$(git rev-parse HEAD)" == "$PSDB_P3_SHA" ]] || { echo "Pinned PSDB SHA changed; regenerate environment and preflight"; return 1; }\n'
text += '[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Tracked PSDB sources changed; commit and repeat preflight"; return 1; }\n'
dest = pathlib.Path(root)/'outputs/psdb_p3.env'
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
    assert dest.read_text() == text, 'Existing environment differs; preserved'
else:
    with dest.open('x') as f: f.write(text)
print('Environment:', dest)
PY
source "$PSDB_WORKTREE/outputs/psdb_p3.env"
[[ "$(git rev-parse HEAD)" == "$PSDB_P3_SHA" ]]
printf 'PSDB-P3 synchronized at %s\nSHA %s\nFormal training NOT_STARTED; test NOT_RUN\n' "$PSDB_P3_WORKTREE" "$PSDB_P3_SHA"
