#!/usr/bin/env bash
# Pin an existing repository's independent worktree without touching its checkout.
set -Eeuo pipefail
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: sync_sdb_p3.sh FULL_SHA [MAIN_REPO] [WORKTREE] [VERIFIED_BUNDLE]'
  exit 0
fi
[[ $# -ge 1 && $# -le 4 ]] || { echo 'Use --help'; exit 2; }
SDB_SHA="$1"
SDB_MAIN="$(realpath -- "${2:-/root/autodl-tmp/projects/Crack_RTDETR}")"
SDB_WORKTREE="$(realpath -m -- "${3:-/root/autodl-tmp/projects/Crack_RTDETR-sdb-p3-v1}")"
SDB_BRANCH=exp-rtdetr-r18-lite-sdb-p3-v1
[[ "$SDB_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full 40-character delivered SHA required'; exit 2; }
[[ "$SDB_MAIN" != "$SDB_WORKTREE" ]] || { echo 'Independent worktree required'; exit 1; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rtdetr
case "$(git -C "$SDB_MAIN" remote get-url origin)" in
 https://github.com/supershaojie/concrete-crack-rtdetr|https://github.com/supershaojie/concrete-crack-rtdetr.git|git@github.com:supershaojie/concrete-crack-rtdetr.git|ssh://git@github.com/supershaojie/concrete-crack-rtdetr.git) ;;
 *) echo 'Unexpected origin; refusing another project'; exit 1 ;;
esac
SDB_OLD_HEAD="$(git -C "$SDB_MAIN" rev-parse HEAD)"
SDB_OLD_STATUS="$(git -C "$SDB_MAIN" status --porcelain)"
if ! git -C "$SDB_MAIN" cat-file -e "$SDB_SHA^{commit}" 2>/dev/null; then
  for SDB_TRY in 1 2 3 4 5; do
    GIT_TERMINAL_PROMPT=0 timeout 120 git -C "$SDB_MAIN" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1 -c http.lowSpeedTime=60 \
      fetch --progress --no-tags --no-write-fetch-head origin "refs/heads/$SDB_BRANCH:refs/remotes/origin/$SDB_BRANCH" || true
    git -C "$SDB_MAIN" cat-file -e "$SDB_SHA^{commit}" 2>/dev/null && break
    echo "Bounded fetch attempt $SDB_TRY/5 did not provide delivered SHA"
  done
fi
if ! git -C "$SDB_MAIN" cat-file -e "$SDB_SHA^{commit}" 2>/dev/null; then
  [[ $# == 4 ]] || { echo 'Object still absent. Supply verified delivery bundle as fourth argument; no checkout changed.'; exit 1; }
  SDB_BUNDLE="$(realpath -- "$4")"
  [[ -f "$SDB_BUNDLE.sha256" ]] || { echo 'Bundle SHA256 sidecar required'; exit 1; }
  (cd "$(dirname -- "$SDB_BUNDLE")" && sha256sum --check "$(basename -- "$SDB_BUNDLE").sha256")
  git -C "$SDB_MAIN" bundle verify "$SDB_BUNDLE"
  git -C "$SDB_MAIN" fetch --no-tags --no-write-fetch-head "$SDB_BUNDLE" "refs/heads/$SDB_BRANCH:refs/sdb-p3-delivery/$SDB_SHA"
fi
git -C "$SDB_MAIN" cat-file -e "$SDB_SHA^{commit}"
git -C "$SDB_MAIN" merge-base --is-ancestor a0459d6a652cb702699087c88fa39a3e4c4087ec "$SDB_SHA"
for SDB_FILE in tools/init_sdb_p3.py tools/check_sdb_p3.py tools/train_sdb_p3.py ultralytics-main/ultralytics/nn/modules/sdb_p3.py; do
  git -C "$SDB_MAIN" cat-file -e "$SDB_SHA:$SDB_FILE"
done
if [[ -e "$SDB_WORKTREE" ]]; then
  [[ -f "$SDB_WORKTREE/.git" ]] || { echo 'Existing directory is not linked worktree; preserved'; exit 1; }
  [[ "$(git -C "$SDB_WORKTREE" rev-parse --show-toplevel)" == "$SDB_WORKTREE" ]]
  [[ "$(git -C "$SDB_WORKTREE" rev-parse --path-format=absolute --git-common-dir)" == "$(git -C "$SDB_MAIN" rev-parse --path-format=absolute --git-common-dir)" ]]
  [[ "$(git -C "$SDB_WORKTREE" rev-parse HEAD)" == "$SDB_SHA" ]] || { echo 'Existing SHA differs; preserved. Choose a new worktree path.'; exit 1; }
  [[ -z "$(git -C "$SDB_WORKTREE" status --porcelain --untracked-files=no)" ]] || { echo 'Existing edits preserved'; exit 1; }
else
  git -C "$SDB_MAIN" worktree add --detach "$SDB_WORKTREE" "$SDB_SHA"
fi
[[ "$(git -C "$SDB_MAIN" rev-parse HEAD)" == "$SDB_OLD_HEAD" ]]
[[ "$(git -C "$SDB_MAIN" status --porcelain)" == "$SDB_OLD_STATUS" ]]
python - "$SDB_WORKTREE" "$SDB_MAIN" "$SDB_SHA" <<'PY'
import pathlib, shlex, sys
root, main, sha = sys.argv[1:]
text = 'source /root/miniconda3/etc/profile.d/conda.sh\nconda activate rtdetr\n'
for key, value in {'SDB_P3_WORKTREE': root, 'SDB_P3_MAIN': main, 'SDB_P3_SHA': sha,
                   'PYTHONPATH': str(pathlib.Path(root)/'ultralytics-main'), 'YOLO_AUTOINSTALL': 'false',
                   'PYTHONUNBUFFERED': '1'}.items():
    text += 'export ' + key + '=' + shlex.quote(value) + '\n'
text += 'cd ' + shlex.quote(root) + '\n'
dest = pathlib.Path(root)/'outputs/sdb_p3.env'
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
    assert dest.read_text() == text, 'Existing environment differs; preserved'
else:
    with dest.open('x') as f: f.write(text)
print('Environment:', dest)
PY
source "$SDB_WORKTREE/outputs/sdb_p3.env"
[[ "$(git rev-parse HEAD)" == "$SDB_P3_SHA" ]]
printf 'SDB-P3 synchronized at %s\nSHA %s\nFormal training NOT_STARTED; test NOT_RUN\n' "$SDB_P3_WORKTREE" "$SDB_P3_SHA"
