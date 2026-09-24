#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 && "$1" =~ ^[0-9a-f]{40}$ ]] || { echo 'Usage: sync_lcd_v1.sh FULL_SHA' >&2; exit 2; }
lcd_main=/root/autodl-tmp/projects/Crack_RTDETR
lcd_py=/root/miniconda3/envs/rtdetr/bin/python
[[ "$(git -C "$lcd_main" remote get-url origin)" == 'https://github.com/supershaojie/concrete-crack-rtdetr.git' ]]
git -C "$lcd_main" fetch origin exp-rtdetr-r18-lite-lcd-v1
[[ "$(git -C "$lcd_main" rev-parse FETCH_HEAD)" == "$1" ]] || { echo 'Remote branch differs from requested SHA' >&2; exit 1; }
lcd_sync=$(mktemp /tmp/lcd-sync-XXXXXX.py)
trap 'rm -f -- "$lcd_sync"' EXIT
git -C "$lcd_main" show "$1:tools/lcd_v1_sync.py" > "$lcd_sync"
"$lcd_py" "$lcd_sync" "$1"
