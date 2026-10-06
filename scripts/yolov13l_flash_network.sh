#!/usr/bin/env bash
# Sourced by AutoDL workers; restore shell options after the accelerator script.
yolov13l_flash_network() {
    local old_flags=$- old_options rc=0
    old_options=$(set +o)
    set +eu
    local turbo_script=${YOLOV13L_NETWORK_TURBO:-/etc/network_turbo}
    if [[ -r "$turbo_script" ]]; then
        source "$turbo_script"
        rc=$?
    fi
    eval "$old_options"
    [[ "$old_flags" == *e* ]] && set -e
    [[ "$old_flags" == *u* ]] && set -u
    if (( rc != 0 )); then
        printf 'AutoDL network_turbo failed with exit %d\n' "$rc" >&2
        return "$rc"
    fi
    local bypass='localhost,127.0.0.1,::1,pypi.org,.pypi.org,files.pythonhosted.org,.pythonhosted.org,pypi.python.org,download.pytorch.org,.pytorch.org'
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$bypass"
    export no_proxy="${no_proxy:+$no_proxy,}$bypass"
}
