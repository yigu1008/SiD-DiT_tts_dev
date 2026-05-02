#!/usr/bin/env bash
# Source this file inside any long-running AMLT bash to keep the job from
# being suspended for inactivity:
#
#   source _heartbeat.sh
#   start_heartbeat "phase-name"   # forks a background ticker
#   ... long-running work ...
#   stop_heartbeat                  # idempotent
#
# Forces line-buffered output for all subsequent python invocations.
# Auto-kills on EXIT/INT/TERM via trap.

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

_HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC:-60}"
_HEARTBEAT_PID=""

start_heartbeat() {
    local label="${1:-work}"
    if [[ -n "${_HEARTBEAT_PID}" ]] && kill -0 "${_HEARTBEAT_PID}" 2>/dev/null; then
        return 0
    fi
    (
        local i=0
        while true; do
            i=$((i + 1))
            echo "[heartbeat] ${label} alive tick=${i} at $(date -u +%FT%TZ)"
            sleep "${_HEARTBEAT_INTERVAL_SEC}"
        done
    ) &
    _HEARTBEAT_PID=$!
    # Make sure children inherit the trap.
    trap '_heartbeat_cleanup' EXIT INT TERM
}

_heartbeat_cleanup() {
    if [[ -n "${_HEARTBEAT_PID}" ]] && kill -0 "${_HEARTBEAT_PID}" 2>/dev/null; then
        kill "${_HEARTBEAT_PID}" 2>/dev/null || true
        wait "${_HEARTBEAT_PID}" 2>/dev/null || true
        _HEARTBEAT_PID=""
    fi
}

stop_heartbeat() {
    _heartbeat_cleanup
}
