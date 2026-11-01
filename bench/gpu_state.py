"""Record the machine state that a measurement depends on.

Clocks cannot be locked on the benchmark machine (no admin rights, and the
GPU also drives a desktop), so every result file carries nvidia-smi readings
taken around the run, the processes holding GPU memory, and the git commit.
"""

import os
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = "/var/tmp/rtx4090.lock"

_FIELDS = [
    "timestamp",
    "name",
    "driver_version",
    "pstate",
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "clocks_event_reasons.active",
]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def nvidia_smi() -> dict:
    out = _run(["nvidia-smi", "--query-gpu=" + ",".join(_FIELDS), "--format=csv,noheader"])
    state = dict(zip(_FIELDS, (v.strip() for v in out.split(",")))) if out else {}
    apps = _run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    )
    state["compute_apps"] = [a.strip() for a in apps.splitlines() if a.strip()]
    return state


def git_commit() -> dict:
    head = _run(["git", "-C", REPO, "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"]))
    cutlass = _run(["git", "-C", os.path.join(REPO, "third_party", "cutlass"), "rev-parse", "HEAD"])
    return {"commit": head, "dirty": dirty, "cutlass_commit": cutlass}


def holds_gpu_lock() -> bool:
    """True if this process (or an ancestor flock) holds the machine-wide GPU lock."""
    import fcntl

    try:
        fd = os.open(LOCK, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Someone holds it; check that it is our parent chain.
        pid = os.getpid()
        held = _run(["lslocks", "--noheadings", "-o", "PID,MODE,PATH"])
        holders = set()
        for line in held.splitlines():
            parts = line.split()
            # A trailing '*' on the mode marks a waiter, not a holder.
            if len(parts) >= 3 and parts[2] == LOCK and not parts[1].endswith("*"):
                holders.add(int(parts[0]))
        while pid > 1:
            if pid in holders:
                return True
            stat = open(f"/proc/{pid}/stat").read()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def header(**extra) -> dict:
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_commit(),
        "holds_gpu_lock": holds_gpu_lock(),
        **extra,
    }
