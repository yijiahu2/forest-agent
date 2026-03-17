from __future__ import annotations

import atexit
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

WORKER_PROC: subprocess.Popen[str] | None = None


def _worker_cmd() -> list[str]:
    return [
        "bash",
        "-lc",
        "source /home/xth/anaconda3/etc/profile.d/conda.sh && conda activate tcd && python -u -m tools.stage_cache_worker",
    ]


def _ensure_worker() -> subprocess.Popen[str]:
    global WORKER_PROC
    if WORKER_PROC is not None and WORKER_PROC.poll() is None:
        return WORKER_PROC

    WORKER_PROC = subprocess.Popen(
        _worker_cmd(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        cwd="/home/xth/forest_agent_project",
    )
    return WORKER_PROC


def _shutdown_worker() -> None:
    global WORKER_PROC
    if WORKER_PROC is None:
        return
    try:
        if WORKER_PROC.stdin:
            WORKER_PROC.stdin.close()
    except Exception:
        pass
    try:
        if WORKER_PROC.poll() is None:
            WORKER_PROC.terminate()
    except Exception:
        pass
    WORKER_PROC = None


atexit.register(_shutdown_worker)


def _call_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    proc = _ensure_worker()
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("stage cache worker stdio unavailable")

    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("stage cache worker exited without response")
    resp = json.loads(line)
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "unknown stage cache worker error"))
    return resp["result"]


def run_stage1_via_worker(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return _call_worker({"op": "stage1", "cfg": cfg})


def run_stage2_via_worker(cfg: Dict[str, Any], m_sem_tif: str) -> Dict[str, Any]:
    return _call_worker({"op": "stage2", "cfg": cfg, "m_sem_tif": m_sem_tif})
