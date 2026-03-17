from __future__ import annotations

import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

from tools.cached_stage_runners import run_stage1_cached, run_stage2_cached


def _write_response(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue

        try:
            req = json.loads(raw)
        except Exception as e:
            _write_response({"ok": False, "error": f"invalid_json: {e}"})
            continue

        op = req.get("op")
        try:
            with redirect_stdout(sys.stderr):
                if op == "stage1":
                    result = run_stage1_cached(req["cfg"])
                elif op == "stage2":
                    result = run_stage2_cached(req["cfg"], req["m_sem_tif"])
                else:
                    raise ValueError(f"unknown op: {op}")
            _write_response({"ok": True, "result": result})
        except Exception as e:
            _write_response({"ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
