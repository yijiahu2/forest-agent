import json
from pathlib import Path
from typing import Dict, Any, Optional


def save_json(obj: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_hint_params(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None

    p = Path(path)
    if not p.exists():
        return None

    data = load_json(str(p))

    for key in ["best_params", "final_params", "suggested_params", "agent_hint_params"]:
        if key in data and isinstance(data[key], dict):
            return data[key]

    return None