from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import optuna

from optuna_layer.io_utils import maybe_load_hint_params, save_json
from optuna_layer.objective import make_objective
from optuna_layer.search_space import build_search_space


DEFAULT_STORAGE = "sqlite:////home/xth/forest_agent_project/outputs/optuna/optuna_single.db"
DEFAULT_OUT_BEST_JSON = "/home/xth/forest_agent_project/outputs/optuna/optuna_single_best.json"


def make_study_name(
    *,
    prefix: str,
    base_config: str,
    explicit_study_name: Optional[str] = None,
) -> str:
    """
    规则：
    1) 若显式传入 --study_name，则直接使用
    2) 否则自动生成唯一 study_name，避免不同运行互相污染
    """
    if explicit_study_name:
        return explicit_study_name

    stem = Path(base_config).stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stem}_{stamp}"


def ensure_storage_parent(storage: str) -> None:
    """
    仅处理 sqlite:///... 这种最常见的本地存储。
    """
    if storage.startswith("sqlite:///"):
        db_path = storage.replace("sqlite:///", "/", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--agent_hint_json", default=None)
    parser.add_argument("--out_best_json", default=DEFAULT_OUT_BEST_JSON)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--resume", action="store_true", help="continue an existing study")
    args = parser.parse_args()

    hint_params = maybe_load_hint_params(args.agent_hint_json)
    search_space = build_search_space(hint_params)

    study_name = make_study_name(
        prefix="forest_agent_single",
        base_config=args.base_config,
        explicit_study_name=args.study_name,
    )

    ensure_storage_parent(args.storage)
    Path(args.out_best_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"[optuna.search] study_name={study_name}")
    print(f"[optuna.search] storage={args.storage}")
    print(f"[optuna.search] n_trials={args.n_trials}")
    print(f"[optuna.search] resume={args.resume}")
    print(f"[optuna.search] agent_hint_json={args.agent_hint_json}")
    print(f"[optuna.search] out_best_json={args.out_best_json}")
    print(f"[optuna.search] hint_params={hint_params}")
    print(f"[optuna.search] search_space={search_space}")

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=bool(args.resume or args.study_name),
    )

    objective = make_objective(
        search_space=search_space,
        base_config_path=args.base_config,
    )

    before_trials = len(study.trials)
    study.optimize(objective, n_trials=args.n_trials)
    after_trials = len(study.trials)

    best = study.best_trial
    result = {
        "study_name": study.study_name,
        "storage": args.storage,
        "resume": bool(args.resume),
        "n_trials_requested": args.n_trials,
        "n_trials_before": before_trials,
        "n_trials_after": after_trials,
        "n_trials_added": after_trials - before_trials,
        "agent_hint_json": args.agent_hint_json,
        "hint_params": hint_params,
        "search_space": search_space,
        "best_params": dict(best.params),
        "best_value": best.value,
        "best_trial_number": best.number,
    }

    save_json(result, args.out_best_json)
    print(f"[optuna.search] saved best json to: {args.out_best_json}")


if __name__ == "__main__":
    main()