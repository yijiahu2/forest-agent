from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import optuna

from optuna_layer.io_utils import maybe_load_hint_params, maybe_load_spatial_context, save_json
from optuna_layer.objective_multi import make_objective_multi
from optuna_layer.search_space import build_search_space


DEFAULT_STORAGE = "sqlite:////home/xth/forest_agent_project/outputs/optuna/optuna_multi.db"
DEFAULT_OUT_BEST_JSON = "/home/xth/forest_agent_project/outputs/optuna/optuna_multi_best.json"


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


def representative_trial_from_pareto(study: optuna.study.Study) -> optuna.trial.FrozenTrial:
    """
    从 Pareto 前沿里选一个 representative trial。
    当前策略：对各目标做简单归一化后取总和最小。
    """
    trials = [t for t in study.best_trials if t.values is not None]
    if not trials:
        raise ValueError("No Pareto-best trials found.")

    n_obj = len(trials[0].values)
    mins = [min(t.values[i] for t in trials) for i in range(n_obj)]
    maxs = [max(t.values[i] for t in trials) for i in range(n_obj)]

    def norm(v: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return (v - lo) / (hi - lo)

    best_trial = None
    best_score = None
    for t in trials:
        score = 0.0
        for i, v in enumerate(t.values):
            score += norm(v, mins[i], maxs[i])
        if best_score is None or score < best_score:
            best_score = score
            best_trial = t

    if best_trial is None:
        raise ValueError("Failed to choose representative Pareto trial.")

    return best_trial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", required=True)
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--agent_hint_json", default=None)
    parser.add_argument("--out_best_json", default=DEFAULT_OUT_BEST_JSON)
    parser.add_argument("--spatial_context_json", default=None)
    parser.add_argument("--storage", default=DEFAULT_STORAGE)
    parser.add_argument("--study_name", default=None)
    parser.add_argument("--resume", action="store_true", help="continue an existing study")
    args = parser.parse_args()

    hint_params = maybe_load_hint_params(args.agent_hint_json)
    spatial_context = maybe_load_spatial_context(args.spatial_context_json)
    search_space = build_search_space(hint_params, spatial_context=spatial_context)

    study_name = make_study_name(
        prefix="forest_agent_multi",
        base_config=args.base_config,
        explicit_study_name=args.study_name,
    )

    ensure_storage_parent(args.storage)
    Path(args.out_best_json).parent.mkdir(parents=True, exist_ok=True)

    print(f"[optuna.search_multi] study_name={study_name}")
    print(f"[optuna.search_multi] storage={args.storage}")
    print(f"[optuna.search_multi] n_trials={args.n_trials}")
    print(f"[optuna.search_multi] resume={args.resume}")
    print(f"[optuna.search_multi] agent_hint_json={args.agent_hint_json}")
    print(f"[optuna.search_multi] out_best_json={args.out_best_json}")
    print(f"[optuna.search_multi] spatial_context_json={args.spatial_context_json}")
    print(f"[optuna.search_multi] hint_params={hint_params}")
    print(f"[optuna.search_multi] search_space={search_space}")

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        directions=["minimize", "minimize", "minimize", "minimize"],
        load_if_exists=bool(args.resume or args.study_name),
    )

    objective = make_objective_multi(
        search_space=search_space,
        base_config_path=args.base_config,
    )

    before_trials = len(study.trials)
    study.optimize(objective, n_trials=args.n_trials)
    after_trials = len(study.trials)

    rep_trial = representative_trial_from_pareto(study)

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
        "spatial_context_json": args.spatial_context_json,
        "spatial_context": spatial_context,
        "search_space": search_space,
        "best_params": dict(rep_trial.params),
        "representative_trial_number": rep_trial.number,
        "representative_values": list(rep_trial.values) if rep_trial.values is not None else None,
        "num_pareto_trials": len(study.best_trials),
        "pareto_trials": [
            {
                "trial_number": t.number,
                "params": dict(t.params),
                "values": list(t.values) if t.values is not None else None,
            }
            for t in study.best_trials
        ],
    }

    save_json(result, args.out_best_json)
    print(f"[optuna.search_multi] saved best json to: {args.out_best_json}")


if __name__ == "__main__":
    main()