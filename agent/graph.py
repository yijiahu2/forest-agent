import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.config_builder import load_yaml, save_yaml  # noqa: E402


# =========================
# 基础工具
# =========================

DEFAULT_AGENT_OUT = "/home/xth/forest_agent_project/outputs/agent/final_summary.json"
DEFAULT_GENERATED_DIR = "/home/xth/forest_agent_project/configs/generated"
DEFAULT_RUNNER = "/home/xth/forest_agent_project/scripts/run_zstreeseg_experiment.py"

SAFE_SEARCH_SPACE = {
    "diam_list": [
        "96,160,256",
        "96,192,320",
        "128,192,320",
        "128,256,320",
    ],
    "tile": [1536, 2048],
    "overlap": [384, 512],
    "tile_overlap": [0.25, 0.35, 0.45],
    "augment": [True, False],
    "iou_merge_thr": [0.18, 0.22, 0.24, 0.28],
    "bsize": [256],
}


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    ensure_parent(Path(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    safe = {}
    for k, allowed in SAFE_SEARCH_SPACE.items():
        if k == "bsize":
            safe["bsize"] = 256
            continue
        v = params.get(k, None)
        if v in allowed:
            safe[k] = v
        else:
            safe[k] = allowed[0]
    safe["augment"] = bool(safe.get("augment", True))
    safe["bsize"] = 256
    return safe


def compute_single_score(metrics: Dict[str, Any]) -> float:
    """
    单目标综合分数：越小越好
    """
    return (
        float(metrics.get("tree_count_error_ratio", 999.0)) * 1.0
        + float(metrics.get("mean_crown_width_error_ratio", 999.0)) * 1.0
        + float(metrics.get("closure_error_abs", 999.0)) * 1.0
        + float(metrics.get("density_error_abs", 999999.0)) / 1000.0
    )


# =========================
# 兼容调用现有模块
# =========================

def try_import_prompt_builder():
    """
    兼容不同 prompts.py 函数名
    """
    import agent.prompts as prompts_mod

    for fn_name in [
        "build_prompt_v2",
        "build_agent_prompt",
        "build_prompt",
        "make_prompt",
    ]:
        fn = getattr(prompts_mod, fn_name, None)
        if callable(fn):
            return fn
    raise AttributeError(
        "No supported prompt builder found in agent/prompts.py. "
        "Expected one of: build_prompt_v2 / build_agent_prompt / build_prompt / make_prompt"
    )


def try_import_detail_extractor():
    """
    兼容不同 detail_tools.py 函数名
    """
    import agent.detail_tools as detail_mod

    for fn_name in [
        "extract_topk_bad_xiaoban",
        "extract_top_k_bad_xiaoban",
        "summarize_topk_bad_xiaoban",
        "load_topk_bad_xiaoban",
    ]:
        fn = getattr(detail_mod, fn_name, None)
        if callable(fn):
            return fn

    # 没有也允许，只是退化
    return None


def try_call_doubao_json(prompt: str) -> Dict[str, Any]:
    """
    兼容不同 doubao_client.py 函数名
    """
    import agent.doubao_client as dc

    candidates = [
        "call_doubao_json",
        "query_doubao_json",
        "generate_json",
        "ask_model_json",
        "chat_json",
    ]
    for name in candidates:
        fn = getattr(dc, name, None)
        if callable(fn):
            return fn(prompt)

    # 兼容：如果只有一个纯文本函数，则尝试 json 解析
    for name in ["call_doubao", "query_doubao", "chat", "ask_model"]:
        fn = getattr(dc, name, None)
        if callable(fn):
            text = fn(prompt)
            if isinstance(text, dict):
                return text
            return json.loads(text)

    raise AttributeError(
        "No supported Doubao client function found in agent/doubao_client.py. "
        "Expected e.g. call_doubao_json / query_doubao_json / call_doubao ..."
    )


# =========================
# config 构建
# =========================

def build_trial_config(
    base_config_path: str,
    params: Dict[str, Any],
    round_idx: int,
    out_dir: str = DEFAULT_GENERATED_DIR,
) -> Dict[str, Any]:
    cfg = load_yaml(base_config_path)
    params = sanitize_params(params)

    base_run_name = cfg.get("run_name", "agent_run")
    run_name = f"{base_run_name}_agent_round_{round_idx:02d}"

    cfg["run_name"] = run_name
    for k, v in params.items():
        cfg[k] = v

    cfg["metrics_json"] = f"/home/xth/forest_agent_project/outputs/{run_name}/metrics.json"
    cfg["details_csv"] = f"/home/xth/forest_agent_project/outputs/{run_name}/details.csv"

    out_path = Path(out_dir) / f"{run_name}.yaml"
    ensure_parent(out_path)
    save_yaml(cfg, str(out_path))

    return {
        "config": cfg,
        "config_path": str(out_path),
        "run_name": run_name,
        "params": params,
    }


# =========================
# 实验执行
# =========================

def run_experiment(config_path: str) -> Dict[str, Any]:
    cmd = [
        "python",
        DEFAULT_RUNNER,
        "--config",
        config_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    print("\n===== AGENT RUN STDOUT =====\n", res.stdout)
    print("\n===== AGENT RUN STDERR =====\n", res.stderr)

    if res.returncode != 0:
        raise RuntimeError(f"Experiment failed:\n{res.stderr}")

    cfg = load_yaml(config_path)
    metrics_json = cfg["metrics_json"]
    details_csv = cfg["details_csv"]

    if not Path(metrics_json).exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_json}")

    metrics = load_json(metrics_json)
    if not metrics:
        raise ValueError(f"metrics.json is empty: {metrics_json}")

    return {
        "metrics": metrics,
        "metrics_json": metrics_json,
        "details_csv": details_csv,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


# =========================
# Agent 逻辑
# =========================

def default_prompt_builder(
    base_cfg: Dict[str, Any],
    current_params: Dict[str, Any],
    latest_metrics: Dict[str, Any],
    bad_xiaoban_summary: Any,
    round_idx: int,
) -> str:
    return f"""
你是林业遥感单木分割参数优化助手。
请基于当前实验结果，输出下一轮 JSON 参数建议。

约束：
1. 只能从以下安全空间中选参数：
   - diam_list: {SAFE_SEARCH_SPACE["diam_list"]}
   - tile: {SAFE_SEARCH_SPACE["tile"]}
   - overlap: {SAFE_SEARCH_SPACE["overlap"]}
   - tile_overlap: {SAFE_SEARCH_SPACE["tile_overlap"]}
   - augment: {SAFE_SEARCH_SPACE["augment"]}
   - iou_merge_thr: {SAFE_SEARCH_SPACE["iou_merge_thr"]}
   - bsize 固定为 256，不允许修改
2. 只输出 JSON，不要输出解释文本
3. JSON 格式：
{{
  "params": {{
    "diam_list": "...",
    "tile": 1536,
    "overlap": 512,
    "tile_overlap": 0.35,
    "augment": true,
    "iou_merge_thr": 0.28,
    "bsize": 256
  }},
  "reason": "简要原因"
}}

当前轮次: {round_idx}
当前参数: {json.dumps(current_params, ensure_ascii=False)}
当前 metrics: {json.dumps(latest_metrics, ensure_ascii=False)}
bad xiaoban 摘要: {json.dumps(bad_xiaoban_summary, ensure_ascii=False)}
patch_id: {base_cfg.get("patch_id", "")}
forest_type: {base_cfg.get("forest_type", "")}
""".strip()


def build_prompt_compat(
    base_cfg: Dict[str, Any],
    current_params: Dict[str, Any],
    latest_metrics: Dict[str, Any],
    bad_xiaoban_summary: Any,
    round_idx: int,
) -> str:
    try:
        fn = try_import_prompt_builder()
        try:
            return fn(
                base_cfg=base_cfg,
                current_params=current_params,
                latest_metrics=latest_metrics,
                bad_xiaoban_summary=bad_xiaoban_summary,
                round_idx=round_idx,
            )
        except TypeError:
            # 回退到更简单签名
            return fn(
                current_params=current_params,
                latest_metrics=latest_metrics,
                bad_xiaoban_summary=bad_xiaoban_summary,
            )
    except Exception:
        return default_prompt_builder(
            base_cfg=base_cfg,
            current_params=current_params,
            latest_metrics=latest_metrics,
            bad_xiaoban_summary=bad_xiaoban_summary,
            round_idx=round_idx,
        )


def extract_bad_xiaoban_summary(details_csv: str, top_k: int = 5) -> Any:
    try:
        fn = try_import_detail_extractor()
        if fn is None:
            return {"message": "detail extractor not found", "details_csv": details_csv}
        try:
            return fn(details_csv=details_csv, top_k=top_k)
        except TypeError:
            return fn(details_csv, top_k)
    except Exception as e:
        return {"message": f"detail extractor failed: {repr(e)}", "details_csv": details_csv}


def normalize_agent_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    if "params" in resp and isinstance(resp["params"], dict):
        params = resp["params"]
    else:
        params = deepcopy(resp)

    params = sanitize_params(params)
    reason = resp.get("reason", "")
    return {"params": params, "reason": reason}


def save_agent_final_summary(
    final_params: dict,
    final_summary: str,
    latest_metrics_json: str = "",
    latest_details_csv: str = "",
    history: Optional[list] = None,
    out_json: str = DEFAULT_AGENT_OUT,
):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "final_params": final_params,
        "best_params": final_params,
        "final_summary": final_summary,
        "latest_metrics_json": latest_metrics_json,
        "latest_details_csv": latest_details_csv,
        "history": history or [],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[agent.graph] final summary saved to {out_json}")


def run_agent_loop(
    base_config_path: str,
    max_rounds: int = 3,
    agent_out_json: str = DEFAULT_AGENT_OUT,
):
    base_cfg = load_yaml(base_config_path)

    current_params = sanitize_params({
        "diam_list": base_cfg.get("diam_list", "96,192,320"),
        "tile": base_cfg.get("tile", 1536),
        "overlap": base_cfg.get("overlap", 512),
        "tile_overlap": base_cfg.get("tile_overlap", 0.35),
        "augment": base_cfg.get("augment", True),
        "iou_merge_thr": base_cfg.get("iou_merge_thr", 0.28),
        "bsize": 256,
    })

    best_params = deepcopy(current_params)
    best_metrics = None
    best_score = None
    latest_metrics_json = ""
    latest_details_csv = ""
    history = []

    for round_idx in range(1, max_rounds + 1):
        prompt = build_prompt_compat(
            base_cfg=base_cfg,
            current_params=current_params,
            latest_metrics=best_metrics or {},
            bad_xiaoban_summary=extract_bad_xiaoban_summary(latest_details_csv, top_k=5) if latest_details_csv else [],
            round_idx=round_idx,
        )

        raw_resp = try_call_doubao_json(prompt)
        norm = normalize_agent_response(raw_resp)
        proposed_params = norm["params"]
        reason = norm["reason"]

        trial_info = build_trial_config(
            base_config_path=base_config_path,
            params=proposed_params,
            round_idx=round_idx,
        )

        run_info = run_experiment(trial_info["config_path"])
        score = compute_single_score(run_info["metrics"])

        hist_item = {
            "round_idx": round_idx,
            "proposed_params": proposed_params,
            "reason": reason,
            "metrics": run_info["metrics"],
            "score": score,
            "metrics_json": run_info["metrics_json"],
            "details_csv": run_info["details_csv"],
            "config_path": trial_info["config_path"],
        }
        history.append(hist_item)

        if best_score is None or score < best_score:
            best_score = score
            best_params = deepcopy(proposed_params)
            best_metrics = deepcopy(run_info["metrics"])
            latest_metrics_json = run_info["metrics_json"]
            latest_details_csv = run_info["details_csv"]

        current_params = deepcopy(proposed_params)

    final_summary = (
        f"Agent completed {max_rounds} rounds. "
        f"Best score={best_score}, best_params={json.dumps(best_params, ensure_ascii=False)}"
    )

    save_agent_final_summary(
        final_params=best_params,
        final_summary=final_summary,
        latest_metrics_json=latest_metrics_json,
        latest_details_csv=latest_details_csv,
        history=history,
        out_json=agent_out_json,
    )

    return {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_score": best_score,
        "latest_metrics_json": latest_metrics_json,
        "latest_details_csv": latest_details_csv,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_config",
        default="/home/xth/forest_agent_project/configs/exp_dom194.yaml",
    )
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--out_json", default=DEFAULT_AGENT_OUT)
    args = parser.parse_args()

    run_agent_loop(
        base_config_path=args.base_config,
        max_rounds=args.max_rounds,
        agent_out_json=args.out_json,
    )


if __name__ == "__main__":
    main()