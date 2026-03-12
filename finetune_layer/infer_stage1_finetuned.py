from __future__ import annotations

import argparse
import copy
from pathlib import Path

from finetune_layer.io_utils import dump_json, dump_yaml, load_yaml, to_bool


def _strip_ckpt_args(extra_args):
    if extra_args is None:
        return []

    if not isinstance(extra_args, list):
        extra_args = [str(extra_args)]

    cleaned = []
    skip_next = False

    for item in extra_args:
        s = str(item).strip()

        if skip_next:
            skip_next = False
            continue

        if s in {"--ckpt", "--checkpoint"}:
            skip_next = True
            continue

        if s.startswith("--ckpt=") or s.startswith("--checkpoint="):
            continue

        cleaned.append(item)

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"]) / "finetuned_infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "base_config": cfg.get("base_config"),
        "ckpt": args.ckpt,
        "can_rerun": False,
        "reason": "",
        "exp_finetuned_yaml": None,
    }

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        summary["reason"] = "训练权重不存在，无法进入回灌阶段。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] ckpt not found.")
        return

    if not to_bool(cfg.get("enable_rerun_after_finetune", False), default=False):
        summary["reason"] = "enable_rerun_after_finetune=false；当前仅执行第五层训练，不回灌 rerun。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] rerun disabled by config.")
        return

    base_cfg_path = cfg.get("base_config")
    if not base_cfg_path:
        summary["reason"] = "缺少 base_config，无法生成 rerun 配置。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] missing base_config.")
        return

    base_cfg = load_yaml(base_cfg_path)

    if not base_cfg.get("stage1_script"):
        summary["reason"] = "base_config 未提供 stage1_script。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] missing stage1_script.")
        return

    new_cfg = copy.deepcopy(base_cfg)

    suffix = str(cfg.get("finetuned_suffix", "_ft_v1"))
    old_run_name = str(base_cfg.get("run_name", "run"))
    new_cfg["run_name"] = old_run_name if old_run_name.endswith(suffix) else old_run_name + suffix

    rerun_out_dir = Path(cfg["output_dir"]) / "rerun_after_finetune"
    rerun_out_dir.mkdir(parents=True, exist_ok=True)

    new_cfg["output_dir"] = str(rerun_out_dir)
    new_cfg["metrics_json"] = str(rerun_out_dir / "metrics.json")
    new_cfg["details_csv"] = str(rerun_out_dir / "details.csv")

    # 只保留一个权重入口
    new_cfg["stage1_ckpt"] = str(ckpt_path.resolve())

    # 清理旧的 ckpt 参数，避免重复注入
    new_cfg["stage1_extra_args"] = _strip_ckpt_args(new_cfg.get("stage1_extra_args", []))

    out_cfg = out_dir / "exp_finetuned.yaml"
    dump_yaml(new_cfg, out_cfg)

    summary["can_rerun"] = True
    summary["reason"] = "已生成可用于 rerun 的 finetuned 配置。"
    summary["exp_finetuned_yaml"] = str(out_cfg)
    summary["rerun_output_dir"] = str(rerun_out_dir)
    summary["rerun_metrics_json"] = new_cfg["metrics_json"]
    summary["rerun_details_csv"] = new_cfg["details_csv"]

    dump_json(summary, out_dir / "integration_summary.json")
    print(f"[OK] finetuned config written: {out_cfg}")


if __name__ == "__main__":
    main()