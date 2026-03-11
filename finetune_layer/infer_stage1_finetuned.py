from __future__ import annotations

import argparse
from pathlib import Path

from finetune_layer.io_utils import dump_json, dump_yaml, load_yaml, to_bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_cfg_path = cfg["base_config"]
    base_cfg = load_yaml(base_cfg_path)

    out_dir = Path(cfg["output_dir"]) / "finetuned_infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "base_config": base_cfg_path,
        "ckpt": args.ckpt,
        "can_rerun": False,
        "reason": "",
        "exp_finetuned_yaml": None,
    }

    if not Path(args.ckpt).exists():
        summary["reason"] = "训练权重不存在，无法进入回灌阶段。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] ckpt not found.")
        return

    if not to_bool(cfg.get("enable_rerun_after_finetune", False), default=False):
        summary["reason"] = "enable_rerun_after_finetune=false；本轮只产出权重与回灌配置。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] rerun disabled by config.")
        return

    suffix = str(cfg.get("finetuned_suffix", "_ft_v1"))
    new_cfg = dict(base_cfg)

    new_cfg["run_name"] = str(base_cfg.get("run_name", "run")) + suffix
    new_cfg["stage1_ckpt"] = args.ckpt

    extra_args = new_cfg.get("stage1_extra_args", [])
    if not isinstance(extra_args, list):
        extra_args = []
    if "--ckpt" not in extra_args and "--checkpoint" not in extra_args:
        extra_args.extend(["--ckpt", args.ckpt])
    new_cfg["stage1_extra_args"] = extra_args

    ft_output_dir = Path(base_cfg["output_dir"]).resolve().parent / f"{Path(base_cfg['output_dir']).name}{suffix}"
    new_cfg["output_dir"] = str(ft_output_dir)
    new_cfg["metrics_json"] = str(ft_output_dir / "metrics.json")
    new_cfg["details_csv"] = str(ft_output_dir / "details.csv")

    new_cfg_path = out_dir / "exp_finetuned.yaml"
    dump_yaml(new_cfg, new_cfg_path)

    summary["can_rerun"] = True
    summary["reason"] = "已生成 finetuned exp 配置，runner 现在会透传 stage1_ckpt / stage1_extra_args。"
    summary["exp_finetuned_yaml"] = str(new_cfg_path)
    dump_json(summary, out_dir / "integration_summary.json")

    print(f"[OK] finetuned config written: {new_cfg_path}")


if __name__ == "__main__":
    main()