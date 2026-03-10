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

    # 默认不启用 rerun，除非你后面显式打开
    if not to_bool(cfg.get("enable_rerun_after_finetune", False), default=False):
        summary["reason"] = "enable_rerun_after_finetune=false；当前仅执行第五层数据与训练，不回灌到 stage1 推理。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] rerun disabled by config.")
        return

    stage1_script = base_cfg.get("stage1_script")
    if not stage1_script:
        summary["reason"] = "base_config 未提供 stage1_script。"
        dump_json(summary, out_dir / "integration_summary.json")
        print("[SKIP] missing stage1_script.")
        return

    # 这里只能写入新配置；是否真正生效取决于 scripts.run_zstreeseg_experiment.py
    # 当前你的 runner 实际并不会把 stage1_ckpt 透传给 stage1_script，因此这里只做兼容保留。
    suffix = str(cfg.get("finetuned_suffix", "_ft_v1"))
    new_run_name = str(base_cfg.get("run_name", "run")) + suffix

    base_cfg["run_name"] = new_run_name
    base_cfg["stage1_ckpt"] = args.ckpt

    # 预留一个显式字段，等你将来改 run_zstreeseg_experiment.py 时直接透传
    extra_args = base_cfg.get("stage1_extra_args", [])
    if not isinstance(extra_args, list):
        extra_args = []
    if "--ckpt" not in extra_args and "--checkpoint" not in extra_args:
        extra_args.extend(["--ckpt", args.ckpt])
    base_cfg["stage1_extra_args"] = extra_args

    new_cfg_path = out_dir / "exp_finetuned.yaml"
    dump_yaml(base_cfg, new_cfg_path)

    summary["can_rerun"] = True
    summary["reason"] = (
        "已生成 finetuned 配置；但只有当 scripts.run_zstreeseg_experiment.py "
        "把 stage1_ckpt / stage1_extra_args 传给 stage1_script 时，微调权重才会真正生效。"
    )
    summary["exp_finetuned_yaml"] = str(new_cfg_path)
    dump_json(summary, out_dir / "integration_summary.json")

    print(f"[OK] finetuned config written: {new_cfg_path}")


if __name__ == "__main__":
    main()