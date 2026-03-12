from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from finetune_layer.io_utils import dump_json, load_yaml


def run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/finetune_dom194.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"config": args.config, "steps": []}

    # 1. 伪标签筛选
    run_cmd([sys.executable, "-m", "finetune_layer.pseudo_label_selector", "--config", args.config])
    pseudo_csv = str(out_dir / "pseudo_select" / "pseudo_candidates.csv")
    replay_csv = str(out_dir / "pseudo_select" / "replay_good_cases.csv")
    summary["steps"].append(
        {"step": "pseudo_select", "pseudo_csv": pseudo_csv, "replay_csv": replay_csv}
    )

    # 2. 构建 ROI 数据集
    run_cmd(
        [
            sys.executable,
            "-m",
            "finetune_layer.build_pseudo_dataset",
            "--config",
            args.config,
            "--pseudo_csv",
            pseudo_csv,
            "--replay_good_csv",
            replay_csv,
        ]
    )
    summary["steps"].append(
        {"step": "build_dataset", "dataset_dir": str(out_dir / "pseudo_dataset")}
    )

    # 3. 训练（tiny 或 external）
    run_cmd([sys.executable, "-m", "finetune_layer.train_stage1_light", "--config", args.config])

    train_summary_path = out_dir / "training" / "train_summary.json"
    if not train_summary_path.exists():
        raise RuntimeError(f"训练完成后未找到 train_summary.json: {train_summary_path}")

    train_summary = load_yaml(str(train_summary_path)) if train_summary_path.suffix in [".yaml", ".yml"] else None
    if train_summary is None:
        import json
        with open(train_summary_path, "r", encoding="utf-8") as f:
            train_summary = json.load(f)

    ckpt = train_summary.get("best_ckpt")
    if not ckpt:
        raise RuntimeError(f"train_summary.json 中未找到 best_ckpt: {train_summary_path}")

    summary["steps"].append(
        {"step": "train_stage1_light", "backend": train_summary.get("backend"), "ckpt": ckpt}
    )

    # 4. 生成 finetuned 配置
    run_cmd(
        [
            sys.executable,
            "-m",
            "finetune_layer.infer_stage1_finetuned",
            "--config",
            args.config,
            "--ckpt",
            ckpt,
        ]
    )
    ft_cfg = str(out_dir / "finetuned_infer" / "exp_finetuned.yaml")
    summary["steps"].append({"step": "make_finetuned_config", "config": ft_cfg})

    enable_rerun = bool(cfg.get("enable_rerun_after_finetune", False))
    summary["enable_rerun_after_finetune"] = enable_rerun

    # 5. 如果开启，则用统一 runner 重跑
    if enable_rerun:
        run_cmd([sys.executable, "-m", "scripts.run_zstreeseg_experiment", "--config", ft_cfg])
        summary["steps"].append({"step": "rerun_experiment", "config": ft_cfg})

        before_csv = cfg.get("details_csv")
        after_cfg = load_yaml(ft_cfg)
        after_csv = str(Path(after_cfg["output_dir"]) / "details.csv")

        if before_csv:
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "finetune_layer.evaluate_finetune_gain",
                    "--before_csv",
                    before_csv,
                    "--after_csv",
                    after_csv,
                    "--out_dir",
                    str(out_dir / "compare"),
                ]
            )
            summary["steps"].append({"step": "compare", "compare_dir": str(out_dir / "compare")})
    else:
        print("[SKIP] rerun after finetune is disabled by config")

    dump_json(summary, out_dir / "finetune_pipeline_summary.json")
    print(f"[OK] finetune pipeline done: {out_dir}")


if __name__ == "__main__":
    main()