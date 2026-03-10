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

    summary = {"config": args.config, "steps": []}

    # 1. 伪标签筛选
    run_cmd([sys.executable, "-m", "finetune_layer.pseudo_label_selector", "--config", args.config])
    pseudo_csv = str(out_dir / "pseudo_select" / "pseudo_candidates.csv")
    replay_csv = str(out_dir / "pseudo_select" / "replay_good_cases.csv")
    summary["steps"].append({"step": "pseudo_select", "pseudo_csv": pseudo_csv, "replay_csv": replay_csv})

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
    summary["steps"].append({"step": "build_dataset", "dataset_dir": str(out_dir / "pseudo_dataset")})

    # 3. 轻量微调
    run_cmd([sys.executable, "-m", "finetune_layer.train_stage1_light", "--config", args.config])
    ckpt = str(out_dir / "training" / "best_stage1_light.pt")
    summary["steps"].append({"step": "train_stage1_light", "ckpt": ckpt})

    # 4. 生成 finetuned 配置
    run_cmd([sys.executable, "-m", "finetune_layer.infer_stage1_finetuned", "--config", args.config, "--ckpt", ckpt])
    ft_cfg = str(out_dir / "finetuned_infer" / "exp_finetuned.yaml")
    summary["steps"].append({"step": "make_finetuned_config", "config": ft_cfg})

    # 5. 用第二层统一 runner 重跑实验
    run_cmd([sys.executable, "-m", "scripts.run_zstreeseg_experiment", "--config", ft_cfg])
    summary["steps"].append({"step": "rerun_experiment", "config": ft_cfg})

    # 6. 对比微调前后
    before_csv = cfg["details_csv"]

    # 这个路径取决于你的 runner 输出位置；第一版先约定为：
    # base_cfg.output_dir/details.csv
    after_csv = str(Path(load_yaml(ft_cfg)["output_dir"]) / "details.csv")

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

    dump_json(summary, out_dir / "finetune_pipeline_summary.json")
    print(f"[OK] finetune pipeline done: {out_dir}")


if __name__ == "__main__":
    main()