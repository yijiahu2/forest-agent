from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from finetune_layer.io_utils import dump_json, load_yaml


def run_cmd(cmd: list[str], cwd: str | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败: {' '.join(cmd)}")


def load_json_file(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


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

    train_summary = load_json_file(train_summary_path)
    summary["train_summary_json"] = str(train_summary_path)

    ckpt = train_summary.get("best_ckpt")
    if not ckpt:
        raise RuntimeError(f"train_summary.json 中未找到 best_ckpt: {train_summary_path}")

    if not Path(ckpt).exists():
        raise RuntimeError(f"best_ckpt 文件不存在: {ckpt}")

    summary["best_ckpt"] = ckpt
    summary["steps"].append(
        {
            "step": "train_stage1_light",
            "backend": train_summary.get("backend"),
            "ckpt": ckpt,
            "train_summary_json": str(train_summary_path),
        }
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

    ft_cfg = out_dir / "finetuned_infer" / "exp_finetuned.yaml"
    if not ft_cfg.exists():
        raise RuntimeError(f"未生成 exp_finetuned.yaml: {ft_cfg}")

    integration_summary_path = out_dir / "finetuned_infer" / "integration_summary.json"
    if integration_summary_path.exists():
        integration_summary = load_json_file(integration_summary_path)
        summary["steps"].append(
            {
                "step": "integration_check",
                "integration_summary_json": str(integration_summary_path),
                "can_rerun": integration_summary.get("can_rerun"),
                "reason": integration_summary.get("reason"),
            }
        )
    else:
        integration_summary = None

    summary["steps"].append({"step": "make_finetuned_config", "config": str(ft_cfg)})

    enable_rerun = to_bool(cfg.get("enable_rerun_after_finetune", False))
    summary["enable_rerun_after_finetune"] = enable_rerun

    # 5. 如果开启，则用统一 runner 重跑
    if enable_rerun:
        if integration_summary is not None and not integration_summary.get("can_rerun", False):
            raise RuntimeError(
                f"enable_rerun_after_finetune=true，但 integration_summary 显示不能 rerun: "
                f"{integration_summary.get('reason', 'unknown reason')}"
            )

        run_cmd([sys.executable, "-m", "scripts.run_zstreeseg_experiment", "--config", str(ft_cfg)])
        summary["steps"].append({"step": "rerun_experiment", "config": str(ft_cfg)})

        before_csv = cfg.get("details_csv")
        after_cfg = load_yaml(str(ft_cfg))

        # 优先使用 finetuned config 中显式写入的 details_csv
        after_csv = after_cfg.get("details_csv")
        if not after_csv:
            after_csv = str(Path(after_cfg["output_dir"]) / "details.csv")
        else:
            after_csv = str(after_csv)

        if not Path(after_csv).exists():
            raise RuntimeError(f"rerun 完成后未找到 after_csv: {after_csv}")

        summary["steps"].append({"step": "rerun_output_check", "after_csv": after_csv})

        if before_csv:
            before_csv = str(before_csv)
            if not Path(before_csv).exists():
                raise RuntimeError(f"before_csv 不存在: {before_csv}")

            compare_dir = out_dir / "compare"
            compare_dir.mkdir(parents=True, exist_ok=True)

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
                    str(compare_dir),
                    "--config",
                    args.config,
                ]
            )
            summary["compare_json"] = str(compare_dir / "finetune_gain_summary.json")
            summary["steps"].append(
                {
                    "step": "compare",
                    "compare_dir": str(compare_dir),
                    "compare_json": str(compare_dir / "finetune_gain_summary.json"),
                    "before_csv": before_csv,
                    "after_csv": after_csv,
                }
            )
        else:
            print("[SKIP] details_csv 缺失，跳过 compare")
            summary["steps"].append(
                {
                    "step": "compare",
                    "status": "skipped",
                    "reason": "details_csv missing in original finetune config",
                }
            )
    else:
        print("[SKIP] rerun after finetune is disabled by config")
        summary["steps"].append(
            {
                "step": "rerun_experiment",
                "status": "skipped",
                "reason": "enable_rerun_after_finetune=false",
            }
        )

    dump_json(summary, out_dir / "finetune_pipeline_summary.json")
    print(f"[OK] finetune pipeline done: {out_dir}")


if __name__ == "__main__":
    main()
