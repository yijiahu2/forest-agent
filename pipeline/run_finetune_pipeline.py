from __future__ import annotations

import argparse
import sys
from pathlib import Path

from finetune_layer.io_utils import (
    assert_exists,
    dump_json,
    load_json,
    load_yaml,
    run_cmd,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/finetune_dom194.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 预检查
    assert_exists(args.config, "finetune config")
    assert_exists(cfg["base_config"], "base_config")
    assert_exists(cfg["details_csv"], "details_csv")
    assert_exists(cfg["metrics_json"], "metrics_json")
    assert_exists(cfg["input_image"], "input_image")
    assert_exists(cfg["xiaoban_shp"], "xiaoban_shp")
    if cfg.get("dem_tif"):
        assert_exists(cfg["dem_tif"], "dem_tif")
    if cfg.get("pseudo_mask_tif"):
        assert_exists(cfg["pseudo_mask_tif"], "pseudo_mask_tif")
    if cfg.get("pseudo_inst_shp"):
        assert_exists(cfg["pseudo_inst_shp"], "pseudo_inst_shp")

    summary = {
        "config": args.config,
        "status": "running",
        "steps": [],
    }

    # 1. 伪标签筛选
    res = run_cmd([sys.executable, "-m", "finetune_layer.pseudo_label_selector", "--config", args.config])
    if res.returncode != 0:
        raise RuntimeError("pseudo_label_selector failed")
    pseudo_csv = str(out_dir / "pseudo_select" / "pseudo_candidates.csv")
    replay_csv = str(out_dir / "pseudo_select" / "replay_good_cases.csv")
    summary["steps"].append({"step": "pseudo_select", "pseudo_csv": pseudo_csv, "replay_csv": replay_csv})

    # 2. 构建 ROI 数据集
    res = run_cmd(
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
    if res.returncode != 0:
        raise RuntimeError("build_pseudo_dataset failed")
    summary["steps"].append({"step": "build_dataset", "dataset_dir": str(out_dir / "pseudo_dataset")})

    # 3. 轻量微调
    res = run_cmd([sys.executable, "-m", "finetune_layer.train_stage1_light", "--config", args.config])
    if res.returncode != 0:
        raise RuntimeError("train_stage1_light failed")

    train_summary_path = out_dir / "training" / "train_summary.json"
    train_summary = load_json(train_summary_path)
    summary["steps"].append({"step": "train_stage1_light", "train_summary_json": str(train_summary_path)})

    if train_summary.get("status") == "skipped_no_masks":
        summary["status"] = "completed_without_training"
        summary["reason"] = "没有可用伪标签 mask，已完成第五层筛选与建集，但未进入真实训练。"
        dump_json(summary, out_dir / "finetune_pipeline_summary.json")
        print("[DONE] completed_without_training")
        return

    ckpt = train_summary.get("best_ckpt")
    if not ckpt or not Path(ckpt).exists():
        summary["status"] = "completed_without_ckpt"
        summary["reason"] = "训练结束但未找到 best ckpt。"
        dump_json(summary, out_dir / "finetune_pipeline_summary.json")
        print("[DONE] completed_without_ckpt")
        return

    # 4. 生成 finetuned 配置 / 兼容性检查
    res = run_cmd([sys.executable, "-m", "finetune_layer.infer_stage1_finetuned", "--config", args.config, "--ckpt", ckpt])
    if res.returncode != 0:
        raise RuntimeError("infer_stage1_finetuned failed")

    integration_summary_path = out_dir / "finetuned_infer" / "integration_summary.json"
    integration = load_json(integration_summary_path)
    summary["steps"].append({"step": "integration_check", "integration_summary_json": str(integration_summary_path)})

    if not integration.get("can_rerun", False):
        summary["status"] = "completed_training_but_rerun_blocked"
        summary["reason"] = integration.get("reason", "当前仓库未接通 finetuned stage1 -> 全流程 rerun。")
        dump_json(summary, out_dir / "finetune_pipeline_summary.json")
        print("[DONE] completed_training_but_rerun_blocked")
        return

    ft_cfg = integration.get("exp_finetuned_yaml")
    if not ft_cfg or not Path(ft_cfg).exists():
        summary["status"] = "completed_training_but_missing_ft_cfg"
        summary["reason"] = "未生成 exp_finetuned.yaml。"
        dump_json(summary, out_dir / "finetune_pipeline_summary.json")
        print("[DONE] completed_training_but_missing_ft_cfg")
        return

    # 5. 用第二层统一 runner 重跑实验
    res = run_cmd([sys.executable, "-m", "scripts.run_zstreeseg_experiment", "--config", ft_cfg])
    if res.returncode != 0:
        raise RuntimeError("rerun experiment failed")
    summary["steps"].append({"step": "rerun_experiment", "config": ft_cfg})

    # 6. 对比微调前后
    before_csv = cfg["details_csv"]
    after_cfg = load_yaml(ft_cfg)
    after_csv = after_cfg["details_csv"]

    res = run_cmd(
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
            "--config",
            args.config,
        ]
    )
    if res.returncode != 0:
        raise RuntimeError("evaluate_finetune_gain failed")

    summary["steps"].append({"step": "compare", "compare_dir": str(out_dir / "compare")})
    summary["status"] = "success"
    dump_json(summary, out_dir / "finetune_pipeline_summary.json")
    print(f"[OK] finetune pipeline done: {out_dir}")


if __name__ == "__main__":
    main()