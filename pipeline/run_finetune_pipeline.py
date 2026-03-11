from __future__ import annotations

import argparse
import sys
from pathlib import Path

from finetune_layer.io_utils import (
    assert_exists,
    dump_json,
    dump_yaml,
    load_json,
    load_yaml,
    run_cmd,
)


def _resolve_from_pipeline_run(cfg: dict) -> dict:
    run_dir = cfg.get("pipeline_run_dir")
    if not run_dir:
        return cfg

    run_dir = Path(run_dir)
    summary_path = run_dir / "pipeline_summary.json"
    state_path = run_dir / "pipeline_state.json"

    if not summary_path.exists() and not state_path.exists():
        raise FileNotFoundError(f"pipeline_run_dir 下未找到 pipeline_summary.json 或 pipeline_state.json: {run_dir}")

    summary = load_json(summary_path) if summary_path.exists() else {}
    state = load_json(state_path) if state_path.exists() else {}

    stages = summary.get("stages", {})
    final_artifacts = summary.get("final_artifacts", {})

    local_refine = stages.get("local_refine", {})
    baseline = stages.get("baseline", {})

    cfg = dict(cfg)

    cfg["details_csv"] = (
        local_refine.get("merged_details_csv")
        or final_artifacts.get("global_details_csv")
        or baseline.get("details_csv")
        or cfg.get("details_csv")
    )
    cfg["metrics_json"] = (
        local_refine.get("merged_metrics_json")
        or final_artifacts.get("global_metrics_json")
        or baseline.get("metrics_json")
        or cfg.get("metrics_json")
    )
    cfg["pseudo_inst_shp"] = (
        local_refine.get("merged_shp")
        or final_artifacts.get("global_inst_shp")
        or baseline.get("inst_shp")
        or cfg.get("pseudo_inst_shp")
    )

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/finetune_dom194.yaml")
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    cfg = _resolve_from_pipeline_run(raw_cfg)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_cfg_path = out_dir / "effective_finetune_config.yaml"
    dump_yaml(cfg, effective_cfg_path)

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
        "effective_config": str(effective_cfg_path),
        "status": "running",
        "steps": [],
        "resolved_inputs": {
            "details_csv": cfg["details_csv"],
            "metrics_json": cfg["metrics_json"],
            "pseudo_inst_shp": cfg.get("pseudo_inst_shp"),
        },
    }

    res = run_cmd(
        [sys.executable, "-m", "finetune_layer.pseudo_label_selector", "--config", str(effective_cfg_path)]
    )
    if res.returncode != 0:
        raise RuntimeError("pseudo_label_selector failed")

    pseudo_csv = str(out_dir / "pseudo_select" / "pseudo_candidates.csv")
    replay_csv = str(out_dir / "pseudo_select" / "replay_good_cases.csv")
    summary["steps"].append({"step": "pseudo_select", "pseudo_csv": pseudo_csv, "replay_csv": replay_csv})

    res = run_cmd(
        [
            sys.executable,
            "-m",
            "finetune_layer.build_pseudo_dataset",
            "--config",
            str(effective_cfg_path),
            "--pseudo_csv",
            pseudo_csv,
            "--replay_good_csv",
            replay_csv,
        ]
    )
    if res.returncode != 0:
        raise RuntimeError("build_pseudo_dataset failed")
    summary["steps"].append({"step": "build_dataset", "dataset_dir": str(out_dir / "pseudo_dataset")})

    res = run_cmd(
        [sys.executable, "-m", "finetune_layer.train_stage1_light", "--config", str(effective_cfg_path)]
    )
    if res.returncode != 0:
        raise RuntimeError("train_stage1_light failed")

    train_summary_path = out_dir / "training" / "train_summary.json"
    train_summary = load_json(train_summary_path)
    summary["steps"].append({"step": "train_stage1_light", "train_summary_json": str(train_summary_path)})

    if train_summary.get("status") == "skipped_no_masks":
        summary["status"] = "completed_without_training"
        summary["reason"] = "没有可用伪标签 mask。"
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

    if not bool(train_summary.get("ckpt_compatible_with_stage1", False)):
        summary["status"] = "trained_but_not_stage1_compatible"
        summary["reason"] = "当前训练产物不是前四层 stage1 可直接复用的权重；已完成建集与训练，但不进入 rerun。"
        summary["artifacts"] = {"best_ckpt": ckpt}
        dump_json(summary, out_dir / "finetune_pipeline_summary.json")
        print("[DONE] trained_but_not_stage1_compatible")
        return

    res = run_cmd(
        [
            sys.executable,
            "-m",
            "finetune_layer.infer_stage1_finetuned",
            "--config",
            str(effective_cfg_path),
            "--ckpt",
            ckpt,
        ]
    )
    if res.returncode != 0:
        raise RuntimeError("infer_stage1_finetuned failed")

    integration_summary_path = out_dir / "finetuned_infer" / "integration_summary.json"
    integration = load_json(integration_summary_path)
    summary["steps"].append({"step": "integration_check", "integration_summary_json": str(integration_summary_path)})

    if not integration.get("can_rerun", False):
        summary["status"] = "completed_training_but_rerun_blocked"
        summary["reason"] = integration.get("reason", "integration blocked")
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

    res = run_cmd([sys.executable, "-m", "scripts.run_zstreeseg_experiment", "--config", ft_cfg])
    if res.returncode != 0:
        raise RuntimeError("rerun experiment failed")
    summary["steps"].append({"step": "rerun_experiment", "config": ft_cfg})

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
            str(effective_cfg_path),
        ]
    )
    if res.returncode != 0:
        raise RuntimeError("evaluate_finetune_gain failed")

    summary["steps"].append({"step": "compare", "compare_dir": str(out_dir / "compare")})
    summary["status"] = "success"
    summary["artifacts"] = {
        "best_ckpt": ckpt,
        "exp_finetuned_yaml": ft_cfg,
        "compare_dir": str(out_dir / "compare"),
    }
    dump_json(summary, out_dir / "finetune_pipeline_summary.json")
    print(f"[OK] finetune pipeline done: {out_dir}")


if __name__ == "__main__":
    main()