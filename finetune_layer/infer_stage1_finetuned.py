from __future__ import annotations

import argparse
from pathlib import Path

from finetune_layer.io_utils import dump_yaml, load_yaml


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

    suffix = str(cfg.get("finetuned_suffix", "_ft_v1"))
    new_run_name = str(base_cfg.get("run_name", "run")) + suffix

    # 这里假设 stage1 脚本未来支持 stage1_ckpt 参数
    # 第一版先把它写进 yaml，runner 里按存在即透传即可
    base_cfg["run_name"] = new_run_name
    base_cfg["stage1_ckpt"] = args.ckpt

    new_cfg_path = out_dir / "exp_finetuned.yaml"
    dump_yaml(base_cfg, new_cfg_path)

    print(f"[OK] finetuned config written: {new_cfg_path}")
    print("后续用 scripts/run_zstreeseg_experiment.py 读取该配置重跑。")


if __name__ == "__main__":
    main()