from __future__ import annotations

import argparse
from pathlib import Path
import copy

from finetune_layer.io_utils import dump_yaml, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    ckpt = str(Path(args.ckpt).resolve())

    new_cfg = copy.deepcopy(cfg)

    # 1. 写入 stage1_ckpt，供 run_zstreeseg_experiment.py 统一透传
    new_cfg["stage1_ckpt"] = ckpt

    # 2. 不再把 --ckpt 塞进 stage1_extra_args，避免重复注入
    #    若原配置已有 stage1_extra_args，则清理其中的 ckpt 相关内容
    extra_args = new_cfg.get("stage1_extra_args", [])
    if extra_args is None:
        extra_args = []

    cleaned = []
    skip_next = False
    for item in extra_args:
        if skip_next:
            skip_next = False
            continue

        s = str(item).strip()
        if s == "--ckpt":
            skip_next = True
            continue
        cleaned.append(item)

    new_cfg["stage1_extra_args"] = cleaned

    # 3. 输出目录加后缀，避免覆盖原实验输出
    suffix = str(cfg.get("finetuned_suffix", "_ft"))
    old_out = str(cfg["output_dir"])
    if not old_out.endswith(suffix):
        new_cfg["output_dir"] = old_out + suffix
    else:
        new_cfg["output_dir"] = old_out

    out_dir = Path(cfg["output_dir"]) / "finetuned_infer"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_cfg = out_dir / "exp_finetuned.yaml"
    dump_yaml(new_cfg, out_cfg)

    print(f"[OK] finetuned config written: {out_cfg}")


if __name__ == "__main__":
    main()