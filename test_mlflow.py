
import mlflow
import random
from pathlib import Path

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("forest_agent_dev")

with mlflow.start_run(run_name="smoke_test_001"):
    # 1. 记录参数
    mlflow.log_param("model", "ZS-TreeSeg")
    mlflow.log_param("compartment_id", "demo_compartment")
    mlflow.log_param("patch_id", "demo_patch_001")
    mlflow.log_param("diam_min", 4)
    mlflow.log_param("diam_max", 10)
    mlflow.log_param("tile_size", 512)

    # 2. 记录指标
    f1 = round(random.uniform(0.75, 0.90), 4)
    iou = round(random.uniform(0.65, 0.85), 4)
    tree_count_error = round(random.uniform(5, 30), 2)

    mlflow.log_metric("F1", f1)
    mlflow.log_metric("IoU", iou)
    mlflow.log_metric("tree_count_error", tree_count_error)

    # 3. 记录标签
    mlflow.set_tag("stage", "smoke_test")
    mlflow.set_tag("forest_type", "dense_mixed")
    mlflow.set_tag("agent_version", "v1")

    # 4. 记录一个artifact文件
    out_dir = Path("artifacts/test_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_file = out_dir / "result.txt"
    txt_file.write_text(
        f"F1={f1}\nIoU={iou}\ntree_count_error={tree_count_error}\n",
        encoding="utf-8"
    )
    mlflow.log_artifact(str(txt_file), artifact_path="outputs")

print("MLflow test run completed.")