import os
import pandas as pd

from main import update_metadata_lists
from train_utils import run_pipeline_multistage, save_artifacts

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", ".."))

artifacts_dir = os.path.join(
    ROOT_DIR, "lead_time_platform_final", "platform", "artifacts_fixed"
)
data_dir = os.path.join(BACKEND_DIR, "data")
csv_path = os.path.join(ROOT_DIR, "shipdata.csv")

df_raw = pd.read_csv(csv_path)

print("Training initial multi-stage models with GA + Asymmetric CQR calibration...")
models, state_bundle, metrics = run_pipeline_multistage(
    df_raw, source="shipdata.csv", use_genetic_tuning=True
)
save_artifacts(artifacts_dir, models, state_bundle, metrics)
update_metadata_lists(df_raw, data_dir)
print("Initial models and metadata lists successfully built and saved!")
for s, info in metrics["stages"].items():
    t = info["test"]
    print(
        f"Stage {s}: R2={t['r2']} MAE={t['mae']} Coverage={t['coverage_q10_q90']} "
        f"BelowBest={t['below_best_rate']} AboveWorst={t['above_worst_rate']}"
    )