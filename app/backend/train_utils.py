"""
Shared training/evaluation utilities for the Lead Time Prediction platform.

Multi-stage remaining-time design + Asymmetric CQR calibration
--------------------------------------------------------------
At each stage s in {0, 1, 2, 3}, the model predicts REMAINING days from the
latest observed milestone to factory arrival:
    total_lead_time = elapsed_days_at_stage_s + remaining_days

Why remaining_days + Asymmetric Conformalized Quantile Regression (CQR):
1. Guarantees total predictions can never fall below already-elapsed days.
2. Removes piecewise-constant tree discretization error on elapsed durations.
3. Asymmetric split/rolling-time CQR calibrates (delta_low, delta_high) on
   out-of-time validation blocks inside the training split so that:
   - optimistic (best_case) is genuinely shorter than actual (~90%+ of outcomes
     take at least as long),
   - realistic (expected) targets the median/center closely,
   - pessimistic (worst_case) is genuinely longer than actual (~90%+ of outcomes
     finish before or at worst_case).
"""

import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)
from data_loader import clean_dataframe  # noqa: E402

CAT_FEATURES = ["country_clean", "supplier_clean", "sap_code_clean"]

# Legacy stage features kept only for backward compatibility with old artifacts.
LEGACY_BASE_FEATURES = [
    "weight",
    "month",
    "is_winter",
    "is_peak",
    "supplier_tier",
    "supplier_hist_mean",
    "recent_3_mean",
    "country_clean",
    "supplier_clean",
    "sap_code_clean",
]

BASE_FEATURES = [
    "weight",
    "log_weight",
    "month",
    "reg_doy_sin",
    "reg_doy_cos",
    "is_winter",
    "is_peak",
    "supplier_count",
    "supplier_tier",
    "supplier_hist_mean",
    "recent_3_mean",
    "country_hist_mean",
    "sap_hist_mean",
    "global_recent_mean",
    "country_clean",
    "supplier_clean",
    "sap_code_clean",
]

STAGE_EXTRA = {
    0: [],
    1: ["dur_reg_to_min", "ministry_month"],
    2: [
        "dur_reg_to_min", "ministry_month",
        "dur_min_to_ship", "elapsed_reg_to_ship", "ship_month",
    ],
    3: [
        "dur_reg_to_min", "ministry_month",
        "dur_min_to_ship", "elapsed_reg_to_ship", "ship_month",
        "dur_ship_to_cust", "elapsed_reg_to_cust", "customs_month",
    ],
}

STAGES = [0, 1, 2, 3]
STAGE_NAMES = {
    0: "order_registration_day_zero",
    1: "post_ministry_approval",
    2: "post_vessel_departure",
    3: "post_customs_arrival",
}
STAGE_LABELS_FA = {
    0: "لحظه ثبت سفارش",
    1: "پس از مجوز وزارتخانه",
    2: "پس از حرکت کشتی از مبدأ",
    3: "پس از ورود به گمرک مقصد",
}

DEFAULT_PARAMS = {
    "iterations": 350,
    "learning_rate": 0.04,
    "depth": 4,
    "l2_leaf_reg": 6.0,
    "random_seed": 42,
    "allow_writing_files": False,
}

DEFAULT_TEST_FRAC = 0.20
TARGET_TAIL_ALPHA = 0.08  # ~8% lower tail + ~8% upper tail => ~84% interval coverage


def feature_cols_for_stage(stage: int, legacy: bool = False) -> list:
    """Ordered feature list = base features + cumulative stage extras."""
    if stage not in STAGE_EXTRA:
        raise ValueError(f"Unknown stage: {stage}")
    base = LEGACY_BASE_FEATURES if legacy else BASE_FEATURES
    return base + STAGE_EXTRA[stage]


def elapsed_for_stage(df: pd.DataFrame, stage: int) -> np.ndarray:
    """Known elapsed days from registration at the start of stage."""
    if stage == 0:
        return np.zeros(len(df), dtype=float)
    if stage == 1:
        return df["dur_reg_to_min"].to_numpy(dtype=float)
    if stage == 2:
        return df["elapsed_reg_to_ship"].to_numpy(dtype=float)
    if stage == 3:
        return df["elapsed_reg_to_cust"].to_numpy(dtype=float)
    raise ValueError(f"Unknown stage: {stage}")


def build_expanding_stats(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """Strictly causal expanding historical statistics.

    df_sorted must be sorted by registery-date. For each row i, only orders
    whose factory-arrive-date < row[i].registery-date are visible, preventing
    future target leakage while capturing supplier, country, SAP, and recent
    macro trends.
    """
    leads = df_sorted["LeadTime"].to_numpy(dtype=float)
    reg_ts = df_sorted["registery-date"].to_numpy()
    arr_ts = df_sorted["factory-arrive-date"].to_numpy()
    sups = df_sorted["supplier_clean"].to_numpy()
    ctrs = df_sorted["country_clean"].to_numpy()
    saps = df_sorted["sap_code_clean"].to_numpy()

    n = len(df_sorted)
    arr_order = np.argsort(arr_ts)
    arr_ptr = 0

    completed_leads = []
    sup_hist = defaultdict(list)
    ctr_hist = defaultdict(list)
    sap_hist = defaultdict(list)

    out_sup_mean, out_sup_std, out_sup_cnt, out_r3 = [], [], [], []
    out_ctr_mean, out_ctr_cnt = [], []
    out_sap_mean, out_sap_cnt = [], []
    out_global_recent = []

    for i in range(n):
        cur_reg = reg_ts[i]
        while arr_ptr < n and arr_ts[arr_order[arr_ptr]] < cur_reg:
            idx = arr_order[arr_ptr]
            lt = leads[idx]
            completed_leads.append(lt)
            sup_hist[sups[idx]].append(lt)
            ctr_hist[ctrs[idx]].append(lt)
            sap_hist[saps[idx]].append(lt)
            arr_ptr += 1

        out_global_recent.append(
            float(np.mean(completed_leads[-20:])) if completed_leads else np.nan
        )

        sh = sup_hist[sups[i]]
        out_sup_cnt.append(len(sh))
        if sh:
            out_sup_mean.append(float(np.mean(sh)))
            out_sup_std.append(float(np.std(sh, ddof=1)) if len(sh) > 1 else 0.0)
            out_r3.append(float(np.mean(sh[-3:])))
        else:
            out_sup_mean.append(np.nan)
            out_sup_std.append(0.0)
            out_r3.append(np.nan)

        ch = ctr_hist[ctrs[i]]
        out_ctr_cnt.append(len(ch))
        out_ctr_mean.append(float(np.mean(ch)) if ch else np.nan)

        ph = sap_hist[saps[i]]
        out_sap_cnt.append(len(ph))
        out_sap_mean.append(float(np.mean(ph)) if ph else np.nan)

    out = df_sorted.copy()
    out["supplier_hist_mean_raw"] = out_sup_mean
    out["supplier_std"] = out_sup_std
    out["supplier_count"] = out_sup_cnt
    out["recent_3_raw"] = out_r3
    out["country_hist_mean_raw"] = out_ctr_mean
    out["country_count"] = out_ctr_cnt
    out["sap_hist_mean_raw"] = out_sap_mean
    out["sap_count"] = out_sap_cnt
    out["global_recent_raw"] = out_global_recent
    return out


def make_feature_frame(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    """Derive smoothed, leakage-free model features from expanded dataframe."""
    df = df.copy()
    df["global_recent_mean"] = df["global_recent_raw"].fillna(global_mean)

    # Hierarchical Bayesian shrinkage to avoid noisy 1-sample spikes
    m_ctr = 3.0
    ctr_cnt = df["country_count"].to_numpy(dtype=float)
    ctr_raw = df["country_hist_mean_raw"].fillna(global_mean).to_numpy(dtype=float)
    df["country_hist_mean"] = (ctr_cnt * ctr_raw + m_ctr * global_mean) / (ctr_cnt + m_ctr)

    m_sap = 3.0
    sap_cnt = df["sap_count"].to_numpy(dtype=float)
    sap_raw = df["sap_hist_mean_raw"].fillna(global_mean).to_numpy(dtype=float)
    df["sap_hist_mean"] = (sap_cnt * sap_raw + m_sap * global_mean) / (sap_cnt + m_sap)

    m_sup = 3.0
    sup_cnt = df["supplier_count"].to_numpy(dtype=float)
    sup_prior = df["country_hist_mean"].to_numpy(dtype=float)
    sup_raw = df["supplier_hist_mean_raw"].fillna(df["country_hist_mean"]).to_numpy(dtype=float)
    df["supplier_hist_mean"] = (sup_cnt * sup_raw + m_sup * sup_prior) / (sup_cnt + m_sup)

    r3_cnt = np.minimum(sup_cnt, 3.0)
    r3_raw = df["recent_3_raw"].fillna(df["supplier_hist_mean"]).to_numpy(dtype=float)
    df["recent_3_mean"] = (
        r3_cnt * r3_raw + 2.0 * df["supplier_hist_mean"].to_numpy(dtype=float)
    ) / (r3_cnt + 2.0)

    # Require >= 3 completed orders before assigning Tier 0 (consistent)
    def _tier(row):
        if row["supplier_count"] < 3:
            return 1
        s = row["supplier_std"]
        return 0 if s < 15 else (1 if s < 30 else 2)

    df["supplier_tier"] = df.apply(_tier, axis=1)

    df["log_weight"] = np.log1p(np.maximum(df["weight"].to_numpy(dtype=float), 0.0))

    m = df["registery-date"].dt.month
    doy = df["registery-date"].dt.dayofyear.to_numpy(dtype=float)
    df["month"] = m
    df["reg_doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["reg_doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    df["is_winter"] = m.isin([12, 1, 2]).astype(int)
    df["is_peak"] = m.isin([11, 12, 1]).astype(int)

    # Milestone-derived columns (selected only by stages that observed them)
    df["ministry_month"] = df["ministry-date"].dt.month
    df["ship_month"] = df["ship-date"].dt.month
    df["customs_month"] = df["costums-arrive-date"].dt.month

    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
    return df


def chronological_split(
    df_sorted: pd.DataFrame, test_frac: float = DEFAULT_TEST_FRAC
) -> tuple:
    """Split a date-sorted frame into (train, test) by position."""
    n = len(df_sorted)
    if n < 10:
        raise ValueError(f"Not enough rows for a train/test split (n={n}).")
    split = int(n * (1.0 - test_frac))
    if split < 5 or n - split < 5:
        raise ValueError(
            f"Split {split}/{n - split} too small; upload more rows or reduce test_frac."
        )
    return df_sorted.iloc[:split].copy(), df_sorted.iloc[split:].copy()


def _fit_single_quantile(X, y, alpha: float, params: dict) -> CatBoostRegressor:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    p.pop("alpha_low", None)
    p.pop("alpha_high", None)
    model = CatBoostRegressor(
        **p,
        loss_function=f"Quantile:alpha={alpha}",
        cat_features=CAT_FEATURES,
        verbose=False,
    )
    model.fit(X, y)
    return model


def train_quantile_models(
    X_train: pd.DataFrame, y_train: np.ndarray, params: dict = None
):
    """Train q10 / q50 / q90 CatBoost quantile regressors."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    alpha_low = float(p.pop("alpha_low", 0.10))
    alpha_high = float(p.pop("alpha_high", 0.90))

    cb_10 = _fit_single_quantile(X_train, y_train, alpha_low, p)
    cb_50 = _fit_single_quantile(X_train, y_train, 0.50, p)
    cb_90 = _fit_single_quantile(X_train, y_train, alpha_high, p)
    return cb_10, cb_50, cb_90


def apply_calibration_and_order(
    raw_low, raw_med, raw_high, delta_low: float, delta_high: float
):
    """Apply CQR offsets to remaining-day predictions and guarantee monotonicity."""
    rem_low = np.maximum(0.0, np.asarray(raw_low, dtype=float) - float(delta_low))
    rem_med = np.maximum(0.0, np.asarray(raw_med, dtype=float))
    rem_high = np.maximum(0.0, np.asarray(raw_high, dtype=float) + float(delta_high))

    rem_low = np.minimum(rem_low, rem_med)
    rem_high = np.maximum(rem_high, rem_med)
    return rem_low, rem_med, rem_high


def evaluate_quantiles(y_true, p10, p50, p90) -> dict:
    """Test/train metrics for the median model + asymmetric interval diagnostics."""
    y_true = np.asarray(y_true, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    p50 = np.asarray(p50, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, p50)))
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs(y_true - p50) / denom)) * 100.0
    return {
        "n": int(len(y_true)),
        "mae": round(float(mean_absolute_error(y_true, p50)), 2),
        "rmse": round(rmse, 2),
        "r2": round(float(r2_score(y_true, p50)), 4),
        "mape": round(mape, 2),
        "medae": round(float(median_absolute_error(y_true, p50)), 2),
        "coverage_q10_q90": round(float(np.mean((y_true >= p10) & (y_true <= p90))), 4),
        "below_best_rate": round(float(np.mean(y_true < p10)), 4),
        "above_worst_rate": round(float(np.mean(y_true > p90)), 4),
        "mean_width_days": round(float(np.mean(p90 - p10)), 2),
    }


def genetic_algorithm_tuning(
    X_train, y_train, X_val, y_val, generations: int = 3, population_size: int = 6
):
    """Optimize CatBoost depth, l2_leaf_reg, learning_rate, and iterations on median MAE+RMSE."""
    rng = np.random.default_rng(42)
    py_rng = random.Random(42)

    population = []
    for _ in range(population_size):
        population.append({
            "depth": int(rng.integers(3, 7)),
            "learning_rate": round(float(rng.uniform(0.02, 0.08)), 4),
            "iterations": int(rng.choice([150, 250, 350, 450])),
            "l2_leaf_reg": round(float(rng.uniform(3.0, 15.0)), 2),
        })

    best_individual = None
    best_score = float("inf")

    for _ in range(generations):
        scores = []
        for ind in population:
            model_med = _fit_single_quantile(X_train, y_train, 0.50, ind)
            p50 = model_med.predict(X_val)
            mae = float(mean_absolute_error(y_val, p50))
            rmse = float(np.sqrt(mean_squared_error(y_val, p50)))
            score = 0.6 * mae + 0.4 * rmse
            scores.append((score, ind))
            if score < best_score:
                best_score = score
                best_individual = dict(ind)

        scores.sort(key=lambda x: x[0])
        survivors = [ind for _, ind in scores[: max(2, population_size // 2)]]

        next_generation = list(survivors)
        while len(next_generation) < population_size:
            parent = py_rng.choice(survivors)
            child = {
                "depth": int(np.clip(parent["depth"] + int(rng.choice([-1, 0, 1])), 3, 6)),
                "learning_rate": round(
                    float(np.clip(parent["learning_rate"] * rng.uniform(0.8, 1.25), 0.02, 0.08)),
                    4,
                ),
                "iterations": int(np.clip(parent["iterations"] + int(rng.choice([-100, 0, 100])), 150, 450)),
                "l2_leaf_reg": round(
                    float(np.clip(parent["l2_leaf_reg"] * rng.uniform(0.75, 1.3), 2.0, 20.0)),
                    2,
                ),
            }
            next_generation.append(child)
        population = next_generation

    return best_individual


def _calibrate_cqr(
    X_train: pd.DataFrame,
    y_rem_train: np.ndarray,
    params: dict,
    alpha_tail: float = TARGET_TAIL_ALPHA,
) -> tuple:
    """Rolling out-of-time Asymmetric CQR calibration inside df_train."""
    n = len(X_train)
    if n < 25:
        return 0.0, 0.0

    cal_start = max(15, int(n * 0.64))
    n_blocks = 3 if (n - cal_start) >= 30 else 1
    blocks = np.array_split(np.arange(cal_start, n), n_blocks)

    err_low_all, err_high_all = [], []
    for blk in blocks:
        if len(blk) == 0 or blk[0] < 12:
            continue
        tr_end = int(blk[0])
        cb_10 = _fit_single_quantile(X_train.iloc[:tr_end], y_rem_train[:tr_end], 0.10, params)
        cb_90 = _fit_single_quantile(X_train.iloc[:tr_end], y_rem_train[:tr_end], 0.90, params)
        p10_cal = cb_10.predict(X_train.iloc[blk])
        p90_cal = cb_90.predict(X_train.iloc[blk])
        y_cal = y_rem_train[blk]
        err_low_all.extend((p10_cal - y_cal).tolist())
        err_high_all.extend((y_cal - p90_cal).tolist())

    if not err_low_all:
        return 0.0, 0.0

    n_cal = len(err_low_all)
    q_level = min(1.0, (1.0 - alpha_tail) * (1.0 + 1.0 / n_cal))
    delta_low = max(0.0, float(np.quantile(np.asarray(err_low_all), q_level)))
    delta_high = max(0.0, float(np.quantile(np.asarray(err_high_all), q_level)))
    return round(delta_low, 2), round(delta_high, 2)


def run_pipeline_multistage(
    df_raw: pd.DataFrame,
    test_frac: float = DEFAULT_TEST_FRAC,
    params: dict = None,
    source: str = "upload",
    stages=None,
    use_genetic_tuning: bool = True,
):
    """Full protocol for every stage: clean -> causal stats -> split -> tune -> CQR -> eval."""
    if stages is None:
        stages = STAGES

    df = clean_dataframe(df_raw)
    df = df.sort_values("registery-date").reset_index(drop=True)
    df = build_expanding_stats(df)
    df_train, df_test = chronological_split(df, test_frac)
    global_mean = float(df_train["LeadTime"].mean())

    df_train = make_feature_frame(df_train, global_mean)
    df_test = make_feature_frame(df_test, global_mean)
    y_train_total = df_train["LeadTime"].to_numpy(dtype=float)
    y_test_total = df_test["LeadTime"].to_numpy(dtype=float)

    val_split_idx = int(len(df_train) * 0.78)

    models_by_stage = {}
    stage_metrics = {}
    calibration = {}
    stage_params_map = {}

    for stage in stages:
        cols = feature_cols_for_stage(stage)
        elapsed_train = elapsed_for_stage(df_train, stage)
        elapsed_test = elapsed_for_stage(df_test, stage)

        y_rem_train = np.maximum(0.0, y_train_total - elapsed_train)

        stage_params = dict(DEFAULT_PARAMS)
        if params:
            stage_params.update(params)
        elif use_genetic_tuning and len(df_train) >= 30 and val_split_idx >= 15:
            X_tr = df_train[cols].iloc[:val_split_idx]
            y_tr = y_rem_train[:val_split_idx]
            X_val = df_train[cols].iloc[val_split_idx:]
            y_val = y_rem_train[val_split_idx:]
            best_ind = genetic_algorithm_tuning(
                X_tr, y_tr, X_val, y_val, generations=3, population_size=6
            )
            if best_ind:
                stage_params.update(best_ind)

        stage_params_map[str(stage)] = {
            k: v for k, v in stage_params.items() if k != "allow_writing_files"
        }

        delta_low, delta_high = _calibrate_cqr(df_train[cols], y_rem_train, stage_params)
        calibration[stage] = {"delta_low": delta_low, "delta_high": delta_high}

        cb_10, cb_50, cb_90 = train_quantile_models(
            df_train[cols], y_rem_train, stage_params
        )
        models_by_stage[stage] = (cb_10, cb_50, cb_90)

        # Evaluate total lead time (elapsed + calibrated remaining) on train & test
        rem_tr_10, rem_tr_50, rem_tr_90 = apply_calibration_and_order(
            cb_10.predict(df_train[cols]),
            cb_50.predict(df_train[cols]),
            cb_90.predict(df_train[cols]),
            delta_low,
            delta_high,
        )
        rem_te_10, rem_te_50, rem_te_90 = apply_calibration_and_order(
            cb_10.predict(df_test[cols]),
            cb_50.predict(df_test[cols]),
            cb_90.predict(df_test[cols]),
            delta_low,
            delta_high,
        )

        test_metrics = evaluate_quantiles(
            y_test_total,
            elapsed_test + rem_te_10,
            elapsed_test + rem_te_50,
            elapsed_test + rem_te_90,
        )
        train_metrics = evaluate_quantiles(
            y_train_total,
            elapsed_train + rem_tr_10,
            elapsed_train + rem_tr_50,
            elapsed_train + rem_tr_90,
        )

        stage_metrics[str(stage)] = {
            "stage": stage,
            "stage_name": STAGE_NAMES[stage],
            "stage_label_fa": STAGE_LABELS_FA[stage],
            "feature_cols": cols,
            "n_features": len(cols),
            "calibration": calibration[stage],
            "params": stage_params_map[str(stage)],
            "test": test_metrics,
            "train": {
                "n": train_metrics["n"],
                "mae": train_metrics["mae"],
                "r2": train_metrics["r2"],
                "coverage_q10_q90": train_metrics["coverage_q10_q90"],
            },
        }

    benchmark_stage = 0 if 0 in stages else min(stages)
    metrics = {
        "protocol": (
            "chronological split by registery-date, causal remaining-days target, "
            "GA hyperparameter tuning, asymmetric rolling-time CQR calibration"
        ),
        "target_mode": "remaining_days",
        "n_total": int(len(df)),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "split_date": str(df_test["registery-date"].min().date()),
        "train_range": [
            str(df_train["registery-date"].min().date()),
            str(df_train["registery-date"].max().date()),
        ],
        "test_range": [
            str(df_test["registery-date"].min().date()),
            str(df_test["registery-date"].max().date()),
        ],
        "params": stage_params_map,
        "stages": stage_metrics,
        "benchmark": "r2",
        "benchmark_stage": benchmark_stage,
        "benchmark_r2": stage_metrics[str(benchmark_stage)]["test"]["r2"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
    }

    history_df = df[
        [
            "supplier_clean",
            "country_clean",
            "sap_code_clean",
            "registery-date",
            "factory-arrive-date",
            "LeadTime",
        ]
    ].copy()
    state_bundle = {
        "history_df": history_df,
        "global_mean": global_mean,
        "calibration": calibration,
        "target_mode": "remaining_days",
    }
    return models_by_stage, state_bundle, metrics


def save_artifacts(artifacts_dir, models_by_stage, state_bundle, metrics):
    """Persist per-stage models + shared history/calibration state + metrics.json."""
    os.makedirs(artifacts_dir, exist_ok=True)
    for stage, (cb_10, cb_50, cb_90) in models_by_stage.items():
        stage_dir = os.path.join(artifacts_dir, f"stage_{stage}")
        os.makedirs(stage_dir, exist_ok=True)
        cb_10.save_model(os.path.join(stage_dir, "model_q10.cbm"))
        cb_50.save_model(os.path.join(stage_dir, "model_q50.cbm"))
        cb_90.save_model(os.path.join(stage_dir, "model_q90.cbm"))
    joblib.dump(state_bundle, os.path.join(artifacts_dir, "model_state.pkl"))
    with open(os.path.join(artifacts_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)