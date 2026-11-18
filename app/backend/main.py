import json
import io
import logging
import os
import sys
import threading
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)
from data_loader import normalize_country  # noqa: E402
from train_utils import (  # noqa: E402
    CAT_FEATURES,
    STAGE_LABELS_FA,
    STAGE_NAMES,
    apply_calibration_and_order,
    feature_cols_for_stage,
    run_pipeline_multistage,
    save_artifacts,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Stage Lead Time Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models_by_stage = {}
state = {}
model_metrics = {}
models_loaded = False
_retrain_lock = threading.Lock()

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BACKEND_DIR, "data")
artifacts_dir = os.path.abspath(
    os.path.join(
        BACKEND_DIR, "..", "..",
        "lead_time_platform_final", "platform", "artifacts_fixed",
    )
)


def update_metadata_lists(df_raw: pd.DataFrame, data_dir: str = DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)

    countries = sorted({
        normalize_country(c)
        for c in df_raw["country"].dropna()
        if str(c).strip()
    })
    suppliers = sorted({
        str(s).strip().upper()
        for s in df_raw["supplier"].dropna()
        if str(s).strip()
    })
    sap_codes = sorted({
        str(x).strip().removesuffix(".0")
        for x in df_raw["sap-code"].dropna()
        if str(x).strip()
    })

    with open(os.path.join(data_dir, "countries.json"), "w", encoding="utf-8") as f:
        json.dump(countries, f, ensure_ascii=False, indent=2)

    with open(os.path.join(data_dir, "suppliers.json"), "w", encoding="utf-8") as f:
        json.dump(suppliers, f, ensure_ascii=False, indent=2)

    with open(os.path.join(data_dir, "sap_codes.json"), "w", encoding="utf-8") as f:
        json.dump(sap_codes, f, ensure_ascii=False, indent=2)


def _stage_model_paths(stage: int):
    d = os.path.join(artifacts_dir, f"stage_{stage}")
    return tuple(os.path.join(d, f"model_q{q}.cbm") for q in (10, 50, 90))


def load_models_with_retry(max_attempts: int = 3, delay_seconds: int = 2):
    """Load all per-stage models, history state and metrics with retry."""
    global models_by_stage, state, model_metrics, models_loaded

    for attempt in range(max_attempts):
        try:
            logger.info(
                "Attempting to load models (attempt %d/%d)", attempt + 1, max_attempts
            )
            state_path = os.path.join(artifacts_dir, "model_state.pkl")
            metrics_path = os.path.join(artifacts_dir, "metrics.json")
            if not os.path.exists(state_path):
                raise FileNotFoundError(f"Model state not found: {state_path}")
            if not os.path.exists(metrics_path):
                raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

            with open(metrics_path, encoding="utf-8") as f:
                loaded_metrics = json.load(f)
            available = sorted(int(s) for s in loaded_metrics.get("stages", {}))

            loaded = {}
            for stage in available:
                paths = _stage_model_paths(stage)
                for p in paths:
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"Model file not found: {p}")
                cbs = []
                for p in paths:
                    m = CatBoostRegressor()
                    m.load_model(p)
                    cbs.append(m)
                loaded[stage] = tuple(cbs)

            if not loaded:
                raise RuntimeError("No stage models found in metrics.json.")

            models_by_stage = loaded
            model_metrics = loaded_metrics
            state = joblib.load(state_path)
            models_loaded = True
            logger.info("Models loaded for stages: %s", sorted(models_by_stage))
            return True

        except Exception as e:
            logger.error("Error loading models (attempt %d): %s", attempt + 1, e)
            models_loaded = False
            if attempt < max_attempts - 1:
                time.sleep(delay_seconds)
            else:
                logger.error("All attempts to load models failed.")
                return False


@app.on_event("startup")
def startup_event():
    logger.info("Starting up Multi-Stage Lead Time Prediction API...")
    if not load_models_with_retry():
        logger.warning("Models failed to load. Server started in degraded mode.")


@app.get("/health")
def health_check():
    if models_loaded:
        target_mode = state.get("target_mode", "legacy_total_leadtime")
        return {
            "status": "healthy",
            "models_loaded": True,
            "stages": sorted(models_by_stage),
            "target_mode": target_mode,
            "needs_retrain": target_mode != "remaining_days",
            "message": "All stage models loaded successfully",
        }
    return {
        "status": "degraded",
        "models_loaded": False,
        "message": "Models not loaded - prediction endpoints may fail",
    }


@app.get("/")
def root():
    return {
        "message": "Multi-Stage Lead Time Prediction API",
        "version": "2.1.0",
        "stages": sorted(models_by_stage) if models_by_stage else [],
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/metrics")
def get_metrics():
    """Benchmark metrics for every stage (held-out test split)."""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    target_mode = state.get("target_mode", "legacy_total_leadtime")
    return {
        "models_loaded": True,
        "target_mode": target_mode,
        "needs_retrain": target_mode != "remaining_days",
        "protocol": model_metrics.get("protocol", ""),
        "benchmark": model_metrics.get("benchmark", "r2"),
        "benchmark_stage": model_metrics.get("benchmark_stage", 0),
        "benchmark_r2": model_metrics.get("benchmark_r2"),
        "n_total": model_metrics.get("n_total"),
        "n_train": model_metrics.get("n_train"),
        "n_test": model_metrics.get("n_test"),
        "split_date": model_metrics.get("split_date"),
        "train_range": model_metrics.get("train_range"),
        "test_range": model_metrics.get("test_range"),
        "stages": model_metrics.get("stages", {}),
        "stage_labels_fa": STAGE_LABELS_FA,
        "trained_at": model_metrics.get("trained_at"),
        "source": model_metrics.get("source"),
    }


class OrderRequest(BaseModel):
    sap_code: str
    country: str
    supplier: str
    weight: float
    registry_date: str
    ministry_date: Optional[str] = None
    ship_date: Optional[str] = None
    customs_date: Optional[str] = None


class RetrainResponse(BaseModel):
    message: str
    benchmark_stage: int
    benchmark_r2: float
    n_total: int
    n_train: int
    n_test: int
    stages: dict


def check_models_loaded():
    if not models_loaded:
        raise HTTPException(
            status_code=503,
            detail="مدل‌ها بارگذاری نشده‌اند. لطفاً وضعیت سرور را بررسی کنید.",
        )


def _parse_optional_date(value, name):
    if value is None or str(value).strip() == "":
        return None
    try:
        d = pd.to_datetime(value)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"فرمت تاریخ برای {name} نامعتبر است (YYYY-MM-DD).",
        )
    if pd.isna(d):
        raise HTTPException(status_code=400, detail=f"تاریخ {name} نامعتبر است.")
    return d


def _resolve_stage_and_dates(order: OrderRequest):
    reg_date = _parse_optional_date(order.registry_date, "registry_date")
    if reg_date is None:
        raise HTTPException(status_code=400, detail="تاریخ ثبت سیستم الزامی است.")

    ministry = _parse_optional_date(order.ministry_date, "ministry_date")
    ship = _parse_optional_date(order.ship_date, "ship_date")
    customs = _parse_optional_date(order.customs_date, "customs_date")

    if customs is not None and (ship is None or ministry is None):
        raise HTTPException(
            status_code=400,
            detail="برای ثبت تاریخ ورود به گمرک، تاریخ مجوز وزارتخانه و حرکت کشتی نیز الزامی است.",
        )
    if ship is not None and ministry is None:
        raise HTTPException(
            status_code=400,
            detail="برای ثبت تاریخ حرکت کشتی، تاریخ مجوز وزارتخانه نیز الزامی است.",
        )

    chain = [
        (name, d)
        for name, d in (
            ("registry_date", reg_date),
            ("ministry_date", ministry),
            ("ship_date", ship),
            ("customs_date", customs),
        )
        if d is not None
    ]
    for i in range(1, len(chain)):
        if chain[i][1] < chain[i - 1][1]:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{chain[i][0]} ({chain[i][1].date()}) نمی‌تواند قبل از "
                    f"{chain[i - 1][0]} ({chain[i - 1][1].date()}) باشد."
                ),
            )

    if customs is not None:
        stage = 3
    elif ship is not None:
        stage = 2
    elif ministry is not None:
        stage = 1
    else:
        stage = 0

    latest = chain[-1][1]
    elapsed_days = int((latest - reg_date).days)
    return stage, reg_date, ministry, ship, customs, elapsed_days


def _build_feature_row(
    order,
    sup_clean: str,
    country_clean: str,
    sap_clean: str,
    stage: int,
    reg_date,
    ministry,
    ship,
    customs,
):
    """Build stage feature row matching train_utils.py (supports both new & legacy artifacts)."""
    history_df = state["history_df"]
    global_mean = float(state.get("global_mean", history_df["LeadTime"].mean()))

    # Strictly causal filter: completed orders before registration when available
    if "factory-arrive-date" in history_df.columns:
        completed = history_df[history_df["factory-arrive-date"] < reg_date].sort_values(
            "factory-arrive-date"
        )
    else:
        completed = history_df[history_df["registery-date"] < reg_date].sort_values(
            "registery-date"
        )

    global_recent = (
        float(completed["LeadTime"].tail(20).mean()) if len(completed) > 0 else global_mean
    )

    # Country shrinkage
    if "country_clean" in completed.columns:
        ctr_past = completed[completed["country_clean"] == country_clean]
        ctr_cnt = len(ctr_past)
        ctr_raw = float(ctr_past["LeadTime"].mean()) if ctr_cnt > 0 else global_mean
    else:
        ctr_cnt, ctr_raw = 0, global_mean
    country_hist_mean = (ctr_cnt * ctr_raw + 3.0 * global_mean) / (ctr_cnt + 3.0)

    # SAP shrinkage
    if "sap_code_clean" in completed.columns:
        sap_past = completed[completed["sap_code_clean"] == sap_clean]
        sap_cnt = len(sap_past)
        sap_raw = float(sap_past["LeadTime"].mean()) if sap_cnt > 0 else global_mean
    else:
        sap_cnt, sap_raw = 0, global_mean
    sap_hist_mean = (sap_cnt * sap_raw + 3.0 * global_mean) / (sap_cnt + 3.0)

    # Supplier shrinkage
    past_sup = completed[completed["supplier_clean"] == sup_clean]
    sup_cnt = len(past_sup)
    is_unseen = sup_cnt == 0
    if not is_unseen:
        sup_raw_mean = float(past_sup["LeadTime"].mean())
        sup_std = float(past_sup["LeadTime"].std(ddof=1)) if sup_cnt > 1 else 0.0
        r3_raw = float(past_sup["LeadTime"].tail(3).mean())
    else:
        sup_raw_mean = country_hist_mean
        sup_std = 0.0
        r3_raw = country_hist_mean

    is_legacy = state.get("target_mode") != "remaining_days"
    if is_legacy:
        sup_mean = float(past_sup["LeadTime"].mean()) if not is_unseen else global_mean
        recent_3_mean = (
            float(past_sup["LeadTime"].tail(3).mean()) if not is_unseen else sup_mean
        )
        tier = (0 if sup_std < 15 else (1 if sup_std < 30 else 2)) if not is_unseen else 1
    else:
        sup_mean = (sup_cnt * sup_raw_mean + 3.0 * country_hist_mean) / (sup_cnt + 3.0)
        r3_cnt = min(sup_cnt, 3)
        recent_3_mean = (r3_cnt * r3_raw + 2.0 * sup_mean) / (r3_cnt + 2.0)
        if sup_cnt < 3:
            tier = 1
        else:
            tier = 0 if sup_std < 15 else (1 if sup_std < 30 else 2)

    month = int(reg_date.month)
    doy = float(reg_date.dayofyear)
    weight = float(order.weight)

    row = {
        "weight": weight,
        "log_weight": float(np.log1p(max(weight, 0.0))),
        "month": month,
        "reg_doy_sin": float(np.sin(2.0 * np.pi * doy / 365.25)),
        "reg_doy_cos": float(np.cos(2.0 * np.pi * doy / 365.25)),
        "is_winter": 1 if month in [12, 1, 2] else 0,
        "is_peak": 1 if month in [11, 12, 1] else 0,
        "supplier_count": sup_cnt,
        "supplier_tier": tier,
        "supplier_hist_mean": sup_mean,
        "recent_3_mean": recent_3_mean,
        "country_hist_mean": country_hist_mean,
        "sap_hist_mean": sap_hist_mean,
        "global_recent_mean": global_recent,
        "country_clean": country_clean,
        "supplier_clean": sup_clean,
        "sap_code_clean": sap_clean,
    }
    if stage >= 1:
        row["dur_reg_to_min"] = int((ministry - reg_date).days)
        row["ministry_month"] = int(ministry.month)
    if stage >= 2:
        row["dur_min_to_ship"] = int((ship - ministry).days)
        row["elapsed_reg_to_ship"] = int((ship - reg_date).days)
        row["ship_month"] = int(ship.month)
    if stage >= 3:
        row["dur_ship_to_cust"] = int((customs - ship).days)
        row["elapsed_reg_to_cust"] = int((customs - reg_date).days)
        row["customs_month"] = int(customs.month)

    stage_info = model_metrics.get("stages", {}).get(str(stage), {})
    cols = stage_info.get("feature_cols") or feature_cols_for_stage(stage, legacy=is_legacy)
    df_row = pd.DataFrame([row])[cols]
    meta = {
        "is_unseen": is_unseen,
        "sup_cnt": sup_cnt,
        "tier": tier,
        "sup_mean": sup_raw_mean if not is_unseen else sup_mean,
        "recent_3_mean": r3_raw if not is_unseen else recent_3_mean,
    }
    return df_row, meta


@app.post("/predict")
def predict_lead_time(order: OrderRequest):
    """Predict total & remaining lead time with calibrated optimistic/realistic/pessimistic bounds."""
    check_models_loaded()

    sap_clean = str(order.sap_code).strip().removesuffix(".0")
    country_raw = str(order.country).strip()
    sup_raw = str(order.supplier).strip()
    if not sap_clean or not country_raw or not sup_raw:
        raise HTTPException(
            status_code=400,
            detail="کد کالا (SAP)، کشور مبدأ و نام تأمین‌کننده الزامی هستند.",
        )
    if not np.isfinite(order.weight) or order.weight <= 0:
        raise HTTPException(status_code=400, detail="وزن محموله باید عددی مثبت باشد.")

    try:
        stage, reg_date, ministry, ship, customs, elapsed = _resolve_stage_and_dates(order)
        if stage not in models_by_stage:
            raise HTTPException(
                status_code=503, detail=f"مدل برای مرحله {stage} در دسترس نیست."
            )

        sup_clean = sup_raw.upper()
        country_clean = normalize_country(country_raw)

        features_df, meta = _build_feature_row(
            order, sup_clean, country_clean, sap_clean,
            stage, reg_date, ministry, ship, customs,
        )

        cb_10, cb_50, cb_90 = models_by_stage[stage]
        pool = Pool(features_df, cat_features=CAT_FEATURES)
        raw_10 = float(cb_10.predict(pool)[0])
        raw_50 = float(cb_50.predict(pool)[0])
        raw_90 = float(cb_90.predict(pool)[0])

        target_mode = state.get("target_mode", "legacy_total_leadtime")
        if target_mode == "remaining_days":
            cal = state.get("calibration", {}).get(stage, {})
            delta_low = float(cal.get("delta_low", 0.0))
            delta_high = float(cal.get("delta_high", 0.0))
            rem_low, rem_med, rem_high = apply_calibration_and_order(
                raw_10, raw_50, raw_90, delta_low, delta_high
            )
            r_low = max(0, int(round(float(rem_low))))
            r_med = max(r_low, int(round(float(rem_med))))
            r_high = max(r_med, int(round(float(rem_high))))
        else:
            # Backward-compatible clamp for legacy total-leadtime artifacts
            r_low = max(0, int(round(raw_10 - elapsed)))
            r_med = max(r_low, int(round(raw_50 - elapsed)))
            r_high = max(r_med, int(round(raw_90 - elapsed)))

        best_total = elapsed + r_low
        expected_total = elapsed + r_med
        worst_total = elapsed + r_high

        best_date = (reg_date + pd.Timedelta(days=best_total)).strftime("%Y-%m-%d")
        expected_date = (reg_date + pd.Timedelta(days=expected_total)).strftime("%Y-%m-%d")
        worst_date = (reg_date + pd.Timedelta(days=worst_total)).strftime("%Y-%m-%d")

        tier_names = {
            0: "دسته طلایی (منظم)",
            1: "دسته نقره‌ای (نوسان متوسط)",
            2: "دسته قرمز (پرنوسان)",
        }
        is_unseen = meta["is_unseen"]
        sup_cnt = meta["sup_cnt"]
        tier = meta["tier"]
        if is_unseen:
            tier_label = "بدون سابقه قبلی"
        elif sup_cnt < 3:
            tier_label = f"کم‌سابقه ({sup_cnt} سفارش تکمیل‌شده)"
        else:
            tier_label = tier_names.get(tier, "نامشخص")

        stage_metrics = model_metrics.get("stages", {}).get(str(stage), {})
        stage_test = stage_metrics.get("test", {})

        return {
            "stage": stage,
            "stage_name": STAGE_NAMES.get(stage),
            "stage_label_fa": STAGE_LABELS_FA.get(stage),
            "elapsed_days": elapsed,
            "best_case_days": best_total,
            "expected_days": expected_total,
            "worst_case_days": worst_total,
            "remaining_best_days": r_low,
            "remaining_expected_days": r_med,
            "remaining_worst_days": r_high,
            "estimated_best_date": best_date,
            "estimated_expected_date": expected_date,
            "estimated_worst_date": worst_date,
            "stage_test_r2": stage_test.get("r2"),
            "stage_test_mae": stage_test.get("mae"),
            "stage_coverage": stage_test.get("coverage_q10_q90"),
            "below_best_rate": stage_test.get("below_best_rate"),
            "above_worst_rate": stage_test.get("above_worst_rate"),
            "is_unseen_supplier": is_unseen,
            "supplier_order_count": sup_cnt,
            "supplier_tier_code": -1 if is_unseen else tier,
            "supplier_tier_name": tier_label,
            "historical_mean": "بدون سابقه" if is_unseen else round(meta["sup_mean"], 1),
            "recent_trend": "بدون سابقه" if is_unseen else round(meta["recent_3_mean"], 1),
            "target_mode": target_mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrain", response_model=RetrainResponse)
def retrain_model(
    file: UploadFile = File(...),
    test_frac: float = Form(0.20),
    use_tuning: bool = Form(True),
):
    """Upload dataset CSV, retrain all stages with GA + CQR calibration, update lists."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="لطفاً یک فایل CSV معتبر انتخاب کنید.")
    if not (0.05 <= float(test_frac) <= 0.50):
        raise HTTPException(
            status_code=400, detail="نسبت داده آزمون باید بین 0.05 و 0.50 باشد."
        )

    if not _retrain_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="یک فرآیند بازآموزی هم‌اکنون در حال اجرا است. لطفاً کمی صبر کنید.",
        )
    try:
        content = file.file.read()
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="حجم فایل بیش از ۵۰ مگابایت است.")
        try:
            df_raw = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"خطا در خواندن CSV: {e}")

        try:
            models, state_bundle, metrics = run_pipeline_multistage(
                df_raw,
                test_frac=float(test_frac),
                source=file.filename,
                use_genetic_tuning=use_tuning,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Retraining failed: %s", e)
            raise HTTPException(status_code=500, detail=f"بازآموزی ناموفق بود: {e}")

        save_artifacts(artifacts_dir, models, state_bundle, metrics)
        update_metadata_lists(df_raw, DATA_DIR)

        if not load_models_with_retry():
            raise HTTPException(
                status_code=500, detail="مدل‌ها ذخیره شدند اما بارگذاری مجدد ناموفق بود."
            )

        logger.info(
            "Retrained on %s: benchmark stage %s R2=%.4f",
            file.filename,
            metrics["benchmark_stage"],
            metrics["benchmark_r2"],
        )
        return {
            "message": "مدل‌ها با منطق زمان باقی‌مانده و کالیبراسیون CQR بازآموزی و فعال شدند.",
            "benchmark_stage": metrics["benchmark_stage"],
            "benchmark_r2": metrics["benchmark_r2"],
            "n_total": metrics["n_total"],
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
            "stages": metrics["stages"],
        }
    finally:
        _retrain_lock.release()


@app.get("/api/countries")
def get_countries():
    path = os.path.join(DATA_DIR, "countries.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/suppliers")
def get_suppliers():
    path = os.path.join(DATA_DIR, "suppliers.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/sap-codes")
def get_sap_codes():
    path = os.path.join(DATA_DIR, "sap_codes.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)