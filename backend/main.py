#!/usr/bin/env python3
"""
Smart Lead Time Prediction Platform — Python ML Inference Engine ("The Brain").

- Binds strictly to 127.0.0.1:8000 (localhost only) so external LAN workstations
  cannot bypass the Bun gateway.
- Resolves resource & persistence paths dynamically for source execution as well
  as PyInstaller (`sys._MEIPASS`) and Nuitka (`sys.frozen`) standalone binaries.
- Zero third-party runtime requirements: executes on Python 3.11 stdlib, and
  automatically uses FastAPI + Uvicorn if installed in the active environment.
"""

from __future__ import annotations

import bisect
import cgi
import csv
import io
import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dynamic Path Resolution (Source / PyInstaller / Nuitka compatible)
# ---------------------------------------------------------------------------

def resolve_bundle_dir() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resolve_runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BUNDLE_DIR = resolve_bundle_dir()
RUNTIME_DIR = resolve_runtime_dir()


def resolve_subdir(env_name: str, folder: str) -> Path:
    override = os.environ.get(env_name, "").strip()
    if override:
        return Path(override).resolve()
    runtime_candidate = RUNTIME_DIR / folder
    if runtime_candidate.exists():
        return runtime_candidate
    bundle_candidate = BUNDLE_DIR / folder
    if bundle_candidate.exists():
        return bundle_candidate
    return runtime_candidate


DATA_DIR = resolve_subdir("LEADTIME_DATA_DIR", "data")
ARTIFACTS_DIR = resolve_subdir("LEADTIME_ARTIFACTS_DIR", "artifacts")

# Always persist retrained outputs in runtime dir if bundle dir is read-only
PERSIST_DATA_DIR = Path(os.environ.get("LEADTIME_DATA_DIR", str(RUNTIME_DIR / "data"))).resolve()
PERSIST_ARTIFACTS_DIR = Path(
    os.environ.get("LEADTIME_ARTIFACTS_DIR", str(RUNTIME_DIR / "artifacts"))
).resolve()

BIND_HOST = "127.0.0.1"  # Enforced localhost-only binding
try:
    BIND_PORT = int(os.environ.get("BACKEND_PORT", "8000"))
except ValueError:
    BIND_PORT = 8000

MS_PER_DAY = 86_400_000
STAGES = (0, 1, 2, 3)

COUNTRY_MAP = {
    "UNITED ARAB EMIRATES": "U.A.E",
    "UAE": "U.A.E",
    "U.A.E.": "U.A.E",
}

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

BASE_FEATURE_COLS = [
    "log_weight",
    "reg_doy_sin",
    "reg_doy_cos",
    "stage_month_sin",
    "stage_month_cos",
    "top_country_1",
    "top_country_2",
    "top_country_3",
    "leg_ewma_level",
    "country_hist_mean",
    "sap_hist_mean",
    "supplier_hist_mean",
    "recent_3_mean",
]

STAGE_EXTRA_COLS = {
    0: [],
    1: ["dur_reg_to_min"],
    2: ["dur_reg_to_min", "dur_min_to_ship"],
    3: ["dur_reg_to_min", "dur_min_to_ship", "dur_ship_to_cust"],
}

LEG_DEFAULTS = (10.6, 48.2, 47.5, 37.6)
LEG_HUBER_DELTAS = (7.0, 14.0, 9.0, 7.0)
ENS_W_HUBER = 0.78
ENS_W_GBDT = 0.12
ENS_W_PROPHET = 0.10
KALMAN_PREV_WEIGHT = 0.10

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

ISO_RE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T\s].*)?$")
DMY_TEXT_RE = re.compile(r"^(\d{1,2})[-/\s]([A-Za-z]{3})[-/\s](\d{2,4})$")
MDY_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")


class ValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Normalization & Date Utilities
# ---------------------------------------------------------------------------

def normalize_country(val: Any) -> str:
    cleaned = str(val or "").strip().upper()
    return COUNTRY_MAP.get(cleaned, cleaned)


def normalize_supplier(val: Any) -> str:
    return str(val or "").strip().upper()


def normalize_sap_code(val: Any) -> str:
    s = str(val or "").strip()
    return s[:-2] if s.endswith(".0") else s


def parse_flexible_date_ms(raw: Any) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    m = ISO_RE.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = datetime(y, mo, d, tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None

    m = DMY_TEXT_RE.match(s)
    if m:
        d = int(m.group(1))
        mo = MONTHS.get(m.group(2).lower())
        y = int(m.group(3))
        if y < 100:
            y += 1900 if y >= 70 else 2000
        if mo is not None:
            try:
                dt = datetime(y, mo, d, tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                return None
        return None

    m = MDY_RE.match(s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = datetime(y, mo, d, tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            return None

    return None


def format_ms_as_iso(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def day_of_year_utc(ms: int) -> int:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.timetuple().tm_yday


def month_utc(ms: int) -> int:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.month


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CleanedShipmentRow:
    sap_code_clean: str
    country_clean: str
    supplier_clean: str
    weight: float
    log_weight: float
    registery_ms: int
    ministry_ms: int
    ship_ms: int
    customs_ms: int
    factory_arrive_ms: int
    registery_date_str: str
    lead_time: float
    dur_reg_to_min: int
    dur_min_to_ship: int
    dur_ship_to_cust: int
    dur_cust_to_fact: int
    elapsed_reg_to_ship: int
    elapsed_reg_to_cust: int
    reg_month: int
    ministry_month: int
    ship_month: int
    customs_month: int
    reg_doy: int

    def leg_value(self, leg_idx: int) -> int:
        if leg_idx == 0:
            return self.dur_reg_to_min
        if leg_idx == 1:
            return self.dur_min_to_ship
        if leg_idx == 2:
            return self.dur_ship_to_cust
        return self.dur_cust_to_fact

    def stage_cutoff_ms(self, stage: int) -> int:
        if stage == 1:
            return self.ministry_ms
        if stage == 2:
            return self.ship_ms
        if stage == 3:
            return self.customs_ms
        return self.registery_ms

    def elapsed_at_stage(self, stage: int) -> int:
        if stage == 1:
            return self.dur_reg_to_min
        if stage == 2:
            return self.elapsed_reg_to_ship
        if stage == 3:
            return self.elapsed_reg_to_cust
        return 0


@dataclass
class GbdtNode:
    feat: int | None = None
    thresh: float | None = None
    left: GbdtNode | None = None
    right: GbdtNode | None = None
    val: float | None = None


@dataclass
class TrainedLegModel:
    leg_idx: int
    weights: list[float]
    l2: float
    gbdt_base: float
    gbdt_lr: float
    gbdt_trees: list[GbdtNode]
    fallback_leg_mean: float


@dataclass
class TrainedStageModel:
    stage: int
    legs: list[TrainedLegModel]
    top_countries: list[str]
    q_low_res: float
    q_high_res: float
    delta_low: float
    delta_high: float


# ---------------------------------------------------------------------------
# Statistics & Math Helpers
# ---------------------------------------------------------------------------

def mean(arr: list[float] | list[int]) -> float:
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def sample_std(arr: list[float] | list[int]) -> float:
    n = len(arr)
    if n <= 1:
        return 0.0
    m = sum(arr) / n
    ss = sum((v - m) * (v - m) for v in arr)
    return math.sqrt(ss / (n - 1))


def quantile(arr: list[float], q: float) -> float:
    if not arr:
        return 0.0
    s = sorted(arr)
    if len(s) == 1:
        return float(s[0])
    q = min(1.0, max(0.0, q))
    pos = (len(s) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def apply_calibration_and_order(
    raw_low: float,
    raw_med: float,
    raw_high: float,
    delta_low: float,
    delta_high: float,
) -> tuple[float, float, float]:
    rem_low = max(0.0, raw_low - delta_low)
    rem_med = max(0.0, raw_med)
    rem_high = max(0.0, raw_high + delta_high)
    rem_low = min(rem_low, rem_med)
    rem_high = max(rem_high, rem_med)
    return rem_low, rem_med, rem_high


def evaluate_quantiles(
    y_true: list[float],
    p10: list[float],
    p50: list[float],
    p90: list[float],
) -> dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {
            "n": 0, "mae": 0.0, "rmse": 0.0, "r2": 0.0, "mape": 0.0,
            "medae": 0.0, "coverage_q10_q90": 0.0, "below_best_rate": 0.0,
            "above_worst_rate": 0.0, "mean_width_days": 0.0,
        }

    abs_errs: list[float] = []
    sq_err_sum = 0.0
    ape_sum = 0.0
    covered = 0
    below = 0
    above = 0
    width_sum = 0.0
    y_mean = mean(y_true)
    ss_tot = 0.0

    for yt, l_v, m_v, h_v in zip(y_true, p10, p50, p90):
        err = yt - m_v
        ae = abs(err)
        abs_errs.append(ae)
        sq_err_sum += err * err
        ape_sum += ae / max(abs(yt), 1.0)
        ss_tot += (yt - y_mean) * (yt - y_mean)
        if l_v <= yt <= h_v:
            covered += 1
        if yt < l_v:
            below += 1
        if yt > h_v:
            above += 1
        width_sum += h_v - l_v

    r2 = 1.0 - sq_err_sum / ss_tot if ss_tot > 0 else 0.0
    return {
        "n": n,
        "mae": round(mean(abs_errs), 2),
        "rmse": round(math.sqrt(sq_err_sum / n), 2),
        "r2": round(r2, 4),
        "mape": round((ape_sum / n) * 100.0, 2),
        "medae": round(quantile(abs_errs, 0.5), 2),
        "coverage_q10_q90": round(covered / n, 4),
        "below_best_rate": round(below / n, 4),
        "above_worst_rate": round(above / n, 4),
        "mean_width_days": round(width_sum / n, 2),
    }


def solve_ridge(
    X: list[list[float]],
    y: list[float],
    l2: float,
    sample_weights: list[float] | None = None,
) -> list[float]:
    n = len(X)
    if n == 0:
        return []
    d = len(X[0])

    means = [0.0] * d
    stds = [1.0] * d
    for j in range(1, d):
        m = sum(row[j] for row in X) / n
        means[j] = m
        sq = sum((row[j] - m) ** 2 for row in X)
        stds[j] = math.sqrt(sq / n) or 1.0

    Xs = [
        [1.0 if j == 0 else (row[j] - means[j]) / stds[j] for j in range(d)]
        for row in X
    ]

    A = [[0.0] * d for _ in range(d)]
    rhs = [0.0] * d

    for i in range(n):
        xi = Xs[i]
        yi = y[i]
        wi = sample_weights[i] if sample_weights is not None else 1.0
        for j in range(d):
            w_xij = wi * xi[j]
            rhs[j] += w_xij * yi
            aj = A[j]
            for k in range(j, d):
                aj[k] += w_xij * xi[k]

    for j in range(d):
        for k in range(j):
            A[j][k] = A[k][j]
    for j in range(1, d):
        A[j][j] += l2

    # Gaussian elimination with partial pivoting
    for i in range(d):
        max_row = max(range(i, d), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]
        rhs[i], rhs[max_row] = rhs[max_row], rhs[i]

        piv = A[i][i] if abs(A[i][i]) > 1e-10 else 1e-10
        ai = A[i]
        for j in range(i, d):
            ai[j] /= piv
        rhs[i] /= piv

        for r in range(d):
            if r == i:
                continue
            factor = A[r][i]
            if factor == 0.0:
                continue
            ar = A[r]
            for j in range(i, d):
                ar[j] -= factor * ai[j]
            rhs[r] -= factor * rhs[i]

    weights = [0.0] * d
    intercept = rhs[0]
    for j in range(1, d):
        w = rhs[j] / stds[j]
        weights[j] = w
        intercept -= w * means[j]
    weights[0] = intercept
    return weights


def fit_huber_ridge(
    X: list[list[float]],
    y: list[float],
    l2: float,
    huber_delta: float,
    iters: int = 6,
) -> list[float]:
    n = len(X)
    if n == 0:
        return []
    d = len(X[0])
    rw = [1.0] * n
    weights = [0.0] * d

    for _ in range(iters):
        weights = solve_ridge(X, y, l2, rw)
        for i in range(n):
            xi = X[i]
            pred = sum(xi[j] * weights[j] for j in range(d))
            abs_err = abs(y[i] - pred)
            rw[i] = 1.0 if abs_err <= huber_delta else huber_delta / abs_err
    return weights


def build_gbdt_tree(
    X: list[list[float]],
    residuals: list[float],
    idxs: list[int],
    sorted_orders: list[list[int]],
    in_node: list[bool],
    depth: int,
    min_leaf: int,
) -> GbdtNode:
    n_total = len(idxs)
    total_sum = sum(residuals[i] for i in idxs)
    mean_val = total_sum / n_total if n_total > 0 else 0.0
    if depth <= 0 or n_total < min_leaf * 2:
        return GbdtNode(val=mean_val)

    for i in idxs:
        in_node[i] = True

    d = len(X[0])
    best_feat = -1
    best_thresh = 0.0
    best_gain = 0.0
    base_score = (total_sum * total_sum) / n_total

    for f in range(d):
        left_sum = 0.0
        left_n = 0
        prev_idx = -1
        for idx in sorted_orders[f]:
            if not in_node[idx]:
                continue
            if prev_idx != -1:
                x_prev = X[prev_idx][f]
                x_curr = X[idx][f]
                if x_prev != x_curr:
                    right_n = n_total - left_n
                    if left_n >= min_leaf and right_n >= min_leaf:
                        right_sum = total_sum - left_sum
                        gain = (left_sum * left_sum) / left_n + (right_sum * right_sum) / right_n - base_score
                        if gain > best_gain:
                            best_gain = gain
                            best_feat = f
                            best_thresh = 0.5 * (x_prev + x_curr)
            left_sum += residuals[idx]
            left_n += 1
            prev_idx = idx

    for i in idxs:
        in_node[i] = False

    if best_feat == -1:
        return GbdtNode(val=mean_val)

    left_idxs = [i for i in idxs if X[i][best_feat] <= best_thresh]
    right_idxs = [i for i in idxs if X[i][best_feat] > best_thresh]
    return GbdtNode(
        feat=best_feat,
        thresh=best_thresh,
        left=build_gbdt_tree(X, residuals, left_idxs, sorted_orders, in_node, depth - 1, min_leaf),
        right=build_gbdt_tree(X, residuals, right_idxs, sorted_orders, in_node, depth - 1, min_leaf),
    )


def predict_gbdt_tree(node: GbdtNode, x: list[float]) -> float:
    while node.val is None:
        assert node.feat is not None and node.thresh is not None
        node = node.left if x[node.feat] <= node.thresh else node.right  # type: ignore[assignment]
    return node.val


def fit_huber_gbdt(
    X: list[list[float]],
    y: list[float],
    n_trees: int = 40,
    lr: float = 0.04,
    depth: int = 2,
    min_leaf: int = 18,
    huber_delta: float = 12.0,
) -> tuple[float, float, list[GbdtNode]]:
    n = len(X)
    if n == 0:
        return 0.0, lr, []
    base = quantile(y, 0.5)
    preds = [base] * n
    trees: list[GbdtNode] = []
    all_idxs = list(range(n))
    d = len(X[0])
    sorted_orders = [sorted(all_idxs, key=lambda i, f=f: X[i][f]) for f in range(d)]
    in_node = [False] * n

    for _ in range(n_trees):
        res = []
        for yi, pi in zip(y, preds):
            err = yi - pi
            if abs(err) <= huber_delta:
                res.append(err)
            else:
                res.append(huber_delta if err > 0 else -huber_delta)
        tree = build_gbdt_tree(X, res, all_idxs, sorted_orders, in_node, depth, min_leaf)
        trees.append(tree)
        for i in range(n):
            preds[i] += lr * predict_gbdt_tree(tree, X[i])

    return base, lr, trees


def predict_huber_gbdt(base: float, lr: float, trees: list[GbdtNode], x: list[float]) -> float:
    total = base
    for tr in trees:
        total += lr * predict_gbdt_tree(tr, x)
    return max(0.0, total)


# ---------------------------------------------------------------------------
# Prophet-Style Causal Additive Time-Series Model
# ---------------------------------------------------------------------------

def prophet_causal_predict_leg(
    pool: list[CleanedShipmentRow],
    query_ms: int,
    leg_idx: int,
    fallback: float,
    seas_cache: dict[tuple[int, int, int], tuple[float, list[float]]] | None = None,
) -> float:
    if not pool:
        return fallback

    w_med = 0.0
    v_med = 0.0
    w_short = 0.0
    v_short = 0.0
    for r in pool:
        age = max(0.0, (query_ms - r.factory_arrive_ms) / MS_PER_DAY)
        wm = math.exp(-age / 240.0)
        ws = math.exp(-age / 90.0)
        val = r.leg_value(leg_idx)
        w_med += wm
        v_med += wm * val
        w_short += ws
        v_short += ws * val

    level_med = v_med / w_med if w_med > 0 else fallback
    level_short = v_short / w_short if w_short > 0 else level_med
    local_level = 0.65 * level_med + 0.35 * level_short
    if len(pool) < 30:
        return local_level

    cache_key = (id(pool[0]), len(pool), leg_idx)
    w_seas: list[float] | None = None
    if seas_cache is not None and cache_key in seas_cache:
        cached_level, cached_w = seas_cache[cache_key]
        if abs(cached_level - local_level) < 1e-6:
            w_seas = cached_w

    if w_seas is None:
        X = []
        y = []
        for r in pool:
            a1 = (2.0 * math.pi * r.reg_doy) / 365.25
            a2 = (4.0 * math.pi * r.reg_doy) / 365.25
            X.append([1.0, math.sin(a1), math.cos(a1), math.sin(a2), math.cos(a2)])
            y.append(r.leg_value(leg_idx) - local_level)
        w_seas = solve_ridge(X, y, 50.0)
        if seas_cache is not None:
            seas_cache[cache_key] = (local_level, w_seas)

    q_doy = day_of_year_utc(query_ms)
    qa1 = (2.0 * math.pi * q_doy) / 365.25
    qa2 = (4.0 * math.pi * q_doy) / 365.25
    seas = (
        w_seas[0]
        + w_seas[1] * math.sin(qa1)
        + w_seas[2] * math.cos(qa1)
        + w_seas[3] * math.sin(qa2)
        + w_seas[4] * math.cos(qa2)
    )
    return max(0.0, local_level + max(-12.0, min(12.0, seas)))


def dot_product(weights: list[float], features: list[float]) -> float:
    return max(0.0, sum(w * f for w, f in zip(weights, features)))


def build_leg_feature_vector(
    pool: list[CleanedShipmentRow],
    sup_clean: str,
    country_clean: str,
    sap_clean: str,
    log_weight: float,
    reg_doy: int,
    query_ms: int,
    stage: int,
    leg_idx: int,
    obs_d1: int,
    obs_d2: int,
    obs_d3: int,
    top_countries: list[str],
    fallback_leg_mean: float,
) -> list[float]:
    if pool:
        g_sum = 0.0
        w_sum = 0.0
        v_sum = 0.0
        ctr_sum = 0.0
        ctr_cnt = 0
        sap_sum = 0.0
        sap_cnt = 0
        sup_vals: list[int] = []
        for p in pool:
            val = p.leg_value(leg_idx)
            g_sum += val
            age = max(0.0, (query_ms - p.factory_arrive_ms) / MS_PER_DAY)
            w = math.exp(-age / 200.0)
            w_sum += w
            v_sum += w * val
            if p.country_clean == country_clean:
                ctr_sum += val
                ctr_cnt += 1
            if p.sap_code_clean == sap_clean:
                sap_sum += val
                sap_cnt += 1
            if p.supplier_clean == sup_clean:
                sup_vals.append(val)
        g_mean = g_sum / len(pool)
        leg_ewma = v_sum / w_sum if w_sum > 0 else g_mean
        ctr_mean = (ctr_sum + 3.0 * leg_ewma) / (ctr_cnt + 3.0)
        sap_mean = (sap_sum + 3.0 * ctr_mean) / (sap_cnt + 3.0)
        sup_mean = (sum(sup_vals) + 2.5 * ctr_mean) / (len(sup_vals) + 2.5)
        r3 = sup_vals[-3:]
        sup_r3 = (sum(r3) + 1.5 * sup_mean) / (len(r3) + 1.5)
    else:
        leg_ewma = fallback_leg_mean
        ctr_mean = fallback_leg_mean
        sap_mean = fallback_leg_mean
        sup_mean = fallback_leg_mean
        sup_r3 = fallback_leg_mean

    stage_month = month_utc(query_ms)
    reg_angle = (2.0 * math.pi * reg_doy) / 365.25
    stage_angle = (2.0 * math.pi * stage_month) / 12.0

    feats = [
        1.0,
        log_weight,
        math.sin(reg_angle),
        math.cos(reg_angle),
        math.sin(stage_angle),
        math.cos(stage_angle),
        *(1.0 if country_clean == tc else 0.0 for tc in top_countries),
        leg_ewma,
        ctr_mean,
        sap_mean,
        sup_mean,
        sup_r3,
    ]
    if stage >= 1:
        feats.append(float(obs_d1))
    if stage >= 2:
        feats.append(float(obs_d2))
    if stage >= 3:
        feats.append(float(obs_d3))
    return feats


def predict_stage_remaining(
    model: TrainedStageModel,
    pool: list[CleanedShipmentRow],
    sup_clean: str,
    country_clean: str,
    sap_clean: str,
    log_weight: float,
    reg_doy: int,
    query_ms: int,
    obs_d1: int,
    obs_d2: int,
    obs_d3: int,
    seas_cache: dict[tuple[int, int, int], tuple[float, list[float]]] | None = None,
) -> float:
    rem_total = 0.0
    for leg in model.legs:
        feats = build_leg_feature_vector(
            pool, sup_clean, country_clean, sap_clean,
            log_weight, reg_doy, query_ms, model.stage, leg.leg_idx,
            obs_d1, obs_d2, obs_d3, model.top_countries, leg.fallback_leg_mean,
        )
        pred_huber = dot_product(leg.weights, feats)
        pred_gbdt = predict_huber_gbdt(leg.gbdt_base, leg.gbdt_lr, leg.gbdt_trees, feats[1:])
        pred_prophet = prophet_causal_predict_leg(
            pool, query_ms, leg.leg_idx, leg.fallback_leg_mean, seas_cache
        )
        rem_total += (
            ENS_W_HUBER * pred_huber
            + ENS_W_GBDT * pred_gbdt
            + ENS_W_PROPHET * pred_prophet
        )
    return max(0.0, rem_total)


def apply_soft_nested_envelope(
    prev_best: float,
    prev_expected: float,
    prev_worst: float,
    raw_best: float,
    expected: float,
    raw_worst: float,
) -> tuple[float, float]:
    b = raw_best
    w = raw_worst
    if abs(expected - prev_expected) <= 22.0:
        if prev_best <= expected:
            b = max(b, prev_best)
        if prev_worst >= expected:
            w = min(w, prev_worst)
    b = min(b, expected)
    w = max(w, expected)
    return b, w


# ---------------------------------------------------------------------------
# CSV Parsing, Validation & Multi-Stage Training
# ---------------------------------------------------------------------------

def parse_csv_and_clean(csv_text: str) -> tuple[list[CleanedShipmentRow], list[str], list[str], list[str]]:
    text = csv_text.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValidationError("فایل CSV خالی است یا هدر ندارد.")

    fieldnames = [f.strip() for f in reader.fieldnames if f is not None]
    reader.fieldnames = fieldnames

    required = [
        "sap-code", "country", "supplier", "weight",
        "registery-date", "ministry-date", "ship-date",
        "costums-arrive-date", "factory-arrive-date", "LeadTime",
    ]
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise ValidationError(f"Missing required columns: {', '.join(missing)}")

    cleaned: list[CleanedShipmentRow] = []
    countries_set: set[str] = set()
    suppliers_set: set[str] = set()
    sap_codes_set: set[str] = set()

    for r in reader:
        if any(r.get(c) is None or str(r.get(c)).strip() == "" for c in required):
            continue

        country_clean = normalize_country(r["country"])
        supplier_clean = normalize_supplier(r["supplier"])
        sap_code_clean = normalize_sap_code(r["sap-code"])

        try:
            weight = float(str(r["weight"]).strip())
            lead_time_raw = float(str(r["LeadTime"]).strip())
        except ValueError as exc:
            raise ValidationError("Invalid numeric values found in weight or LeadTime.") from exc

        if not math.isfinite(weight) or not math.isfinite(lead_time_raw):
            raise ValidationError("Invalid numeric values found in weight or LeadTime.")
        if weight <= 0:
            raise ValidationError("All rows must have positive weight.")

        reg_ms = parse_flexible_date_ms(r["registery-date"])
        if reg_ms is None:
            raise ValidationError("Invalid or unparseable dates found in column: registery-date")
        min_ms = parse_flexible_date_ms(r["ministry-date"])
        if min_ms is None:
            raise ValidationError("Invalid or unparseable dates found in column: ministry-date")
        ship_ms = parse_flexible_date_ms(r["ship-date"])
        if ship_ms is None:
            raise ValidationError("Invalid or unparseable dates found in column: ship-date")
        cust_ms = parse_flexible_date_ms(r["costums-arrive-date"])
        if cust_ms is None:
            raise ValidationError("Invalid or unparseable dates found in column: costums-arrive-date")
        fact_ms = parse_flexible_date_ms(r["factory-arrive-date"])
        if fact_ms is None:
            raise ValidationError("Invalid or unparseable dates found in column: factory-arrive-date")

        dur_reg_to_min = round((min_ms - reg_ms) / MS_PER_DAY)
        dur_min_to_ship = round((ship_ms - min_ms) / MS_PER_DAY)
        dur_ship_to_cust = round((cust_ms - ship_ms) / MS_PER_DAY)
        dur_cust_to_fact = round((fact_ms - cust_ms) / MS_PER_DAY)

        if dur_reg_to_min < 0:
            raise ValidationError("Negative milestone duration detected in dur_reg_to_min.")
        if dur_min_to_ship < 0:
            raise ValidationError("Negative milestone duration detected in dur_min_to_ship.")
        if dur_ship_to_cust < 0:
            raise ValidationError("Negative milestone duration detected in dur_ship_to_cust.")
        if dur_cust_to_fact < 0:
            raise ValidationError("Negative milestone duration detected in dur_cust_to_fact.")

        computed_lead = round((fact_ms - reg_ms) / MS_PER_DAY)
        if computed_lead != round(lead_time_raw):
            raise ValidationError("Integrity check failed: LeadTime mismatch.")

        countries_set.add(country_clean)
        suppliers_set.add(supplier_clean)
        sap_codes_set.add(sap_code_clean)

        cleaned.append(
            CleanedShipmentRow(
                sap_code_clean=sap_code_clean,
                country_clean=country_clean,
                supplier_clean=supplier_clean,
                weight=weight,
                log_weight=math.log1p(max(weight, 0.0)),
                registery_ms=reg_ms,
                ministry_ms=min_ms,
                ship_ms=ship_ms,
                customs_ms=cust_ms,
                factory_arrive_ms=fact_ms,
                registery_date_str=format_ms_as_iso(reg_ms),
                lead_time=lead_time_raw,
                dur_reg_to_min=dur_reg_to_min,
                dur_min_to_ship=dur_min_to_ship,
                dur_ship_to_cust=dur_ship_to_cust,
                dur_cust_to_fact=dur_cust_to_fact,
                elapsed_reg_to_ship=round((ship_ms - reg_ms) / MS_PER_DAY),
                elapsed_reg_to_cust=round((cust_ms - reg_ms) / MS_PER_DAY),
                reg_month=month_utc(reg_ms),
                ministry_month=month_utc(min_ms),
                ship_month=month_utc(ship_ms),
                customs_month=month_utc(cust_ms),
                reg_doy=day_of_year_utc(reg_ms),
            )
        )

    if not cleaned:
        raise ValidationError("Dataset has no complete rows after removing missing values.")

    cleaned.sort(key=lambda row: row.registery_ms)
    return cleaned, sorted(countries_set), sorted(suppliers_set), sorted(sap_codes_set)


def train_single_stage(
    train_rows: list[CleanedShipmentRow],
    stage: int,
    top_countries: list[str],
    tune_l2: bool = True,
) -> TrainedStageModel:
    stage_l2_default = 12.0 if stage <= 1 else 25.0
    legs: list[TrainedLegModel] = []
    leg_x: list[list[list[float]]] = []
    leg_y: list[list[float]] = []

    row_pools: list[list[CleanedShipmentRow]] = []
    row_cutoffs: list[int] = []
    arrived_so_far: list[CleanedShipmentRow] = []
    arrive_keys: list[int] = []

    for r in train_rows:
        cut = r.stage_cutoff_ms(stage)
        pos = bisect.bisect_left(arrive_keys, cut)
        row_pools.append(arrived_so_far[:pos])
        row_cutoffs.append(cut)
        # Insert current row sorted by factory_arrive_ms for subsequent causal queries
        ins = bisect.bisect_right(arrive_keys, r.factory_arrive_ms)
        arrive_keys.insert(ins, r.factory_arrive_ms)
        arrived_so_far.insert(ins, r)

    for leg_idx in range(stage, 4):
        fallback_leg_mean = (
            mean([r.leg_value(leg_idx) for r in train_rows])
            if train_rows
            else LEG_DEFAULTS[leg_idx]
        )
        huber_delta = LEG_HUBER_DELTAS[leg_idx]

        Xtr: list[list[float]] = []
        ytr: list[float] = []
        for i, r in enumerate(train_rows):
            Xtr.append(
                build_leg_feature_vector(
                    row_pools[i],
                    r.supplier_clean,
                    r.country_clean,
                    r.sap_code_clean,
                    r.log_weight,
                    r.reg_doy,
                    row_cutoffs[i],
                    stage,
                    leg_idx,
                    r.dur_reg_to_min,
                    r.dur_min_to_ship,
                    r.dur_ship_to_cust,
                    top_countries,
                    fallback_leg_mean,
                )
            )
            ytr.append(float(r.leg_value(leg_idx)))

        chosen_l2 = stage_l2_default
        val_start = max(15, int(len(train_rows) * 0.72))
        if tune_l2 and len(train_rows) >= 30 and val_start < len(train_rows) - 4:
            best_mae = float("inf")
            cands = (10.0, 15.0, 25.0) if stage <= 1 else (18.0, 25.0, 40.0)
            for cand_l2 in cands:
                w_cand = fit_huber_ridge(Xtr[:val_start], ytr[:val_start], cand_l2, huber_delta)
                err_sum = sum(
                    abs(ytr[i] - dot_product(w_cand, Xtr[i]))
                    for i in range(val_start, len(train_rows))
                )
                val_mae = err_sum / (len(train_rows) - val_start)
                if val_mae < best_mae:
                    best_mae = val_mae
                    chosen_l2 = cand_l2

        weights = fit_huber_ridge(Xtr, ytr, chosen_l2, huber_delta)
        gbdt_base, gbdt_lr, gbdt_trees = fit_huber_gbdt(
            [x[1:] for x in Xtr],
            ytr,
            n_trees=40,
            lr=0.04,
            depth=2,
            min_leaf=18,
            huber_delta=huber_delta,
        )

        legs.append(
            TrainedLegModel(
                leg_idx=leg_idx,
                weights=weights,
                l2=chosen_l2,
                gbdt_base=gbdt_base,
                gbdt_lr=gbdt_lr,
                gbdt_trees=gbdt_trees,
                fallback_leg_mean=fallback_leg_mean,
            )
        )
        leg_x.append(Xtr)
        leg_y.append(ytr)

    # Fast 4-block rolling out-of-time conformal residuals over tail 45% of train_rows
    cal_start = max(20, int(len(train_rows) * 0.55))
    block_size = max(20, math.ceil((len(train_rows) - cal_start) / 4))
    residuals: list[float] = []
    seas_cache: dict[tuple[int, int, int], tuple[float, list[float]]] = {}

    for b_start in range(cal_start, len(train_rows), block_size):
        b_end = min(len(train_rows), b_start + block_size)
        sub_legs = []
        for idx, leg in enumerate(legs):
            huber_delta = LEG_HUBER_DELTAS[leg.leg_idx]
            w_sub = fit_huber_ridge(leg_x[idx][:b_start], leg_y[idx][:b_start], leg.l2, huber_delta)
            g_base, g_lr, g_trees = fit_huber_gbdt(
                [x[1:] for x in leg_x[idx][:b_start]],
                leg_y[idx][:b_start],
                n_trees=24,
                lr=0.05,
                depth=2,
                min_leaf=18,
                huber_delta=huber_delta,
            )
            sub_legs.append((leg.leg_idx, leg.fallback_leg_mean, w_sub, g_base, g_lr, g_trees))

        for i in range(b_start, b_end):
            r = train_rows[i]
            cut = row_cutoffs[i]
            pool = row_pools[i]
            rem_pred = 0.0
            for idx, (l_idx, fb_mean, w_sub, g_base, g_lr, g_trees) in enumerate(sub_legs):
                feat = leg_x[idx][i]
                ph = dot_product(w_sub, feat)
                pg = predict_huber_gbdt(g_base, g_lr, g_trees, feat[1:])
                pp = prophet_causal_predict_leg(pool, cut, l_idx, fb_mean, seas_cache)
                rem_pred += ENS_W_HUBER * ph + ENS_W_GBDT * pg + ENS_W_PROPHET * pp
            true_rem = max(0.0, r.lead_time - r.elapsed_at_stage(stage))
            residuals.append(true_rem - rem_pred)

    q_low_res = -20.0
    q_high_res = 20.0
    if len(residuals) >= 6:
        alpha_low = 0.055 if stage == 0 else 0.065
        alpha_high = 1.0 - alpha_low
        q_low_res = round(quantile(residuals, alpha_low), 2)
        q_high_res = round(quantile(residuals, alpha_high), 2)

    return TrainedStageModel(
        stage=stage,
        legs=legs,
        top_countries=top_countries,
        q_low_res=q_low_res,
        q_high_res=q_high_res,
        delta_low=0.0,
        delta_high=0.0,
    )


def run_pipeline_multistage(
    csv_text: str,
    test_frac: float = 0.2,
    source: str = "upload",
    tune_l2: bool = True,
) -> dict[str, Any]:
    rows, countries, suppliers, sap_codes = parse_csv_and_clean(csv_text)

    n = len(rows)
    if n < 10:
        raise ValidationError(f"Not enough rows for a train/test split (n={n}).")
    split = int(n * (1.0 - test_frac))
    if split < 5 or n - split < 5:
        raise ValidationError(
            f"Split {split}/{n - split} too small; upload more rows or reduce test_frac."
        )

    df_train = rows[:split]
    df_test = rows[split:]

    ctr_counts: dict[str, int] = {}
    for r in df_train:
        ctr_counts[r.country_clean] = ctr_counts.get(r.country_clean, 0) + 1
    top_countries = [
        k for k, _ in sorted(ctr_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    ]

    eval_stage_models: dict[int, TrainedStageModel] = {
        stage: train_single_stage(df_train, stage, top_countries, tune_l2)
        for stage in STAGES
    }

    def evaluate_sequential(
        subset: list[CleanedShipmentRow],
        base_pool: list[CleanedShipmentRow],
    ) -> dict[int, dict[str, Any]]:
        sorted_pool = sorted(base_pool, key=lambda r: r.factory_arrive_ms)
        arrive_keys = [r.factory_arrive_ms for r in sorted_pool]
        seas_cache: dict[tuple[int, int, int], tuple[float, list[float]]] = {}

        y_true = [r.lead_time for r in subset]
        stage_p10: dict[int, list[float]] = {s: [] for s in STAGES}
        stage_p50: dict[int, list[float]] = {s: [] for s in STAGES}
        stage_p90: dict[int, list[float]] = {s: [] for s in STAGES}

        for row in subset:
            prev_b = 0.0
            prev_e = 0.0
            prev_w = 999999.0

            for stage in STAGES:
                model = eval_stage_models[stage]
                cut = row.stage_cutoff_ms(stage)
                pos = bisect.bisect_left(arrive_keys, cut)
                pool = sorted_pool[:pos]
                el = row.elapsed_at_stage(stage)

                raw_rem = predict_stage_remaining(
                    model,
                    pool,
                    row.supplier_clean,
                    row.country_clean,
                    row.sap_code_clean,
                    row.log_weight,
                    row.reg_doy,
                    cut,
                    row.dur_reg_to_min,
                    row.dur_min_to_ship,
                    row.dur_ship_to_cust,
                    seas_cache,
                )

                raw_tot = el + raw_rem
                exp_tot = (
                    raw_tot
                    if stage == 0
                    else max(float(el), (1.0 - KALMAN_PREV_WEIGHT) * raw_tot + KALMAN_PREV_WEIGHT * prev_e)
                )
                adj_rem = max(0.0, exp_tot - el)
                lo_rem = max(0.0, adj_rem + model.q_low_res)
                hi_rem = max(adj_rem, adj_rem + model.q_high_res)
                cal10, _, cal90 = apply_calibration_and_order(
                    lo_rem, adj_rem, hi_rem, model.delta_low, model.delta_high
                )

                b_tot = el + cal10
                w_tot = el + cal90
                if stage > 0:
                    b_tot, w_tot = apply_soft_nested_envelope(
                        prev_b, prev_e, prev_w, b_tot, exp_tot, w_tot
                    )

                stage_p10[stage].append(b_tot)
                stage_p50[stage].append(exp_tot)
                stage_p90[stage].append(w_tot)

                prev_b = b_tot
                prev_e = exp_tot
                prev_w = w_tot

        return {
            s: evaluate_quantiles(y_true, stage_p10[s], stage_p50[s], stage_p90[s])
            for s in STAGES
        }

    train_evals = evaluate_sequential(df_train, df_train)
    test_evals = evaluate_sequential(df_test, rows)

    stage_metrics: dict[str, dict[str, Any]] = {}
    stage_models: dict[int, TrainedStageModel] = {}

    for stage in STAGES:
        cols = [*BASE_FEATURE_COLS, *STAGE_EXTRA_COLS[stage]]
        eval_model = eval_stage_models[stage]
        tr_e = train_evals[stage]
        te_e = test_evals[stage]
        stage_metrics[str(stage)] = {
            "stage": stage,
            "stage_name": STAGE_NAMES[stage],
            "stage_label_fa": STAGE_LABELS_FA[stage],
            "feature_cols": cols,
            "n_features": len(cols),
            "calibration": {
                "delta_low": eval_model.delta_low,
                "delta_high": eval_model.delta_high,
            },
            "params": {
                "ensemble": "ProphetTS(0.10)+HuberGBDT(0.12)+HuberRidge(0.78)",
                "leg_l2": [leg.l2 for leg in eval_model.legs],
            },
            "test": te_e,
            "train": {
                "n": tr_e["n"],
                "mae": tr_e["mae"],
                "r2": tr_e["r2"],
                "coverage_q10_q90": tr_e["coverage_q10_q90"],
            },
        }
        stage_models[stage] = train_single_stage(rows, stage, top_countries, tune_l2)

    benchmark_stage = 0
    metrics = {
        "protocol": (
            "chronological split by registery-date, leg-decomposed remaining-days ensemble "
            "(Prophet EWMA+Fourier time-series + Huber GBDT + Huber IRLS Ridge), "
            "Kalman stage stabilization & nested out-of-time conformal calibration"
        ),
        "target_mode": "remaining_days",
        "n_total": len(rows),
        "n_train": len(df_train),
        "n_test": len(df_test),
        "split_date": df_test[0].registery_date_str,
        "train_range": [df_train[0].registery_date_str, df_train[-1].registery_date_str],
        "test_range": [df_test[0].registery_date_str, df_test[-1].registery_date_str],
        "stages": stage_metrics,
        "benchmark": "r2",
        "benchmark_stage": benchmark_stage,
        "benchmark_r2": stage_metrics[str(benchmark_stage)]["test"]["r2"],
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": source,
    }

    return {
        "rows": rows,
        "countries": countries,
        "suppliers": suppliers,
        "sap_codes": sap_codes,
        "metrics": metrics,
        "stage_models": stage_models,
    }


def resolve_stage_and_dates(order: dict[str, Any]) -> dict[str, Any]:
    reg_raw = order.get("registry_date")
    reg_ms = parse_flexible_date_ms(reg_raw)
    if reg_ms is None:
        if not reg_raw or not str(reg_raw).strip():
            raise ValidationError("تاریخ ثبت سیستم الزامی است.")
        raise ValidationError("فرمت تاریخ برای registry_date نامعتبر است (YYYY-MM-DD).")

    def parse_opt(val: Any, name: str) -> int | None:
        if val is None or not str(val).strip():
            return None
        ms = parse_flexible_date_ms(val)
        if ms is None:
            raise ValidationError(f"فرمت تاریخ برای {name} نامعتبر است (YYYY-MM-DD).")
        return ms

    ministry_ms = parse_opt(order.get("ministry_date"), "ministry_date")
    ship_ms = parse_opt(order.get("ship_date"), "ship_date")
    customs_ms = parse_opt(order.get("customs_date"), "customs_date")

    if customs_ms is not None and (ship_ms is None or ministry_ms is None):
        raise ValidationError(
            "برای ثبت تاریخ ورود به گمرک، تاریخ مجوز وزارتخانه و حرکت کشتی نیز الزامی است."
        )
    if ship_ms is not None and ministry_ms is None:
        raise ValidationError(
            "برای ثبت تاریخ حرکت کشتی، تاریخ مجوز وزارتخانه نیز الزامی است."
        )

    chain: list[tuple[str, int]] = [("registry_date", reg_ms)]
    if ministry_ms is not None:
        chain.append(("ministry_date", ministry_ms))
    if ship_ms is not None:
        chain.append(("ship_date", ship_ms))
    if customs_ms is not None:
        chain.append(("customs_date", customs_ms))

    for i in range(1, len(chain)):
        if chain[i][1] < chain[i - 1][1]:
            raise ValidationError(
                f"{chain[i][0]} ({format_ms_as_iso(chain[i][1])}) نمی‌تواند قبل از "
                f"{chain[i - 1][0]} ({format_ms_as_iso(chain[i - 1][1])}) باشد."
            )

    if customs_ms is not None:
        stage = 3
    elif ship_ms is not None:
        stage = 2
    elif ministry_ms is not None:
        stage = 1
    else:
        stage = 0

    latest_ms = chain[-1][1]
    elapsed_days = round((latest_ms - reg_ms) / MS_PER_DAY)
    return {
        "stage": stage,
        "reg_ms": reg_ms,
        "ministry_ms": ministry_ms,
        "ship_ms": ship_ms,
        "customs_ms": customs_ms,
        "elapsed_days": elapsed_days,
    }


def predict_lead_time(
    order: dict[str, Any],
    sorted_history_by_arrive: list[CleanedShipmentRow],
    arrive_keys: list[int],
    metrics: dict[str, Any],
    stage_models: dict[int, TrainedStageModel],
) -> dict[str, Any]:
    sap_clean = normalize_sap_code(order.get("sap_code"))
    country_raw = str(order.get("country") or "").strip()
    sup_raw = str(order.get("supplier") or "").strip()

    if not sap_clean or not country_raw or not sup_raw:
        raise ValidationError("کد کالا (SAP)، کشور مبدأ و نام تأمین‌کننده الزامی هستند.")

    try:
        weight = float(order.get("weight", 0))
    except (TypeError, ValueError) as exc:
        raise ValidationError("وزن محموله باید عددی مثبت باشد.") from exc

    if not math.isfinite(weight) or weight <= 0:
        raise ValidationError("وزن محموله باید عددی مثبت باشد.")

    dates_info = resolve_stage_and_dates(order)
    stage = dates_info["stage"]
    reg_ms = dates_info["reg_ms"]
    ministry_ms = dates_info["ministry_ms"]
    ship_ms = dates_info["ship_ms"]
    customs_ms = dates_info["customs_ms"]
    elapsed_days = dates_info["elapsed_days"]

    stage_info = (metrics.get("stages") or {}).get(str(stage))
    stage_model = stage_models.get(stage)
    if not stage_info or stage_model is None:
        raise ValidationError(f"مدل برای مرحله {stage} در دسترس نیست.", 503)

    sup_clean = normalize_supplier(sup_raw)
    country_clean = normalize_country(country_raw)

    global_mean = (
        mean([r.lead_time for r in sorted_history_by_arrive])
        if sorted_history_by_arrive
        else 115.0
    )

    pos_reg = bisect.bisect_left(arrive_keys, reg_ms)
    completed_at_reg = sorted_history_by_arrive[:pos_reg]
    if not completed_at_reg and sorted_history_by_arrive:
        completed_at_reg = sorted_history_by_arrive[: min(20, len(sorted_history_by_arrive))]

    ctr_past = [r.lead_time for r in completed_at_reg if r.country_clean == country_clean]
    ctr_cnt = len(ctr_past)
    ctr_raw = mean(ctr_past) if ctr_cnt > 0 else global_mean
    country_hist_mean = (ctr_cnt * ctr_raw + 3.0 * global_mean) / (ctr_cnt + 3.0)

    past_sup = [r.lead_time for r in completed_at_reg if r.supplier_clean == sup_clean]
    sup_cnt = len(past_sup)
    is_unseen = sup_cnt == 0

    sup_raw_mean = country_hist_mean
    sup_std = 0.0
    r3_raw = country_hist_mean
    if not is_unseen:
        sup_raw_mean = mean(past_sup)
        sup_std = sample_std(past_sup) if sup_cnt > 1 else 0.0
        r3_raw = mean(past_sup[-3:])

    sup_mean = (sup_cnt * sup_raw_mean + 3.0 * country_hist_mean) / (sup_cnt + 3.0)
    r3_cnt = min(sup_cnt, 3)
    recent_3_mean = (r3_cnt * r3_raw + 2.0 * sup_mean) / (r3_cnt + 2.0)

    tier = 1
    if sup_cnt >= 3:
        tier = 0 if sup_std < 15.0 else (1 if sup_std < 30.0 else 2)

    d1 = round((ministry_ms - reg_ms) / MS_PER_DAY) if ministry_ms is not None else 0
    d2 = round((ship_ms - ministry_ms) / MS_PER_DAY) if (ship_ms is not None and ministry_ms is not None) else 0
    d3 = round((customs_ms - ship_ms) / MS_PER_DAY) if (customs_ms is not None and ship_ms is not None) else 0

    log_weight = math.log1p(max(weight, 0.0))
    reg_doy = day_of_year_utc(reg_ms)

    stage_dates = [reg_ms, ministry_ms, ship_ms, customs_ms]
    stage_elapsed = [0, d1, d1 + d2, d1 + d2 + d3]

    prev_best = 0.0
    prev_expected = 0.0
    prev_worst = 999999.0

    for s in range(stage + 1):
        m = stage_models.get(s)
        if m is None:
            continue
        query_ms = stage_dates[s] if stage_dates[s] is not None else reg_ms
        el = stage_elapsed[s]
        pos = bisect.bisect_left(arrive_keys, query_ms)
        pool = sorted_history_by_arrive[:pos]
        if not pool and sorted_history_by_arrive:
            pool = sorted_history_by_arrive[: min(20, len(sorted_history_by_arrive))]

        raw_rem = predict_stage_remaining(
            m, pool, sup_clean, country_clean, sap_clean,
            log_weight, reg_doy, query_ms, d1, d2, d3,
        )
        raw_total = el + raw_rem
        expected_tot = (
            raw_total
            if s == 0
            else max(float(el), (1.0 - KALMAN_PREV_WEIGHT) * raw_total + KALMAN_PREV_WEIGHT * prev_expected)
        )
        adj_rem = max(0.0, expected_tot - el)
        raw_low = max(0.0, adj_rem + m.q_low_res)
        raw_high = max(adj_rem, adj_rem + m.q_high_res)
        rem_low, _, rem_high = apply_calibration_and_order(
            raw_low, adj_rem, raw_high, m.delta_low, m.delta_high
        )

        best_tot = el + rem_low
        worst_tot = el + rem_high
        if s > 0:
            best_tot, worst_tot = apply_soft_nested_envelope(
                prev_best, prev_expected, prev_worst, best_tot, expected_tot, worst_tot
            )

        prev_best = best_tot
        prev_expected = expected_tot
        prev_worst = worst_tot

    best_total = max(elapsed_days, round(prev_best))
    expected_total = max(best_total, round(prev_expected))
    worst_total = max(expected_total, round(prev_worst))

    r_low = max(0, best_total - elapsed_days)
    r_med = max(r_low, expected_total - elapsed_days)
    r_high = max(r_med, worst_total - elapsed_days)

    best_date = format_ms_as_iso(reg_ms + best_total * MS_PER_DAY)
    expected_date = format_ms_as_iso(reg_ms + expected_total * MS_PER_DAY)
    worst_date = format_ms_as_iso(reg_ms + worst_total * MS_PER_DAY)

    tier_names = {
        0: "دسته طلایی (منظم)",
        1: "دسته نقره‌ای (نوسان متوسط)",
        2: "دسته قرمز (پرنوسان)",
    }
    if is_unseen:
        tier_label = "بدون سابقه قبلی"
    elif sup_cnt < 3:
        tier_label = f"کم‌سابقه ({sup_cnt} سفارش تکمیل‌شده)"
    else:
        tier_label = tier_names.get(tier, "نامشخص")

    stage_test = stage_info.get("test") or {}

    return {
        "stage": stage,
        "stage_name": STAGE_NAMES[stage],
        "stage_label_fa": STAGE_LABELS_FA[stage],
        "elapsed_days": elapsed_days,
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
        "historical_mean": "بدون سابقه" if is_unseen else round(sup_raw_mean, 1),
        "recent_trend": "بدون سابقه" if is_unseen else round(recent_3_mean, 1),
        "target_mode": "remaining_days",
    }


# ---------------------------------------------------------------------------
# Engine State & Lifecycle
# ---------------------------------------------------------------------------

class EngineState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.retrain_lock = threading.Lock()
        self.models_loaded = False
        self.rows: list[CleanedShipmentRow] = []
        self.sorted_by_arrive: list[CleanedShipmentRow] = []
        self.arrive_keys: list[int] = []
        self.metrics: dict[str, Any] | None = None
        self.stage_models: dict[int, TrainedStageModel] = {}
        self.countries: list[str] = []
        self.suppliers: list[str] = []
        self.sap_codes: list[str] = []
        self.init_error: str | None = None

    @staticmethod
    def _read_json_if_exists(path: Path) -> Any:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    def initialize(self) -> None:
        t0 = time.perf_counter()
        csv_path = DATA_DIR / "shipdata.csv"
        if not csv_path.exists():
            # Fallback to parent/shipdata.csv if launched during migration
            alt = BUNDLE_DIR.parent / "shipdata.csv"
            if alt.exists():
                csv_path = alt

        if not csv_path.exists():
            self.init_error = f"Dataset not found at {csv_path}"
            print(f"[LeadTime-Python] {self.init_error}", file=sys.stderr)
            return

        try:
            csv_text = csv_path.read_text(encoding="utf-8")
            result = run_pipeline_multistage(csv_text, test_frac=0.2, source="shipdata.csv", tune_l2=True)

            countries = self._read_json_if_exists(DATA_DIR / "countries.json") or result["countries"]
            suppliers = self._read_json_if_exists(DATA_DIR / "suppliers.json") or result["suppliers"]
            sap_codes = self._read_json_if_exists(DATA_DIR / "sap_codes.json") or result["sap_codes"]

            sorted_arr = sorted(result["rows"], key=lambda r: r.factory_arrive_ms)
            arr_keys = [r.factory_arrive_ms for r in sorted_arr]

            with self.lock:
                self.rows = result["rows"]
                self.sorted_by_arrive = sorted_arr
                self.arrive_keys = arr_keys
                self.metrics = result["metrics"]
                self.stage_models = result["stage_models"]
                self.countries = countries
                self.suppliers = suppliers
                self.sap_codes = sap_codes
                self.models_loaded = True
                self.init_error = None

            dt = time.perf_counter() - t0
            print(
                f"[LeadTime-Python] Ready in {dt:.2f}s | rows={len(self.rows)} | "
                f"suppliers={len(self.suppliers)} | stages=0,1,2,3",
                flush=True,
            )
        except Exception as exc:
            self.init_error = str(exc)
            print(f"[LeadTime-Python] Initialization failed: {exc}", file=sys.stderr, flush=True)

    def get_health(self) -> dict[str, Any]:
        with self.lock:
            if self.models_loaded and self.metrics is not None:
                target_mode = self.metrics.get("target_mode", "remaining_days")
                return {
                    "service": "leadtime-python-ml",
                    "status": "healthy",
                    "models_loaded": True,
                    "stages": sorted(int(k) for k in self.metrics.get("stages", {}).keys()),
                    "target_mode": target_mode,
                    "needs_retrain": target_mode != "remaining_days",
                    "n_total": self.metrics.get("n_total"),
                    "message": "All stage models loaded successfully",
                }
            return {
                "service": "leadtime-python-ml",
                "status": "degraded",
                "models_loaded": False,
                "message": self.init_error or "Models not loaded - initializing or missing data",
            }

    def get_metadata(self) -> dict[str, Any]:
        with self.lock:
            return {
                "countries": list(self.countries),
                "suppliers": list(self.suppliers),
                "sap_codes": list(self.sap_codes),
                "stages": list(STAGES),
                "stage_labels_fa": STAGE_LABELS_FA,
                "n_total": len(self.rows),
                "models_loaded": self.models_loaded,
                "trained_at": self.metrics.get("trained_at") if self.metrics else None,
                "source": self.metrics.get("source") if self.metrics else None,
            }

    def get_metrics(self) -> tuple[int, dict[str, Any]]:
        with self.lock:
            if not self.models_loaded or self.metrics is None:
                return 503, {"detail": "Models not loaded."}
            m = self.metrics
            target_mode = m.get("target_mode", "remaining_days")
            return 200, {
                "models_loaded": True,
                "target_mode": target_mode,
                "needs_retrain": target_mode != "remaining_days",
                "protocol": m.get("protocol", ""),
                "benchmark": m.get("benchmark", "r2"),
                "benchmark_stage": m.get("benchmark_stage", 0),
                "benchmark_r2": m.get("benchmark_r2"),
                "n_total": m.get("n_total"),
                "n_train": m.get("n_train"),
                "n_test": m.get("n_test"),
                "split_date": m.get("split_date"),
                "train_range": m.get("train_range"),
                "test_range": m.get("test_range"),
                "stages": m.get("stages", {}),
                "stage_labels_fa": STAGE_LABELS_FA,
                "trained_at": m.get("trained_at"),
                "source": m.get("source"),
            }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.models_loaded or self.metrics is None:
                raise ValidationError(
                    "مدل‌ها بارگذاری نشده‌اند. لطفاً وضعیت سرور را بررسی کنید.", 503
                )
            return predict_lead_time(
                payload,
                self.sorted_by_arrive,
                self.arrive_keys,
                self.metrics,
                self.stage_models,
            )

    def batch_predict_orders(self, orders: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(orders, list) or not orders:
            raise ValidationError("لیست سفارش‌ها خالی است.")
        if len(orders) > 500:
            raise ValidationError("حداکثر ۵۰۰ سفارش در هر درخواست دسته‌ای مجاز است.")

        with self.lock:
            if not self.models_loaded or self.metrics is None:
                raise ValidationError(
                    "مدل‌ها بارگذاری نشده‌اند. لطفاً وضعیت سرور را بررسی کنید.", 503
                )
            hist = self.sorted_by_arrive
            keys = self.arrive_keys
            metrics = self.metrics
            models = self.stage_models

        results: list[dict[str, Any]] = []
        succeeded = 0
        failed = 0
        for idx, order in enumerate(orders):
            if not isinstance(order, dict):
                failed += 1
                results.append({"row": idx + 1, "ok": False, "error": "فرمت رکورد نامعتبر است."})
                continue
            try:
                pred = predict_lead_time(order, hist, keys, metrics, models)
                succeeded += 1
                results.append({
                    "row": idx + 1,
                    "ok": True,
                    "input": {
                        "sap_code": normalize_sap_code(order.get("sap_code")),
                        "country": normalize_country(order.get("country")),
                        "supplier": normalize_supplier(order.get("supplier")),
                        "weight": order.get("weight"),
                        "registry_date": order.get("registry_date"),
                    },
                    "prediction": pred,
                })
            except ValidationError as exc:
                failed += 1
                results.append({"row": idx + 1, "ok": False, "error": exc.message})

        return {
            "count": len(orders),
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
        }

    def batch_predict_csv(self, csv_text: str) -> dict[str, Any]:
        text = csv_text.lstrip("\ufeff")
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise ValidationError("فایل CSV خالی است یا هدر ندارد.")

        fieldnames = [f.strip() for f in reader.fieldnames if f is not None]
        reader.fieldnames = fieldnames

        def pick(row: dict[str, Any], *keys: str) -> Any:
            for k in keys:
                if k in row and row[k] is not None and str(row[k]).strip() != "":
                    return str(row[k]).strip()
            return None

        orders: list[dict[str, Any]] = []
        for row in reader:
            if not any(str(v or "").strip() for v in row.values()):
                continue
            w_raw = pick(row, "weight", "Weight")
            try:
                w_val = float(w_raw) if w_raw is not None else float("nan")
            except ValueError:
                w_val = float("nan")
            orders.append({
                "sap_code": pick(row, "sap_code", "sap-code", "SAP") or "",
                "country": pick(row, "country", "Country", "principle") or "",
                "supplier": pick(row, "supplier", "Supplier") or "",
                "weight": w_val,
                "registry_date": pick(row, "registry_date", "registery-date", "registry-date") or "",
                "ministry_date": pick(row, "ministry_date", "ministry-date"),
                "ship_date": pick(row, "ship_date", "ship-date"),
                "customs_date": pick(row, "customs_date", "costums-arrive-date", "customs-arrive-date"),
            })

        return self.batch_predict_orders(orders)

    def retrain(self, csv_text: str, filename: str, test_frac: float = 0.2) -> dict[str, Any]:
        if not self.retrain_lock.acquire(blocking=False):
            raise ValidationError(
                "یک فرآیند بازآموزی هم‌اکنون در حال اجرا است. لطفاً کمی صبر کنید.", 429
            )
        try:
            result = run_pipeline_multistage(
                csv_text, test_frac=test_frac, source=filename, tune_l2=True
            )
            sorted_arr = sorted(result["rows"], key=lambda r: r.factory_arrive_ms)
            arr_keys = [r.factory_arrive_ms for r in sorted_arr]

            with self.lock:
                self.rows = result["rows"]
                self.sorted_by_arrive = sorted_arr
                self.arrive_keys = arr_keys
                self.countries = result["countries"]
                self.suppliers = result["suppliers"]
                self.sap_codes = result["sap_codes"]
                self.metrics = result["metrics"]
                self.stage_models = result["stage_models"]
                self.models_loaded = True
                self.init_error = None

            # Persist updated registries, dataset & metrics
            try:
                PERSIST_DATA_DIR.mkdir(parents=True, exist_ok=True)
                (PERSIST_DATA_DIR / "countries.json").write_text(
                    json.dumps(result["countries"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (PERSIST_DATA_DIR / "suppliers.json").write_text(
                    json.dumps(result["suppliers"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (PERSIST_DATA_DIR / "sap_codes.json").write_text(
                    json.dumps(result["sap_codes"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (PERSIST_DATA_DIR / "shipdata.csv").write_text(csv_text, encoding="utf-8")

                PERSIST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                (PERSIST_ARTIFACTS_DIR / "metrics.json").write_text(
                    json.dumps(result["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as write_err:
                print(f"[LeadTime-Python] Warning: could not persist artifacts: {write_err}", file=sys.stderr)

            m = result["metrics"]
            return {
                "message": "مدل‌ها با منطق زمان باقی‌مانده و کالیبراسیون CQR بازآموزی و فعال شدند.",
                "benchmark_stage": m["benchmark_stage"],
                "benchmark_r2": m["benchmark_r2"],
                "n_total": m["n_total"],
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "stages": m["stages"],
            }
        finally:
            self.retrain_lock.release()


ENGINE = EngineState()


# ---------------------------------------------------------------------------
# HTTP Server (FastAPI+Uvicorn when installed, stdlib ThreadingHTTPServer fallback)
# ---------------------------------------------------------------------------

def parse_multipart_upload(headers: Any, body: bytes) -> tuple[str, bytes, dict[str, str]]:
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValidationError("فرمت درخواست باید multipart/form-data باشد.")

    env = {
        "REQUEST_METHOD": "POST",
        "CONTENT_TYPE": content_type,
        "CONTENT_LENGTH": str(len(body)),
    }
    form = cgi.FieldStorage(fp=io.BytesIO(body), environ=env, keep_blank_values=True)
    fields: dict[str, str] = {}
    filename = ""
    file_bytes = b""

    for key in form.keys():
        item = form[key]
        if isinstance(item, list):
            item = item[0]
        if getattr(item, "filename", None):
            filename = str(item.filename)
            file_bytes = item.file.read()
        else:
            fields[key] = str(item.value)

    if not filename or not filename.lower().endswith(".csv"):
        raise ValidationError("لطفاً یک فایل CSV معتبر انتخاب کنید.")
    if not file_bytes:
        raise ValidationError("فایل CSV انتخاب‌شده خالی است.")
    return filename, file_bytes, fields


class LeadTimeRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep stdout clean; log only errors via stderr when needed
        return

    def _send_json(self, status_code: int, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self, max_bytes: int = 50 * 1024 * 1024) -> bytes:
        cl_header = self.headers.get("Content-Length")
        if not cl_header:
            return b""
        try:
            length = int(cl_header)
        except ValueError as exc:
            raise ValidationError("Invalid Content-Length header.") from exc
        if length < 0 or length > max_bytes:
            raise ValidationError("حجم درخواست بیش از حد مجاز (۵۰ مگابایت) است.", 413)
        return self.rfile.read(length)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/" or path == "/info":
            self._send_json(200, {
                "service": "Lead Time ML Inference Engine",
                "version": "3.0.0",
                "endpoints": ["/health", "/metadata", "/metrics", "/predict", "/batch-predict", "/retrain"],
            })
        elif path == "/health":
            self._send_json(200, ENGINE.get_health())
        elif path == "/metadata":
            self._send_json(200, ENGINE.get_metadata())
        elif path == "/metrics":
            code, payload = ENGINE.get_metrics()
            self._send_json(code, payload)
        elif path == "/countries":
            self._send_json(200, ENGINE.get_metadata()["countries"])
        elif path == "/suppliers":
            self._send_json(200, ENGINE.get_metadata()["suppliers"])
        elif path == "/sap-codes":
            self._send_json(200, ENGINE.get_metadata()["sap_codes"])
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown endpoint: {path}"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        try:
            body = self._read_body()
            ctype = (self.headers.get("Content-Type") or "").lower()

            if path == "/predict":
                try:
                    data = json.loads(body.decode("utf-8")) if body else {}
                except Exception as exc:
                    raise ValidationError("بدنه JSON نامعتبر است.") from exc
                if not isinstance(data, dict):
                    raise ValidationError("بدنه درخواست باید یک آبجکت JSON باشد.")
                self._send_json(200, ENGINE.predict(data))

            elif path == "/batch-predict":
                if "multipart/form-data" in ctype:
                    _, file_bytes, _ = parse_multipart_upload(self.headers, body)
                    try:
                        csv_text = file_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        csv_text = file_bytes.decode("latin-1")
                    self._send_json(200, ENGINE.batch_predict_csv(csv_text))
                else:
                    try:
                        data = json.loads(body.decode("utf-8")) if body else {}
                    except Exception as exc:
                        raise ValidationError("بدنه JSON نامعتبر است.") from exc
                    orders = data if isinstance(data, list) else data.get("orders")
                    if not isinstance(orders, list):
                        raise ValidationError("فیلد orders باید آرایه‌ای از سفارش‌ها باشد.")
                    self._send_json(200, ENGINE.batch_predict_orders(orders))

            elif path == "/retrain":
                filename, file_bytes, fields = parse_multipart_upload(self.headers, body)
                try:
                    test_frac = float(fields.get("test_frac", "0.20"))
                except ValueError as exc:
                    raise ValidationError("نسبت داده آزمون نامعتبر است.") from exc
                if not math.isfinite(test_frac) or test_frac < 0.05 or test_frac > 0.5:
                    raise ValidationError("نسبت داده آزمون باید بین 0.05 و 0.50 باشد.")
                try:
                    csv_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    csv_text = file_bytes.decode("latin-1")
                self._send_json(200, ENGINE.retrain(csv_text, filename, test_frac=test_frac))

            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"detail": f"Unknown endpoint: {path}"})

        except ValidationError as exc:
            self._send_json(exc.status_code, {"detail": exc.message})
        except Exception as exc:
            self._send_json(500, {"detail": f"خطای داخلی موتور یادگیری ماشین: {exc}"})


def try_create_fastapi_app() -> Any:
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.responses import JSONResponse
    except ImportError:
        return None

    api = FastAPI(title="Lead Time ML Inference Engine", version="3.0.0")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return ENGINE.get_health()

    @api.get("/metadata")
    def metadata() -> dict[str, Any]:
        return ENGINE.get_metadata()

    @api.get("/metrics")
    def metrics() -> JSONResponse:
        code, payload = ENGINE.get_metrics()
        return JSONResponse(status_code=code, content=payload)

    @api.get("/countries")
    def countries() -> list[str]:
        return ENGINE.get_metadata()["countries"]

    @api.get("/suppliers")
    def suppliers() -> list[str]:
        return ENGINE.get_metadata()["suppliers"]

    @api.get("/sap-codes")
    def sap_codes() -> list[str]:
        return ENGINE.get_metadata()["sap_codes"]

    @api.post("/predict")
    async def predict_endpoint(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValidationError("بدنه درخواست باید یک آبجکت JSON باشد.")
            return ENGINE.predict(body)
        except ValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @api.post("/batch-predict")
    async def batch_predict_endpoint(request: Request) -> dict[str, Any]:
        ctype = (request.headers.get("content-type") or "").lower()
        try:
            if "multipart/form-data" in ctype:
                form = await request.form()
                file = form.get("file")
                if file is None or not hasattr(file, "read"):
                    raise ValidationError("لطفاً یک فایل CSV معتبر انتخاب کنید.")
                fname = str(getattr(file, "filename", "") or "")
                if not fname.lower().endswith(".csv"):
                    raise ValidationError("لطفاً یک فایل CSV معتبر انتخاب کنید.")
                raw_bytes = await file.read()
                csv_text = raw_bytes.decode("utf-8", errors="replace")
                return ENGINE.batch_predict_csv(csv_text)
            body = await request.json()
            orders = body if isinstance(body, list) else (body.get("orders") if isinstance(body, dict) else None)
            if not isinstance(orders, list):
                raise ValidationError("فیلد orders باید آرایه‌ای از سفارش‌ها باشد.")
            return ENGINE.batch_predict_orders(orders)
        except ValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    @api.post("/retrain")
    async def retrain_endpoint(
        file: UploadFile = File(...),
        test_frac: float = Form(0.20),
    ) -> dict[str, Any]:
        try:
            if not file.filename or not file.filename.lower().endswith(".csv"):
                raise ValidationError("لطفاً یک فایل CSV معتبر انتخاب کنید.")
            if not math.isfinite(test_frac) or test_frac < 0.05 or test_frac > 0.5:
                raise ValidationError("نسبت داده آزمون باید بین 0.05 و 0.50 باشد.")
            raw_bytes = await file.read()
            csv_text = raw_bytes.decode("utf-8", errors="replace")
            return ENGINE.retrain(csv_text, file.filename, test_frac=test_frac)
        except ValidationError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return api


app = try_create_fastapi_app()


def main() -> None:
    init_thread = threading.Thread(target=ENGINE.initialize, name="ml-engine-init", daemon=True)
    init_thread.start()
    if app is not None:
        try:
            import uvicorn  # type: ignore
            print(f"[LeadTime-Python] Starting Uvicorn on http://{BIND_HOST}:{BIND_PORT}", flush=True)
            uvicorn.run(app, host=BIND_HOST, port=BIND_PORT, log_level="warning")
            return
        except ImportError:
            pass

    server = ThreadingHTTPServer((BIND_HOST, BIND_PORT), LeadTimeRequestHandler)
    print(f"[LeadTime-Python] Listening on http://{BIND_HOST}:{BIND_PORT} (stdlib HTTP)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
