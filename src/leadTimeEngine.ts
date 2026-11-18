import { parse } from "csv-parse/sync";

export const COUNTRY_MAP: Record<string, string> = {
  "UNITED ARAB EMIRATES": "U.A.E",
  UAE: "U.A.E",
  "U.A.E.": "U.A.E",
};

export const STAGE_NAMES: Record<number, string> = {
  0: "order_registration_day_zero",
  1: "post_ministry_approval",
  2: "post_vessel_departure",
  3: "post_customs_arrival",
};

export const STAGE_LABELS_FA: Record<number, string> = {
  0: "لحظه ثبت سفارش",
  1: "پس از مجوز وزارتخانه",
  2: "پس از حرکت کشتی از مبدأ",
  3: "پس از ورود به گمرک مقصد",
};

export const STAGES = [0, 1, 2, 3] as const;
export const TARGET_TAIL_ALPHA = 0.08;

const MONTHS: Record<string, number> = {
  jan: 0,
  feb: 1,
  mar: 2,
  apr: 3,
  may: 4,
  jun: 5,
  jul: 6,
  aug: 7,
  sep: 8,
  oct: 9,
  nov: 10,
  dec: 11,
};

const MS_PER_DAY = 86_400_000;

export interface CleanedShipmentRow {
  sapCodeClean: string;
  countryClean: string;
  supplierClean: string;
  weight: number;
  logWeight: number;
  registeryMs: number;
  ministryMs: number;
  shipMs: number;
  customsMs: number;
  factoryArriveMs: number;
  registeryDateStr: string;
  leadTime: number;
  durRegToMin: number;
  durMinToShip: number;
  durShipToCust: number;
  durCustToFact: number;
  elapsedRegToShip: number;
  elapsedRegToCust: number;
  regMonth: number;
  ministryMonth: number;
  shipMonth: number;
  customsMonth: number;
  regDoy: number;
}

export interface StageEvalMetrics {
  n: number;
  mae: number;
  rmse: number;
  r2: number;
  mape: number;
  medae: number;
  coverage_q10_q90: number;
  below_best_rate: number;
  above_worst_rate: number;
  mean_width_days: number;
}

export interface StageInfo {
  stage: number;
  stage_name: string;
  stage_label_fa: string;
  feature_cols: string[];
  n_features: number;
  calibration: {
    delta_low: number;
    delta_high: number;
  };
  params?: Record<string, unknown>;
  test: StageEvalMetrics;
  train: {
    n: number;
    mae: number;
    r2: number;
    coverage_q10_q90: number;
  };
}

export interface GbdtNode {
  feat?: number;
  thresh?: number;
  left?: GbdtNode;
  right?: GbdtNode;
  val?: number;
}

export interface TrainedLegModel {
  legIdx: number;
  weights: number[];
  l2: number;
  gbdtBase: number;
  gbdtLr: number;
  gbdtTrees: GbdtNode[];
  fallbackLegMean: number;
}

export interface TrainedStageModel {
  stage: number;
  legs: TrainedLegModel[];
  topCountries: string[];
  qLowRes: number;
  qHighRes: number;
  deltaLow: number;
  deltaHigh: number;
}

export interface MetricsBundle {
  protocol: string;
  target_mode: string;
  n_total: number;
  n_train: number;
  n_test: number;
  split_date: string;
  train_range: [string, string];
  test_range: [string, string];
  params?: Record<string, unknown>;
  stages: Record<string, StageInfo>;
  benchmark: string;
  benchmark_stage: number;
  benchmark_r2: number;
  trained_at: string;
  source: string;
}

export interface OrderInput {
  sap_code: string;
  country: string;
  supplier: string;
  weight: number;
  registry_date: string;
  ministry_date?: string | null;
  ship_date?: string | null;
  customs_date?: string | null;
}

export class ValidationError extends Error {
  statusCode: number;
  constructor(message: string, statusCode = 400) {
    super(message);
    this.statusCode = statusCode;
  }
}

export function normalizeCountry(val: string): string {
  const cleaned = String(val).trim().toUpperCase();
  return COUNTRY_MAP[cleaned] ?? cleaned;
}

export function normalizeSupplier(val: string): string {
  return String(val).trim().toUpperCase();
}

export function normalizeSapCode(val: string): string {
  return String(val).trim().replace(/\.0$/, "");
}

export function parseFlexibleDateMs(raw: unknown): number | null {
  if (raw === null || raw === undefined) return null;
  const s = String(raw).trim();
  if (!s) return null;

  const isoMatch = /^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[T\s].*)?$/.exec(s);
  if (isoMatch) {
    const year = Number(isoMatch[1]);
    const month = Number(isoMatch[2]) - 1;
    const day = Number(isoMatch[3]);
    const d = new Date(Date.UTC(year, month, day));
    if (
      d.getUTCFullYear() === year &&
      d.getUTCMonth() === month &&
      d.getUTCDate() === day
    ) {
      return d.getTime();
    }
    return null;
  }

  const dmyTextMatch = /^(\d{1,2})[-/\s]([A-Za-z]{3})[-/\s](\d{2,4})$/.exec(s);
  if (dmyTextMatch) {
    const day = Number(dmyTextMatch[1]);
    const month = MONTHS[dmyTextMatch[2].toLowerCase()];
    let year = Number(dmyTextMatch[3]);
    if (year < 100) {
      year += year >= 70 ? 1900 : 2000;
    }
    if (month !== undefined) {
      const d = new Date(Date.UTC(year, month, day));
      if (
        d.getUTCFullYear() === year &&
        d.getUTCMonth() === month &&
        d.getUTCDate() === day
      ) {
        return d.getTime();
      }
    }
    return null;
  }

  const mdyMatch = /^(\d{1,2})\/(\d{1,2})\/(\d{4})$/.exec(s);
  if (mdyMatch) {
    const month = Number(mdyMatch[1]) - 1;
    const day = Number(mdyMatch[2]);
    const year = Number(mdyMatch[3]);
    const d = new Date(Date.UTC(year, month, day));
    if (
      d.getUTCFullYear() === year &&
      d.getUTCMonth() === month &&
      d.getUTCDate() === day
    ) {
      return d.getTime();
    }
    return null;
  }

  const parsed = Date.parse(s);
  if (Number.isNaN(parsed)) return null;
  const d = new Date(parsed);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
}

export function formatMsAsIsoDate(ms: number): string {
  const d = new Date(ms);
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function dayOfYearUtc(ms: number): number {
  const d = new Date(ms);
  const start = Date.UTC(d.getUTCFullYear(), 0, 0);
  return Math.floor((ms - start) / MS_PER_DAY);
}

export function parseCsvAndClean(csvText: string): {
  rows: CleanedShipmentRow[];
  countries: string[];
  suppliers: string[];
  sapCodes: string[];
} {
  let records: Record<string, string>[];
  try {
    records = parse(csvText, {
      columns: true,
      skip_empty_lines: true,
      bom: true,
      trim: true,
    }) as Record<string, string>[];
  } catch (err) {
    throw new ValidationError(
      `خطا در خواندن CSV: ${err instanceof Error ? err.message : String(err)}`
    );
  }

  if (records.length === 0) {
    throw new ValidationError("فایل CSV خالی است.");
  }

  const required = [
    "sap-code",
    "country",
    "supplier",
    "weight",
    "registery-date",
    "ministry-date",
    "ship-date",
    "costums-arrive-date",
    "factory-arrive-date",
    "LeadTime",
  ];
  const cols = Object.keys(records[0]);
  const missing = required.filter((c) => !cols.includes(c));
  if (missing.length > 0) {
    throw new ValidationError(`Missing required columns: ${missing.join(", ")}`);
  }

  const cleaned: CleanedShipmentRow[] = [];
  const countriesSet = new Set<string>();
  const suppliersSet = new Set<string>();
  const sapCodesSet = new Set<string>();

  for (const r of records) {
    const hasMissing = required.some(
      (c) => r[c] === undefined || r[c] === null || String(r[c]).trim() === ""
    );
    if (hasMissing) continue;

    const countryClean = normalizeCountry(r["country"]);
    const supplierClean = normalizeSupplier(r["supplier"]);
    const sapCodeClean = normalizeSapCode(r["sap-code"]);

    const weight = Number(r["weight"]);
    const leadTimeRaw = Number(r["LeadTime"]);
    if (!Number.isFinite(weight) || !Number.isFinite(leadTimeRaw)) {
      throw new ValidationError("Invalid numeric values found in weight or LeadTime.");
    }
    if (weight <= 0) {
      throw new ValidationError("All rows must have positive weight.");
    }

    const registeryMs = parseFlexibleDateMs(r["registery-date"]);
    if (registeryMs === null) {
      throw new ValidationError("Invalid or unparseable dates found in column: registery-date");
    }
    const ministryMs = parseFlexibleDateMs(r["ministry-date"]);
    if (ministryMs === null) {
      throw new ValidationError("Invalid or unparseable dates found in column: ministry-date");
    }
    const shipMs = parseFlexibleDateMs(r["ship-date"]);
    if (shipMs === null) {
      throw new ValidationError("Invalid or unparseable dates found in column: ship-date");
    }
    const customsMs = parseFlexibleDateMs(r["costums-arrive-date"]);
    if (customsMs === null) {
      throw new ValidationError("Invalid or unparseable dates found in column: costums-arrive-date");
    }
    const factoryArriveMs = parseFlexibleDateMs(r["factory-arrive-date"]);
    if (factoryArriveMs === null) {
      throw new ValidationError("Invalid or unparseable dates found in column: factory-arrive-date");
    }

    const durRegToMin = Math.round((ministryMs - registeryMs) / MS_PER_DAY);
    const durMinToShip = Math.round((shipMs - ministryMs) / MS_PER_DAY);
    const durShipToCust = Math.round((customsMs - shipMs) / MS_PER_DAY);
    const durCustToFact = Math.round((factoryArriveMs - customsMs) / MS_PER_DAY);

    if (durRegToMin < 0) {
      throw new ValidationError("Negative milestone duration detected in dur_reg_to_min.");
    }
    if (durMinToShip < 0) {
      throw new ValidationError("Negative milestone duration detected in dur_min_to_ship.");
    }
    if (durShipToCust < 0) {
      throw new ValidationError("Negative milestone duration detected in dur_ship_to_cust.");
    }
    if (durCustToFact < 0) {
      throw new ValidationError("Negative milestone duration detected in dur_cust_to_fact.");
    }

    const computedLead = Math.round((factoryArriveMs - registeryMs) / MS_PER_DAY);
    if (computedLead !== Math.round(leadTimeRaw)) {
      throw new ValidationError("Integrity check failed: LeadTime mismatch.");
    }

    countriesSet.add(countryClean);
    suppliersSet.add(supplierClean);
    sapCodesSet.add(sapCodeClean);

    const regDate = new Date(registeryMs);
    cleaned.push({
      sapCodeClean,
      countryClean,
      supplierClean,
      weight,
      logWeight: Math.log1p(Math.max(weight, 0)),
      registeryMs,
      ministryMs,
      shipMs,
      customsMs,
      factoryArriveMs,
      registeryDateStr: formatMsAsIsoDate(registeryMs),
      leadTime: leadTimeRaw,
      durRegToMin,
      durMinToShip,
      durShipToCust,
      durCustToFact,
      elapsedRegToShip: Math.round((shipMs - registeryMs) / MS_PER_DAY),
      elapsedRegToCust: Math.round((customsMs - registeryMs) / MS_PER_DAY),
      regMonth: regDate.getUTCMonth() + 1,
      ministryMonth: new Date(ministryMs).getUTCMonth() + 1,
      shipMonth: new Date(shipMs).getUTCMonth() + 1,
      customsMonth: new Date(customsMs).getUTCMonth() + 1,
      regDoy: dayOfYearUtc(registeryMs),
    });
  }

  if (cleaned.length === 0) {
    throw new ValidationError("Dataset has no complete rows after removing missing values.");
  }

  cleaned.sort((a, b) => a.registeryMs - b.registeryMs);

  return {
    rows: cleaned,
    countries: Array.from(countriesSet).sort(),
    suppliers: Array.from(suppliersSet).sort(),
    sapCodes: Array.from(sapCodesSet).sort(),
  };
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (const v of arr) sum += v;
  return sum / arr.length;
}

function sampleStd(arr: number[]): number {
  if (arr.length <= 1) return 0;
  const m = mean(arr);
  let ss = 0;
  for (const v of arr) ss += (v - m) * (v - m);
  return Math.sqrt(ss / (arr.length - 1));
}

function quantile(arr: number[], q: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * Math.min(1, Math.max(0, q));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  const frac = pos - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

export function elapsedForRowAtStage(row: CleanedShipmentRow, stage: number): number {
  if (stage === 0) return 0;
  if (stage === 1) return row.durRegToMin;
  if (stage === 2) return row.elapsedRegToShip;
  if (stage === 3) return row.elapsedRegToCust;
  return 0;
}

function stageMonthForRow(row: CleanedShipmentRow, stage: number): number {
  if (stage === 1) return row.ministryMonth;
  if (stage === 2) return row.shipMonth;
  if (stage === 3) return row.customsMonth;
  return row.regMonth;
}

export function applyCalibrationAndOrder(
  rawLow: number,
  rawMed: number,
  rawHigh: number,
  deltaLow: number,
  deltaHigh: number
): [number, number, number] {
  let remLow = Math.max(0, rawLow - deltaLow);
  const remMed = Math.max(0, rawMed);
  let remHigh = Math.max(0, rawHigh + deltaHigh);
  remLow = Math.min(remLow, remMed);
  remHigh = Math.max(remHigh, remMed);
  return [remLow, remMed, remHigh];
}

function roundTo(val: number, decimals: number): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

export function evaluateQuantiles(
  yTrue: number[],
  p10: number[],
  p50: number[],
  p90: number[]
): StageEvalMetrics {
  const n = yTrue.length;
  if (n === 0) {
    return {
      n: 0,
      mae: 0,
      rmse: 0,
      r2: 0,
      mape: 0,
      medae: 0,
      coverage_q10_q90: 0,
      below_best_rate: 0,
      above_worst_rate: 0,
      mean_width_days: 0,
    };
  }

  const absErrs: number[] = [];
  let sqErrSum = 0;
  let apeSum = 0;
  let covered = 0;
  let below = 0;
  let above = 0;
  let widthSum = 0;
  const yMean = mean(yTrue);
  let ssTot = 0;

  for (let i = 0; i < n; i++) {
    const err = yTrue[i] - p50[i];
    const absErr = Math.abs(err);
    absErrs.push(absErr);
    sqErrSum += err * err;
    apeSum += absErr / Math.max(Math.abs(yTrue[i]), 1.0);
    ssTot += (yTrue[i] - yMean) * (yTrue[i] - yMean);
    if (yTrue[i] >= p10[i] && yTrue[i] <= p90[i]) covered++;
    if (yTrue[i] < p10[i]) below++;
    if (yTrue[i] > p90[i]) above++;
    widthSum += p90[i] - p10[i];
  }

  const r2 = ssTot > 0 ? 1 - sqErrSum / ssTot : 0;

  return {
    n,
    mae: roundTo(mean(absErrs), 2),
    rmse: roundTo(Math.sqrt(sqErrSum / n), 2),
    r2: roundTo(r2, 4),
    mape: roundTo((apeSum / n) * 100, 2),
    medae: roundTo(quantile(absErrs, 0.5), 2),
    coverage_q10_q90: roundTo(covered / n, 4),
    below_best_rate: roundTo(below / n, 4),
    above_worst_rate: roundTo(above / n, 4),
    mean_width_days: roundTo(widthSum / n, 2),
  };
}

const LEG_DEFAULTS = [10.6, 48.2, 47.5, 37.6];
const LEG_HUBER_DELTAS = [7, 14, 9, 7];
const ENS_W_HUBER = 0.78;
const ENS_W_GBDT = 0.12;
const ENS_W_PROPHET = 0.10;
const KALMAN_PREV_WEIGHT = 0.10;

function legValue(row: CleanedShipmentRow, legIdx: number): number {
  if (legIdx === 0) return row.durRegToMin;
  if (legIdx === 1) return row.durMinToShip;
  if (legIdx === 2) return row.durShipToCust;
  return row.durCustToFact;
}

function stageCutoffMs(row: CleanedShipmentRow, stage: number): number {
  if (stage === 1) return row.ministryMs;
  if (stage === 2) return row.shipMs;
  if (stage === 3) return row.customsMs;
  return row.registeryMs;
}

function solveRidge(
  X: number[][],
  y: number[],
  l2: number,
  sampleWeights?: number[]
): number[] {
  const n = X.length;
  const d = X[0].length;
  if (n === 0) return new Array(d).fill(0);

  const means = new Array(d).fill(0);
  const stds = new Array(d).fill(1);
  for (let j = 1; j < d; j++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += X[i][j];
    const m = sum / n;
    means[j] = m;
    let sq = 0;
    for (let i = 0; i < n; i++) sq += (X[i][j] - m) * (X[i][j] - m);
    stds[j] = Math.sqrt(sq / n) || 1.0;
  }

  const Xs = X.map((row) =>
    row.map((v, j) => (j === 0 ? 1.0 : (v - means[j]) / stds[j]))
  );

  const A: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const rhs: number[] = new Array(d).fill(0);

  for (let i = 0; i < n; i++) {
    const xi = Xs[i];
    const yi = y[i];
    const wi = sampleWeights ? sampleWeights[i] : 1.0;
    for (let j = 0; j < d; j++) {
      rhs[j] += wi * xi[j] * yi;
      for (let k = 0; k < d; k++) {
        A[j][k] += wi * xi[j] * xi[k];
      }
    }
  }

  for (let j = 1; j < d; j++) {
    A[j][j] += l2;
  }

  for (let i = 0; i < d; i++) {
    let maxRow = i;
    for (let r = i + 1; r < d; r++) {
      if (Math.abs(A[r][i]) > Math.abs(A[maxRow][i])) maxRow = r;
    }
    [A[i], A[maxRow]] = [A[maxRow], A[i]];
    [rhs[i], rhs[maxRow]] = [rhs[maxRow], rhs[i]];

    const piv = Math.abs(A[i][i]) > 1e-10 ? A[i][i] : 1e-10;
    for (let j = i; j < d; j++) A[i][j] /= piv;
    rhs[i] /= piv;

    for (let r = 0; r < d; r++) {
      if (r === i) continue;
      const factor = A[r][i];
      for (let j = i; j < d; j++) A[r][j] -= factor * A[i][j];
      rhs[r] -= factor * rhs[i];
    }
  }

  const weights = new Array(d).fill(0);
  let intercept = rhs[0];
  for (let j = 1; j < d; j++) {
    weights[j] = rhs[j] / stds[j];
    intercept -= weights[j] * means[j];
  }
  weights[0] = intercept;
  return weights;
}

function fitHuberRidgeRegression(
  X: number[][],
  y: number[],
  l2: number,
  huberDelta: number,
  iters = 6
): number[] {
  const n = X.length;
  const d = X[0].length;
  if (n === 0) return new Array(d).fill(0);

  const rw = new Array(n).fill(1.0);
  let weights = new Array(d).fill(0);

  for (let it = 0; it < iters; it++) {
    weights = solveRidge(X, y, l2, rw);
    for (let i = 0; i < n; i++) {
      let pred = 0;
      for (let j = 0; j < d; j++) pred += X[i][j] * weights[j];
      const absErr = Math.abs(y[i] - pred);
      rw[i] = absErr <= huberDelta ? 1.0 : huberDelta / absErr;
    }
  }
  return weights;
}

function buildGbdtTree(
  X: number[][],
  residuals: number[],
  idxs: number[],
  depth: number,
  minLeaf: number
): GbdtNode {
  let sum = 0;
  for (const i of idxs) sum += residuals[i];
  const meanVal = idxs.length > 0 ? sum / idxs.length : 0;
  if (depth <= 0 || idxs.length < minLeaf * 2) {
    return { val: meanVal };
  }

  let bestFeat = -1;
  let bestThresh = 0;
  let bestGain = 0;
  const nTotal = idxs.length;
  const d = X[0].length;

  for (let f = 0; f < d; f++) {
    const sorted = [...idxs].sort((a, b) => X[a][f] - X[b][f]);
    let leftSum = 0;
    let leftN = 0;
    for (let k = 0; k < sorted.length - 1; k++) {
      const idx = sorted[k];
      leftSum += residuals[idx];
      leftN++;
      const nextIdx = sorted[k + 1];
      if (X[idx][f] === X[nextIdx][f]) continue;
      const rightN = nTotal - leftN;
      if (leftN < minLeaf || rightN < minLeaf) continue;
      const rightSum = sum - leftSum;
      const gain =
        (leftSum * leftSum) / leftN +
        (rightSum * rightSum) / rightN -
        (sum * sum) / nTotal;
      if (gain > bestGain) {
        bestGain = gain;
        bestFeat = f;
        bestThresh = 0.5 * (X[idx][f] + X[nextIdx][f]);
      }
    }
  }

  if (bestFeat === -1) {
    return { val: meanVal };
  }

  const leftIdxs: number[] = [];
  const rightIdxs: number[] = [];
  for (const i of idxs) {
    if (X[i][bestFeat] <= bestThresh) leftIdxs.push(i);
    else rightIdxs.push(i);
  }

  return {
    feat: bestFeat,
    thresh: bestThresh,
    left: buildGbdtTree(X, residuals, leftIdxs, depth - 1, minLeaf),
    right: buildGbdtTree(X, residuals, rightIdxs, depth - 1, minLeaf),
  };
}

function predictGbdtTree(node: GbdtNode, x: number[]): number {
  if (node.val !== undefined) return node.val;
  return x[node.feat!] <= node.thresh!
    ? predictGbdtTree(node.left!, x)
    : predictGbdtTree(node.right!, x);
}

function fitHuberGbdt(
  X: number[][],
  y: number[],
  nTrees = 40,
  lr = 0.04,
  depth = 2,
  minLeaf = 18,
  huberDelta = 12
): { base: number; lr: number; trees: GbdtNode[] } {
  if (X.length === 0) return { base: 0, lr, trees: [] };
  const base = quantile(y, 0.5);
  const preds = new Array(y.length).fill(base);
  const trees: GbdtNode[] = [];
  const allIdxs = y.map((_, i) => i);

  for (let t = 0; t < nTrees; t++) {
    const res = y.map((yi, i) => {
      const err = yi - preds[i];
      if (Math.abs(err) <= huberDelta) return err;
      return err > 0 ? huberDelta : -huberDelta;
    });
    const tree = buildGbdtTree(X, res, allIdxs, depth, minLeaf);
    trees.push(tree);
    for (let i = 0; i < y.length; i++) {
      preds[i] += lr * predictGbdtTree(tree, X[i]);
    }
  }
  return { base, lr, trees };
}

function predictHuberGbdt(
  base: number,
  lr: number,
  trees: GbdtNode[],
  x: number[]
): number {
  let sum = base;
  for (const tr of trees) {
    sum += lr * predictGbdtTree(tr, x);
  }
  return Math.max(0, sum);
}

// Prophet-style causal additive time-series model:
// damped multi-horizon EWMA local level + annual K=2 Fourier seasonality.
function prophetCausalPredictLeg(
  pool: CleanedShipmentRow[],
  queryMs: number,
  legIdx: number,
  fallback: number
): number {
  if (pool.length === 0) return fallback;
  let wMed = 0;
  let vMed = 0;
  let wShort = 0;
  let vShort = 0;
  for (const r of pool) {
    const age = Math.max(0, (queryMs - r.factoryArriveMs) / MS_PER_DAY);
    const wm = Math.exp(-age / 240);
    const ws = Math.exp(-age / 90);
    const val = legValue(r, legIdx);
    wMed += wm;
    vMed += wm * val;
    wShort += ws;
    vShort += ws * val;
  }
  const levelMed = wMed > 0 ? vMed / wMed : fallback;
  const levelShort = wShort > 0 ? vShort / wShort : levelMed;
  const localLevel = 0.65 * levelMed + 0.35 * levelShort;
  if (pool.length < 30) return localLevel;

  const X: number[][] = [];
  const y: number[] = [];
  for (const r of pool) {
    const a1 = (2 * Math.PI * r.regDoy) / 365.25;
    const a2 = (4 * Math.PI * r.regDoy) / 365.25;
    X.push([1, Math.sin(a1), Math.cos(a1), Math.sin(a2), Math.cos(a2)]);
    y.push(legValue(r, legIdx) - localLevel);
  }
  const wSeas = solveRidge(X, y, 50);
  const qDoy = dayOfYearUtc(queryMs);
  const qa1 = (2 * Math.PI * qDoy) / 365.25;
  const qa2 = (4 * Math.PI * qDoy) / 365.25;
  const seas =
    wSeas[0] +
    wSeas[1] * Math.sin(qa1) +
    wSeas[2] * Math.cos(qa1) +
    wSeas[3] * Math.sin(qa2) +
    wSeas[4] * Math.cos(qa2);
  return Math.max(0, localLevel + Math.max(-12, Math.min(12, seas)));
}

function dotProduct(weights: number[], features: number[]): number {
  let sum = 0;
  for (let i = 0; i < weights.length; i++) {
    sum += weights[i] * (features[i] ?? 0);
  }
  return Math.max(0, sum);
}

function buildLegFeatureVector(
  pool: CleanedShipmentRow[],
  supClean: string,
  countryClean: string,
  sapClean: string,
  logWeight: number,
  regDoy: number,
  queryMs: number,
  stage: number,
  legIdx: number,
  obsD1: number,
  obsD2: number,
  obsD3: number,
  topCountries: string[],
  fallbackLegMean: number
): number[] {
  const gVals = pool.map((r) => legValue(r, legIdx));
  const gMean = gVals.length > 0 ? mean(gVals) : fallbackLegMean;

  let wSum = 0;
  let vSum = 0;
  for (const p of pool) {
    const age = Math.max(0, (queryMs - p.factoryArriveMs) / MS_PER_DAY);
    const w = Math.exp(-age / 200);
    wSum += w;
    vSum += w * legValue(p, legIdx);
  }
  const legEwma = wSum > 0 ? vSum / wSum : gMean;

  const ctrVals = pool
    .filter((r) => r.countryClean === countryClean)
    .map((r) => legValue(r, legIdx));
  const ctrSum = ctrVals.reduce((a, b) => a + b, 0);
  const ctrMean = (ctrSum + 3.0 * legEwma) / (ctrVals.length + 3.0);

  const sapVals = pool
    .filter((r) => r.sapCodeClean === sapClean)
    .map((r) => legValue(r, legIdx));
  const sapSum = sapVals.reduce((a, b) => a + b, 0);
  const sapMean = (sapSum + 3.0 * ctrMean) / (sapVals.length + 3.0);

  const supVals = pool
    .filter((r) => r.supplierClean === supClean)
    .map((r) => legValue(r, legIdx));
  const supSum = supVals.reduce((a, b) => a + b, 0);
  const supMean = (supSum + 2.5 * ctrMean) / (supVals.length + 2.5);

  const r3 = supVals.slice(-3);
  const r3Sum = r3.reduce((a, b) => a + b, 0);
  const supR3 = (r3Sum + 1.5 * supMean) / (r3.length + 1.5);

  const stageMonth = new Date(queryMs).getUTCMonth() + 1;
  const regAngle = (2 * Math.PI * regDoy) / 365.25;
  const stageAngle = (2 * Math.PI * stageMonth) / 12.0;

  const feats = [
    1.0,
    logWeight,
    Math.sin(regAngle),
    Math.cos(regAngle),
    Math.sin(stageAngle),
    Math.cos(stageAngle),
    ...topCountries.map((tc) => (countryClean === tc ? 1.0 : 0.0)),
    legEwma,
    ctrMean,
    sapMean,
    supMean,
    supR3,
  ];

  if (stage >= 1) feats.push(obsD1);
  if (stage >= 2) feats.push(obsD2);
  if (stage >= 3) feats.push(obsD3);
  return feats;
}

function predictStageRemaining(
  model: TrainedStageModel,
  pool: CleanedShipmentRow[],
  supClean: string,
  countryClean: string,
  sapClean: string,
  logWeight: number,
  regDoy: number,
  queryMs: number,
  obsD1: number,
  obsD2: number,
  obsD3: number
): number {
  let remTotal = 0;
  for (const leg of model.legs) {
    const feats = buildLegFeatureVector(
      pool,
      supClean,
      countryClean,
      sapClean,
      logWeight,
      regDoy,
      queryMs,
      model.stage,
      leg.legIdx,
      obsD1,
      obsD2,
      obsD3,
      model.topCountries,
      leg.fallbackLegMean
    );
    const predHuber = dotProduct(leg.weights, feats);
    const predGbdt = predictHuberGbdt(
      leg.gbdtBase,
      leg.gbdtLr,
      leg.gbdtTrees,
      feats.slice(1)
    );
    const predProphet = prophetCausalPredictLeg(
      pool,
      queryMs,
      leg.legIdx,
      leg.fallbackLegMean
    );
    remTotal +=
      ENS_W_HUBER * predHuber +
      ENS_W_GBDT * predGbdt +
      ENS_W_PROPHET * predProphet;
  }
  return Math.max(0, remTotal);
}

function applySoftNestedEnvelope(
  prevBest: number,
  prevExpected: number,
  prevWorst: number,
  rawBest: number,
  expected: number,
  rawWorst: number
): [number, number] {
  let b = rawBest;
  let w = rawWorst;
  // Apply monotonic nested bounds when stage shift stays within normal uncertainty
  if (Math.abs(expected - prevExpected) <= 22) {
    if (prevBest <= expected) b = Math.max(b, prevBest);
    if (prevWorst >= expected) w = Math.min(w, prevWorst);
  }
  b = Math.min(b, expected);
  w = Math.max(w, expected);
  return [b, w];
}

export function resolveStageAndDates(order: OrderInput): {
  stage: number;
  regMs: number;
  ministryMs: number | null;
  shipMs: number | null;
  customsMs: number | null;
  elapsedDays: number;
} {
  const regMs = parseFlexibleDateMs(order.registry_date);
  if (regMs === null) {
    if (!order.registry_date || !String(order.registry_date).trim()) {
      throw new ValidationError("تاریخ ثبت سیستم الزامی است.");
    }
    throw new ValidationError("فرمت تاریخ برای registry_date نامعتبر است (YYYY-MM-DD).");
  }

  const parseOpt = (val: string | null | undefined, name: string): number | null => {
    if (val === null || val === undefined || String(val).trim() === "") return null;
    const ms = parseFlexibleDateMs(val);
    if (ms === null) {
      throw new ValidationError(`فرمت تاریخ برای ${name} نامعتبر است (YYYY-MM-DD).`);
    }
    return ms;
  };

  const ministryMs = parseOpt(order.ministry_date, "ministry_date");
  const shipMs = parseOpt(order.ship_date, "ship_date");
  const customsMs = parseOpt(order.customs_date, "customs_date");

  if (customsMs !== null && (shipMs === null || ministryMs === null)) {
    throw new ValidationError(
      "برای ثبت تاریخ ورود به گمرک، تاریخ مجوز وزارتخانه و حرکت کشتی نیز الزامی است."
    );
  }
  if (shipMs !== null && ministryMs === null) {
    throw new ValidationError(
      "برای ثبت تاریخ حرکت کشتی، تاریخ مجوز وزارتخانه نیز الزامی است."
    );
  }

  const chain: Array<{ name: string; ms: number }> = [{ name: "registry_date", ms: regMs }];
  if (ministryMs !== null) chain.push({ name: "ministry_date", ms: ministryMs });
  if (shipMs !== null) chain.push({ name: "ship_date", ms: shipMs });
  if (customsMs !== null) chain.push({ name: "customs_date", ms: customsMs });

  for (let i = 1; i < chain.length; i++) {
    if (chain[i].ms < chain[i - 1].ms) {
      throw new ValidationError(
        `${chain[i].name} (${formatMsAsIsoDate(chain[i].ms)}) نمی‌تواند قبل از ${chain[i - 1].name} (${formatMsAsIsoDate(chain[i - 1].ms)}) باشد.`
      );
    }
  }

  let stage = 0;
  if (customsMs !== null) stage = 3;
  else if (shipMs !== null) stage = 2;
  else if (ministryMs !== null) stage = 1;

  const latestMs = chain[chain.length - 1].ms;
  const elapsedDays = Math.round((latestMs - regMs) / MS_PER_DAY);
  return { stage, regMs, ministryMs, shipMs, customsMs, elapsedDays };
}

export function predictLeadTime(
  order: OrderInput,
  historyRows: CleanedShipmentRow[],
  metrics: MetricsBundle,
  stageModels?: Record<number, TrainedStageModel>
) {
  const sapClean = normalizeSapCode(order.sap_code ?? "");
  const countryRaw = String(order.country ?? "").trim();
  const supRaw = String(order.supplier ?? "").trim();

  if (!sapClean || !countryRaw || !supRaw) {
    throw new ValidationError("کد کالا (SAP)، کشور مبدأ و نام تأمین‌کننده الزامی هستند.");
  }
  if (!Number.isFinite(order.weight) || order.weight <= 0) {
    throw new ValidationError("وزن محموله باید عددی مثبت باشد.");
  }

  const { stage, regMs, ministryMs, shipMs, customsMs, elapsedDays } =
    resolveStageAndDates(order);

  const stageInfo = metrics.stages?.[String(stage)];
  const stageModel = stageModels?.[stage];
  if (!stageInfo || !stageModel) {
    throw new ValidationError(`مدل برای مرحله ${stage} در دسترس نیست.`, 503);
  }

  const supClean = normalizeSupplier(supRaw);
  const countryClean = normalizeCountry(countryRaw);

  const globalMean =
    historyRows.length > 0
      ? mean(historyRows.map((r) => r.leadTime))
      : 115.0;

  let completedAtReg = historyRows
    .filter((r) => r.factoryArriveMs < regMs)
    .sort((a, b) => a.factoryArriveMs - b.factoryArriveMs);

  if (completedAtReg.length === 0 && historyRows.length > 0) {
    completedAtReg = historyRows.slice(0, Math.min(20, historyRows.length));
  }

  const ctrPast = completedAtReg.filter((r) => r.countryClean === countryClean);
  const ctrCnt = ctrPast.length;
  const ctrRaw = ctrCnt > 0 ? mean(ctrPast.map((r) => r.leadTime)) : globalMean;
  const countryHistMean = (ctrCnt * ctrRaw + 3.0 * globalMean) / (ctrCnt + 3.0);

  const pastSup = completedAtReg.filter((r) => r.supplierClean === supClean);
  const supCnt = pastSup.length;
  const isUnseen = supCnt === 0;

  let supRawMean = countryHistMean;
  let supStd = 0;
  let r3Raw = countryHistMean;
  if (!isUnseen) {
    const supLeads = pastSup.map((r) => r.leadTime);
    supRawMean = mean(supLeads);
    supStd = supCnt > 1 ? sampleStd(supLeads) : 0;
    r3Raw = mean(supLeads.slice(-3));
  }

  const supMean = (supCnt * supRawMean + 3.0 * countryHistMean) / (supCnt + 3.0);
  const r3Cnt = Math.min(supCnt, 3);
  const recent3Mean = (r3Cnt * r3Raw + 2.0 * supMean) / (r3Cnt + 2.0);

  let tier = 1;
  if (supCnt >= 3) {
    tier = supStd < 15 ? 0 : supStd < 30 ? 1 : 2;
  }

  const d1 = ministryMs !== null ? Math.round((ministryMs - regMs) / MS_PER_DAY) : 0;
  const d2 =
    shipMs !== null && ministryMs !== null
      ? Math.round((shipMs - ministryMs) / MS_PER_DAY)
      : 0;
  const d3 =
    customsMs !== null && shipMs !== null
      ? Math.round((customsMs - shipMs) / MS_PER_DAY)
      : 0;
  const logWeight = Math.log1p(Math.max(order.weight, 0));
  const regDoy = dayOfYearUtc(regMs);

  const stageDates = [regMs, ministryMs, shipMs, customsMs];
  const stageElapsed = [0, d1, d1 + d2, d1 + d2 + d3];

  let prevBest = 0;
  let prevExpected = 0;
  let prevWorst = 999999;

  for (let s = 0; s <= stage; s++) {
    const m = stageModels?.[s];
    const queryMs = stageDates[s] ?? regMs;
    const el = stageElapsed[s];
    let pool = historyRows
      .filter((r) => r.factoryArriveMs < queryMs)
      .sort((a, b) => a.factoryArriveMs - b.factoryArriveMs);
    if (pool.length === 0 && historyRows.length > 0) {
      pool = historyRows.slice(0, Math.min(20, historyRows.length));
    }

    if (!m) continue;
    const rawRem = predictStageRemaining(
      m,
      pool,
      supClean,
      countryClean,
      sapClean,
      logWeight,
      regDoy,
      queryMs,
      d1,
      d2,
      d3
    );

    const rawTotal = el + rawRem;
    const expectedTotal =
      s === 0
        ? rawTotal
        : Math.max(el, (1 - KALMAN_PREV_WEIGHT) * rawTotal + KALMAN_PREV_WEIGHT * prevExpected);
    const adjRem = Math.max(0, expectedTotal - el);
    const rawLow = Math.max(0, adjRem + m.qLowRes);
    const rawHigh = Math.max(adjRem, adjRem + m.qHighRes);
    const [remLow, , remHigh] = applyCalibrationAndOrder(
      rawLow,
      adjRem,
      rawHigh,
      m.deltaLow,
      m.deltaHigh
    );

    let bestTot = el + remLow;
    let worstTot = el + remHigh;
    if (s > 0) {
      [bestTot, worstTot] = applySoftNestedEnvelope(
        prevBest,
        prevExpected,
        prevWorst,
        bestTot,
        expectedTotal,
        worstTot
      );
    }

    prevBest = bestTot;
    prevExpected = expectedTotal;
    prevWorst = worstTot;
  }

  const bestTotal = Math.max(elapsedDays, Math.round(prevBest));
  const expectedTotal = Math.max(bestTotal, Math.round(prevExpected));
  const worstTotal = Math.max(expectedTotal, Math.round(prevWorst));

  const rLow = Math.max(0, bestTotal - elapsedDays);
  const rMed = Math.max(rLow, expectedTotal - elapsedDays);
  const rHigh = Math.max(rMed, worstTotal - elapsedDays);

  const bestDate = formatMsAsIsoDate(regMs + bestTotal * MS_PER_DAY);
  const expectedDate = formatMsAsIsoDate(regMs + expectedTotal * MS_PER_DAY);
  const worstDate = formatMsAsIsoDate(regMs + worstTotal * MS_PER_DAY);

  const tierNames: Record<number, string> = {
    0: "دسته طلایی (منظم)",
    1: "دسته نقره‌ای (نوسان متوسط)",
    2: "دسته قرمز (پرنوسان)",
  };

  let tierLabel: string;
  if (isUnseen) {
    tierLabel = "بدون سابقه قبلی";
  } else if (supCnt < 3) {
    tierLabel = `کم‌سابقه (${supCnt} سفارش تکمیل‌شده)`;
  } else {
    tierLabel = tierNames[tier] ?? "نامشخص";
  }

  const stageTest = stageInfo.test ?? {};

  return {
    stage,
    stage_name: STAGE_NAMES[stage],
    stage_label_fa: STAGE_LABELS_FA[stage],
    elapsed_days: elapsedDays,
    best_case_days: bestTotal,
    expected_days: expectedTotal,
    worst_case_days: worstTotal,
    remaining_best_days: rLow,
    remaining_expected_days: rMed,
    remaining_worst_days: rHigh,
    estimated_best_date: bestDate,
    estimated_expected_date: expectedDate,
    estimated_worst_date: worstDate,
    stage_test_r2: stageTest.r2,
    stage_test_mae: stageTest.mae,
    stage_coverage: stageTest.coverage_q10_q90,
    below_best_rate: stageTest.below_best_rate,
    above_worst_rate: stageTest.above_worst_rate,
    is_unseen_supplier: isUnseen,
    supplier_order_count: supCnt,
    supplier_tier_code: isUnseen ? -1 : tier,
    supplier_tier_name: tierLabel,
    historical_mean: isUnseen ? "بدون سابقه" : roundTo(supRawMean, 1),
    recent_trend: isUnseen ? "بدون سابقه" : roundTo(recent3Mean, 1),
    target_mode: "remaining_days",
  };
}

const BASE_FEATURE_COLS = [
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
];

const STAGE_EXTRA_COLS: Record<number, string[]> = {
  0: [],
  1: ["dur_reg_to_min"],
  2: ["dur_reg_to_min", "dur_min_to_ship"],
  3: ["dur_reg_to_min", "dur_min_to_ship", "dur_ship_to_cust"],
};

function trainSingleStage(
  trainRows: CleanedShipmentRow[],
  stage: number,
  topCountries: string[],
  tuneL2: boolean
): TrainedStageModel {
  const stageL2Default = stage <= 1 ? 12 : 25;
  const legs: TrainedLegModel[] = [];
  const legX: number[][][] = [];
  const legY: number[][] = [];
  const rowPools: CleanedShipmentRow[][] = [];
  const rowCutoffs: number[] = [];

  for (let i = 0; i < trainRows.length; i++) {
    const r = trainRows[i];
    const cut = stageCutoffMs(r, stage);
    const pool = trainRows
      .slice(0, i)
      .filter((p) => p.factoryArriveMs < cut)
      .sort((a, b) => a.factoryArriveMs - b.factoryArriveMs);
    rowPools.push(pool);
    rowCutoffs.push(cut);
  }

  for (let legIdx = stage; legIdx < 4; legIdx++) {
    const fallbackLegMean =
      trainRows.length > 0
        ? mean(trainRows.map((r) => legValue(r, legIdx)))
        : LEG_DEFAULTS[legIdx];
    const huberDelta = LEG_HUBER_DELTAS[legIdx];

    const Xtr: number[][] = [];
    const ytr: number[] = [];
    for (let i = 0; i < trainRows.length; i++) {
      const r = trainRows[i];
      Xtr.push(
        buildLegFeatureVector(
          rowPools[i],
          r.supplierClean,
          r.countryClean,
          r.sapCodeClean,
          r.logWeight,
          r.regDoy,
          rowCutoffs[i],
          stage,
          legIdx,
          r.durRegToMin,
          r.durMinToShip,
          r.durShipToCust,
          topCountries,
          fallbackLegMean
        )
      );
      ytr.push(legValue(r, legIdx));
    }

    let chosenL2 = stageL2Default;
    const valStart = Math.max(15, Math.floor(trainRows.length * 0.72));
    if (tuneL2 && trainRows.length >= 30 && valStart < trainRows.length - 4) {
      let bestMae = Infinity;
      for (const candL2 of stage <= 1 ? [10, 15, 25] : [18, 25, 40]) {
        const wCand = fitHuberRidgeRegression(
          Xtr.slice(0, valStart),
          ytr.slice(0, valStart),
          candL2,
          huberDelta
        );
        let errSum = 0;
        for (let i = valStart; i < trainRows.length; i++) {
          errSum += Math.abs(ytr[i] - dotProduct(wCand, Xtr[i]));
        }
        const valMae = errSum / (trainRows.length - valStart);
        if (valMae < bestMae) {
          bestMae = valMae;
          chosenL2 = candL2;
        }
      }
    }

    const weights = fitHuberRidgeRegression(Xtr, ytr, chosenL2, huberDelta);
    const gbdt = fitHuberGbdt(
      Xtr.map((x) => x.slice(1)),
      ytr,
      40,
      0.04,
      2,
      18,
      huberDelta
    );

    legs.push({
      legIdx,
      weights,
      l2: chosenL2,
      gbdtBase: gbdt.base,
      gbdtLr: gbdt.lr,
      gbdtTrees: gbdt.trees,
      fallbackLegMean,
    });
    legX.push(Xtr);
    legY.push(ytr);
  }

  // Fast 4-block rolling out-of-time conformal residuals over tail 45% of trainRows
  const calStart = Math.max(20, Math.floor(trainRows.length * 0.55));
  const blockSize = Math.max(20, Math.ceil((trainRows.length - calStart) / 4));
  const residuals: number[] = [];

  for (let bStart = calStart; bStart < trainRows.length; bStart += blockSize) {
    const bEnd = Math.min(trainRows.length, bStart + blockSize);
    const subLegs = legs.map((leg, idx) => {
      const huberDelta = LEG_HUBER_DELTAS[leg.legIdx];
      return {
        legIdx: leg.legIdx,
        fallbackLegMean: leg.fallbackLegMean,
        weights: fitHuberRidgeRegression(
          legX[idx].slice(0, bStart),
          legY[idx].slice(0, bStart),
          leg.l2,
          huberDelta
        ),
        gbdt: fitHuberGbdt(
          legX[idx].slice(0, bStart).map((x) => x.slice(1)),
          legY[idx].slice(0, bStart),
          24,
          0.05,
          2,
          18,
          huberDelta
        ),
      };
    });

    for (let i = bStart; i < bEnd; i++) {
      const r = trainRows[i];
      const cut = rowCutoffs[i];
      const pool = rowPools[i];
      let remPred = 0;
      for (let idx = 0; idx < subLegs.length; idx++) {
        const sub = subLegs[idx];
        const feat = legX[idx][i];
        const ph = dotProduct(sub.weights, feat);
        const pg = predictHuberGbdt(
          sub.gbdt.base,
          sub.gbdt.lr,
          sub.gbdt.trees,
          feat.slice(1)
        );
        const pp = prophetCausalPredictLeg(
          pool,
          cut,
          sub.legIdx,
          sub.fallbackLegMean
        );
        remPred += ENS_W_HUBER * ph + ENS_W_GBDT * pg + ENS_W_PROPHET * pp;
      }
      const trueRem = Math.max(0, r.leadTime - elapsedForRowAtStage(r, stage));
      residuals.push(trueRem - remPred);
    }
  }

  let qLowRes = -20;
  let qHighRes = 20;
  if (residuals.length >= 6) {
    const alphaLow = stage === 0 ? 0.055 : 0.065;
    const alphaHigh = 1.0 - alphaLow;
    qLowRes = roundTo(quantile(residuals, alphaLow), 2);
    qHighRes = roundTo(quantile(residuals, alphaHigh), 2);
  }

  return {
    stage,
    legs,
    topCountries,
    qLowRes,
    qHighRes,
    deltaLow: 0,
    deltaHigh: 0,
  };
}

export function runPipelineMultistage(
  csvText: string,
  testFrac = 0.2,
  source = "upload",
  tuneL2 = true
): {
  rows: CleanedShipmentRow[];
  countries: string[];
  suppliers: string[];
  sapCodes: string[];
  metrics: MetricsBundle;
  stageModels: Record<number, TrainedStageModel>;
} {
  const { rows, countries, suppliers, sapCodes } = parseCsvAndClean(csvText);

  const n = rows.length;
  if (n < 10) {
    throw new ValidationError(`Not enough rows for a train/test split (n=${n}).`);
  }
  const split = Math.floor(n * (1.0 - testFrac));
  if (split < 5 || n - split < 5) {
    throw new ValidationError(
      `Split ${split}/${n - split} too small; upload more rows or reduce test_frac.`
    );
  }

  const dfTrain = rows.slice(0, split);
  const dfTest = rows.slice(split);

  const ctrCounts = new Map<string, number>();
  for (const r of dfTrain) {
    ctrCounts.set(r.countryClean, (ctrCounts.get(r.countryClean) ?? 0) + 1);
  }
  const topCountries = Array.from(ctrCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map((x) => x[0]);

  const evalStageModels: Record<number, TrainedStageModel> = {};
  for (const stage of STAGES) {
    evalStageModels[stage] = trainSingleStage(dfTrain, stage, topCountries, tuneL2);
  }

  const evaluateSequential = (
    subset: CleanedShipmentRow[],
    basePool: CleanedShipmentRow[]
  ): Record<number, StageEvalMetrics> => {
    const yTrue = subset.map((r) => r.leadTime);
    const stageP10: number[][] = [[], [], [], []];
    const stageP50: number[][] = [[], [], [], []];
    const stageP90: number[][] = [[], [], [], []];

    for (const row of subset) {
      let prevB = 0;
      let prevE = 0;
      let prevW = 999999;

      for (const stage of STAGES) {
        const model = evalStageModels[stage];
        const cut = stageCutoffMs(row, stage);
        const pool = basePool
          .filter((r) => r.factoryArriveMs < cut)
          .sort((a, b) => a.factoryArriveMs - b.factoryArriveMs);
        const el = elapsedForRowAtStage(row, stage);
        const rawRem = predictStageRemaining(
          model,
          pool,
          row.supplierClean,
          row.countryClean,
          row.sapCodeClean,
          row.logWeight,
          row.regDoy,
          cut,
          row.durRegToMin,
          row.durMinToShip,
          row.durShipToCust
        );

        const rawTot = el + rawRem;
        const expTot =
          stage === 0
            ? rawTot
            : Math.max(el, (1 - KALMAN_PREV_WEIGHT) * rawTot + KALMAN_PREV_WEIGHT * prevE);
        const adjRem = Math.max(0, expTot - el);
        const loRem = Math.max(0, adjRem + model.qLowRes);
        const hiRem = Math.max(adjRem, adjRem + model.qHighRes);
        const [cal10, , cal90] = applyCalibrationAndOrder(
          loRem,
          adjRem,
          hiRem,
          model.deltaLow,
          model.deltaHigh
        );

        let bTot = el + cal10;
        let wTot = el + cal90;
        if (stage > 0) {
          [bTot, wTot] = applySoftNestedEnvelope(
            prevB,
            prevE,
            prevW,
            bTot,
            expTot,
            wTot
          );
        }

        stageP10[stage].push(bTot);
        stageP50[stage].push(expTot);
        stageP90[stage].push(wTot);

        prevB = bTot;
        prevE = expTot;
        prevW = wTot;
      }
    }

    return {
      0: evaluateQuantiles(yTrue, stageP10[0], stageP50[0], stageP90[0]),
      1: evaluateQuantiles(yTrue, stageP10[1], stageP50[1], stageP90[1]),
      2: evaluateQuantiles(yTrue, stageP10[2], stageP50[2], stageP90[2]),
      3: evaluateQuantiles(yTrue, stageP10[3], stageP50[3], stageP90[3]),
    };
  };

  const trainEvals = evaluateSequential(dfTrain, dfTrain);
  const testEvals = evaluateSequential(dfTest, rows);

  const stageMetrics: Record<string, StageInfo> = {};
  const stageModels: Record<number, TrainedStageModel> = {};

  for (const stage of STAGES) {
    const cols = [...BASE_FEATURE_COLS, ...STAGE_EXTRA_COLS[stage]];
    const evalModel = evalStageModels[stage];
    const trainEval = trainEvals[stage];
    const testEval = testEvals[stage];

    stageMetrics[String(stage)] = {
      stage,
      stage_name: STAGE_NAMES[stage],
      stage_label_fa: STAGE_LABELS_FA[stage],
      feature_cols: cols,
      n_features: cols.length,
      calibration: {
        delta_low: evalModel.deltaLow,
        delta_high: evalModel.deltaHigh,
      },
      params: {
        ensemble: "ProphetTS(0.10)+HuberGBDT(0.12)+HuberRidge(0.78)",
        leg_l2: evalModel.legs.map((l) => l.l2),
      },
      test: testEval,
      train: {
        n: trainEval.n,
        mae: trainEval.mae,
        r2: trainEval.r2,
        coverage_q10_q90: trainEval.coverage_q10_q90,
      },
    };

    stageModels[stage] = trainSingleStage(rows, stage, topCountries, tuneL2);
  }

  const benchmarkStage = 0;
  const metrics: MetricsBundle = {
    protocol:
      "chronological split by registery-date, leg-decomposed remaining-days ensemble (Prophet EWMA+Fourier time-series + Huber GBDT + Huber IRLS Ridge), Kalman stage stabilization & nested out-of-time conformal calibration",
    target_mode: "remaining_days",
    n_total: rows.length,
    n_train: dfTrain.length,
    n_test: dfTest.length,
    split_date: dfTest[0].registeryDateStr,
    train_range: [dfTrain[0].registeryDateStr, dfTrain[dfTrain.length - 1].registeryDateStr],
    test_range: [dfTest[0].registeryDateStr, dfTest[dfTest.length - 1].registeryDateStr],
    stages: stageMetrics,
    benchmark: "r2",
    benchmark_stage: benchmarkStage,
    benchmark_r2: stageMetrics[String(benchmarkStage)].test.r2,
    trained_at: new Date().toISOString(),
    source,
  };

  return { rows, countries, suppliers, sapCodes, metrics, stageModels };
}
