import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import cors from "cors";
import express from "express";
import type { NextFunction, Request, Response } from "express";
import multer from "multer";
import {
  STAGE_LABELS_FA,
  STAGES,
  ValidationError,
  predictLeadTime,
  runPipelineMultistage,
  type CleanedShipmentRow,
  type MetricsBundle,
  type OrderInput,
  type TrainedStageModel,
} from "./src/leadTimeEngine.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT_DIR = __dirname;
const FRONTEND_INDEX = path.join(ROOT_DIR, "app", "frontend", "index.html");
const DATA_DIR = path.join(ROOT_DIR, "app", "backend", "data");
const ARTIFACTS_DIR = path.join(
  ROOT_DIR,
  "lead_time_platform_final",
  "platform",
  "artifacts_fixed"
);
const METRICS_PATH = path.join(ARTIFACTS_DIR, "metrics.json");
const CSV_PATH = path.join(ROOT_DIR, "shipdata.csv");

interface AppState {
  modelsLoaded: boolean;
  rows: CleanedShipmentRow[];
  metrics: MetricsBundle | null;
  stageModels: Record<number, TrainedStageModel>;
  countries: string[];
  suppliers: string[];
  sapCodes: string[];
}

const state: AppState = {
  modelsLoaded: false,
  rows: [],
  metrics: null,
  stageModels: {},
  countries: [],
  suppliers: [],
  sapCodes: [],
};

let retrainInProgress = false;

function readJsonIfExists<T>(filePath: string): T | null {
  try {
    if (!fs.existsSync(filePath)) return null;
    const raw = fs.readFileSync(filePath, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

function initializeState(): void {
  try {
    if (fs.existsSync(CSV_PATH)) {
      const csvText = fs.readFileSync(CSV_PATH, "utf-8");
      const result = runPipelineMultistage(csvText, 0.2, "shipdata.csv", true);
      state.rows = result.rows;
      state.countries =
        readJsonIfExists<string[]>(path.join(DATA_DIR, "countries.json")) ??
        result.countries;
      state.suppliers =
        readJsonIfExists<string[]>(path.join(DATA_DIR, "suppliers.json")) ??
        result.suppliers;
      state.sapCodes =
        readJsonIfExists<string[]>(path.join(DATA_DIR, "sap_codes.json")) ??
        result.sapCodes;
      state.metrics = result.metrics;
      state.stageModels = result.stageModels;
      state.modelsLoaded = true;
    }

    console.log(
      `[LeadTime] Loaded ${state.rows.length} historical shipments, ${state.suppliers.length} suppliers, stages=${Object.keys(state.metrics?.stages ?? {}).join(",")}`
    );
  } catch (err) {
    console.error("[LeadTime] Failed to initialize state:", err);
    state.modelsLoaded = false;
  }
}

initializeState();

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

app.get("/", (_req: Request, res: Response) => {
  res.sendFile(FRONTEND_INDEX);
});

app.get("/api", (_req: Request, res: Response) => {
  res.json({
    message: "Multi-Stage Lead Time Prediction API",
    version: "2.1.0",
    stages: state.metrics ? Object.keys(state.metrics.stages).map(Number).sort() : [],
    health: "/health",
    metrics: "/metrics",
  });
});

app.get("/health", (_req: Request, res: Response) => {
  if (state.modelsLoaded && state.metrics) {
    const targetMode = state.metrics.target_mode || "remaining_days";
    res.json({
      status: "healthy",
      models_loaded: true,
      stages: Object.keys(state.metrics.stages).map(Number).sort(),
      target_mode: targetMode,
      needs_retrain: targetMode !== "remaining_days",
      message: "All stage models loaded successfully",
    });
    return;
  }
  res.json({
    status: "degraded",
    models_loaded: false,
    message: "Models not loaded - prediction endpoints may fail",
  });
});

app.get("/metrics", (_req: Request, res: Response) => {
  if (!state.modelsLoaded || !state.metrics) {
    res.status(503).json({ detail: "Models not loaded." });
    return;
  }
  const m = state.metrics;
  const targetMode = m.target_mode || "remaining_days";
  res.json({
    models_loaded: true,
    target_mode: targetMode,
    needs_retrain: targetMode !== "remaining_days",
    protocol: m.protocol || "",
    benchmark: m.benchmark || "r2",
    benchmark_stage: m.benchmark_stage ?? 0,
    benchmark_r2: m.benchmark_r2,
    n_total: m.n_total,
    n_train: m.n_train,
    n_test: m.n_test,
    split_date: m.split_date,
    train_range: m.train_range,
    test_range: m.test_range,
    stages: m.stages || {},
    stage_labels_fa: STAGE_LABELS_FA,
    trained_at: m.trained_at,
    source: m.source,
  });
});

app.get("/api/countries", (_req: Request, res: Response) => {
  res.json(state.countries);
});

app.get("/api/suppliers", (_req: Request, res: Response) => {
  res.json(state.suppliers);
});

app.get("/api/sap-codes", (_req: Request, res: Response) => {
  res.json(state.sapCodes);
});

app.post("/predict", (req: Request, res: Response) => {
  if (!state.modelsLoaded || !state.metrics) {
    res.status(503).json({
      detail: "مدل‌ها بارگذاری نشده‌اند. لطفاً وضعیت سرور را بررسی کنید.",
    });
    return;
  }

  try {
    const body = req.body as Partial<OrderInput>;
    const order: OrderInput = {
      sap_code: String(body.sap_code ?? ""),
      country: String(body.country ?? ""),
      supplier: String(body.supplier ?? ""),
      weight: Number(body.weight),
      registry_date: String(body.registry_date ?? ""),
      ministry_date: body.ministry_date ?? null,
      ship_date: body.ship_date ?? null,
      customs_date: body.customs_date ?? null,
    };

    const result = predictLeadTime(
      order,
      state.rows,
      state.metrics,
      state.stageModels
    );
    res.json(result);
  } catch (err) {
    if (err instanceof ValidationError) {
      res.status(err.statusCode).json({ detail: err.message });
      return;
    }
    res.status(400).json({
      detail: err instanceof Error ? err.message : String(err),
    });
  }
});

app.post("/retrain", upload.single("file"), (req: Request, res: Response) => {
  const file = req.file;
  if (!file || !file.originalname.toLowerCase().endsWith(".csv")) {
    res.status(400).json({ detail: "لطفاً یک فایل CSV معتبر انتخاب کنید." });
    return;
  }

  const testFracRaw = req.body?.test_frac ?? "0.20";
  const testFrac = Number(testFracRaw);
  if (!Number.isFinite(testFrac) || testFrac < 0.05 || testFrac > 0.5) {
    res.status(400).json({ detail: "نسبت داده آزمون باید بین 0.05 و 0.50 باشد." });
    return;
  }

  const useTuning = String(req.body?.use_tuning ?? "true") !== "false";

  if (retrainInProgress) {
    res.status(429).json({
      detail: "یک فرآیند بازآموزی هم‌اکنون در حال اجرا است. لطفاً کمی صبر کنید.",
    });
    return;
  }

  retrainInProgress = true;
  try {
    const csvText = file.buffer.toString("utf-8");
    const result = runPipelineMultistage(
      csvText,
      testFrac,
      file.originalname,
      useTuning
    );

    state.rows = result.rows;
    state.countries = result.countries;
    state.suppliers = result.suppliers;
    state.sapCodes = result.sapCodes;
    state.metrics = result.metrics;
    state.stageModels = result.stageModels;
    state.modelsLoaded = true;

    try {
      fs.mkdirSync(DATA_DIR, { recursive: true });
      fs.writeFileSync(
        path.join(DATA_DIR, "countries.json"),
        JSON.stringify(result.countries, null, 2),
        "utf-8"
      );
      fs.writeFileSync(
        path.join(DATA_DIR, "suppliers.json"),
        JSON.stringify(result.suppliers, null, 2),
        "utf-8"
      );
      fs.writeFileSync(
        path.join(DATA_DIR, "sap_codes.json"),
        JSON.stringify(result.sapCodes, null, 2),
        "utf-8"
      );
      fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
      fs.writeFileSync(
        METRICS_PATH,
        JSON.stringify(result.metrics, null, 2),
        "utf-8"
      );
    } catch (writeErr) {
      console.warn("[LeadTime] Could not persist retrained metadata to disk:", writeErr);
    }

    res.json({
      message: "مدل‌ها با منطق زمان باقی‌مانده و کالیبراسیون CQR بازآموزی و فعال شدند.",
      benchmark_stage: result.metrics.benchmark_stage,
      benchmark_r2: result.metrics.benchmark_r2,
      n_total: result.metrics.n_total,
      n_train: result.metrics.n_train,
      n_test: result.metrics.n_test,
      stages: result.metrics.stages,
    });
  } catch (err) {
    if (err instanceof ValidationError) {
      res.status(err.statusCode).json({ detail: err.message });
      return;
    }
    res.status(500).json({
      detail: `بازآموزی ناموفق بود: ${err instanceof Error ? err.message : String(err)}`,
    });
  } finally {
    retrainInProgress = false;
  }
});

// Express error handler (handles multer file size limit errors etc.)
app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  if (err && typeof err === "object" && "code" in err && err.code === "LIMIT_FILE_SIZE") {
    res.status(400).json({ detail: "حجم فایل بیش از ۵۰ مگابایت است." });
    return;
  }
  console.error("[LeadTime] Unhandled error:", err);
  res.status(500).json({ detail: "خطای داخلی سرور" });
});

const PORT = 3000;
const HOST = "0.0.0.0";

const isMain =
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(__filename);

if (isMain && process.env.NODE_ENV !== "test") {
  app.listen(PORT, HOST, () => {
    console.log(
      `[LeadTime] Server listening on http://${HOST}:${PORT} (stages: ${STAGES.join(",")})`
    );
  });
}

export default app;
