# Smart Lead Time Prediction Platform (LAN Client–Server Edition)

Production-ready, LAN-based Client–Server supply chain lead time forecasting platform.
Client workstations run a stateless, codeless web UI in any browser over the company LAN while all machine learning models, historical registries, and datasets stay exclusively on the server host.

## Architecture

```text
Workstation Browser (LAN)
        │  HTTP (minified HTML/CSS/JS, zero sourcemaps, zero datasets)
        ▼
gateway/server.ts (Bun LAN Gateway — 0.0.0.0:3000)
        │  Reverse proxy /api/* (127.0.0.1 only)
        ▼
backend/main.py (Python ML Inference Engine — 127.0.0.1:8000)
        ├── data/        (shipdata.csv, countries.json, sap_codes.json, suppliers.json)
        └── artifacts/   (metrics.json)
```

- **`backend/` ("The Brain")**: Python ML service bound strictly to `127.0.0.1:8000` (localhost-only, unreachable directly from external LAN workstations). Runs on Python 3.11+ stdlib out of the box and automatically uses FastAPI + Uvicorn when installed via `backend/requirements.txt`. Implements the multi-stage leg-decomposed ensemble (`Prophet EWMA+Fourier + Huber GBDT + Huber IRLS Ridge`), Kalman stage smoothing, and nested out-of-time conformal bounds.
- **`gateway/` (Bun/TypeScript LAN Gateway)**: Binds `0.0.0.0:3000` (configurable via `.env`), serves minified, sourcemap-free `frontend/` bundles, strictly blocks access to source files/datasets/sourcemaps, validates and reverse-proxies `/api/*` requests to `http://127.0.0.1:8000/*`, and automatically manages/spawns the local Python backend when `AUTO_SPAWN_BACKEND=true`.
- **`frontend/`**: Stateless client UI (`index.html`, `styles.css`, `app.ts`) communicating only through `/api/*` endpoints with single-order prediction, batch CSV prediction, and model retraining.

## Directory Layout

```text
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── data/
│   │   ├── shipdata.csv
│   │   ├── countries.json
│   │   ├── sap_codes.json
│   │   └── suppliers.json
│   └── artifacts/
│       └── metrics.json
├── gateway/
│   ├── server.ts
│   ├── build.ts
│   ├── package.json
│   ├── tsconfig.json
│   ├── bun.lock
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.ts
├── package.json
├── tsconfig.json
└── README.md
```

## API Endpoints

All external requests go through the gateway (`/api/*`):

- `GET /api/health` — Engine readiness and loaded stage status
- `GET /api/metadata` — Normalized `countries`, `suppliers`, `sap_codes`, and stage labels
- `GET /api/metrics` — Stage evaluation metrics and training metadata
- `POST /api/predict` — Single-order progressive forecast (Stages 0–3)
- `POST /api/batch-predict` — Batch forecast via JSON (`{ orders: [...] }`) or `multipart/form-data` CSV upload
- `POST /api/retrain` — Retrain all 4 stages from an uploaded `multipart/form-data` CSV dataset

## Local Development

```bash
# Bundle frontend assets (no sourcemaps) and start gateway + Python ML engine
npm run dev
```

Or run components separately:

```bash
# Terminal 1: Python ML Backend (127.0.0.1:8000)
python3 backend/main.py

# Terminal 2: Bun LAN Gateway (0.0.0.0:3000)
cd gateway
cp .env.example .env
# Set AUTO_SPAWN_BACKEND=false if running backend separately
bun run dev
```

## Compiling Standalone Source-Free Binaries (LAN Server Deployment)

### 1. Compile the Python ML Engine (`backend/`)

Optional: install FastAPI/Uvicorn and PyInstaller or Nuitka on your build host:

```bash
pip install -r backend/requirements.txt pyinstaller
```

**Option A — PyInstaller:**

```bash
pyinstaller --onefile --name leadtime-backend \
  --add-data "backend/data:data" \
  --add-data "backend/artifacts:artifacts" \
  backend/main.py
```

**Option B — Nuitka:**

```bash
python -m nuitka --onefile --output-filename=leadtime-backend \
  --include-data-dir=backend/data=data \
  --include-data-dir=backend/artifacts=artifacts \
  backend/main.py
```

(`backend/main.py` resolves `sys._MEIPASS`, `sys.frozen`, and relative paths automatically.)

### 2. Build the UI & Compile the Bun Gateway (`gateway/`)

```bash
cd gateway
bun run build.ts
bun build --compile --minify server.ts --outfile dist/leadtime-gateway
```

Because `build.ts` outputs minified assets with `sourcemap: "none"` and updates `embeddedAssets.ts`, the resulting `dist/leadtime-gateway` binary requires zero external `frontend/` files.

### 3. Deploy to Your LAN Server

Copy into a single server directory:

```text
deploy/
├── leadtime-gateway      (or leadtime-gateway.exe)
├── leadtime-backend      (or leadtime-backend.exe)
├── data/                 (writable folder for shipdata.csv & JSON registries)
├── artifacts/            (writable folder for metrics.json)
└── .env                  (HOST=0.0.0.0, PORT=3000, BACKEND_URL=http://127.0.0.1:8000)
```

Run `./leadtime-gateway` on the LAN host (port `3000` open on the host firewall; port `8000` closed externally). Client workstations simply open `http://<SERVER-LAN-IP>:3000` in their browser.
