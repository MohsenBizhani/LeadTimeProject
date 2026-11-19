import fs from "node:fs";
import path from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import {
  EMBEDDED_APP_JS,
  EMBEDDED_INDEX_HTML,
  EMBEDDED_STYLES_CSS,
} from "./embeddedAssets.ts";

interface GatewayConfig {
  host: string;
  port: number;
  backendUrl: string;
  autoSpawnBackend: boolean;
  proxyTimeoutMs: number;
  heavyProxyTimeoutMs: number;
}

const MAX_BODY_BYTES = 50 * 1024 * 1024;

function parseEnvFile(filePath: string): Record<string, string> {
  const out: Record<string, string> = {};
  try {
    if (!fs.existsSync(filePath)) return out;
    const raw = fs.readFileSync(filePath, "utf-8");
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eq = trimmed.indexOf("=");
      if (eq <= 0) continue;
      const key = trimmed.slice(0, eq).trim();
      let val = trimmed.slice(eq + 1).trim();
      if (
        (val.startsWith('"') && val.endsWith('"')) ||
        (val.startsWith("'") && val.endsWith("'"))
      ) {
        val = val.slice(1, -1);
      }
      out[key] = val;
    }
  } catch {
    // Ignore unreadable .env files
  }
  return out;
}

function loadConfig(): GatewayConfig {
  const execDir = path.dirname(process.execPath);
  const moduleDir = import.meta.dir;
  const rootDir = path.resolve(moduleDir, "..");

  const envFromFiles: Record<string, string> = {
    ...parseEnvFile(path.join(rootDir, ".env")),
    ...parseEnvFile(path.join(moduleDir, ".env")),
    ...parseEnvFile(path.join(execDir, ".env")),
  };

  const getSetting = (key: string): string | undefined =>
    process.env[key]?.trim() || envFromFiles[key]?.trim();

  // Default to port 3000; avoid unintended inherited container PORT=8080 unless explicitly set in .env or GATEWAY_PORT
  const explicitGatewayPort = getSetting("GATEWAY_PORT");
  const filePort = envFromFiles.PORT?.trim();
  const procPort = process.env.PORT?.trim();
  const rawPort =
    explicitGatewayPort ||
    filePort ||
    (procPort && procPort !== "8080" ? procPort : undefined) ||
    "3000";

  const port = Number.parseInt(rawPort, 10);
  const resolvedPort = Number.isFinite(port) && port > 0 && port < 65536 ? port : 3000;

  const host = getSetting("HOST") || "0.0.0.0";
  const backendUrl = (getSetting("BACKEND_URL") || "http://127.0.0.1:8000").replace(/\/+$/, "");
  const autoSpawnBackend = (getSetting("AUTO_SPAWN_BACKEND") || "true").toLowerCase() !== "false";

  const proxyTimeoutRaw = Number.parseInt(getSetting("PROXY_TIMEOUT_MS") || "15000", 10);
  const heavyTimeoutRaw = Number.parseInt(getSetting("HEAVY_PROXY_TIMEOUT_MS") || "120000", 10);

  return {
    host,
    port: resolvedPort,
    backendUrl,
    autoSpawnBackend,
    proxyTimeoutMs: Number.isFinite(proxyTimeoutRaw) && proxyTimeoutRaw > 0 ? proxyTimeoutRaw : 15000,
    heavyProxyTimeoutMs:
      Number.isFinite(heavyTimeoutRaw) && heavyTimeoutRaw > 0 ? heavyTimeoutRaw : 120000,
  };
}

interface StaticAsset {
  content: string;
  contentType: string;
  cacheControl: string;
}

function loadStaticAssets(): Map<string, StaticAsset> {
  const diskPublic = [
    path.join(import.meta.dir, "public"),
    path.join(path.dirname(process.execPath), "public"),
  ].find((dir) => fs.existsSync(path.join(dir, "index.html")));

  const readOrFallback = (relPath: string, fallback: string): string => {
    if (!diskPublic) return fallback;
    try {
      const full = path.join(diskPublic, relPath);
      if (fs.existsSync(full)) {
        return fs.readFileSync(full, "utf-8");
      }
    } catch {
      // Fallback to embedded string
    }
    return fallback;
  };

  const map = new Map<string, StaticAsset>();
  const indexHtml = readOrFallback("index.html", EMBEDDED_INDEX_HTML);
  const stylesCss = readOrFallback(path.join("assets", "styles.css"), EMBEDDED_STYLES_CSS);
  const appJs = readOrFallback(path.join("assets", "app.js"), EMBEDDED_APP_JS);

  map.set("/", {
    content: indexHtml,
    contentType: "text/html; charset=utf-8",
    cacheControl: "no-cache",
  });
  map.set("/index.html", {
    content: indexHtml,
    contentType: "text/html; charset=utf-8",
    cacheControl: "no-cache",
  });
  map.set("/assets/styles.css", {
    content: stylesCss,
    contentType: "text/css; charset=utf-8",
    cacheControl: "public, max-age=3600",
  });
  map.set("/assets/app.js", {
    content: appJs,
    contentType: "application/javascript; charset=utf-8",
    cacheControl: "public, max-age=3600",
  });

  return map;
}

const SECURITY_HEADERS: Record<string, string> = {
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "SAMEORIGIN",
  "Referrer-Policy": "no-referrer",
};

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...SECURITY_HEADERS,
    },
  });
}

const ALLOWED_API_ROUTES: Record<
  string,
  { methods: readonly string[]; upstream: string; heavy?: boolean }
> = {
  "/api/health": { methods: ["GET"], upstream: "/health" },
  "/api/metadata": { methods: ["GET"], upstream: "/metadata" },
  "/api/metrics": { methods: ["GET"], upstream: "/metrics" },
  "/api/countries": { methods: ["GET"], upstream: "/countries" },
  "/api/suppliers": { methods: ["GET"], upstream: "/suppliers" },
  "/api/sap-codes": { methods: ["GET"], upstream: "/sap-codes" },
  "/api/predict": { methods: ["POST"], upstream: "/predict" },
  "/api/batch-predict": { methods: ["POST"], upstream: "/batch-predict", heavy: true },
  "/api/retrain": { methods: ["POST"], upstream: "/retrain", heavy: true },
};

async function validateAndReadBody(
  req: Request,
  routePath: string
): Promise<{ ok: true; body: ArrayBuffer | undefined } | { ok: false; response: Response }> {
  if (req.method === "GET" || req.method === "HEAD") {
    return { ok: true, body: undefined };
  }

  const contentLengthHeader = req.headers.get("content-length");
  if (contentLengthHeader) {
    const cl = Number(contentLengthHeader);
    if (!Number.isFinite(cl) || cl < 0) {
      return { ok: false, response: jsonResponse({ detail: "Invalid Content-Length." }, 400) };
    }
    if (cl > MAX_BODY_BYTES) {
      return {
        ok: false,
        response: jsonResponse({ detail: "حجم درخواست بیش از ۵۰ مگابایت است." }, 413),
      };
    }
  }

  const ctype = (req.headers.get("content-type") || "").toLowerCase();
  if (routePath === "/api/predict" && !ctype.includes("application/json")) {
    return {
      ok: false,
      response: jsonResponse({ detail: "نوع محتوای درخواست باید application/json باشد." }, 415),
    };
  }
  if (routePath === "/api/retrain" && !ctype.includes("multipart/form-data")) {
    return {
      ok: false,
      response: jsonResponse({ detail: "نوع محتوای بازآموزی باید multipart/form-data باشد." }, 415),
    };
  }
  if (
    routePath === "/api/batch-predict" &&
    !ctype.includes("application/json") &&
    !ctype.includes("multipart/form-data")
  ) {
    return {
      ok: false,
      response: jsonResponse(
        { detail: "درخواست دسته‌ای باید application/json یا multipart/form-data باشد." },
        415
      ),
    };
  }

  const buf = await req.arrayBuffer();
  if (buf.byteLength > MAX_BODY_BYTES) {
    return {
      ok: false,
      response: jsonResponse({ detail: "حجم درخواست بیش از ۵۰ مگابایت است." }, 413),
    };
  }

  if (ctype.includes("application/json")) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(new TextDecoder().decode(buf));
    } catch {
      return { ok: false, response: jsonResponse({ detail: "فرمت JSON نامعتبر است." }, 400) };
    }

    if (routePath === "/api/predict") {
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return {
          ok: false,
          response: jsonResponse({ detail: "بدنه درخواست باید یک آبجکت JSON باشد." }, 400),
        };
      }
      const p = parsed as Record<string, unknown>;
      const w = Number(p.weight);
      if (
        !String(p.sap_code ?? "").trim() ||
        !String(p.country ?? "").trim() ||
        !String(p.supplier ?? "").trim() ||
        !String(p.registry_date ?? "").trim() ||
        !Number.isFinite(w) ||
        w <= 0
      ) {
        return {
          ok: false,
          response: jsonResponse(
            { detail: "کد SAP، کشور مبدأ، تأمین‌کننده، وزن مثبت و تاریخ ثبت الزامی هستند." },
            400
          ),
        };
      }
    } else if (routePath === "/api/batch-predict") {
      const orders = Array.isArray(parsed)
        ? parsed
        : parsed && typeof parsed === "object"
          ? (parsed as Record<string, unknown>).orders
          : null;
      if (!Array.isArray(orders) || orders.length === 0) {
        return {
          ok: false,
          response: jsonResponse({ detail: "لیست orders نامعتبر یا خالی است." }, 400),
        };
      }
    }
  }

  return { ok: true, body: buf };
}

async function proxyToBackend(
  req: Request,
  upstreamPath: string,
  config: GatewayConfig,
  heavy = false
): Promise<Response> {
  const url = new URL(req.url);
  const targetUrl = `${config.backendUrl}${upstreamPath}${url.search}`;
  const validated = await validateAndReadBody(req, url.pathname);
  if (!validated.ok) return validated.response;

  const headers = new Headers();
  const ctype = req.headers.get("content-type");
  if (ctype) headers.set("content-type", ctype);
  headers.set("accept", "application/json");
  headers.set("x-forwarded-for", req.headers.get("x-forwarded-for") || "lan-client");

  const timeoutMs = heavy ? config.heavyProxyTimeoutMs : config.proxyTimeoutMs;

  try {
    const upstreamRes = await fetch(targetUrl, {
      method: req.method,
      headers,
      body: validated.body,
      signal: AbortSignal.timeout(timeoutMs),
    });

    const resBody = await upstreamRes.arrayBuffer();
    return new Response(resBody, {
      status: upstreamRes.status,
      headers: {
        "Content-Type": upstreamRes.headers.get("content-type") || "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        ...SECURITY_HEADERS,
      },
    });
  } catch (err) {
    const isTimeout =
      err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError");
    if (isTimeout) {
      return jsonResponse(
        {
          detail: "زمان پاسخ‌گویی موتور تحلیل به پایان رسید.",
          code: "BACKEND_TIMEOUT",
        },
        504
      );
    }
    return jsonResponse(
      {
        detail: "موتور تحلیل پایتون در دسترس نیست یا در حال راه‌اندازی است.",
        code: "BACKEND_UNAVAILABLE",
      },
      503
    );
  }
}

async function probeBackend(
  backendUrl: string
): Promise<"ours" | "foreign" | "unreachable"> {
  try {
    const res = await fetch(`${backendUrl}/health`, {
      signal: AbortSignal.timeout(1200),
    });
    const text = await res.text();
    try {
      const json = JSON.parse(text) as { service?: string; models_loaded?: unknown };
      if (
        json.service === "leadtime-python-ml" ||
        typeof json.models_loaded === "boolean"
      ) {
        return "ours";
      }
    } catch {
      // Not JSON from our backend
    }
    return "foreign";
  } catch {
    return "unreachable";
  }
}

async function isPortFreeOnLoopback(port: number): Promise<boolean> {
  const net = await import("node:net");
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.unref();
    srv.once("error", () => resolve(false));
    srv.listen(port, "127.0.0.1", () => {
      srv.close(() => resolve(true));
    });
  });
}

async function resolveSpawnBackendPort(preferredUrl: string): Promise<string> {
  const parsed = new URL(preferredUrl);
  const preferredPort = Number.parseInt(parsed.port || "8000", 10);
  if (await isPortFreeOnLoopback(preferredPort)) {
    return preferredUrl;
  }
  for (let candidate = 8001; candidate <= 8020; candidate++) {
    if (await isPortFreeOnLoopback(candidate)) {
      return `http://127.0.0.1:${candidate}`;
    }
  }
  return preferredUrl;
}

function findBackendCommand(): { cmd: string; args: string[]; cwd: string } | null {
  const execDir = path.dirname(process.execPath);
  const gatewayDir = import.meta.dir;
  const rootDir = path.resolve(gatewayDir, "..");

  const binCandidates = [
    path.join(execDir, "leadtime-backend"),
    path.join(execDir, "leadtime-backend.exe"),
    path.join(rootDir, "backend", "dist", "leadtime-backend"),
    path.join(rootDir, "backend", "dist", "leadtime-backend.exe"),
  ];
  for (const bin of binCandidates) {
    if (fs.existsSync(bin)) {
      return { cmd: bin, args: [], cwd: path.dirname(bin) };
    }
  }

  const pyCandidates = [
    path.join(rootDir, "backend", "main.py"),
    path.join(execDir, "backend", "main.py"),
  ];
  for (const pyScript of pyCandidates) {
    if (fs.existsSync(pyScript)) {
      const pyBin = process.platform === "win32" ? "python" : "python3";
      return { cmd: pyBin, args: [pyScript], cwd: path.dirname(pyScript) };
    }
  }

  return null;
}

function spawnBackendIfNeeded(config: GatewayConfig): ChildProcess | null {
  if (!config.autoSpawnBackend) return null;
  const spec = findBackendCommand();
  if (!spec) {
    console.warn("[Gateway] Could not locate backend/main.py or compiled backend binary.");
    return null;
  }
  const backendPort = new URL(config.backendUrl).port || "8000";
  console.log(
    `[Gateway] Spawning Python ML backend on 127.0.0.1:${backendPort}: ${spec.cmd} ${spec.args.join(" ")}`
  );
  const child = spawn(spec.cmd, spec.args, {
    cwd: spec.cwd,
    env: {
      ...process.env,
      BACKEND_PORT: backendPort,
    },
    stdio: ["ignore", "inherit", "inherit"],
  });
  child.on("exit", (code, signal) => {
    if (code !== null && code !== 0) {
      console.error(`[Gateway] Backend exited with code ${code}`);
    } else if (signal) {
      console.log(`[Gateway] Backend terminated by signal ${signal}`);
    }
  });
  return child;
}

const config = loadConfig();
const staticAssets = loadStaticAssets();
let backendChild: ChildProcess | null = null;

const backendReadyInit = (async () => {
  const probe = await probeBackend(config.backendUrl);
  if (probe === "ours") {
    console.log(`[Gateway] Connected to existing Python ML backend at ${config.backendUrl}`);
    return;
  }
  if (probe === "foreign") {
    const fallbackUrl = await resolveSpawnBackendPort(config.backendUrl);
    console.log(
      `[Gateway] Port on ${config.backendUrl} is occupied by another service; routing internal ML engine to ${fallbackUrl}`
    );
    config.backendUrl = fallbackUrl;
  }
  backendChild = spawnBackendIfNeeded(config);
})();

const server = Bun.serve({
  hostname: config.host,
  port: config.port,
  maxRequestBodySize: MAX_BODY_BYTES,
  async fetch(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const pathname = url.pathname;

    // 1. Gateway + Backend health probe convenience endpoint
    if (pathname === "/health" && req.method === "GET") {
      await backendReadyInit;
      return proxyToBackend(req, "/health", config, false);
    }

    // 2. Strictly allowlisted /api/* reverse proxy to localhost-only Python ML engine
    if (pathname.startsWith("/api/") || pathname === "/api") {
      if (pathname === "/api") {
        return jsonResponse({
          service: "Lead Time LAN Gateway",
          version: "3.0.0",
          endpoints: Object.keys(ALLOWED_API_ROUTES),
        });
      }
      const route = ALLOWED_API_ROUTES[pathname];
      if (!route) {
        return jsonResponse({ detail: `Unknown API route: ${pathname}` }, 404);
      }
      if (!route.methods.includes(req.method)) {
        return jsonResponse({ detail: `Method ${req.method} not allowed` }, 405);
      }
      await backendReadyInit;
      return proxyToBackend(req, route.upstream, config, Boolean(route.heavy));
    }

    // 3. Static bundled & minified frontend assets only (no source, dataset, or sourcemap exposure)
    if (req.method === "GET" || req.method === "HEAD") {
      const asset = staticAssets.get(pathname);
      if (asset) {
        return new Response(req.method === "HEAD" ? null : asset.content, {
          status: 200,
          headers: {
            "Content-Type": asset.contentType,
            "Cache-Control": asset.cacheControl,
            ...SECURITY_HEADERS,
          },
        });
      }
    }

    return jsonResponse({ detail: "Not Found" }, 404);
  },
});

console.log(`[Gateway] LAN server listening on http://${config.host}:${server.port}`);

function shutdown(signal: string): void {
  console.log(`[Gateway] Received ${signal}, shutting down...`);
  if (backendChild && !backendChild.killed) {
    backendChild.kill("SIGTERM");
  }
  server.stop(true);
  process.exit(0);
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));
