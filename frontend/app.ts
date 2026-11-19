interface PredictResponse {
  stage: number;
  stage_name: string;
  stage_label_fa: string;
  elapsed_days: number;
  best_case_days: number;
  expected_days: number;
  worst_case_days: number;
  remaining_best_days: number;
  remaining_expected_days: number;
  remaining_worst_days: number;
  estimated_best_date: string;
  estimated_expected_date: string;
  estimated_worst_date: string;
  stage_test_mae?: number;
  supplier_order_count?: number;
  supplier_tier_name?: string;
  historical_mean?: number | string;
  recent_trend?: number | string;
}

interface MetadataResponse {
  countries: string[];
  suppliers: string[];
  sap_codes: string[];
  n_total?: number;
  models_loaded?: boolean;
}

interface HealthResponse {
  status: string;
  models_loaded: boolean;
  message?: string;
}

interface BatchResultRow {
  row: number;
  ok: boolean;
  error?: string;
  input?: {
    sap_code: string;
    country: string;
    supplier: string;
    weight: number;
    registry_date: string;
  };
  prediction?: PredictResponse;
}

interface BatchPredictResponse {
  count: number;
  succeeded: number;
  failed: number;
  results: BatchResultRow[];
}

const API_BASE = "/api";
let allSuppliers: string[] = [];
let lastBatchResults: BatchResultRow[] = [];

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing DOM element #${id}`);
  return el as T;
}

function getTodayIso(): string {
  const now = new Date();
  const y = now.getFullYear();
  const m = String(now.getMonth() + 1).padStart(2, "0");
  const d = String(now.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showAlert(msg: string, type: "error" | "warn" | "success" = "error"): void {
  const el = byId<HTMLDivElement>("globalAlert");
  if (!msg) {
    el.className = "alert-banner";
    el.textContent = "";
    return;
  }
  el.className = `alert-banner show ${type}`;
  el.textContent = msg;
}

function parseApiError(errBody: unknown, statusCode: number): string {
  if (!errBody || typeof errBody !== "object") return `HTTP ${statusCode}`;
  const obj = errBody as { detail?: unknown; message?: unknown };
  if (typeof obj.detail === "string") return obj.detail;
  if (typeof obj.message === "string") return obj.message;
  if (Array.isArray(obj.detail)) {
    return obj.detail
      .map((e: { loc?: unknown[]; msg?: string }) => {
        const field = Array.isArray(e.loc) && e.loc.length > 0 ? String(e.loc[e.loc.length - 1]) : "فیلد";
        return `${field}: ${e.msg ?? ""}`;
      })
      .join(" | ");
  }
  return `HTTP ${statusCode}`;
}

function formatDatePair(isoStr: string): string {
  if (!isoStr) return "-";
  const d = new Date(`${isoStr}T00:00:00`);
  if (Number.isNaN(d.getTime())) return isoStr;
  try {
    const fa = new Intl.DateTimeFormat("fa-IR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(d);
    return `${fa} (${isoStr})`;
  } catch {
    return isoStr;
  }
}

function syncMilestoneControls(): void {
  const regVal = byId<HTMLInputElement>("registry_date").value.trim();
  const minEl = byId<HTMLInputElement>("ministry_date");
  const minVal = minEl.value.trim();
  const shipEl = byId<HTMLInputElement>("ship_date");
  const shipWrap = byId<HTMLDivElement>("wrap_ship_date");
  const custEl = byId<HTMLInputElement>("customs_date");
  const custWrap = byId<HTMLDivElement>("wrap_customs_date");

  if (!minVal) {
    shipEl.value = "";
    shipEl.disabled = true;
    shipWrap.classList.add("disabled");
  } else {
    shipEl.disabled = false;
    shipWrap.classList.remove("disabled");
  }

  if (!shipEl.value.trim()) {
    custEl.value = "";
    custEl.disabled = true;
    custWrap.classList.add("disabled");
  } else {
    custEl.disabled = false;
    custWrap.classList.remove("disabled");
  }

  let activeStage = 0;
  const chip = byId<HTMLSpanElement>("liveStageChip");
  if (custEl.value.trim()) {
    activeStage = 3;
    chip.textContent = "مرحله فعلی فرم: پس از ورود به گمرک مقصد (مرحله ۳)";
  } else if (shipEl.value.trim()) {
    activeStage = 2;
    chip.textContent = "مرحله فعلی فرم: پس از حرکت کشتی از مبدأ (مرحله ۲)";
  } else if (minVal) {
    activeStage = 1;
    chip.textContent = "مرحله فعلی فرم: پس از مجوز وزارتخانه (مرحله ۱)";
  } else {
    activeStage = 0;
    chip.textContent = "مرحله فعلی فرم: لحظه ثبت سفارش (مرحله ۰)";
  }

  document.querySelectorAll("#stageStepper .step-pill").forEach((pill) => {
    const step = Number(pill.getAttribute("data-step"));
    pill.classList.remove("done", "current");
    if (step < activeStage || (step === 0 && regVal && activeStage > 0)) {
      pill.classList.add("done");
    } else if (step === activeStage) {
      pill.classList.add("current");
    }
  });
}

function initCustomDatePicker(): void {
  const popover = byId<HTMLDivElement>("datePickerPopover");
  const monthSel = byId<HTMLSelectElement>("dpMonthSelect");
  const yearSel = byId<HTMLSelectElement>("dpYearSelect");
  const daysGrid = byId<HTMLDivElement>("dpDaysGrid");
  const prevBtn = byId<HTMLButtonElement>("dpPrevMonth");
  const nextBtn = byId<HTMLButtonElement>("dpNextMonth");
  const todayBtn = byId<HTMLButtonElement>("dpTodayBtn");
  const closeBtn = byId<HTMLButtonElement>("dpCloseBtn");

  const MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];

  MONTH_NAMES.forEach((mName, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = mName;
    monthSel.appendChild(opt);
  });

  const curYear = new Date().getFullYear();
  for (let y = 2018; y <= curYear + 3; y++) {
    const opt = document.createElement("option");
    opt.value = String(y);
    opt.textContent = String(y);
    yearSel.appendChild(opt);
  }

  let activeInput: HTMLInputElement | null = null;
  let viewYear = curYear;
  let viewMonth = new Date().getMonth();

  function parseIso(val: string): Date | null {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec((val || "").trim());
    if (!m) return null;
    const y = Number(m[1]);
    const mo = Number(m[2]) - 1;
    const d = Number(m[3]);
    const dt = new Date(y, mo, d);
    if (dt.getFullYear() !== y || dt.getMonth() !== mo || dt.getDate() !== d) return null;
    return dt;
  }

  function closePicker(): void {
    popover.classList.remove("open");
    activeInput = null;
  }

  function selectDate(y: number, m: number, d: number): void {
    if (!activeInput) return;
    const iso = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
    activeInput.value = iso;
    syncMilestoneControls();
    closePicker();
  }

  function renderCalendar(): void {
    monthSel.value = String(viewMonth);
    yearSel.value = String(viewYear);
    daysGrid.innerHTML = "";

    const firstDay = new Date(viewYear, viewMonth, 1).getDay();
    const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
    const prevMonthDays = new Date(viewYear, viewMonth, 0).getDate();
    const todayIso = getTodayIso();
    const selectedVal = activeInput ? activeInput.value.trim() : "";

    for (let i = firstDay - 1; i >= 0; i--) {
      const dayNum = prevMonthDays - i;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "dp-day muted";
      btn.textContent = String(dayNum);
      btn.addEventListener("click", () => {
        const prevMo = viewMonth === 0 ? 11 : viewMonth - 1;
        const prevYr = viewMonth === 0 ? viewYear - 1 : viewYear;
        selectDate(prevYr, prevMo, dayNum);
      });
      daysGrid.appendChild(btn);
    }

    for (let d = 1; d <= daysInMonth; d++) {
      const iso = `${viewYear}-${String(viewMonth + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "dp-day";
      if (iso === todayIso) btn.classList.add("today");
      if (iso === selectedVal) btn.classList.add("selected");
      btn.textContent = String(d);
      btn.addEventListener("click", () => selectDate(viewYear, viewMonth, d));
      daysGrid.appendChild(btn);
    }

    const totalCells = firstDay + daysInMonth;
    const trailing = totalCells % 7 === 0 ? 0 : 7 - (totalCells % 7);
    for (let d = 1; d <= trailing; d++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "dp-day muted";
      btn.textContent = String(d);
      btn.addEventListener("click", () => {
        const nextMo = viewMonth === 11 ? 0 : viewMonth + 1;
        const nextYr = viewMonth === 11 ? viewYear + 1 : viewYear;
        selectDate(nextYr, nextMo, d);
      });
      daysGrid.appendChild(btn);
    }
  }

  function positionPopover(anchorEl: Element): void {
    const rect = anchorEl.getBoundingClientRect();
    const popW = 300;
    const popH = popover.offsetHeight || 310;
    let left = Math.min(Math.max(8, rect.left), window.innerWidth - popW - 8);
    let top = rect.bottom + 8;
    if (top + popH > window.innerHeight - 8 && rect.top - popH - 8 > 8) {
      top = rect.top - popH - 8;
    }
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
  }

  function openPickerFor(inputEl: HTMLInputElement | null): void {
    if (!inputEl || inputEl.disabled) return;
    activeInput = inputEl;
    const parsed = parseIso(inputEl.value);
    if (parsed) {
      viewYear = parsed.getFullYear();
      viewMonth = parsed.getMonth();
    } else {
      const order = ["registry_date", "ministry_date", "ship_date", "customs_date"];
      const idx = order.indexOf(inputEl.id);
      let refDate: Date | null = null;
      for (let i = idx - 1; i >= 0; i--) {
        refDate = parseIso(byId<HTMLInputElement>(order[i]).value);
        if (refDate) break;
      }
      const base = refDate || new Date();
      viewYear = base.getFullYear();
      viewMonth = base.getMonth();
    }
    renderCalendar();
    popover.classList.add("open");
    positionPopover(inputEl.closest(".input-with-clear") || inputEl);
  }

  prevBtn.addEventListener("click", () => {
    if (viewMonth === 0) {
      viewMonth = 11;
      viewYear--;
    } else {
      viewMonth--;
    }
    renderCalendar();
  });

  nextBtn.addEventListener("click", () => {
    if (viewMonth === 11) {
      viewMonth = 0;
      viewYear++;
    } else {
      viewMonth++;
    }
    renderCalendar();
  });

  monthSel.addEventListener("change", () => {
    viewMonth = Number(monthSel.value);
    renderCalendar();
  });

  yearSel.addEventListener("change", () => {
    viewYear = Number(yearSel.value);
    renderCalendar();
  });

  todayBtn.addEventListener("click", () => {
    const now = new Date();
    selectDate(now.getFullYear(), now.getMonth(), now.getDate());
  });

  closeBtn.addEventListener("click", closePicker);

  document.querySelectorAll<HTMLInputElement>(".date-input").forEach((input) => {
    input.addEventListener("click", (e) => {
      e.stopPropagation();
      openPickerFor(input);
    });
    input.addEventListener("focus", () => openPickerFor(input));
  });

  document.querySelectorAll<HTMLButtonElement>(".cal-trigger-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = btn.getAttribute("data-cal-for");
      if (!id) return;
      const input = document.getElementById(id) as HTMLInputElement | null;
      if (input && !input.disabled) {
        input.focus();
        openPickerFor(input);
      }
    });
  });

  document.addEventListener("click", (e) => {
    if (!popover.classList.contains("open")) return;
    const target = e.target as Node;
    if (popover.contains(target)) return;
    if (activeInput && activeInput.closest(".input-with-clear")?.contains(target)) return;
    closePicker();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && popover.classList.contains("open")) {
      closePicker();
    }
  });

  window.addEventListener("resize", () => {
    if (popover.classList.contains("open") && activeInput) {
      positionPopover(activeInput.closest(".input-with-clear") || activeInput);
    }
  });
}

async function loadMetadata(): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/metadata`);
    if (!res.ok) return;
    const data = (await res.json()) as MetadataResponse;
    const countries = Array.isArray(data.countries) ? data.countries : [];
    const suppliers = Array.isArray(data.suppliers) ? data.suppliers : [];
    const sapCodes = Array.isArray(data.sap_codes) ? data.sap_codes : [];

    byId<HTMLDataListElement>("countryList").innerHTML = countries
      .map((c) => `<option value="${escapeHtml(c)}">`)
      .join("");
    byId<HTMLDataListElement>("sapList").innerHTML = sapCodes
      .map((x) => `<option value="${escapeHtml(x)}">`)
      .join("");

    allSuppliers = suppliers;
    const supList = byId<HTMLDataListElement>("supplierList");
    if (allSuppliers.length <= 250) {
      supList.innerHTML = allSuppliers
        .map((s) => `<option value="${escapeHtml(s)}">`)
        .join("");
    }
  } catch {
    // Non-fatal while backend is warming up
  }
}

let metadataLoaded = false;

async function checkHealth(): Promise<void> {
  const dot = byId<HTMLSpanElement>("serverDot");
  const text = byId<HTMLSpanElement>("serverStatusText");
  try {
    const hRes = await fetch(`${API_BASE}/health`);
    const h = (await hRes.json()) as HealthResponse;
    if (!hRes.ok || !h.models_loaded) {
      dot.className = "status-dot";
      text.textContent = "سرویس در حال آماده‌سازی مدل‌ها است...";
      window.setTimeout(() => void checkHealth(), 2500);
      return;
    }
    dot.className = "status-dot ok";
    text.textContent = "متصل به سرویس پیش‌بینی هوشمند";
    if (!metadataLoaded) {
      metadataLoaded = true;
      await loadMetadata();
    }
  } catch {
    dot.className = "status-dot err";
    text.textContent = "در انتظار راه‌اندازی موتور تحلیل...";
    window.setTimeout(() => void checkHealth(), 2500);
  }
}

function renderSingleResult(data: PredictResponse): void {
  byId("resBest").innerHTML = `${data.best_case_days}<small>روز</small>`;
  byId("resExpected").innerHTML = `${data.expected_days}<small>روز</small>`;
  byId("resWorst").innerHTML = `${data.worst_case_days}<small>روز</small>`;

  byId("resBestDate").textContent = formatDatePair(data.estimated_best_date);
  byId("resExpectedDate").textContent = formatDatePair(data.estimated_expected_date);
  byId("resWorstDate").textContent = formatDatePair(data.estimated_worst_date);

  byId("remBest").innerHTML = `${data.remaining_best_days}<small>روز</small>`;
  byId("remExpected").innerHTML = `${data.remaining_expected_days}<small>روز</small>`;
  byId("remWorst").innerHTML = `${data.remaining_worst_days}<small>روز</small>`;

  byId("stageBannerText").textContent = `مرحله محاسبه: ${data.stage_label_fa} (مرحله ${data.stage})`;
  byId("stageBannerSub").textContent = `زمان سپری‌شده: ${data.elapsed_days} روز`;

  const scaleMax = Math.max(data.worst_case_days * 1.1, 1);
  const pctElapsed = Math.min(100, (data.elapsed_days / scaleMax) * 100);
  const pctBest = Math.min(100, (data.best_case_days / scaleMax) * 100);
  const pctExp = Math.min(100, (data.expected_days / scaleMax) * 100);
  const pctWorst = Math.min(100, (data.worst_case_days / scaleMax) * 100);

  byId("barElapsed").style.width = `${pctElapsed}%`;
  byId("barInterval").style.right = `${pctBest}%`;
  byId("barInterval").style.width = `${Math.max(2, pctWorst - pctBest)}%`;
  byId("barExpected").style.right = `calc(${pctExp}% - 2px)`;
  byId("rangeSummaryLabel").textContent =
    `بازه محتمل: ${data.best_case_days} تا ${data.worst_case_days} روز (محتمل‌ترین: ${data.expected_days} روز)`;

  byId("intelTier").textContent = data.supplier_tier_name || "-";
  byId("intelCount").textContent =
    data.supplier_order_count != null ? `${data.supplier_order_count} سفارش` : "-";
  byId("intelMean").textContent =
    typeof data.historical_mean === "number"
      ? `${data.historical_mean} روز`
      : data.historical_mean || "-";
  byId("intelTrend").textContent =
    typeof data.recent_trend === "number"
      ? `${data.recent_trend} روز`
      : data.recent_trend || "-";
  byId("intelStageAcc").textContent =
    data.stage_test_mae != null ? `${data.stage_test_mae} روز` : "-";

  byId("resultsPanel").classList.add("show");
}

function renderBatchResults(data: BatchPredictResponse): void {
  lastBatchResults = data.results || [];
  const wrap = byId<HTMLDivElement>("batchTableWrap");
  const tbody = byId<HTMLTableSectionElement>("batchTableBody");
  const dlBtn = byId<HTMLButtonElement>("batchDownloadBtn");

  if (!lastBatchResults.length) {
    wrap.classList.add("hidden");
    dlBtn.classList.add("hidden");
    return;
  }

  tbody.innerHTML = lastBatchResults
    .slice(0, 100)
    .map((item) => {
      if (!item.ok || !item.prediction) {
        return `<tr>
          <td>${item.row}</td>
          <td colspan="7" class="batch-err">${escapeHtml(item.error || "خطا در ردیف")}</td>
        </tr>`;
      }
      const p = item.prediction;
      const inp = item.input;
      return `<tr>
        <td>${item.row}</td>
        <td dir="ltr">${escapeHtml(inp?.sap_code ?? "-")}</td>
        <td dir="ltr">${escapeHtml(inp?.supplier ?? "-")}</td>
        <td>مرحله ${p.stage}</td>
        <td>${p.best_case_days} روز</td>
        <td><strong>${p.expected_days} روز</strong></td>
        <td>${p.worst_case_days} روز</td>
        <td dir="ltr">${escapeHtml(p.estimated_expected_date)}</td>
      </tr>`;
    })
    .join("");

  wrap.classList.remove("hidden");
  dlBtn.classList.remove("hidden");
}

function downloadBatchCsv(): void {
  if (!lastBatchResults.length) return;
  const headers = [
    "row", "status", "sap_code", "country", "supplier", "weight", "registry_date",
    "stage", "elapsed_days", "best_case_days", "expected_days", "worst_case_days",
    "estimated_best_date", "estimated_expected_date", "estimated_worst_date", "error",
  ];
  const csvCell = (val: unknown): string => {
    const s = String(val ?? "");
    return `"${s.replace(/"/g, '""')}"`;
  };
  const lines = [headers.join(",")];
  for (const r of lastBatchResults) {
    const p = r.prediction;
    const i = r.input;
    lines.push(
      [
        r.row,
        r.ok ? "ok" : "error",
        i?.sap_code ?? "",
        i?.country ?? "",
        i?.supplier ?? "",
        i?.weight ?? "",
        i?.registry_date ?? "",
        p?.stage ?? "",
        p?.elapsed_days ?? "",
        p?.best_case_days ?? "",
        p?.expected_days ?? "",
        p?.worst_case_days ?? "",
        p?.estimated_best_date ?? "",
        p?.estimated_expected_date ?? "",
        p?.estimated_worst_date ?? "",
        r.error ?? "",
      ]
        .map(csvCell)
        .join(",")
    );
  }
  const blob = new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `leadtime-predictions-${getTodayIso()}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function bootstrap(): void {
  byId("year").textContent = String(new Date().getFullYear());
  byId<HTMLInputElement>("registry_date").value = getTodayIso();

  ["registry_date", "ministry_date", "ship_date", "customs_date"].forEach((id) => {
    const el = byId<HTMLInputElement>(id);
    el.addEventListener("change", syncMilestoneControls);
    el.addEventListener("input", syncMilestoneControls);
  });

  byId<HTMLInputElement>("supplier").addEventListener("input", (e) => {
    if (!allSuppliers.length || allSuppliers.length <= 250) return;
    const query = (e.target as HTMLInputElement).value.trim().toUpperCase();
    const supList = byId<HTMLDataListElement>("supplierList");
    if (query.length >= 1) {
      const matches = allSuppliers.filter((s) => s.includes(query)).slice(0, 25);
      supList.innerHTML = matches.map((s) => `<option value="${escapeHtml(s)}">`).join("");
    } else {
      supList.innerHTML = "";
    }
  });

  document.querySelectorAll<HTMLButtonElement>(".clear-field-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const field = btn.getAttribute("data-field");
      if (!field) return;
      const el = document.getElementById(field) as HTMLInputElement | null;
      if (el && !el.disabled) el.value = "";
      syncMilestoneControls();
    });
  });

  byId("clearAllBtn").addEventListener("click", () => {
    showAlert("");
    ["sap_code", "country", "supplier", "weight", "ministry_date", "ship_date", "customs_date"].forEach(
      (id) => {
        byId<HTMLInputElement>(id).value = "";
      }
    );
    byId<HTMLInputElement>("registry_date").value = getTodayIso();
    syncMilestoneControls();
    byId("resultsPanel").classList.remove("show");
  });

  byId("predictBtn").addEventListener("click", async () => {
    showAlert("");
    const sap_code = byId<HTMLInputElement>("sap_code").value.trim();
    const country = byId<HTMLInputElement>("country").value.trim();
    const supplier = byId<HTMLInputElement>("supplier").value.trim();
    const weight = parseFloat(byId<HTMLInputElement>("weight").value);
    const registry_date = byId<HTMLInputElement>("registry_date").value.trim();
    const ministry_date = byId<HTMLInputElement>("ministry_date").value.trim() || null;
    const ship_date = byId<HTMLInputElement>("ship_date").value.trim() || null;
    const customs_date = byId<HTMLInputElement>("customs_date").value.trim() || null;

    if (!sap_code || !country || !supplier) {
      showAlert("لطفاً کد کالا (SAP)، کشور مبدأ و نام تأمین‌کننده را وارد کنید.");
      return;
    }
    if (!Number.isFinite(weight) || weight <= 0) {
      showAlert("لطفاً وزن محموله را به‌صورت عددی بزرگتر از صفر وارد کنید.");
      return;
    }
    if (!registry_date) {
      showAlert("لطفاً تاریخ ثبت سیستم را انتخاب کنید.");
      return;
    }

    const isoPattern = /^\d{4}-\d{2}-\d{2}$/;
    const fields: Array<[string, string | null]> = [
      ["تاریخ ثبت سیستم", registry_date],
      ["تاریخ مجوز وزارتخانه", ministry_date],
      ["تاریخ حرکت کشتی", ship_date],
      ["تاریخ ورود به گمرک", customs_date],
    ];
    for (const [label, val] of fields) {
      if (val && !isoPattern.test(val)) {
        showAlert(`فرمت ${label} باید به‌صورت YYYY-MM-DD باشد (مثلاً 2024-06-15).`);
        return;
      }
    }

    const activeDates = fields.filter((x): x is [string, string] => Boolean(x[1]));
    for (let i = 1; i < activeDates.length; i++) {
      if (activeDates[i][1] < activeDates[i - 1][1]) {
        showAlert(`${activeDates[i][0]} نمی‌تواند قبل از ${activeDates[i - 1][0]} باشد.`);
        return;
      }
    }

    const btn = byId<HTMLButtonElement>("predictBtn");
    btn.disabled = true;
    btn.textContent = "در حال محاسبه...";

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sap_code,
          country,
          supplier,
          weight,
          registry_date,
          ministry_date,
          ship_date,
          customs_date,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(parseApiError(data, response.status));
      }
      renderSingleResult(data as PredictResponse);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      showAlert(`خطا در پیش‌بینی: ${msg}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "محاسبه پیش‌بینی و بازه خطا";
    }
  });

  byId("batchPredictBtn").addEventListener("click", async () => {
    const fileInput = byId<HTMLInputElement>("batchFile");
    const status = byId<HTMLDivElement>("batchStatus");
    const btn = byId<HTMLButtonElement>("batchPredictBtn");
    if (!fileInput.files || !fileInput.files.length) {
      showAlert("لطفاً ابتدا یک فایل CSV برای پیش‌بینی دسته‌ای انتخاب کنید.");
      return;
    }
    showAlert("");
    const form = new FormData();
    form.append("file", fileInput.files[0]);

    btn.disabled = true;
    btn.textContent = "در حال پردازش...";
    status.textContent = "در حال ارسال و محاسبه پیش‌بینی دسته‌ای...";

    try {
      const res = await fetch(`${API_BASE}/batch-predict`, { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(parseApiError(data, res.status));
      const parsed = data as BatchPredictResponse;
      status.textContent = `تعداد کل: ${parsed.count} | موفق: ${parsed.succeeded} | خطادار: ${parsed.failed}`;
      renderBatchResults(parsed);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      status.textContent = `خطا: ${msg}`;
      showAlert(`خطا در پیش‌بینی دسته‌ای: ${msg}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "اجرای پیش‌بینی دسته‌ای";
    }
  });

  byId("batchDownloadBtn").addEventListener("click", downloadBatchCsv);

  byId("retrainBtn").addEventListener("click", async () => {
    const fileInput = byId<HTMLInputElement>("datasetFile");
    const status = byId<HTMLDivElement>("retrainStatus");
    const btn = byId<HTMLButtonElement>("retrainBtn");
    if (!fileInput.files || !fileInput.files.length) {
      showAlert("لطفاً ابتدا یک فایل CSV انتخاب کنید.");
      return;
    }
    showAlert("");
    const form = new FormData();
    form.append("file", fileInput.files[0]);

    btn.disabled = true;
    btn.textContent = "در حال بازآموزی...";
    status.textContent = "در حال اجرای بازآموزی چندمرحله‌ای و کالیبراسیون...";

    try {
      const res = await fetch(`${API_BASE}/retrain`, { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(parseApiError(data, res.status));

      const typed = data as {
        n_total?: number;
        stages?: Record<string, { test?: { mae?: number } }>;
      };
      const s0Mae = typed.stages?.["0"]?.test?.mae;
      status.textContent =
        s0Mae != null
          ? `بازآموزی با موفقیت انجام شد. تعداد کل: ${typed.n_total} | خطای میانگین (MAE) مرحله ۰: ${s0Mae} روز`
          : `بازآموزی با موفقیت انجام شد. تعداد کل: ${typed.n_total}`;
      showAlert("مدل‌های جدید با منطق زمان باقی‌مانده و کالیبراسیون جدید با موفقیت جایگزین شدند.", "success");
      await Promise.all([loadMetadata(), checkHealth()]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      status.textContent = `خطا: ${msg}`;
      showAlert(`خطا در بازآموزی: ${msg}`, "error");
    } finally {
      btn.disabled = false;
      btn.textContent = "بازآموزی مدل";
    }
  });

  initCustomDatePicker();
  syncMilestoneControls();
  void loadMetadata();
  void checkHealth();
}

bootstrap();
