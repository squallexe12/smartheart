/* =============================================================
   SmartHeart -- in-browser inference + UI controller
   Mirrors the Python preprocessing & scoring contract exactly.
   ============================================================= */

const STATE = {
  model: null,
  inputs: {},        // raw values keyed by feature name
  contribLabels: [], // human-readable label per encoded column index
  predictTimer: null,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/* ---------- bootstrap ---------- */

init().catch((err) => {
  console.error("[SmartHeart] init failed:", err);
  const card = $("#result-card");
  if (card) {
    card.innerHTML = `<div style="color:#FF8B8C;padding:18px;font-size:14px;line-height:1.5">
      Failed to load model. <code style="background:rgba(255,93,95,0.1);padding:2px 6px;border-radius:4px;font-family:'JetBrains Mono',monospace;font-size:12px">${escapeHtml(
        err.message
      )}</code>
    </div>`;
  }
});

async function init() {
  const res = await fetch("./public/model.json", { cache: "no-cache" });
  if (!res.ok) throw new Error(`model.json HTTP ${res.status}`);
  STATE.model = await res.json();

  // hydrate hero stat strip with live values from the artifact
  hydrateHeroStats();

  // build encoded-column labels (used by contribution display)
  STATE.contribLabels = buildEncodedLabels(STATE.model);

  // build inputs into the three group cards
  buildInputs();

  // initial predict
  runPredict(true);

  // global wiring
  $("#reset-btn")?.addEventListener("click", resetToDefaults);
}

/* ---------- helpers: labels & lookups ---------- */

function getFeature(name) {
  return STATE.model.features.find((f) => f.name === name);
}

function getOptionLabel(featName, value) {
  const f = getFeature(featName);
  if (!f || !Array.isArray(f.options)) return String(value);
  const o = f.options.find((opt) => String(opt.value) === String(value));
  return o ? o.label : String(value);
}

function buildEncodedLabels(model) {
  // Same ordering as encoded_feature_order; produce readable strings.
  return model.preprocessing.encoded_feature_order.map((col) => {
    if (model.preprocessing.numeric_features.includes(col)) {
      return { feature: col, kind: "numeric", label: getFeature(col).label };
    }
    if (model.preprocessing.binary_features.includes(col)) {
      return { feature: col, kind: "binary", label: getFeature(col).label };
    }
    // one-hot: ${feat}_${value}
    const idx = col.indexOf("_");
    const feat = col.slice(0, idx);
    const value = col.slice(idx + 1);
    const featDef = getFeature(feat);
    const opt = featDef.options.find((o) => String(o.value) === value);
    return {
      feature: feat,
      kind: "onehot",
      value,
      label: `${featDef.label}: ${opt ? opt.label : value}`,
    };
  });
}

/* ---------- hero stats ---------- */

function hydrateHeroStats() {
  const m = STATE.model.metrics;
  const set = (key, value) => {
    const el = document.querySelector(`[data-stat="${key}"]`);
    if (el) el.textContent = value;
  };
  set("acc", (m.test_accuracy * 100).toFixed(1));
  set("auc", m.test_auc.toFixed(3));
  set("f1",  m.test_f1.toFixed(3));
}

/* ---------- build inputs ---------- */

function buildInputs() {
  $$("[data-fields]").forEach((container) => {
    const names = container.dataset.fields.split(",").map((s) => s.trim());
    names.forEach((name) => {
      const f = getFeature(name);
      if (!f) return;
      const node = buildField(f);
      container.appendChild(node);
      STATE.inputs[name] = f.default;
    });
  });
}

function buildField(f) {
  const wrap = document.createElement("div");
  wrap.className = "field";
  wrap.dataset.feature = f.name;

  // is this a binary segmented control? (sex / fbs / exang have exactly 2 options)
  const isBinarySeg =
    f.kind === "categorical" &&
    Array.isArray(f.options) &&
    f.options.length === 2;

  // is this a numeric slider?
  const isNumeric = f.kind === "numeric";

  // label row
  const lab = document.createElement("div");
  lab.className = "field-label";
  const labText = document.createElement("span");
  labText.className = "lab";
  labText.textContent = f.label;
  lab.appendChild(labText);

  if (isNumeric) {
    const valChip = document.createElement("span");
    valChip.className = "val";
    valChip.dataset.role = "val";
    valChip.textContent = formatNumeric(f.default, f.step);
    lab.appendChild(valChip);
  }
  wrap.appendChild(lab);

  if (isNumeric) {
    const rwrap = document.createElement("div");
    rwrap.className = "range-wrap";

    const range = document.createElement("input");
    range.type = "range";
    range.className = "range";
    range.min = f.min;
    range.max = f.max;
    range.step = f.step;
    range.value = f.default;
    setRangeProgress(range);

    range.addEventListener("input", () => {
      const v = parseFloat(range.value);
      STATE.inputs[f.name] = v;
      $(`[data-role="val"]`, wrap).textContent = formatNumeric(v, f.step);
      setRangeProgress(range);
      schedulePredict();
    });

    rwrap.appendChild(range);

    const scale = document.createElement("div");
    scale.className = "range-scale";
    const left = document.createElement("span");
    left.textContent = formatNumeric(f.min, f.step);
    const right = document.createElement("span");
    right.textContent = formatNumeric(f.max, f.step);
    scale.appendChild(left);
    scale.appendChild(right);

    wrap.appendChild(rwrap);
    wrap.appendChild(scale);
  } else if (isBinarySeg) {
    const seg = document.createElement("div");
    seg.className = "segmented";
    f.options.forEach((opt, i) => {
      const id = `seg-${f.name}-${i}`;
      const input = document.createElement("input");
      input.type = "radio";
      input.name = `seg-${f.name}`;
      input.id = id;
      input.value = String(opt.value);
      input.checked = Number(opt.value) === Number(f.default);
      input.addEventListener("change", () => {
        STATE.inputs[f.name] = Number(input.value);
        schedulePredict();
      });

      const label = document.createElement("label");
      label.setAttribute("for", id);
      label.textContent = opt.label;

      seg.appendChild(input);
      seg.appendChild(label);
    });
    wrap.appendChild(seg);
  } else {
    // styled native select
    const sWrap = document.createElement("div");
    sWrap.className = "select-wrap";

    const sel = document.createElement("select");
    sel.className = "select";
    sel.setAttribute("aria-label", f.label);
    f.options.forEach((opt) => {
      const o = document.createElement("option");
      o.value = String(opt.value);
      o.textContent = opt.label;
      if (Number(opt.value) === Number(f.default)) o.selected = true;
      sel.appendChild(o);
    });
    sel.addEventListener("change", () => {
      STATE.inputs[f.name] = Number(sel.value);
      schedulePredict();
    });

    const chev = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    chev.setAttribute("viewBox", "0 0 16 16");
    chev.setAttribute("fill", "none");
    chev.setAttribute("stroke", "currentColor");
    chev.setAttribute("stroke-width", "1.6");
    chev.setAttribute("class", "select-chev");
    chev.innerHTML = `<path d="m3 6 5 5 5-5" stroke-linecap="round" stroke-linejoin="round"/>`;

    sWrap.appendChild(sel);
    sWrap.appendChild(chev);
    wrap.appendChild(sWrap);
  }

  return wrap;
}

function formatNumeric(v, step) {
  // Show decimals if step < 1 (e.g. oldpeak), else integer.
  if (step && step < 1) return Number(v).toFixed(1);
  return String(Math.round(Number(v)));
}

function setRangeProgress(range) {
  const min = parseFloat(range.min);
  const max = parseFloat(range.max);
  const val = parseFloat(range.value);
  const pct = ((val - min) / (max - min)) * 100;
  range.style.setProperty("--p", `${pct}%`);
}

/* ---------- reset ---------- */

function resetToDefaults() {
  STATE.model.features.forEach((f) => {
    STATE.inputs[f.name] = f.default;
  });

  // sync UI
  $$("[data-feature]").forEach((wrap) => {
    const name = wrap.dataset.feature;
    const f = getFeature(name);
    const range = $("input.range", wrap);
    if (range) {
      range.value = f.default;
      setRangeProgress(range);
      $(`[data-role="val"]`, wrap).textContent = formatNumeric(f.default, f.step);
    }
    const radios = $$(`input[type="radio"]`, wrap);
    if (radios.length) {
      radios.forEach((r) => (r.checked = Number(r.value) === Number(f.default)));
    }
    const sel = $("select.select", wrap);
    if (sel) sel.value = String(f.default);
  });

  runPredict(true);
}

/* ---------- prediction (debounced) ---------- */

function schedulePredict() {
  if (STATE.predictTimer) clearTimeout(STATE.predictTimer);
  STATE.predictTimer = setTimeout(() => runPredict(false), 150);
}

function runPredict(immediate) {
  const t0 = performance.now();
  const enc = encodePatient(STATE.inputs, STATE.model);
  const z = standardize(enc, STATE.model);
  const w = STATE.model.weights;
  const b = STATE.model.bias;

  let score = b;
  const contribs = new Array(z.length);
  for (let i = 0; i < z.length; i++) {
    contribs[i] = z[i] * w[i];
    score += contribs[i];
  }
  const proba = sigmoid(score);
  const t1 = performance.now();

  renderResult(proba, contribs, t1 - t0, immediate);
}

/* ---------- inference primitives (mirrors Python exactly) ---------- */

function encodePatient(inputs, model) {
  const order = model.preprocessing.encoded_feature_order;
  const cats = model.preprocessing.categorical_features;
  const numericNames = model.preprocessing.numeric_features;
  const binaryNames = model.preprocessing.binary_features;
  const vec = new Array(order.length);

  for (let i = 0; i < order.length; i++) {
    const col = order[i];

    if (numericNames.includes(col)) {
      vec[i] = Number(inputs[col]);
      continue;
    }
    if (binaryNames.includes(col)) {
      vec[i] = Number(inputs[col]);
      continue;
    }
    // one-hot: drop-first. column name is `${feat}_${value}`.
    const us = col.indexOf("_");
    const feat = col.slice(0, us);
    const value = col.slice(us + 1);
    const raw = inputs[feat];

    // categorical_features[feat] is the ordered list; first value is dropped.
    const allValues = cats[feat].map(String);
    // sanity: value present in remaining categories (not the first)
    void allValues;
    vec[i] = String(raw) === value ? 1 : 0;
  }

  return vec;
}

function standardize(vec, model) {
  // Only the first 6 columns (numeric) get standardized.
  const out = new Array(vec.length);
  const mean = model.preprocessing.numeric_mean;
  const std = model.preprocessing.numeric_std;
  const nNum = mean.length; // 6

  for (let i = 0; i < vec.length; i++) {
    if (i < nNum) {
      out[i] = (vec[i] - mean[i]) / std[i];
    } else {
      out[i] = vec[i];
    }
  }
  return out;
}

function sigmoid(z) {
  const clipped = Math.max(-500, Math.min(500, z));
  return 1 / (1 + Math.exp(-clipped));
}

/* ---------- render result ---------- */

function renderResult(proba, contribs, latencyMs, immediate) {
  const pct = proba * 100;
  const band = riskBand(pct);

  // gauge
  const gaugeVal = $("#gauge-value");
  if (gaugeVal) gaugeVal.textContent = pct.toFixed(1);
  const ring = $("#gauge-ring");
  if (ring) {
    const circumference = 2 * Math.PI * 68; // r = 68
    const offset = circumference * (1 - proba);
    ring.setAttribute("stroke-dasharray", String(circumference));
    ring.setAttribute("stroke-dashoffset", String(offset));
    ring.setAttribute("stroke", band.stroke);
    ring.style.filter = `drop-shadow(0 0 10px ${band.glow})`;
  }
  const sub = $("#gauge-band-mini");
  if (sub) sub.textContent = band.minor;

  // band pill
  const pill = $("#band-pill");
  if (pill) {
    pill.classList.remove("band-pill--low", "band-pill--mod", "band-pill--elev", "band-pill--high");
    pill.classList.add(band.cls);
  }
  const label = $("#band-label");
  if (label) label.textContent = band.label + " risk";
  const blurb = $("#band-blurb");
  if (blurb) blurb.textContent = band.blurb;

  // contributions: top 3 by absolute value
  const list = $("#contrib-list");
  if (list) {
    const ranked = contribs
      .map((v, i) => ({ v, i }))
      .filter((x) => STATE.contribLabels[x.i].kind !== "onehot" || x.v !== 0) // drop zero one-hots
      .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
      .slice(0, 3);

    const maxAbs = ranked.reduce((m, x) => Math.max(m, Math.abs(x.v)), 0) || 1;

    list.innerHTML = "";
    ranked.forEach(({ v, i }) => {
      const meta = STATE.contribLabels[i];
      const up = v > 0;
      const li = document.createElement("li");
      li.className = "contrib";

      const arrow = document.createElement("span");
      arrow.className = `contrib-arrow ${up ? "up" : "dn"}`;
      arrow.textContent = up ? "↑" : "↓";

      const txt = document.createElement("div");
      txt.className = "contrib-label";
      txt.textContent = meta.label;

      const val = document.createElement("span");
      val.className = "contrib-value font-mono";
      const sign = up ? "+" : "−";
      val.innerHTML = `<span class="${up ? "pos" : "neg"}">${sign}${Math.abs(v).toFixed(2)}</span>`;

      const bar = document.createElement("div");
      bar.className = "contrib-bar";
      const fill = document.createElement("div");
      fill.className = `contrib-bar-fill ${up ? "up" : "dn"}`;
      bar.appendChild(fill);

      li.appendChild(arrow);
      li.appendChild(txt);
      li.appendChild(val);
      li.appendChild(bar);
      list.appendChild(li);

      // animate fill on next frame
      requestAnimationFrame(() => {
        fill.style.width = `${(Math.abs(v) / maxAbs) * 100}%`;
      });
    });

    if (ranked.length === 0) {
      list.innerHTML = `<li class="text-[12.5px] text-text-lo">No active contributions yet.</li>`;
    }
  }

  // latency stamp
  const lat = $("#latency-stamp");
  if (lat) lat.textContent = `${latencyMs.toFixed(1)} ms`;

  // animate card lift (skip on first render)
  if (!immediate) {
    const card = $("#result-card");
    if (card) {
      card.classList.remove("updating");
      // force reflow then re-add
      void card.offsetWidth;
      card.classList.add("updating");
    }
  }
}

function riskBand(pct) {
  if (pct < 25) {
    return {
      cls: "band-pill--low",
      label: "Low",
      minor: "Low risk",
      blurb:
        "Model estimates a low probability of coronary artery disease for these inputs.",
      stroke: "#3DDC97",
      glow: "rgba(61,220,151,0.55)",
    };
  }
  if (pct < 50) {
    return {
      cls: "band-pill--mod",
      label: "Moderate",
      minor: "Moderate risk",
      blurb:
        "A non-trivial probability of coronary artery disease. Consider discussing with a clinician.",
      stroke: "#F4B860",
      glow: "rgba(244,184,96,0.55)",
    };
  }
  if (pct < 75) {
    return {
      cls: "band-pill--elev",
      label: "Elevated",
      minor: "Elevated risk",
      blurb:
        "The model leans toward disease. The signals listed below most influenced this estimate.",
      stroke: "#FF8A3D",
      glow: "rgba(255,138,61,0.55)",
    };
  }
  return {
    cls: "band-pill--high",
    label: "High",
    minor: "High risk",
    blurb:
      "The model strongly suggests coronary artery disease for these inputs. Educational estimate, not a diagnosis.",
    stroke: "#FF5D5F",
    glow: "rgba(255,93,95,0.55)",
  };
}

/* ---------- util ---------- */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[c]));
}
