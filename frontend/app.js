const $ = (id) => document.getElementById(id);
const COLORS = ["#2a78b8", "#e8833a", "#3aa57a", "#b04fa0", "#8a8f98", "#c9a227"];
let chart = null;

function activateTab(name) {
  const target = document.getElementById(`panel-${name}`);
  if (!target) return;
  document.querySelectorAll(".tab").forEach((button) => {
    const active = button.dataset.tab === name;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    const active = panel.id === `panel-${name}`;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });
  history.replaceState(null, "", `#${name}`);
}

document.querySelectorAll(".tab").forEach((button, index, all) => {
  button.addEventListener("click", () => activateTab(button.dataset.tab));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowRight", "ArrowUp", "ArrowLeft"].includes(event.key)) return;
    event.preventDefault();
    const delta = ["ArrowDown", "ArrowRight"].includes(event.key) ? 1 : -1;
    const next = all[(index + delta + all.length) % all.length];
    next.focus();
    activateTab(next.dataset.tab);
  });
});
document.querySelectorAll("[data-open-tab]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    activateTab(link.dataset.openTab);
    document.querySelector(`#tab-${link.dataset.openTab}`).focus();
  });
});
const initialTab = location.hash.slice(1);
if (document.getElementById(`panel-${initialTab}`)) activateTab(initialTab);

function fmt(v, d = 3) { return v === null || v === undefined ? "-" : (typeof v === "number" ? Number(v.toFixed(d)).toString() : v); }
function table(el, head, rows, selectedIdx = -1) {
  el.innerHTML = "<thead><tr>" + head.map((h) => `<th>${h}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map((r, i) => `<tr class="${i === selectedIdx ? "sel" : ""}">` + r.map((c) => `<td>${c}</td>`).join("") + "</tr>").join("") + "</tbody>";
}
function setStatus(text, cls) { $("status").innerHTML = text ? `<div class="msg ${cls}">${text}</div>` : ""; }

async function runAnalysis() {
  const ks = [...document.querySelectorAll(".ks input:checked")].map((c) => c.value).join(",");
  const params = new URLSearchParams({ start: $("start").value, end: $("end").value, distance: $("distance").value, k: ks });
  $("run").disabled = true; setStatus("Running analysis… (the first request can take up to a minute while a free server wakes up)", "info");
  try {
    const res = await fetch(`${window.API_BASE}/api/analyze?${params}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    activateTab("results");
    render(data); setStatus("", "");
  } catch (e) {
    setStatus(`Error: ${e.message}. Is the backend running at ${window.API_BASE}?`, "err");
  } finally { $("run").disabled = false; }
}

function render(d) {
  $("results").hidden = false;
  $("empty-results").hidden = true;
  const sel = d.metrics.find((m) => m.selected && (m.method === "kmeans" || m.method === "kmedians"));
  if (!sel) throw new Error("No selected model was returned. Try another setting.");
  const obj = sel.method === "kmeans" ? ["WCSS", fmt(sel.wcss_euclidean, 1)] : ["L1 objective", fmt(sel.l1_objective, 1)];
  $("stats").innerHTML = [
    ["Profiles", d.profiles], ["Selected k", d.selected_k], ["Silhouette", fmt(sel.silhouette)],
    ["Davies-Bouldin", fmt(sel.davies_bouldin_euclidean)], [obj[0], obj[1]],
    ["Stability (ARI)", `${fmt(d.stability.adjusted_rand_mean)} ± ${fmt(d.stability.adjusted_rand_std)}`],
    ["PCA variance (2-D)", (d.pca_variance_2d * 100).toFixed(1) + "%"],
  ].map(([k, v]) => `<div class="stat"><b>${v}</b><span>${k}</span></div>`).join("");

  const clusters = [...new Set(d.points.map((p) => p.cluster))].sort();
  const datasets = clusters.map((c, i) => ({
    label: `Cluster ${c}`, backgroundColor: COLORS[i % COLORS.length] + "bb", pointRadius: 3.5,
    data: d.points.filter((p) => p.cluster === c).map((p) => ({ x: p.pca_1, y: p.pca_2, s: `${p.station_name} ${p.year}-${String(p.month).padStart(2, "0")}` })),
  }));
  if (chart) chart.destroy();
  if (window.Chart) {
    chart = new Chart($("pca"), { type: "scatter", data: { datasets }, options: {
      maintainAspectRatio: false, scales: { x: { title: { display: true, text: "PC1" }, grid: { color: "#edf1f2" } }, y: { title: { display: true, text: "PC2" }, grid: { color: "#edf1f2" } } },
      plugins: { legend: { position: "bottom", labels: { usePointStyle: true, boxWidth: 7, color: "#657983", font: { size: 10 } } }, tooltip: { callbacks: { label: (c) => `${c.raw.s}  (${c.raw.x.toFixed(2)}, ${c.raw.y.toFixed(2)})` } } } } });
  } else {
    const canvas = $("pca");
    const ctx = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
    ctx.scale(ratio, ratio); ctx.clearRect(0, 0, rect.width, rect.height);
    const points = datasets.flatMap((set, i) => set.data.map((p) => ({ ...p, color: COLORS[i % COLORS.length] })));
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    points.forEach((p) => { ctx.beginPath(); ctx.fillStyle = p.color; ctx.globalAlpha = .72; ctx.arc(24 + ((p.x - minX) / (maxX - minX || 1)) * (rect.width - 48), rect.height - 24 - ((p.y - minY) / (maxY - minY || 1)) * (rect.height - 48), 4, 0, 2 * Math.PI); ctx.fill(); });
    ctx.globalAlpha = 1;
  }

  const rows = d.metrics.map((m) => [m.method, m.k, fmt(m.wcss_euclidean, 1), fmt(m.l1_objective, 1), fmt(m.silhouette), fmt(m.davies_bouldin_euclidean),
    (m.min_cluster_fraction * 100).toFixed(1) + "%", fmt(m.stability_ari_mean), fmt(m.ari_vs_partition)]);
  table($("metrics"), ["Method", "k", "WCSS", "L1", "Silhouette", "Davies-Bouldin", "Min group", "Stability ARI", "ARI vs partition"], rows,
    d.metrics.findIndex((m) => m.selected && (m.method === "kmeans" || m.method === "kmedians")));
  table($("clusters"), ["Cluster", "Profiles", "Mean TMAX °C", "Mean TMIN °C", "Mean precip. mm", "Mean coverage"],
    d.clusters.map((c) => [c.cluster, c.profiles, c.tmax_c, c.tmin_c, c.precipitation_mm, c.coverage]));
  const sc = d.station_counts;
  table($("stations"), ["Station", ...sc.clusters.map((c) => `Cluster ${c}`)], sc.stations.map((s, i) => [s, ...sc.counts[i]]));
}
$("run").addEventListener("click", runAnalysis);
