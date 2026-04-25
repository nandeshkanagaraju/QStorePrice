// QStorePrice AI Dashboard — polls /admin/* endpoints and renders KPIs.

const POLL_MS = 2000;
const SCENARIOS = ["STABLE_WEEK", "BUSY_WEEKEND", "FARMER_WEEK", "TREND_WEEK", "CRISIS_WEEK"];

const $ = (id) => document.getElementById(id);

async function fetchJSON(url) {
    try {
        const r = await fetch(url, { headers: { "Accept": "application/json" } });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return await r.json();
    } catch (e) {
        return { error: e.message };
    }
}

function fmt(v, digits = 3) {
    if (v === null || v === undefined || Number.isNaN(v)) return "--";
    return Number(v).toFixed(digits);
}

function setStatus(ok, text) {
    const dot = $("status-dot");
    dot.classList.toggle("bad", !ok);
    $("status-text").textContent = text;
}

function renderKPIs(d) {
    if (!d || d.error || !d.summary) {
        setStatus(false, "Server unreachable");
        return;
    }
    setStatus(true, "Online");
    const s = d.summary;
    $("kpi-wrr").textContent = fmt(s.wrr_mean);
    $("kpi-wrr-max").textContent = fmt(s.wrr_max);
    $("kpi-quality").textContent = fmt(s.quality_mean);
    $("kpi-episodes").textContent = s.episodes_total ?? 0;
    $("kpi-steps").textContent = s.steps_total ?? 0;
    $("kpi-violations").textContent = s.violations_total ?? 0;
    const passPct = ((s.constitutional_pass_rate ?? 1) * 100).toFixed(0);
    $("kpi-const").textContent = `${passPct}%`;
    if (typeof d.uptime_seconds === "number") {
        $("uptime").textContent = `uptime ${formatUptime(d.uptime_seconds)}`;
    }
}

function formatUptime(sec) {
    sec = Math.floor(sec);
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

function renderScenarioBars(d) {
    const el = $("scenario-bars");
    if (!d || !d.by_scenario || Object.keys(d.by_scenario).length === 0) {
        el.innerHTML = `<p class="empty">No episodes recorded yet. Run a session to populate.</p>`;
        return;
    }
    const rows = SCENARIOS.map(name => {
        const bucket = d.by_scenario[name];
        const wrr = bucket ? bucket.wrr_mean : 0;
        const n = bucket ? bucket.n : 0;
        const pct = Math.max(0, Math.min(100, wrr * 100));
        const cls = wrr >= 0.7 ? "good" : (wrr >= 0.3 ? "" : "warn");
        return `
        <div class="bar-row">
            <div>${name}<span style="color:var(--text-muted);"> (${n})</span></div>
            <div class="bar-track"><div class="bar-fill ${cls}" style="width:${pct}%"></div></div>
            <div class="bar-value">${fmt(wrr)}</div>
        </div>`;
    }).join("");
    el.innerHTML = rows;
}

function renderEpisodeTable(d) {
    const tbody = $("episodes-body");
    const eps = (d && d.recent_episodes) || [];
    if (eps.length === 0) {
        tbody.innerHTML = `<tr><td colspan="10" class="empty">Awaiting episodes...</td></tr>`;
        return;
    }
    tbody.innerHTML = eps.slice().reverse().map(e => `
        <tr>
            <td>${e.scenario}</td>
            <td>${e.agent_type}</td>
            <td>${fmt(e.wrr)}</td>
            <td>${fmt(e.r1_pricing)}</td>
            <td>${fmt(e.r2_farmer)}</td>
            <td>${fmt(e.r3_trend)}</td>
            <td>${fmt(e.brief_quality_score)}</td>
            <td>${e.anti_hack_violations}</td>
            <td class="${e.constitutional_passed ? 'pass' : 'fail'}">${e.constitutional_passed ? 'PASS' : 'FAIL'}</td>
            <td>${e.steps}</td>
        </tr>
    `).join("");
}

function renderRewardCurve(d) {
    const canvas = $("reward-curve");
    if (!canvas.getContext) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.clientWidth || canvas.width;
    const H = canvas.height;
    canvas.width = W;
    ctx.clearRect(0, 0, W, H);

    const steps = (d && d.recent_steps) || [];
    if (steps.length < 2) {
        ctx.fillStyle = "rgba(138, 149, 168, 0.6)";
        ctx.font = "13px Outfit, sans-serif";
        ctx.fillText("No step rewards yet.", 12, H / 2);
        return;
    }

    const rewards = steps.map(s => s.reward);
    const minR = Math.min(...rewards, -0.05);
    const maxR = Math.max(...rewards, 0.05);
    const range = (maxR - minR) || 1;
    const padX = 18, padY = 16;
    const innerW = W - padX * 2;
    const innerH = H - padY * 2;

    // Zero baseline
    const zeroY = padY + innerH * (maxR / range);
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.beginPath();
    ctx.moveTo(padX, zeroY);
    ctx.lineTo(W - padX, zeroY);
    ctx.stroke();

    // Curve
    const grad = ctx.createLinearGradient(0, padY, 0, H - padY);
    grad.addColorStop(0, "rgba(92, 200, 255, 0.9)");
    grad.addColorStop(1, "rgba(155, 140, 255, 0.4)");
    ctx.strokeStyle = grad;
    ctx.lineWidth = 2;
    ctx.beginPath();
    rewards.forEach((r, i) => {
        const x = padX + (i / (rewards.length - 1)) * innerW;
        const y = padY + ((maxR - r) / range) * innerH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
}

function renderTasks(tasks) {
    const el = $("tasks");
    const items = (tasks && tasks.tasks) || [];
    if (items.length === 0) {
        el.innerHTML = SCENARIOS.map((s, i) =>
            `<div class="task-chip"><div class="level">Level ${i}</div><div class="name">${s}</div></div>`
        ).join("");
        return;
    }
    el.innerHTML = items.map(t =>
        `<div class="task-chip"><div class="level">Level ${t.level}</div><div class="name">${t.name}</div></div>`
    ).join("");
}

async function poll() {
    const [dash, tasks] = await Promise.all([
        fetchJSON("/admin/dashboard"),
        fetchJSON("/admin/tasks"),
    ]);
    renderKPIs(dash);
    renderScenarioBars(dash);
    renderEpisodeTable(dash);
    renderRewardCurve(dash);
    renderTasks(tasks);
}

document.addEventListener("DOMContentLoaded", () => {
    poll();
    setInterval(poll, POLL_MS);
});
