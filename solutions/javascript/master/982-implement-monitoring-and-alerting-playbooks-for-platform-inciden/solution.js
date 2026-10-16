/** Task 982 - Implement monitoring and alerting playbooks for platform incidents.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Services publish one minute of metrics at a time (request count, error count, latency samples).
 * The evaluator runs four checks every minute: a multi-window SLO burn rate (both the 10-minute and
 * the 5-minute error rate must exceed 14.4x the budget), a p99 latency guardrail, and a no-data
 * watchdog for services that stopped reporting. Firing alerts are de-duplicated per rule and
 * service, opened once, auto-resolved when the condition clears, and each one drives a playbook
 * whose steps are audited - the auto-remediation step can be disabled by policy, and a failing step
 * escalates instead of continuing.
 * Simplified: metrics are generated from a seeded incident script (no Prometheus) and the playbook
 * "execution" only records step outcomes. Verified: the alarm minute, the burn rate and p99 against
 * direct recomputation, de-duplication and resolution, playbook order/gating/escalation, the
 * no-data boundary and determinism.
 */
"use strict";
const assert = require("node:assert");

const SLO = { target: 0.999, longWindow: 10, shortWindow: 5, burnThreshold: 14.4 };
const LATENCY_BUDGET_MS = 800;
const INCIDENT = { start: 30, end: 44 };
const SERVICES = ["api", "worker", "silent"];
const PLAYBOOKS = {
  "burn-rate": [
    { step: "acknowledge", mode: "auto" },
    { step: "inspect-recent-deploys", mode: "auto" },
    { step: "rollback-last-deploy", mode: "autoRemediation" },
    { step: "page-oncall", mode: "escalate" },
  ],
  "latency-p99": [
    { step: "acknowledge", mode: "auto" },
    { step: "check-saturation", mode: "auto" },
    { step: "scale-out", mode: "autoRemediation" },
  ],
  "no-data": [{ step: "notify-data-owner", mode: "escalate" }],
};
const SEVERITY_RANK = { critical: 0, warning: 1 };

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, unit: () => Number(next() >> 11n) / 2 ** 53, int: (n) => Number(next() % BigInt(n)) };
}

/** Metric stream per service; `api` degrades between INCIDENT.start and INCIDENT.end (inclusive). */
function makeSeries(seed, { minutes = 70, services = SERVICES, incident = INCIDENT } = {}) {
  const rng = splitmix64(seed);
  const series = {};
  for (const service of services) {
    if (service === "silent") { series[service] = []; continue; }
    series[service] = Array.from({ length: minutes }, (_, minute) => {
      const broken = service === "api" && minute >= incident.start && minute <= incident.end;
      const errorRate = service === "worker" ? 0.0005 : broken ? 0.2 : 0.0005;
      const base = service === "worker" ? 90 : broken ? 950 : 120;
      return { minute, requests: 200, errors: Math.round(200 * errorRate * (0.8 + 0.4 * rng.unit())),
        latencies: Array.from({ length: 20 }, () => Math.round(base * (0.7 + 0.6 * rng.unit()))) };
    });
  }
  return series;
}

const percentile = (values, fraction) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length === 0 ? 0 : sorted[Math.max(0, Math.ceil(fraction * sorted.length) - 1)];
};

/** Aggregate one trailing window; `null` when the service published nothing in it. */
function windowStats(samples, minute, width) {
  const rows = samples.filter((row) => row.minute > minute - width && row.minute <= minute);
  if (rows.length === 0) return null;
  const requests = rows.reduce((sum, row) => sum + row.requests, 0);
  const errors = rows.reduce((sum, row) => sum + row.errors, 0);
  const latencies = rows.flatMap((row) => row.latencies);
  return { requests, errors, errorRate: requests === 0 ? 1 : errors / requests, p99: percentile(latencies, 0.99) };
}

/** Every rule evaluated at one minute; the burn rate needs both windows over the threshold. */
function evaluate(series, minute) {
  const alerts = [];
  for (const [service, samples] of Object.entries(series)) {
    const long = windowStats(samples, minute, SLO.longWindow);
    const short = windowStats(samples, minute, SLO.shortWindow);
    if (long === null || short === null || short.requests === 0) {
      alerts.push({ rule: "no-data", service, severity: "warning", value: 0, threshold: 0 });
      continue;
    }
    const budget = 1 - SLO.target;
    const burnLong = long.errorRate / budget;
    const burnShort = short.errorRate / budget;
    if (burnLong > SLO.burnThreshold && burnShort > SLO.burnThreshold) {
      alerts.push({ rule: "burn-rate", service, severity: "critical", value: burnLong, threshold: SLO.burnThreshold,
        detail: burnShort });
    }
    if (short.p99 > LATENCY_BUDGET_MS) {
      alerts.push({ rule: "latency-p99", service, severity: "warning", value: short.p99,
        threshold: LATENCY_BUDGET_MS, detail: short.errorRate });
    }
  }
  return alerts;
}

/** Run a playbook: honour the auto-remediation policy and stop after a failing step. */
function runPlaybook(alert, { autoRemediation = true, failingStep = null } = {}) {
  const executed = [];
  let escalated = false;
  for (const step of PLAYBOOKS[alert.rule] || []) {
    if (step.mode === "autoRemediation" && !autoRemediation) {
      executed.push({ ...step, outcome: "skipped" });
      continue;
    }
    if (step.step === failingStep) {
      executed.push({ ...step, outcome: "failed" });
      escalated = true;
      break;
    }
    executed.push({ ...step, outcome: step.mode === "escalate" ? "escalated" : "ok" });
  }
  return { playbook: alert.rule, executed, escalated };
}

/** Minute-by-minute monitoring loop with per-(rule, service) de-duplication and auto-resolution. */
function runMonitoring(series, { minutes = 70, autoRemediation = true, failingStep = null } = {}) {
  const open = new Map();
  const alerts = [];
  const timeline = [];
  for (let minute = 6; minute <= minutes; minute += 1) {
    const firing = evaluate(series, minute);
    const firingKeys = new Set(firing.map((alert) => `${alert.rule}|${alert.service}`));
    const opened = [];
    for (const alert of firing) {
      const key = `${alert.rule}|${alert.service}`;
      if (open.has(key)) continue;
      open.set(key, alert);
      alert.firedAt = minute;
      alert.playbook = runPlaybook(alert, { autoRemediation, failingStep });
      alerts.push(alert);
      opened.push(key);
    }
    const resolved = [];
    for (const [key, alert] of open) {
      if (firingKeys.has(key) || alert.rule === "no-data") continue;
      open.delete(key);
      alert.resolvedAt = minute;
      resolved.push(key);
    }
    timeline.push({ minute, opened, resolved });
  }
  return { alerts, timeline, stillOpen: [...open.keys()] };
}

function main() {
  const series = makeSeries(982);
  const run = runMonitoring(series);

  // 1. Happy path: the outage raises exactly one critical alert, and only after it starts.
  const critical = run.alerts.filter((alert) => alert.severity === "critical");
  assert.strictEqual(critical.length, 1, "one critical alert for the whole outage");
  assert.strictEqual(critical[0].service, "api", "the failing service is named");
  assert.strictEqual(critical[0].firedAt, INCIDENT.start, `fires at minute ${INCIDENT.start}, got ${critical[0].firedAt}`);
  assert(critical[0].value > SLO.burnThreshold, "the reported burn rate is over the threshold");
  assert(!run.alerts.some((alert) => alert.service === "worker" && alert.rule === "burn-rate"),
    "the healthy service never burns budget");
  assert(run.alerts.some((alert) => alert.rule === "latency-p99" && alert.service === "api"), "the latency rule fires");

  // 2. Oracle: burn rate and p99 recomputed straight from the raw samples.
  const apiSamples = series.api;
  const windowErrors = (minute, width) => apiSamples.filter((row) => row.minute > minute - width && row.minute <= minute)
    .reduce((sum, row) => sum + row.errors, 0);
  const windowRequests = (minute, width) => apiSamples.filter((row) => row.minute > minute - width && row.minute <= minute)
    .reduce((sum, row) => sum + row.requests, 0);
  const expectedBurn = (windowErrors(INCIDENT.start, 10) / windowRequests(INCIDENT.start, 10)) / (1 - SLO.target);
  assert(Math.abs(critical[0].value - expectedBurn) < 1e-9, `burn ${critical[0].value} vs ${expectedBurn}`);
  const latencyAlert = run.alerts.find((alert) => alert.rule === "latency-p99");
  const windowLatencies = apiSamples.filter((row) => row.minute > latencyAlert.firedAt - 5 && row.minute <= latencyAlert.firedAt)
    .flatMap((row) => row.latencies).sort((a, b) => a - b);
  assert.strictEqual(latencyAlert.value, windowLatencies[Math.ceil(0.99 * windowLatencies.length) - 1],
    "p99 matches an independent sorted percentile");
  assert(windowStats(series.api, 10, 5).errorRate < 0.001, "the quiet window stays clean");

  // 3. De-duplication: one alert per rule and service, timeline strictly ordered.
  const keys = run.alerts.map((alert) => `${alert.rule}|${alert.service}`);
  assert.strictEqual(new Set(keys).size, keys.length, "no duplicate alerts");
  const openedEvents = run.timeline.flatMap((entry) => entry.opened);
  assert.strictEqual(openedEvents.length, run.alerts.length, "every alert opens exactly once");
  for (let i = 1; i < run.timeline.length; i += 1) {
    assert(run.timeline[i].minute === run.timeline[i - 1].minute + 1, "the timeline is contiguous");
  }
  assert.strictEqual(run.timeline.filter((entry) => entry.opened.includes("burn-rate|api")).length, 1, "one opening");

  // 4. Auto-resolution: the critical alert closes once the incident is over.
  assert(critical[0].resolvedAt > INCIDENT.end, `resolved at ${critical[0].resolvedAt}, after minute ${INCIDENT.end}`);
  assert(critical[0].resolvedAt > INCIDENT.end && critical[0].resolvedAt <= INCIDENT.end + SLO.longWindow,
    `resolved at ${critical[0].resolvedAt}, within one long window of the recovery`);
  assert(run.timeline.some((entry) => entry.minute === critical[0].resolvedAt && entry.resolved.includes("burn-rate|api")),
    "the resolution is on the timeline");
  assert(!run.stillOpen.includes("burn-rate|api"), "the critical alert is not left open");

  // 5. Playbooks: ordered steps, policy gating, and escalation on failure.
  const playbook = critical[0].playbook;
  assert.deepStrictEqual(playbook.executed.map((step) => step.step),
    ["acknowledge", "inspect-recent-deploys", "rollback-last-deploy", "page-oncall"], "playbook order");
  assert(!playbook.escalated && playbook.executed.at(-1).outcome === "escalated", "the last step pages on-call");
  const manual = runMonitoring(series, { autoRemediation: false });
  const manualCritical = manual.alerts.find((alert) => alert.rule === "burn-rate");
  assert(manualCritical.playbook.executed.some((step) => step.outcome === "skipped"), "policy can disable remediation");
  assert(manualCritical.playbook.executed.length === 4, "skipped steps stay in the audit trail");
  const broken = runMonitoring(series, { failingStep: "rollback-last-deploy" });
  const brokenPlaybook = broken.alerts.find((alert) => alert.rule === "burn-rate").playbook;
  assert(brokenPlaybook.escalated && brokenPlaybook.executed.length === 3, "a failing step stops the run and escalates");
  assert.deepStrictEqual(runMonitoring(series).alerts.find((a) => a.rule === "no-data").playbook.executed.map((s) => s.step),
    ["notify-data-owner"], "the no-data playbook only notifies");

  // 6. No-data boundary: a silent service is watched and never auto-resolves.
  const silent = run.alerts.filter((alert) => alert.service === "silent");
  assert.strictEqual(silent.length, 1, "one open alert for the silent service");
  assert.strictEqual(silent[0].rule, "no-data", "it is the watchdog, not a burn rate");
  assert.strictEqual(silent[0].resolvedAt, undefined, "a silent service never clears itself");
  assert(run.stillOpen.includes("no-data|silent"), "the watchdog stays open at the end of the window");
  assert.deepStrictEqual(windowStats([], 10, 5), null, "no samples, no statistics");

  // 7. Severity ordering and grouping in the report.
  const ranked = [...run.alerts].sort((a, b) => (SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity])
    || (a.service < b.service ? -1 : 1));
  assert.strictEqual(ranked[0].severity, "critical", "critical alerts lead the report");
  assert.deepStrictEqual([...new Set(ranked.map((alert) => alert.service))], ["api", "silent"], "grouped by service");
  assert(run.alerts.every((alert) => PLAYBOOKS[alert.rule] !== undefined), "every rule has a playbook");

  // 8. Determinism and boundaries: same seed, same incident; an empty stream is all no-data.
  assert(JSON.stringify(runMonitoring(makeSeries(982))) === JSON.stringify(run), "monitoring is reproducible");
  const other = runMonitoring(makeSeries(983));
  assert(other.alerts.find((alert) => alert.rule === "burn-rate").value !== critical[0].value, "another seed shifts metrics");
  assert.strictEqual(other.alerts.find((alert) => alert.rule === "burn-rate").firedAt, INCIDENT.start, "the alarm minute is stable");
  const nothing = runMonitoring({ api: [] }, { minutes: 8 });
  assert.strictEqual(nothing.alerts.length, 1, "an empty stream yields the watchdog only");
  assert.strictEqual(evaluate({ api: [] }, 6)[0].rule, "no-data", "the watchdog is the only rule that can fire blind");
  assert.deepStrictEqual(runPlaybook({ rule: "unknown" }).executed, [], "an unknown rule has no playbook");

  console.log(`monitoring: ${run.alerts.length} alerts (${run.alerts.map((a) => `${a.rule}:${a.service}@${a.firedAt}`).join(", ")}), ` +
    `critical burn ${critical[0].value.toFixed(1)}x resolved at ${critical[0].resolvedAt}, playbook ` +
    `${playbook.executed.length} steps`);
  console.log("PASS");
}
module.exports = { SLO, LATENCY_BUDGET_MS, INCIDENT, SERVICES, PLAYBOOKS, SEVERITY_RANK, splitmix64, makeSeries,
  percentile, windowStats, evaluate, runPlaybook, runMonitoring };
if (require.main === module) main();
