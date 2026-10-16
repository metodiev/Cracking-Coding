/** Task 937 - Implement deployment pipeline with canary analysis using online metrics and rollback.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A release pipeline ramps a canary through traffic stages (5% -> 20% -> 50% -> 100%). Each stage
 * accumulates *online* metrics per window - error count, crash count and latency samples for the
 * baseline and canary slices - and decides between four verdicts: `continue` while the canary slice
 * has fewer than `minSamples` observations, `rollback` when a hard gate (crash rate), the absolute
 * error-rate budget, the p99 latency guardrail or a one-sided z-test fires, `ramp` to the next
 * stage, and `promote` when the last stage is clean. A rollback freezes the pipeline, so traffic
 * returns to the baseline release and no later window can re-ramp it.
 *
 * Simplified: traffic slices are simulated from explicit per-release quality parameters instead of
 * a real LB, windows are fixed-size and the verdict is a single decision per window (no Bayesian
 * posterior over time). Verified: promotion of a clean canary, rollback of regressing, crashing and
 * slow canaries, the minimum-sample guard, ramp monotonicity, idempotent freezing after a verdict,
 * determinism for a fixed seed, and agreement with an independent exact binomial test - the online
 * z-test's verdicts are cross-checked against P[X >= k] recomputed by a pmf recurrence.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the only source of randomness; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}
const unit = (rng) => Number(rng.next() >> 11n) / 2 ** 53;

/** Independent statistic: exact one-sided P[X >= k] for X ~ Binomial(n, p), summed by pmf recurrence. */
function binomialTail(k, n, p) {
  if (k <= 0) return 1;
  if (k > n) return 0;
  const q = 1 - p;
  let pmf = q ** n;
  let tail = 0;
  for (let i = 0; i <= n; i += 1) {
    if (i >= k) tail += pmf;
    pmf = (pmf * (n - i) * p) / ((i + 1) * q);
  }
  return Math.min(1, tail);
}

/** Abramowitz & Stegun 7.1.26 - the normal CDF behind the z-test. */
function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const y = 1 - poly * Math.exp(-x * x);
  return x < 0 ? -y : y;
}
const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));
const Z_CRIT = 2.3263;

const newAcc = () => ({ n: 0, errors: 0, crashes: 0, latencies: [] });

function observe(acc, rng, release) {
  const failed = unit(rng) < release.errorRate;
  acc.n += 1;
  if (failed) acc.errors += 1;
  if (unit(rng) < release.crashRate) acc.crashes += 1;
  acc.latencies.push(release.latencyBase + Math.floor(unit(rng) * release.latencySpread));
}

function summarize(acc) {
  const sorted = [...acc.latencies].sort((a, b) => a - b);
  return {
    n: acc.n, errors: acc.errors, rate: acc.errors / Math.max(1, acc.n),
    crashRate: acc.crashes / Math.max(1, acc.n),
    p99: sorted.length ? sorted[Math.max(0, Math.ceil(0.99 * sorted.length) - 1)] : 0,
  };
}

class Pipeline {
  constructor(config) {
    this.cfg = Object.assign({ stages: [5, 20, 50, 100], perWindow: 800, minSamples: 200, alpha: 0.01,
      guardrail: 0.02, latencyFactor: 1.5, latencyBudget: 400, maxWindows: 40 }, config);
    this.status = "analyzing";
  }

  start(baselineRelease, canaryRelease) {
    this.baselineRelease = baselineRelease;
    this.canaryRelease = canaryRelease;
    this.stage = 0;
    this.weight = this.cfg.stages[0];
    this.status = "analyzing";
    this.trace = [];
    this.acc = { baseline: newAcc(), canary: newAcc() };
    return this;
  }

  /** One analysis window: `perWindow` baseline requests plus the canary's weighted share. */
  window(rng) {
    const { perWindow } = this.cfg;
    for (let i = 0; i < perWindow; i += 1) observe(this.acc.baseline, rng, this.baselineRelease);
    const canary = Math.round((perWindow * this.weight) / 100);
    for (let i = 0; i < canary; i += 1) observe(this.acc.canary, rng, this.canaryRelease);
    return this;
  }

  /** Verdict for the accumulated window: gates first, then the one-sided z-test. */
  analyze() {
    const { alpha, guardrail, latencyFactor, latencyBudget, minSamples } = this.cfg;
    const b = summarize(this.acc.baseline);
    const c = summarize(this.acc.canary);
    const pb = Math.min(0.5, Math.max(1e-6, b.rate));
    const se = Math.sqrt(pb * (1 - pb) * (1 / Math.max(1, c.n) + 1 / Math.max(1, b.n)));
    const z = se > 0 ? (c.rate - b.rate) / se : 0;
    const stats = { n: c.n, errors: c.errors, rate: c.rate, baselineRate: b.rate, p99: c.p99, z,
      p: binomialTail(c.errors, c.n, pb), oneSided: 1 - normCdf(z) };
    if (c.n < minSamples) return { decision: "continue", reason: `need ${minSamples} canary samples, have ${c.n}`, stats };
    if (c.crashRate > guardrail) return { decision: "rollback", reason: "hard gate: crash rate over budget", stats };
    if (c.rate - b.rate > guardrail) return { decision: "rollback", reason: "absolute error budget exceeded", stats };
    if (c.p99 > latencyBudget || c.p99 > b.p99 * latencyFactor) return { decision: "rollback", reason: "p99 latency guardrail", stats };
    if (z > Z_CRIT) return { decision: "rollback", reason: `z=${z.toFixed(2)} rejects equal quality at alpha=${alpha}`, stats };
    if (this.stage === this.cfg.stages.length - 1) return { decision: "promote", reason: "final stage clean", stats };
    return { decision: "ramp", reason: "no regression detected", stats };
  }

  decide() {
    if (this.status !== "analyzing") return { decision: "frozen", reason: `pipeline already ${this.status}` };
    const verdict = this.analyze();
    this.trace.push({ stage: this.stage, weight: this.weight, ...verdict });
    if (verdict.decision === "rollback") this.status = "rolled-back";
    else if (verdict.decision === "promote") { this.status = "promoted"; this.weight = 100; }
    else if (verdict.decision === "ramp") { this.stage += 1; this.weight = this.cfg.stages[this.stage]; }
    return verdict;
  }
}

/** Drive windows until the pipeline reaches a verdict; a canary must not run forever. */
function rollout(config, seed) {
  const pipeline = new Pipeline(config).start(config.baselineRelease, config.canaryRelease);
  const rng = splitmix64(seed);
  let windows = 0;
  while (pipeline.status === "analyzing") {
    pipeline.window(rng);
    pipeline.decide();
    windows += 1;
    assert(windows <= pipeline.cfg.maxWindows, `canary must reach a verdict within ${pipeline.cfg.maxWindows} windows`);
  }
  pipeline.windows = windows;
  pipeline.rng = rng;
  return pipeline;
}

const release = (errorRate = 0.01, crashRate = 0, latencyBase = 100, latencySpread = 40) =>
  ({ errorRate, crashRate, latencyBase, latencySpread });
const verdicts = (pipeline) => pipeline.trace.map((t) => t.decision);

function main() {
  const baseline = release();
  const healthy = { baselineRelease: baseline, canaryRelease: release() };

  // 1. Happy path: a clean canary ramps through every stage and is promoted at weight 100.
  const good = rollout(healthy, 937);
  assert(good.status === "promoted" && good.weight === 100, `clean canary is promoted, got ${good.status}`);
  assert(verdicts(good).includes("ramp") && verdicts(good).at(-1) === "promote", "the last verdict is a promotion");
  const weights = good.trace.map((t) => t.weight);
  assert(weights.every((w, i) => i === 0 || w >= weights[i - 1]), `traffic ramps monotonically: ${weights}`);
  assert(new Set(weights).size > 2 && weights[0] === 5, "several traffic stages are exercised from 5% up");

  // 2. Independent oracle: an exact binomial test agrees with every window of the clean run.
  for (const entry of good.trace) assert(entry.stats.p >= good.cfg.alpha, `exact test agrees at window ${entry.stats.n}`);
  assert(good.trace.every((t) => t.stats.p >= 0 && t.stats.p <= 1), "p-values stay in range");

  // 3. Regression: a canary with 3x the error rate is rolled back before the final stage.
  const bad = rollout({ baselineRelease: baseline, canaryRelease: release(0.03) }, 937);
  assert(bad.status === "rolled-back" && bad.stage < bad.cfg.stages.length - 1, `regressing canary rolls back early, got ${bad.status}`);
  const decision = bad.trace.at(-1);
  assert(decision.stats.p < bad.cfg.alpha && decision.stats.z > Z_CRIT, `exact test and z-test agree: p=${decision.stats.p}`);
  assert(decision.reason.includes("rejects equal quality"), `the online test rejects the canary: ${decision.reason}`);
  assert(!verdicts(bad).includes("promote"), "a regressing canary is never promoted");
  assert(bad.weight === bad.cfg.stages[bad.stage], "the rollout stops at the last approved stage weight");

  // 4. Absolute budget: a badly broken canary is caught by the error-rate budget, not the z-test.
  const broken = rollout({ baselineRelease: baseline, canaryRelease: release(0.08) }, 5);
  assert(broken.status === "rolled-back" && broken.trace.at(-1).reason === "absolute error budget exceeded", "error budget gates hard regressions");

  // 5. Boundary: the minimum-sample guard blocks any verdict below `minSamples` observations.
  const tiny = rollout({ ...healthy, perWindow: 10, minSamples: 2 }, 3);
  assert(tiny.trace[0].decision === "continue" && tiny.trace[0].stats.n === 1, "one canary sample is not enough to decide");
  assert(tiny.trace[1].stats.n === 2 && tiny.trace[1].decision !== "continue", "the verdict lands exactly at minSamples");
  assert.throws(() => rollout({ ...healthy, minSamples: 5000, maxWindows: 5 }, 3), /must reach a verdict/, "starved canary is detected");

  // 6. Hard gate: crashes trip a rollback even when the error rate looks identical.
  const crashing = rollout({ baselineRelease: baseline, canaryRelease: release(0.01, 0.05) }, 11);
  assert(crashing.status === "rolled-back" && crashing.trace.at(-1).reason.startsWith("hard gate"), "crash gate rolls back");

  // 7. Latency guardrail: same error rate, three times the latency base.
  const slow = rollout({ baselineRelease: baseline, canaryRelease: release(0.01, 0, 300, 40) }, 12);
  assert(slow.status === "rolled-back" && slow.trace.at(-1).reason === "p99 latency guardrail", "latency guardrail rolls back");

  // 8. Boundary: a single-stage config gives the canary all traffic and promotes on one clean window.
  const direct = rollout({ ...healthy, stages: [100] }, 937);
  assert(direct.weight === 100 && direct.windows === 1 && direct.status === "promoted", "single-stage rollout promotes immediately");

  // 9. Determinism and freezing: the same seed replays the same trace, and a verdict is idempotent.
  const replay = rollout(healthy, 937);
  assert(JSON.stringify(replay.trace) === JSON.stringify(good.trace), "the same seed reproduces the same trace");
  assert(JSON.stringify(rollout({ baselineRelease: baseline, canaryRelease: release(0.03) }, 938).trace) !== JSON.stringify(bad.trace), "a different seed changes the samples");
  const frozen = replay.window(replay.rng).decide();
  assert(frozen.decision === "frozen" && replay.status === "promoted" && replay.weight === 100, "a promoted pipeline is frozen");
  const frozenBad = bad.window(bad.rng).decide();
  assert(frozenBad.decision === "frozen" && bad.status === "rolled-back" && bad.trace.length === bad.windows, "a rolled-back pipeline is frozen");

  console.log(`canary: good=${good.windows} windows bad=${bad.windows} broken=${broken.windows} crashing=${crashing.windows} slow=${slow.windows} direct=${direct.windows}`);
  console.log("PASS");
}
module.exports = { splitmix64, binomialTail, normCdf, Pipeline, rollout, summarize, release, Z_CRIT };
if (require.main === module) main();
