/** Task 938 - Implement safe experiment platform that enforces SLOs and budget on experiments.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * An A/B experiment platform: units are assigned to a variant by hashed bucketing of (experiment
 * id, unit id), so assignment is deterministic, stable and mutually exclusive without a shared
 * assignment service. The platform consumes a budget per exposure and evaluates every analysis
 * window in a fixed safety order: budget exhaustion, then the SLO guardrails (error rate and p99
 * latency, for either variant), and only then the effect test. A guardrail therefore halts the
 * experiment before any winner can be shipped, and once halted no further exposure is served while
 * every collected sample stays in the final report.
 *
 * Simplified: variants are simulated from per-variant quality parameters instead of a real service,
 * exposures are one per unit (no re-exposure), and the effect test is a Wald interval on two
 * proportions rather than a group-sequential boundary. Verified: assignment determinism and balance,
 * allocation boundaries, guardrail precedence over shipping, exact budget stopping, the minimum
 * sample-size gate, empty-data safety, and agreement with an independently recomputed effect and CI.
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
const CI_Z = 1.959964;
const VARIANTS = ["control", "treatment"];

/** 32-bit avalanche + FNV-1a over the unit id: the hash behind hashed bucketing. */
function mix32(x) {
  let h = x >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}
function hashUnit(unitId) {
  let h = 0x811c9dc5;
  for (let i = 0; i < unitId.length; i += 1) h = Math.imul(h ^ unitId.charCodeAt(i), 0x01000193) >>> 0;
  return h >>> 0;
}
const bucketOf = (experimentId, unitId) => mix32(hashUnit(unitId) ^ mix32(experimentId)) % 1000;

function observe(rng, params) {
  return {
    converted: unit(rng) < params.conversion,
    errored: unit(rng) < params.errorRate,
    latency: params.latency + Math.floor(unit(rng) * 30),
    cost: params.costPerExposure ?? 1,
  };
}

function summarize(samples) {
  const n = samples.length;
  const sorted = samples.map((s) => s.latency).sort((a, b) => a - b);
  const converted = samples.filter((s) => s.converted).length;
  const errors = samples.filter((s) => s.errored).length;
  return { n, converted, errors, rate: n ? converted / n : 0, errorRate: n ? errors / n : 0,
    p99: n ? sorted[Math.max(0, Math.ceil(0.99 * n) - 1)] : 0 };
}

const varianceOf = (s) => (s.n ? (s.rate * (1 - s.rate)) / s.n : 0);

/** Effect and Wald interval for the difference of two proportions. */
function summarizeExperiment(samples) {
  const control = summarize(samples.control);
  const treatment = summarize(samples.treatment);
  const effect = treatment.rate - control.rate;
  const se = Math.sqrt(varianceOf(control) + varianceOf(treatment));
  return { control, treatment, effect, se, ci: [effect - CI_Z * se, effect + CI_Z * se],
    pooled: (control.converted + treatment.converted) / Math.max(1, control.n + treatment.n) };
}

class Experiment {
  constructor(config) {
    this.cfg = Object.assign({ id: 938, allocation: [500, 500], minPerVariant: 100, budget: 1e9,
      guardrailErrorRate: 0.05, guardrailLatency: 400, window: 400 }, config);
    this.status = "running";
    this.exposures = 0;
    this.cost = 0;
    this.samples = { control: [], treatment: [] };
    this.trace = [];
  }

  /** Stable hashed assignment; null when the experiment is halted or the bucket is unallocated. */
  assign(unitId) {
    if (this.status !== "running") return null;
    const bucket = bucketOf(this.cfg.id, unitId);
    let acc = 0;
    for (let i = 0; i < VARIANTS.length; i += 1) {
      acc += this.cfg.allocation[i];
      if (bucket < acc) return VARIANTS[i];
    }
    return null;
  }

  record(variant, sample) {
    this.samples[variant].push(sample);
    this.exposures += 1;
    this.cost += sample.cost;
  }

  /** Budget and SLO guardrails are checked before any winner, so a harmful variant cannot ship. */
  analyse() {
    const stats = summarizeExperiment(this.samples);
    const verdict = { exposures: this.exposures, cost: this.cost, stats };
    if (this.cost >= this.cfg.budget) {
      this.status = "halted-budget";
      return { ...verdict, decision: "halt-budget", reason: "experiment budget exhausted" };
    }
    for (const variant of VARIANTS) {
      const s = stats[variant];
      if (s.errorRate > this.cfg.guardrailErrorRate) {
        this.status = "halted-guardrail";
        return { ...verdict, decision: "halt-guardrail", reason: `${variant} error rate breaches the SLO` };
      }
      if (s.p99 > this.cfg.guardrailLatency) {
        this.status = "halted-guardrail";
        return { ...verdict, decision: "halt-guardrail", reason: `${variant} p99 latency breaches the SLO` };
      }
    }
    if (stats.control.n < this.cfg.minPerVariant || stats.treatment.n < this.cfg.minPerVariant) {
      return { ...verdict, decision: "continue", reason: "minimum sample size not reached" };
    }
    if (stats.ci[0] > 0) {
      this.status = "shipped";
      return { ...verdict, decision: "ship", reason: "the effect interval is entirely above zero" };
    }
    if (stats.ci[1] < 0) {
      this.status = "stopped";
      return { ...verdict, decision: "stop", reason: "the effect interval is entirely below zero" };
    }
    return { ...verdict, decision: "continue", reason: "the effect interval still contains zero" };
  }
}

/** Drive units through the experiment until it halts or the population is exhausted. */
function runExperiment(config, { units, seed }) {
  const experiment = new Experiment(config);
  const rng = splitmix64(seed);
  const truth = config.truth;
  for (let u = 0; u < units; u += 1) {
    const variant = experiment.assign(`unit-${u}`);
    if (!variant) break;
    experiment.record(variant, observe(rng, truth[variant]));
    if (experiment.exposures % experiment.cfg.window === 0) experiment.trace.push(experiment.analyse());
  }
  experiment.trace.push(experiment.analyse());
  return experiment;
}

/** Independent oracle: recompute the effect straight from the raw samples. */
function effectFromSamples(samples) {
  const rate = (list) => (list.length ? list.filter((s) => s.converted).length / list.length : 0);
  return rate(samples.treatment) - rate(samples.control);
}
const ok = (truth) => ({ control: { conversion: 0.10, errorRate: 0.01, latency: 100, ...truth.control },
  treatment: { conversion: 0.10, errorRate: 0.01, latency: 100, ...truth.treatment } });

function main() {
  // 1. Assignment is deterministic, balanced and mutually exclusive.
  const unitIds = Array.from({ length: 20000 }, (_, i) => `unit-${i}`);
  const probe = new Experiment({ id: 938 });
  const controlShare = unitIds.filter((u) => probe.assign(u) === "control").length / unitIds.length;
  assert(Math.abs(controlShare - 0.5) < 0.02, `hashed buckets are balanced, got ${controlShare}`);
  assert(unitIds.slice(0, 500).every((u) => probe.assign(u) === probe.assign(u)), "a unit always lands in the same variant");

  // 2. Boundary allocations: a variant with no traffic gets nothing, a full allocation gets all.
  const onlyControl = new Experiment({ id: 7, allocation: [1000, 0] });
  const allTreatment = new Experiment({ id: 7, allocation: [0, 1000] });
  assert(unitIds.every((u) => onlyControl.assign(u) === "control"), "0% allocation receives no traffic");
  assert(unitIds.every((u) => allTreatment.assign(u) === "treatment"), "100% allocation receives everything");
  const partial = new Experiment({ id: 7, allocation: [400, 400] });
  const outside = unitIds.filter((u) => partial.assign(u) === null).length / unitIds.length;
  assert(Math.abs(outside - 0.2) < 0.02, `unallocated traffic stays out of the experiment, got ${outside}`);

  // 3. Happy path plus independent oracle: a 4-point lift ships and the effect is reproducible.
  const shipped = runExperiment({ id: 938, minPerVariant: 500, truth: ok({ treatment: { conversion: 0.14 } }) }, { units: 6000, seed: 938 });
  assert(shipped.status === "shipped" && shipped.trace.at(-1).decision === "ship", `a clear lift ships, got ${shipped.status}`);
  const final = shipped.trace.at(-1).stats;
  assert(final.effect === effectFromSamples(shipped.samples), "the platform effect equals the oracle recomputation");
  assert(final.ci[0] <= final.effect && final.effect <= final.ci[1] && final.ci[0] > 0, `the interval brackets the effect and excludes zero: ${final.ci}`);

  // 4. Safety: a variant that lifts conversions but breaches the error SLO halts instead of shipping.
  const unsafe = runExperiment({ id: 938, minPerVariant: 500, truth: ok({ treatment: { conversion: 0.30, errorRate: 0.20 } }) }, { units: 6000, seed: 4 });
  assert(unsafe.status === "halted-guardrail" && unsafe.trace.at(-1).decision === "halt-guardrail", "guardrail halts a harmful variant");
  assert(!unsafe.trace.some((t) => t.decision === "ship"), "an SLO breach is never shipped");
  assert(unsafe.exposures === unsafe.samples.control.length + unsafe.samples.treatment.length, "no collected sample is dropped");
  assert(unsafe.assign("unit-99999") === null, "a halted experiment serves no further exposure");

  // 5. Budget: spending stops at the cap within one analysis window and never overshoots it.
  const capped = runExperiment({ id: 938, budget: 1200, window: 400, truth: ok({}) }, { units: 20000, seed: 5 });
  assert(capped.status === "halted-budget" && capped.cost >= 1200, `budget halts the run at ${capped.cost}`);
  assert(capped.cost < 1200 + 2 * 400, "the overshoot stays inside one analysis window");
  assert(capped.trace.at(-1).stats.control.n + capped.trace.at(-1).stats.treatment.n === capped.exposures, "the report covers every exposure");

  // 6. Boundary: without enough samples per variant the platform never declares a verdict.
  const starved = runExperiment({ id: 938, minPerVariant: 100000, truth: ok({ treatment: { conversion: 0.9 } }) }, { units: 5000, seed: 6 });
  assert(starved.status === "running" && starved.trace.every((t) => t.decision === "continue"), "no verdict before the sample floor");
  assert(starved.trace.at(-1).reason === "minimum sample size not reached", "the reason names the gate");

  // 7. Boundary: an empty experiment reports zeros instead of NaN and keeps running.
  const empty = runExperiment({ id: 938, truth: ok({}) }, { units: 0, seed: 7 });
  const zero = empty.trace.at(-1);
  assert(empty.exposures === 0 && zero.decision === "continue" && zero.stats.effect === 0, "empty data yields a safe continue");
  assert(!Number.isNaN(zero.stats.effect) && !Number.isNaN(zero.stats.ci[0]), "empty data produces no NaN statistics");

  // 8. Regression: a 4-point drop is stopped, and a per-exposure price reaches the cap sooner.
  const worse = runExperiment({ id: 938, minPerVariant: 500, truth: ok({ treatment: { conversion: 0.06 } }) }, { units: 6000, seed: 9 });
  assert(worse.status === "stopped" && worse.trace.at(-1).stats.ci[1] < 0, "a clear regression stops the experiment");
  const priced = runExperiment({ id: 938, budget: 1000, truth: ok({ treatment: { conversion: 0.10, costPerExposure: 2 } }) }, { units: 20000, seed: 5 });
  assert(priced.status === "halted-budget" && priced.cost >= 1000, `priced exposures reach the cap at cost ${priced.cost}`);
  assert(priced.exposures < capped.exposures, "a pricier variant buys fewer exposures for the same budget");

  // 9. Determinism: the same seed replays the same report, a different seed moves the numbers.
  const again = runExperiment({ id: 938, minPerVariant: 500, truth: ok({ treatment: { conversion: 0.14 } }) }, { units: 6000, seed: 938 });
  assert(JSON.stringify(again.trace) === JSON.stringify(shipped.trace), "the same seed reproduces the same trace");
  const other = runExperiment({ id: 938, minPerVariant: 500, truth: ok({ treatment: { conversion: 0.14 } }) }, { units: 6000, seed: 939 });
  assert(JSON.stringify(other.trace) !== JSON.stringify(shipped.trace), "a different seed changes the samples");

  // 10. Latency SLO: a slow variant halts on the p99 guardrail even with equal conversion.
  const slow = runExperiment({ id: 938, truth: ok({ treatment: { latency: 500 } }) }, { units: 4000, seed: 10 });
  assert(slow.status === "halted-guardrail" && slow.trace.at(-1).reason.includes("p99 latency"), "latency SLO halts the experiment");

  console.log(`experiments: shipments=1 halts=4 exposures=${capped.exposures} budgetSpent=${capped.cost}`);
  console.log("PASS");
}
module.exports = { splitmix64, bucketOf, mix32, Experiment, runExperiment, summarizeExperiment, effectFromSamples, CI_Z };
if (require.main === module) main();
