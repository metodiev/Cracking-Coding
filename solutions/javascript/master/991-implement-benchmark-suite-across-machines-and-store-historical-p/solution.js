/** Task 991 - Implement benchmark suite across machines and store historical performance.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A harness that records benchmark runs from several machines and judges them against history. Each
 * machine carries a calibration factor measured once against a reference workload, so a raw
 * millisecond figure is divided by that factor before it is stored as `normalizedMs`; comparisons
 * then only see the contribution of the code, not of the box it ran on. The history is append-only,
 * and a verdict needs a minimum number of samples on both sides of the comparison, which keeps a
 * single noisy run from being reported as a regression.
 *
 * What is simplified: machines are simulated -- a run's raw time comes from a seeded model instead of
 * a real timer -- because a clock cannot be reproduced. Verified: normalization and calibration con-
 * sistency across machines, regression/improvement/insufficient-samples verdicts, refusal of unknown
 * or mis-calibrated machines, median agreement with a brute-force oracle, append-only history and
 * byte-identical replay of the whole simulated suite.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the file's only randomness source, so every simulated run is byte-reproducible. */
function splitmix64(seed) {
  let s = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    s = BigInt.asUintN(64, s + 0x9e3779b97f4a7c15n);
    let z = BigInt.asUintN(64, (s ^ (s >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)), unit: () => Number(next() % 1000000n) / 1000000 };
}

function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

const MACHINES = [
  { name: "ref-box", factor: 1.0 }, { name: "slow-rig", factor: 2.4 },
  { name: "fast-ci", factor: 0.6 }, { name: "laptop", factor: 1.7 },
];
const factorOf = (machines, name) => {
  const m = machines.find((x) => x.name === name);
  return m ? m.factor : NaN;
};

/** Append-only history. An entry is rejected when the machine is unknown or badly calibrated. */
function recordRun(history, entry, machines = MACHINES) {
  const factor = factorOf(machines, entry.machine);
  if (!Number.isFinite(factor)) return { ok: false, reason: "unknown-machine", history };
  if (!(factor > 0)) return { ok: false, reason: "invalid-calibration", history };
  if (!(entry.rawMs > 0)) return { ok: false, reason: "invalid-measurement", history };
  const next = history.concat([{ id: history.length, machine: entry.machine, benchmark: entry.benchmark,
    rawMs: entry.rawMs, normalizedMs: entry.rawMs / factor }]);
  return { ok: true, reason: "recorded", history: next, entry: next[next.length - 1] };
}

const median = (xs) => {
  const a = xs.slice().sort((x, y) => x - y);
  const mid = a.length >> 1;
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
};

/** Independent oracle: rank-based selection by counting, rather than sorting and picking a middle slot. */
function bruteMedian(xs) {
  if (xs.length === 0) return NaN;
  const values = Array.from(new Set(xs)).sort((a, b) => a - b);
  const at = (k) => {
    let seen = 0;
    for (const v of values) {
      let count = 0;
      for (const x of xs) if (x === v) count += 1;
      if (seen + count > k) return v;
      seen += count;
    }
    return NaN;
  };
  const mid = xs.length >> 1;
  return xs.length % 2 ? at(mid) : (at(mid - 1) + at(mid)) / 2;
}

/** Median over a group of runs; `window` limits how many of the most recent runs are used. */
function medianNormalized(history, benchmark, { machine = null, window = 5 } = {}) {
  const rows = history.filter((h) => h.benchmark === benchmark && (machine === null || h.machine === machine));
  const recent = rows.slice(Math.max(0, rows.length - window));
  return { samples: recent.length, medianMs: recent.length === 0 ? NaN : median(recent.map((r) => r.normalizedMs)) };
}

/**
 * Verdict for one benchmark: compare the median of the most recent `window` runs of the "current"
 * half against the earlier "baseline" half. Fewer than `minSamples` on either side is not a verdict.
 */
function assess(history, benchmark, { tolerance = 0.1, minSamples = 3, window = 5 } = {}) {
  const rows = history.filter((h) => h.benchmark === benchmark);
  const cut = Math.floor(rows.length / 2);
  const baseline = medianNormalized(rows.slice(0, cut), benchmark, { window });
  const current = medianNormalized(rows.slice(cut), benchmark, { window });
  if (baseline.samples < minSamples || current.samples < minSamples) {
    return { verdict: "insufficient-samples", baseline, current, delta: NaN };
  }
  const delta = (current.medianMs - baseline.medianMs) / baseline.medianMs;
  const verdict = delta > tolerance ? "regression" : delta < -tolerance ? "improvement" : "stable";
  return { verdict, baseline, current, delta };
}

const BENCHES = [{ name: "sort-10k", baseMs: 100 }, { name: "search-1m", baseMs: 250 }];
/** Simulated measurement: base cost times machine speed times a seeded jitter of at most `jitter`. */
function measure(machine, bench, rng, jitter = 0.08) {
  const noise = 1 + (rng.unit() * 2 - 1) * jitter;
  return Math.round(bench.baseMs * machine.factor * noise * 1000) / 1000;
}

/** Simulate `rounds` rounds of every benchmark on every machine, in a fixed order. */
function simulate(seed, rounds = 4, { machines = MACHINES, benches = BENCHES, jitter = 0.08,
  growth = () => 1 } = {}) {
  const rng = splitmix64(seed);
  let history = [];
  for (let r = 0; r < rounds; r += 1) {
    const scale = growth(r);
    for (const bench of benches) for (const machine of machines) {
      history = recordRun(history, { machine: machine.name, benchmark: bench.name,
        rawMs: measure(machine, { name: bench.name, baseMs: bench.baseMs * scale }, rng, jitter) }, machines).history;
    }
  }
  return { history, digest: fnv1a(JSON.stringify(history)) };
}

function main() {
  const rng = splitmix64(0x991);
  let history = [];
  for (const machine of MACHINES) history = recordRun(history, { machine: machine.name,
    benchmark: "sort-10k", rawMs: measure(machine, BENCHES[0], rng) }, MACHINES).history;
  // 1. Happy path: normalized times are raw / factor, so calibration hides machine speed.
  assert.deepStrictEqual(history.map((h) => h.normalizedMs), history.map((h) => h.rawMs / factorOf(MACHINES, h.machine)),
    "normalization divides by the machine factor");
  assert.deepStrictEqual(history.map((h) => h.id), [0, 1, 2, 3], "history ids are monotone");
  // 2. Calibration invariant: the machines' raw times differ a lot, their normalized medians do not.
  const rawSpread = Math.max(...history.map((h) => h.rawMs)) / Math.min(...history.map((h) => h.rawMs));
  const normSpread = Math.max(...history.map((h) => h.normalizedMs)) / Math.min(...history.map((h) => h.normalizedMs));
  assert.ok(rawSpread > 3 && normSpread < rawSpread / 3, `calibration narrows the spread (${rawSpread.toFixed(2)}x -> ${normSpread.toFixed(2)}x)`);
  // 3. Boundary: fewer samples than required yields no verdict, and an empty history has no baseline.
  const sparse = assess(history, "sort-10k", { minSamples: 3 });
  assert.deepStrictEqual([sparse.verdict, Number.isNaN(sparse.delta)], ["insufficient-samples", true],
    "too few samples is not a verdict");
  assert.strictEqual(medianNormalized([], "sort-10k").medianMs, NaN, "empty history has no median");
  // 4. Negative: unknown machines and non-positive calibrations are refused without touching history.
  const unknown = recordRun(history, { machine: "ghost", benchmark: "sort-10k", rawMs: 10 });
  const badCal = recordRun(history, { machine: "sabotaged", benchmark: "sort-10k", rawMs: 10 },
    MACHINES.concat([{ name: "sabotaged", factor: 0 }]));
  assert.deepStrictEqual([unknown.reason, badCal.reason, unknown.history.length], ["unknown-machine", "invalid-calibration", history.length],
    "bad runs are refused and history is untouched");
  // 5. Oracle: the median helper agrees with the explicit definition on every window of the fixture.
  const fixture = simulate(0x991, 4).history;
  const xs = fixture.map((h) => h.normalizedMs);
  for (let n = 1; n <= xs.length; n += 1) {
    assert.strictEqual(median(xs.slice(0, n)), bruteMedian(xs.slice(0, n)), `median matches at n=${n}`);
  }
  // 6. Stable workload: with no drift the suite is judged stable, never a regression.
  const stable = assess(fixture, "sort-10k", { tolerance: 0.1, minSamples: 2 });
  assert.deepStrictEqual([stable.verdict, Math.abs(stable.delta) < 0.1], ["stable", true], "noise stays inside tolerance");
  // 7. Regression: a workload that really slows down is reported, with a delta of the right sign.
  const slowed = simulate(0x991, 6, { growth: (r) => (r < 3 ? 1 : 1.35) }).history;
  const verdict = assess(slowed, "sort-10k", { tolerance: 0.1, minSamples: 2 });
  assert.deepStrictEqual([verdict.verdict, verdict.delta > 0.2], ["regression", true], "sustained slowdown is a regression");
  const sped = assess(simulate(0x991, 6, { growth: (r) => (r < 3 ? 1 : 0.6) }).history, "sort-10k",
    { tolerance: 0.1, minSamples: 2 });
  assert.strictEqual(sped.verdict, "improvement", "sustained speedup is an improvement");
  // 8. Append-only and reproducible: a longer history starts with the shorter one, ids never repeat,
  // and the whole simulation replays byte for byte.
  const short = simulate(0x991, 2), long = simulate(0x991, 4);
  assert.deepStrictEqual(long.history.slice(0, short.history.length), short.history, "history prefix is immutable");
  assert.strictEqual(new Set(long.history.map((h) => h.id)).size, long.history.length, "ids are unique");
  assert.strictEqual(simulate(0x991, 4).digest, long.digest, "simulation replays byte for byte");
  assert.strictEqual(simulate(0x991, 4).digest === simulate(0x992, 4).digest, false, "a different seed is a different run set");
  console.log(`runs=${long.history.length} verdict=${verdict.verdict} delta=${verdict.delta.toFixed(3)} digest=${long.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, recordRun, median, bruteMedian, medianNormalized, assess, measure,
  simulate, MACHINES, BENCHES };
