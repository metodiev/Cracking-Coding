/** Task 951 - Implement end-to-end reproducible benchmarking harness for all exercises in repo.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * In-process benchmark harness for the repo's exercise suite. Each exercise declares sized cases and
 * one implementation per language; implementations get a Counter and report counted steps instead of
 * wall-clock time, because a clock is the one thing a shared machine cannot reproduce. Case inputs come
 * from a splitmix64 stream derived from (seed, exercise id, size), so `runOnce(seed)` is a pure function
 * of its seed and its whole row set collapses to one digest.
 * Simplified: no subprocesses, timers or per-language toolchains -- "languages" are implementations in
 * this file and cost is counted operations, not nanoseconds. Verified: row shape, byte-identical repeats
 * for one seed, seed sensitivity, cross-implementation agreement per case, closed-form operation oracles,
 * linear-vs-binary asymptotic separation, aggregate medians, and detection of measured-cost drift.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the only source of randomness in this file; identical stream on every run. */
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
/** FNV-1a: the harness' digest primitive. */
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
/** Order-independent serialisation: keys are sorted, so a digest never depends on insertion order. */
function stableStringify(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(stableStringify).join(",") + "]";
  return "{" + Object.keys(value).sort()
    .map((k) => JSON.stringify(k) + ":" + stableStringify(value[k])).join(",") + "}";
}
const digestOf = (value) => fnv1a(stableStringify(value));
const hex32 = (n) => (n >>> 0).toString(16).padStart(8, "0");

/** Counted steps replace the clock: one comparison is one step, so every run is bit-reproducible. */
class Counter {
  constructor() { this.ops = 0; this.cmp = 0; }
  step(n = 1) { this.ops += n; return this; }
  compare(a, b) { this.ops += 1; this.cmp += 1; return a < b ? -1 : a > b ? 1 : 0; }
}
function sumLinear(input, ctr) {
  let total = 0;
  for (const x of input.xs) { ctr.step(); total += x; }
  return total;
}
const sumReduce = (input, ctr) => input.xs.reduce((a, x) => { ctr.step(); return a + x; }, 0);
function insertionSort(input, ctr) {
  const a = input.xs.slice();
  for (let i = 1; i < a.length; i += 1) {
    const key = a[i];
    let j = i - 1;
    while (j >= 0 && ctr.compare(a[j], key) > 0) { a[j + 1] = a[j]; j -= 1; }
    a[j + 1] = key;
  }
  return a;
}
function mergeSort(input, ctr) {
  const a = input.xs.slice();
  const buffer = new Array(a.length);
  const sortRange = (lo, hi) => {
    if (hi - lo < 2) return;
    const mid = (lo + hi) >> 1;
    sortRange(lo, mid); sortRange(mid, hi);
    let i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
      if (ctr.compare(a[i], a[j]) <= 0) { buffer[k] = a[i]; i += 1; } else { buffer[k] = a[j]; j += 1; }
      k += 1;
    }
    while (i < mid) { buffer[k] = a[i]; i += 1; k += 1; }
    while (j < hi) { buffer[k] = a[j]; j += 1; k += 1; }
    for (let t = lo; t < hi; t += 1) a[t] = buffer[t];
  };
  sortRange(0, a.length);
  return a;
}
function linearSearch(input, ctr) {
  for (let i = 0; i < input.xs.length; i += 1) { ctr.step(); if (input.xs[i] === input.target) return i; }
  return -1;
}
function binarySearch(input, ctr) {
  let lo = 0, hi = input.xs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const cmp = ctr.compare(input.xs[mid], input.target);
    if (cmp === 0) return mid;
    if (cmp < 0) lo = mid + 1; else hi = mid - 1;
  }
  return -1;
}
const sumInput = (spec, rng) => ({ xs: Array.from({ length: spec.n }, () => rng.int(1000)) });
/** Distinct ascending values: uniqueness makes both search implementations return one index. */
function searchInput(spec, rng) {
  const xs = [];
  let v = 0;
  for (let i = 0; i < spec.n; i += 1) { v += 1 + rng.int(4); xs.push(v); }
  const target = xs.length === 0 ? 0
    : spec.look === "missing" ? xs[xs.length - 1] + 1
      : spec.look === "last" ? xs[xs.length - 1] : xs[xs.length >> 1];
  return { xs, target };
}
const EXERCISES = [
  { id: "e-sum", tier: "beginner", cases: [{ n: 0 }, { n: 8 }, { n: 64 }], makeInput: sumInput,
    impls: { "js-loop": sumLinear, "js-reduce": sumReduce } },
  { id: "e-sort", tier: "intermediate", cases: [{ n: 1 }, { n: 16 }, { n: 64 }], makeInput: sumInput,
    impls: { "js-insertion": insertionSort, "js-merge": mergeSort } },
  { id: "e-search", tier: "advanced", makeInput: searchInput,
    cases: [{ n: 0, look: "missing" }, { n: 16, look: "middle" }, { n: 128, look: "last" }],
    impls: { "js-linear": linearSearch, "js-binary": binarySearch } },
];
/** Run the suite. Inputs depend only on (seed, exercise, case), so all `reps` rows of a case share
 * one input and one answer digest -- precisely what makes measurement drift visible. */
function runOnce(seed, { exercises = EXERCISES, reps = 3 } = {}) {
  const rows = [];
  for (const ex of exercises) {
    for (const spec of ex.cases) {
      const caseSeed = (seed ^ fnv1a(ex.id) ^ Math.imul(spec.n + 1, 0x9e3779b1)) >>> 0;
      const input = ex.makeInput(spec, splitmix64(caseSeed));
      const inDigest = hex32(digestOf(input));
      for (let rep = 0; rep < reps; rep += 1) {
        for (const lang of Object.keys(ex.impls).sort()) {
          const ctr = new Counter();
          const value = ex.impls[lang](input, ctr);
          rows.push({ suite: ex.id, lang, n: spec.n, rep, ops: ctr.ops, cmp: ctr.cmp, inDigest, tier: ex.tier,
            outDigest: hex32(digestOf(value)) });
        }
      }
    }
  }
  return { seed, reps, rows, digest: hex32(digestOf(rows)),
    totalOps: rows.reduce((a, r) => a + r.ops, 0) };
}
function median(values) {
  const a = values.slice().sort((x, y) => x - y);
  const mid = a.length >> 1;
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}
/** Per (suite, implementation, size) median of the repeated measurements. */
function aggregate(run) {
  const groups = new Map();
  for (const r of run.rows) {
    const key = `${r.suite}/${r.lang}/n=${r.n}`;
    if (!groups.has(key)) groups.set(key, { ops: [], outDigest: r.outDigest, inDigest: r.inDigest });
    groups.get(key).ops.push(r.ops);
  }
  const stats = new Map();
  for (const [key, g] of groups) stats.set(key, { ops: median(g.ops), samples: g.ops.length, outDigest: g.outDigest, inDigest: g.inDigest });
  return stats;
}
function firstRowDifference(a, b) {
  for (let i = 0; i < Math.min(a.rows.length, b.rows.length); i += 1) {
    if (stableStringify(a.rows[i]) !== stableStringify(b.rows[i])) {
      const r = a.rows[i];
      return `${r.suite}/${r.lang}/n=${r.n} rep=${r.rep}: ops ${r.ops} vs ${b.rows[i].ops}`;
    }
  }
  return `row count ${a.rows.length} vs ${b.rows.length}`;
}
/** The harness' own gate: two runs of one seed must hash identically, and it must name the drifter. */
function verifyReproducible(exercises, { seed = 20240951, reps = 2 } = {}) {
  const [first, second] = [0, 1].map(() => runOnce(seed, { exercises, reps }));
  return first.digest === second.digest
    ? { ok: true, detail: `identical row digest ${first.digest}` }
    : { ok: false, detail: `digest drift: ${firstRowDifference(first, second)}` };
}
/** A shared, call-order dependent PRNG stream: the harness' nondeterminism canary. */
const drift = (() => { const rng = splitmix64(7); return () => 1 + rng.int(1 << 20); })();
const JITTER_SUITE = [{ id: "e-jitter", tier: "master", cases: [{ n: 8 }, { n: 16 }], makeInput: sumInput,
  impls: { "js-jitter": (input, ctr) => { ctr.step(drift()); return sumLinear(input, ctr); } } }];

function main() {
  const seed = 20240951;
  const [run, repeat, other] = [seed, seed, seed + 1].map((s) => runOnce(s));
  const stats = aggregate(run);
  // 1. Happy path: every (suite, implementation, size) group is measured exactly `reps` times.
  const expected = EXERCISES.reduce((a, ex) => a + ex.cases.length * Object.keys(ex.impls).length, 0) * run.reps;
  const perGroup = new Map();
  for (const r of run.rows) {
    const k = `${r.suite}/${r.lang}/n=${r.n}`;
    perGroup.set(k, (perGroup.get(k) || 0) + 1);
  }
  assert(run.rows.length === expected && [...perGroup.values()].every((c) => c === run.reps),
    `row shape: ${run.rows.length} rows, each group measured ${run.reps} times`);
  // 2. Reproducibility: one seed reproduces every row byte for byte, and the run digest agrees.
  assert.deepStrictEqual(repeat.rows, run.rows, "same seed reproduces every row");
  assert.strictEqual(repeat.digest, run.digest, "run digest is stable");
  // 3. Seed sensitivity: generated inputs follow the seed, so the digest has to move with it.
  assert.notStrictEqual(other.digest, run.digest, "a different seed yields a different run digest");
  assert(run.rows.some((r, i) => r.inDigest !== other.rows[i].inDigest), "generated inputs follow the seed");
  // 4. Differential agreement per case, and generators emit exactly the declared item counts.
  const answers = new Map();
  for (const r of run.rows) {
    const k = `${r.suite}/n=${r.n}/rep=${r.rep}`;
    answers.set(k, (answers.get(k) || new Set()).add(r.outDigest));
  }
  const generatorsMatch = EXERCISES.every((ex) => ex.cases.every((s) => ex.makeInput(s, splitmix64(1)).xs.length === s.n));
  assert([...answers.values()].every((s) => s.size === 1) && generatorsMatch,
    "all implementations agree on every case and generators emit exactly n items");
  // 5. Boundary: a zero-length case still yields rows -- zero steps and the empty-sum answer.
  const zeroSum = run.rows.find((r) => r.suite === "e-sum" && r.n === 0 && r.lang === "js-loop");
  assert(zeroSum.ops === 0 && zeroSum.outDigest === hex32(digestOf(0)) && stats.get("e-sum/js-reduce/n=0").ops === 0,
    "an empty input costs zero steps, sums to 0, and aggregates to 0");
  // 6. Closed-form operation oracles, derived by hand rather than by the harness.
  const ascending = { xs: Array.from({ length: 64 }, (_, i) => i) };
  const descending = { xs: Array.from({ length: 64 }, (_, i) => 64 - i) };
  const [sortedCase, reversedCase, mergeCase, sumCase] = [0, 1, 2, 3].map(() => new Counter());
  insertionSort(ascending, sortedCase);  insertionSort(descending, reversedCase);
  mergeSort(descending, mergeCase);
  assert(sortedCase.cmp === 63, "insertion sort on sorted input compares n-1 times");
  assert(reversedCase.cmp === (64 * 63) / 2, "insertion sort on reversed input compares n(n-1)/2 times");
  assert(mergeCase.cmp >= (64 * 6) / 2 && mergeCase.cmp <= 64 * 6,
    `merge sort comparisons ${mergeCase.cmp} lie in [n*log2(n)/2, n*log2(n)]`);
  assert(sumLinear(ascending, sumCase) === (63 * 64) / 2 && sumCase.ops === 64, "linear sum matches n(n-1)/2 in n steps");
  // 7. Asymptotic separation on the largest search case, measured outside the harness.
  const probe = { xs: Array.from({ length: 128 }, (_, i) => 3 * i), target: 300 };
  const binaryCase = new Counter();
  const linearCase = new Counter();
  assert(binarySearch(probe, binaryCase) === 100 && linearSearch(probe, linearCase) === 100,
    "both searches locate index 100");
  assert(binaryCase.ops <= Math.ceil(Math.log2(128)) + 1 && binaryCase.ops * 10 < linearCase.ops,
    `binary probes ${binaryCase.ops} <= ceil(log2 n)+1, linear steps ${linearCase.ops}`);
  // 8. The harness catches an implementation whose measured cost drifts between runs.
  const jitter = verifyReproducible(JITTER_SUITE, { seed });
  const clean = verifyReproducible([EXERCISES[0]], { seed });
  assert(jitter.ok === false && jitter.detail.includes("js-jitter"), `drift named: ${jitter.detail}`);
  assert(clean.ok === true, `a clean suite is reproducible: ${clean.detail}`);
  // 9. Aggregation: medians collapse repetitions, plus the helper's own parity boundary.
  assert(stats.get("e-sum/js-loop/n=64").ops === 64 && stats.get("e-search/js-binary/n=128").samples === run.reps,
    "repeated measurements collapse to the closed form and keep their samples");
  assert(median([9, 1, 5]) === 5 && median([1, 2, 3, 4]) === 2.5, "median of odd and even sample sets");
  const separation = stats.get("e-search/js-linear/n=128").ops / stats.get("e-search/js-binary/n=128").ops;
  console.log(`benchmark: exercises=${EXERCISES.length} rows=${run.rows.length} totalOps=${run.totalOps} `
    + `digest=${run.digest} linear/binary(128)=${separation.toFixed(1)}x`);
  console.log("PASS");
}
module.exports = { splitmix64, fnv1a, stableStringify, digestOf, hex32, Counter, sumLinear, sumReduce,
  insertionSort, mergeSort, linearSearch, binarySearch, sumInput, searchInput, EXERCISES, runOnce, median,
  aggregate, firstRowDifference, verifyReproducible, JITTER_SUITE };
if (require.main === module) main();
