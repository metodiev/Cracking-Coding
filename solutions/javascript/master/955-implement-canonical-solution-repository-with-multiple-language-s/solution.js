/** Task 955 - Implement canonical solution repository with multiple language solutions and performance baselines.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * The canonical registry model: every task lists one translation per language (four required), and the
 * audit does three jobs. (1) Coverage -- a task missing a translation is reported instead of shipping a
 * lopsided problem. (2) Differential correctness -- all translations of a task must produce the same
 * canonical answer on the hand-written samples and on seeded random inputs, so the digest of a task is
 * language independent. (3) Performance baselines -- each language has a constant-factor multiplier over
 * the reference translation's counted operations, and a translation that blows its budget is a
 * reported drift instead of a silent slowdown.
 *
 * Simplified: a "language solution" is an independent in-process implementation rather than a compiled
 * source file, and cost is counted steps rather than nanoseconds. Verified: full coverage, a missing
 * translation reported by name, cross-translation agreement incl. an injected divergence, baseline
 * scaling arithmetic (Java 1x, Python 3x the reference), the drift verdict for a quadratic translation,
 * closed-form operation counts for the linear implementations, empty/degenerate samples, digest
 * independence from listing order, and deterministic re-audits.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
class Counter {
  constructor() { this.ops = 0; }
  step(n = 1) { this.ops += n; return this; }
}
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
function stableStringify(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(stableStringify).join(",") + "]";
  return "{" + Object.keys(value).sort().map((k) => JSON.stringify(k) + ":" + stableStringify(value[k])).join(",") + "}";
}
const digestOf = (value) => (fnv1a(stableStringify(value)) >>> 0).toString(16).padStart(8, "0");
const runOps = (impl, input) => { const ctr = new Counter(); const value = impl(input, ctr); return { value, ops: ctr.ops }; };

// --- task "max-subarray": four translations, one of them deliberately quadratic -------------------
const kadaneFor = (xs, ctr) => {
  if (xs.length === 0) return 0;
  let best = xs[0], run = xs[0];
  for (let i = 1; i < xs.length; i += 1) { ctr.step(); run = Math.max(xs[i], run + xs[i]); best = Math.max(best, run); }
  return best;
};
const kadaneWhile = (xs, ctr) => {
  if (xs.length === 0) return 0;
  let best = xs[0], run = xs[0], i = 1;
  while (i < xs.length) { ctr.step(); run = run + xs[i] > xs[i] ? run + xs[i] : xs[i]; best = run > best ? run : best; i += 1; }
  return best;
};
/** Divide and conquer: the best sum is left, right, or a crossing window -> Theta(n log n). */
function divideSubarray(xs, ctr) {
  if (xs.length === 0) return 0;
  const solve = (lo, hi) => {
    if (hi - lo === 1) { ctr.step(); return { best: xs[lo], left: xs[lo], right: xs[lo], total: xs[lo] }; }
    const mid = (lo + hi) >> 1;
    const a = solve(lo, mid), b = solve(mid, hi);
    ctr.step(2);
    return { best: Math.max(a.best, b.best, a.right + b.left), left: Math.max(a.left, a.total + b.left),
      right: Math.max(b.right, b.total + a.right), total: a.total + b.total };
  };
  return solve(0, xs.length).best;
}
const quadraticSubarray = (xs, ctr) => {
  let best = xs.length ? -Infinity : 0;
  for (let i = 0; i < xs.length; i += 1) {
    let sum = 0;
    for (let j = i; j < xs.length; j += 1) { ctr.step(); sum += xs[j]; if (sum > best) best = sum; }
  }
  return best;
};
// --- task "sort-ints": four translations over values in [0, 1000] ---------------------------------
function insertionSort(xs, ctr) {
  const a = xs.slice();
  for (let i = 1; i < a.length; i += 1) {
    const key = a[i];
    let j = i - 1;
    while (j >= 0) { ctr.step(); if (a[j] <= key) break; a[j + 1] = a[j]; j -= 1; }
    a[j + 1] = key;
  }
  return a;
}
function mergeSort(xs, ctr) {
  if (xs.length < 2) return xs.slice();
  const mid = xs.length >> 1;
  const left = mergeSort(xs.slice(0, mid), ctr), right = mergeSort(xs.slice(mid), ctr);
  const out = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) { ctr.step(); out.push(left[i] <= right[j] ? left[i++] : right[j++]); }
  return out.concat(left.slice(i)).concat(right.slice(j));
}
const countingSort = (xs, ctr) => {
  const counts = new Array(1001).fill(0);
  for (const x of xs) { ctr.step(); counts[x] += 1; }
  const out = [];
  for (let v = 0; v <= 1000; v += 1) for (let k = 0; k < counts[v]; k += 1) out.push(v);
  return out;
};
const selectionSort = (xs, ctr) => {
  const a = xs.slice();
  for (let i = 0; i < a.length; i += 1) {
    let min = i;
    for (let j = i + 1; j < a.length; j += 1) { ctr.step(); if (a[j] < a[min]) min = j; }
    [a[i], a[min]] = [a[min], a[i]];
  }
  return a;
};

const LANGUAGES = ["python", "java", "javascript", "csharp"];
const LANGUAGE_MULTIPLIER = { python: 3, java: 1, javascript: 1.5, csharp: 1.1 };
const TOLERANCE = 0.35;
const ints = (size, rng, span = 2000) => Array.from({ length: size }, () => (rng.int(span) - span / 2));
const REGISTRY = [
  { id: "max-subarray", reference: "python", samples: [[-2, 1, -3, 4, -1, 2, 1, -5, 4], [1], [-5], [], [3, -1, 3]],
    makeInput: (size, rng) => ints(size, rng),
    translations: { python: kadaneFor, java: kadaneWhile, javascript: divideSubarray, csharp: quadraticSubarray } },
  { id: "sort-ints", reference: "javascript", samples: [[5, 3, 3, 1], [], [7], [0, 1000, 999, 0]],
    makeInput: (size, rng) => Array.from({ length: size }, () => rng.int(1001)),
    translations: { python: insertionSort, java: mergeSort, javascript: countingSort, csharp: selectionSort } },
];

/** Reference operations at `size`, scaled by the per-language constant factor -> the stored baselines. */
function baselines(task, size, seed) {
  const input = task.makeInput(size, makeRng((seed ^ fnv1a(task.id)) >>> 0));
  const { ops } = runOps(task.translations[task.reference], input);
  return Object.fromEntries(LANGUAGES.map((lang) => [lang, { ops: Math.ceil(ops * LANGUAGE_MULTIPLIER[lang]), tolerance: TOLERANCE }]));
}
/** Audit one task: coverage, cross-translation agreement, and measured ops against the baselines. */
function auditTask(task, { size = 256, seed = 20240955 } = {}) {
  const table = baselines(task, size, seed);
  const missing = LANGUAGES.filter((lang) => !task.translations[lang]).map((lang) => `${task.id}:${lang}`);
  const cases = [...task.samples.map((xs) => ({ kind: "sample", xs })), ...[0, 1, 5, 17].map((n) => ({ kind: `n=${n}`, xs: task.makeInput(n, makeRng(n)) }))];
  const divergences = [];
  let agreements = 0;
  for (const [index, item] of cases.entries()) {
    const answers = LANGUAGES.filter((l) => task.translations[l])
      .map((lang) => ({ lang, ...runOps(task.translations[lang], item.xs) }));
    const first = answers[0];
    for (const answer of answers) {
      if (stableStringify(answer.value) === stableStringify(first.value)) agreements += 1;
      else divergences.push({ task: task.id, case: index, lang: answer.lang, expected: first.value, got: answer.value });
    }
  }
  const input = task.makeInput(size, makeRng((seed ^ fnv1a(task.id)) >>> 0));
  const measured = Object.fromEntries(LANGUAGES.filter((l) => task.translations[l])
    .map((lang) => [lang, runOps(task.translations[lang], input).ops]));
  const drift = Object.entries(measured).filter(([lang, ops]) => ops > table[lang].ops * (1 + table[lang].tolerance))
    .map(([lang, ops]) => ({ task: task.id, lang, ops, baseline: table[lang].ops, drift: Number((ops / table[lang].ops - 1).toFixed(3)) }));
  const digest = digestOf({ task: task.id, answers: cases.map((item) => runOps(task.translations[task.reference], item.xs).value) });
  return { id: task.id, missing, divergences, drift, table, measured, agreements, expectedAgreements: cases.length * 4, digest };
}
/** Whole-registry audit; drift is expected for cheap-vs-expensive translations, so it is reported. */
function auditRegistry(registry, options) {
  const tasks = registry.map((task) => auditTask(task, options));
  return { tasks, missing: tasks.flatMap((t) => t.missing), divergences: tasks.flatMap((t) => t.divergences),
    drift: tasks.flatMap((t) => t.drift), digest: digestOf(tasks.map((t) => [t.id, t.digest]).sort((a, b) => (a[0] < b[0] ? -1 : 1))),
    translations: registry.reduce((a, t) => a + Object.keys(t.translations).length, 0) };
}

function main() {
  const good = auditRegistry(REGISTRY);
  // 1. Coverage: every task ships all four translations, 8 in total, and nothing is reported missing.
  assert(good.translations === 8 && good.missing.length === 0, `coverage missing=${good.missing.join(",")}`);
  assert.deepStrictEqual(good.tasks.map((t) => t.id), ["max-subarray", "sort-ints"], "task order is the registry order");
  // 2. Differential agreement: 5 samples + 4 generated sizes per task, all four translations each.
  assert(good.tasks.every((t) => t.agreements === t.expectedAgreements), `agreements ${good.tasks.map((t) => t.agreements)}`);
  assert(good.divergences.length === 0, `divergences ${JSON.stringify(good.divergences)}`);
  // 3. Negative: a task that drops a translation is reported by name, and a wrong answer is a divergence.
  const incomplete = [REGISTRY[0], { ...REGISTRY[1], translations: { ...REGISTRY[1].translations, csharp: undefined } }];
  const trimmed = auditRegistry(incomplete);
  assert.deepStrictEqual(trimmed.missing, ["sort-ints:csharp"], `missing ${trimmed.missing.join(",")}`);
  const broken = [{ ...REGISTRY[0], translations: { ...REGISTRY[0].translations, java: (xs) => (xs.length === 0 ? 1 : kadaneWhile(xs, new Counter())) } }];
  const caught = auditRegistry(broken);
  assert(caught.divergences.some((d) => d.task === "max-subarray" && d.case === 3 && d.lang === "java"),
    `the empty sample divergence is caught: ${JSON.stringify(caught.divergences)}`);
  // 4. Baselines: Java is 1x the reference, Python 3x, JavaScript 1.5x -- by independent arithmetic.
  const reference = good.tasks[0];
  assert(reference.table.java.ops === reference.measured.python && reference.table.python.ops === 3 * reference.measured.python,
    `baseline scaling java=${reference.table.java.ops} python=${reference.table.python.ops}`);
  // 5. Drift verdict: the quadratic translation is flagged, the reference translation is not.
  const driftKeys = good.drift.map((d) => `${d.task}:${d.lang}`);
  assert(driftKeys.includes("max-subarray:csharp") && driftKeys.includes("sort-ints:java"), `drift ${driftKeys.join(",")}`);
  assert(!driftKeys.includes("max-subarray:python") && !driftKeys.includes("sort-ints:javascript"), "references stay in budget");
  assert(good.drift.every((d) => d.drift > 0.35), "every reported drift exceeds the tolerance");
  // 6. Closed-form operation counts for the linear implementations (oracle, not self-reference).
  assert([[0, 0], [1, 0], [7, 6], [64, 63]].every(([n, expected]) => runOps(kadaneFor, ints(n, makeRng(n))).ops === expected),
    "kadane costs n-1 steps");
  assert(runOps(countingSort, Array.from({ length: 40 }, (_, i) => i)).ops === 40, "counting sort costs n steps");
  assert(runOps(kadaneFor, ints(64, makeRng(9))).ops > runOps(kadaneFor, ints(8, makeRng(9))).ops, "cost is monotone in n");
  assert(runOps(quadraticSubarray, ints(64, makeRng(1))).ops === (64 * 65) / 2, "the quadratic translation costs n(n+1)/2");
  // 7. Boundaries: empty and single-element samples agree across translations and tasks.
  assert(REGISTRY.every((task) => task.samples.some((s) => s.length === 0) && task.samples.some((s) => s.length === 1)),
    "every task carries an empty and a single-element sample");
  assert.deepStrictEqual(runOps(divideSubarray, []).value, 0, "empty max-subarray is 0");
  assert.deepStrictEqual(runOps(mergeSort, []).value, [], "empty sort returns an empty array");
  assert.deepStrictEqual(runOps(selectionSort, [7]).value, [7], "a single element is already sorted");
  // 8. Independent oracle + digest stability: brute force agrees with the reference, order is irrelevant.
  const rng = makeRng(20240955);
  let oracleAgreements = 0;
  for (let trial = 0; trial < 200; trial += 1) {
    const xs = ints(rng.int(12), rng);
    if (runOps(kadaneFor, xs).value === runOps(quadraticSubarray, xs).value) oracleAgreements += 1;
  }
  assert.strictEqual(oracleAgreements, 200, `oracle disagreements ${200 - oracleAgreements}`);
  const reordered = [{ ...REGISTRY[1], translations: { csharp: selectionSort, javascript: countingSort, java: mergeSort, python: insertionSort } }, REGISTRY[0]];
  assert.strictEqual(auditRegistry(reordered).digest, good.digest, "the registry digest ignores listing order");
  console.log(`canonical-repo: tasks=${good.tasks.length} translations=${good.translations} drift=${driftKeys.join(",")} `
    + `digest=${good.digest}`);
  console.log("PASS");
}

module.exports = { makeRng, Counter, fnv1a, stableStringify, digestOf, runOps, kadaneFor, kadaneWhile,
  divideSubarray, quadraticSubarray, insertionSort, mergeSort, countingSort, selectionSort, LANGUAGES,
  LANGUAGE_MULTIPLIER, TOLERANCE, ints, REGISTRY, baselines, auditTask, auditRegistry };
if (require.main === module) main();
