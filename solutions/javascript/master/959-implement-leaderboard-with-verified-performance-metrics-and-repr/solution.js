/** Task 959 - Implement leaderboard with verified performance metrics and reproducible runs.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A leaderboard that refuses unverifiable numbers. Every submitted run names a task, a solution code, a
 * language, a harness version and an input seed; the verifier re-executes that exact configuration
 * locally and compares both the result digest and the operation count. A run is rejected when the
 * digest does not reproduce, when the claimed ops differ from the reproduced ones, when the measured
 * time is below the physical floor implied by the ops and the language's throughput, when the harness
 * version or code is unknown, or when the same (task, code, language, seed) is submitted twice. Only
 * accepted runs enter the ranking, which scores each (task, code, language) by the median of its runs
 * and breaks ties by submission order; the top entries are then re-verified once more.
 *
 * Simplified: measurements are counted operations divided by a per-language throughput constant rather
 * than real timings, so the whole pipeline is deterministic and replayable. Verified: every rejection
 * category with its reason, accepted-run arithmetic, median semantics (odd and even), the ranking order
 * against a hand-computed order, re-verification of the podium, exclusion of rejected runs, the
 * empty/single/non-positive boundaries, and identical leaderboards on replay.
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
  compare(a, b) { this.step(); return a < b ? -1 : a > b ? 1 : 0; }
}
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
const digestOf = (value) => (fnv1a(JSON.stringify(value)) >>> 0).toString(16).padStart(8, "0");

const countingSort = (xs, ctr) => {
  const counts = new Array(1000).fill(0);
  for (const x of xs) { ctr.step(); counts[x] += 1; }
  const out = [];
  for (let v = 0; v < 1000; v += 1) for (let k = 0; k < counts[v]; k += 1) out.push(v);
  return out;
};
function mergeSort(xs, ctr) {
  if (xs.length < 2) return xs.slice();
  const mid = xs.length >> 1;
  const left = mergeSort(xs.slice(0, mid), ctr), right = mergeSort(xs.slice(mid), ctr);
  const out = [];
  let i = 0, j = 0;
  while (i < left.length && j < right.length) { ctr.step(); out.push(left[i] <= right[j] ? left[i++] : right[j++]); }
  return out.concat(left.slice(i)).concat(right.slice(j));
}
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
const CODES = { counting: countingSort, merge: mergeSort, insertion: insertionSort };
const OPS_PER_MS = { java: 200, javascript: 120, python: 40 };
const HARNESS_VERSION = 3;
const TASKS = { "sort-ints": { size: 256, input: (seed) => Array.from({ length: 256 }, (_, i) => (seed * 37 + i * 101) % 1000) } };

/** Re-execute the declared configuration: this is the only source of truth for digest and ops. */
function referenceRun(taskId, codeId, seed) {
  if (!TASKS[taskId]) throw new Error(`unknown task ${taskId}`);
  if (!CODES[codeId]) throw new Error(`unknown code ${codeId}`);
  const ctr = new Counter();
  const value = CODES[codeId](TASKS[taskId].input(seed), ctr);
  return { digest: digestOf(value), ops: ctr.ops };
}
const measuredMs = (ops, language) => Number((ops / OPS_PER_MS[language]).toFixed(4));
const median = (values) => {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : Number(((sorted[mid - 1] + sorted[mid]) / 2).toFixed(4));
};
/** Verdict for one run; `context.seen` makes duplicate detection order dependent but deterministic. */
function verifyRun(run, context) {
  const reject = (reason) => ({ ok: false, reason, runId: run.runId });
  if (run.harnessVersion !== HARNESS_VERSION) return reject(`unsupported harness version ${run.harnessVersion}`);
  if (!TASKS[run.task]) return reject(`unknown task ${run.task}`);
  if (!CODES[run.code]) return reject(`unknown code ${run.code}`);
  if (!OPS_PER_MS[run.language]) return reject(`unknown language ${run.language}`);
  if (!(run.ops > 0) || !(run.ms > 0)) return reject("non-positive measurement");
  const reference = referenceRun(run.task, run.code, run.seed);
  if (run.digest !== reference.digest) return reject("digest does not reproduce");
  if (run.ops !== reference.ops) return reject(`ops mismatch (claimed ${run.ops}, reproduced ${reference.ops})`);
  if (run.ms < measuredMs(run.ops, run.language) * 0.9) return reject("impossible throughput");
  const key = `${run.task}/${run.code}/${run.language}/${run.seed}`;
  if (context.seen.has(key)) return reject("duplicate run");
  context.seen.add(key);
  return { ok: true, runId: run.runId, key, digest: reference.digest, ops: reference.ops };
}
/** Leaderboard identity: one row per (task, code, language); the seed only deduplicates repeats. */
const boardKey = (run) => `${run.task}/${run.code}/${run.language}`;
/** Verify every run, rank the accepted ones, and re-verify the podium. */
function leaderboard(runs, { topN = 3 } = {}) {
  const context = { seen: new Set() };
  const verdicts = runs.map((run) => ({ run, ...verifyRun(run, context) }));
  const accepted = verdicts.filter((v) => v.ok);
  const byKey = new Map();
  for (const verdict of accepted) {
    const key = boardKey(verdict.run);
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key).push(verdict.run);
  }
  const entries = [...byKey.entries()].map(([key, list]) => {
    const ordered = list.slice().sort((a, b) => (a.submittedAt - b.submittedAt) || (a.runId < b.runId ? -1 : 1));
    return { key, task: ordered[0].task, code: ordered[0].code, language: ordered[0].language, runs: ordered.length,
      medianMs: median(ordered.map((r) => r.ms)), bestMs: Math.min(...ordered.map((r) => r.ms)), firstSubmittedAt: ordered[0].submittedAt };
  }).sort((a, b) => (a.medianMs - b.medianMs) || (a.firstSubmittedAt - b.firstSubmittedAt) || (a.key < b.key ? -1 : 1));
  const reverified = entries.slice(0, topN).map((entry) => {
    const source = accepted.find((v) => boardKey(v.run) === entry.key).run;
    const again = referenceRun(source.task, source.code, source.seed);
    return { key: entry.key, reproducible: again.digest === source.digest && again.ops === source.ops };
  });
  return { verdicts, entries: entries.map((entry, index) => ({ rank: index + 1, ...entry })), accepted: accepted.length,
    rejected: verdicts.filter((v) => !v.ok), reverified };
}

function main() {
  const rng = makeRng(20240959);
  /** A legitimate submission: digest and ops come from the same reference run the verifier uses. */
  const legit = (runId, code, language, seed, submittedAt) => {
    const reference = referenceRun("sort-ints", code, seed);
    return { runId, task: "sort-ints", code, language, seed, submittedAt, harnessVersion: HARNESS_VERSION,
      digest: reference.digest, ops: reference.ops, ms: measuredMs(reference.ops, language) };
  };
  const runs = [
    legit("r1", "counting", "java", 1, 10),
    legit("r2", "counting", "java", 2, 12),
    legit("r3", "counting", "java", 3, 14),
    legit("r4", "merge", "javascript", 1, 20),
    legit("r5", "insertion", "python", 1, 30),
    { ...legit("r6", "merge", "java", 4, 40), digest: "deadbeef" },
    { ...legit("r7", "counting", "java", 5, 50), ops: 4 },
    { ...legit("r8", "merge", "java", 6, 60), ms: 0.0001 },
    { runId: "r9", task: "sort-ints", code: "bogus", language: "java", seed: 7, submittedAt: 70,
      harnessVersion: HARNESS_VERSION, digest: "x", ops: 10, ms: 1 },
    { ...legit("r10", "counting", "java", 1, 80) },
    { ...legit("r11", "counting", "java", 9, 90), harnessVersion: HARNESS_VERSION - 1 },
  ];
  const board = leaderboard(runs);
  // 1. Happy path: five verifiable runs are accepted, six are rejected, and the totals add up.
  assert(board.accepted === 5 && board.rejected.length === 6 && board.verdicts.length === runs.length,
    `accepted=${board.accepted} rejected=${board.rejected.length}`);
  assert(board.entries.length === 3 && board.entries.every((e) => e.runs >= 1), "one entry per (task, code, language)");
  // 2. Every rejection category is reported with its own reason.
  const reasons = board.rejected.map((v) => v.reason);
  assert(reasons.some((r) => r === "digest does not reproduce"), `reasons ${reasons.join(" | ")}`);
  assert(reasons.some((r) => r.startsWith("ops mismatch")), "fabricated ops are caught");
  assert(reasons.includes("impossible throughput") && reasons.includes("duplicate run"), "speed floor and duplicates caught");
  assert(reasons.some((r) => r === "unknown code bogus") && reasons.some((r) => r.startsWith("unsupported harness version")),
    "unknown code and stale harness version caught");
  assert(board.rejected.some((v) => v.runId === "r7") && board.rejected.some((v) => v.runId === "r10"), "the offenders are named");
  // 3. Re-verification of the podium reproduces digest and ops for every top entry.
  assert(board.reverified.length === 3 && board.reverified.every((v) => v.reproducible),
    `podium re-verification ${JSON.stringify(board.reverified)}`);
  // 4. Independent order: the board must equal a ranking rebuilt from the accepted runs by hand.
  const acceptedRuns = board.verdicts.filter((v) => v.ok).map((v) => v.run);
  const byHand = [...new Set(acceptedRuns.map((r) => `${r.code}/${r.language}`))]
    .map((key) => ({ key, ms: median(acceptedRuns.filter((r) => `${r.code}/${r.language}` === key).map((r) => r.ms)) }))
    .sort((a, b) => a.ms - b.ms);
  assert(board.entries.map((e) => `${e.code}/${e.language}`).join(",") === byHand.map((e) => e.key).join(","),
    `ranking ${board.entries.map((e) => `${e.rank}:${e.code}/${e.language}=${e.medianMs}`).join(" ")}`);
  assert(board.entries.map((e) => e.medianMs).join(",") === byHand.map((e) => e.ms).join(","), "medians match the hand rebuild");
  assert(board.entries.every((e, i) => i === 0 || board.entries[i - 1].medianMs <= e.medianMs), "medians are ascending");
  // 5. Median semantics: odd and even samples, plus the closed form for the counting runs.
  assert(median([3, 1, 2]) === 2 && median([4, 1, 3, 2]) === 2.5 && median([7]) === 7, "median boundaries");
  const counting = board.entries[0];
  assert(counting.medianMs === measuredMs(referenceRun("sort-ints", "counting", 1).ops, "java"),
    `median ms ${counting.medianMs} equals ops / throughput`);
  assert(counting.bestMs === counting.medianMs, "identical runs share one median");
  // 6. Rejected runs never reach the board, and the seed actually drives the digest.
  const acceptedKeys = new Set(board.verdicts.filter((v) => v.ok).map((v) => boardKey(v.run)));
  assert(board.entries.every((e) => acceptedKeys.has(e.key)), "every entry traces back to an accepted run");
  const rejectedOnly = [...new Set(board.rejected.map((v) => boardKey(v.run)))].filter((k) => !acceptedKeys.has(k));
  assert(rejectedOnly.length > 0 && board.entries.every((e) => !rejectedOnly.includes(e.key)),
    `rejected-only keys get no row: ${rejectedOnly.join(",")}`);
  assert(referenceRun("sort-ints", "merge", 1).digest !== referenceRun("sort-ints", "merge", 2).digest, "the seed moves the digest");
  assert.deepStrictEqual(referenceRun("sort-ints", "merge", 1), referenceRun("sort-ints", "merge", 1), "re-execution is byte-identical");
  // 7. Boundaries: no runs, one run, and a non-positive measurement.
  const empty = leaderboard([]);
  assert(empty.entries.length === 0 && empty.accepted === 0 && empty.reverified.length === 0, "an empty board is valid");
  const single = leaderboard([legit("solo", "merge", "java", 11, 1)]);
  assert(single.entries.length === 1 && single.entries[0].rank === 1 && single.entries[0].medianMs === measuredMs(referenceRun("sort-ints", "merge", 11).ops, "java"),
    "a lone run ranks first");
  const zero = leaderboard([{ ...legit("zero", "merge", "java", 12, 1), ops: 0, ms: 0 }]);
  assert(zero.rejected.length === 1 && zero.rejected[0].reason === "non-positive measurement", "zero measurements are rejected");
  assert.throws(() => referenceRun("sort-ints", "bogus", 1), /unknown code bogus/, "the reference itself guards unknown codes");
  // 8. Determinism: the whole pipeline replays identically, and an extra seeded run verifies too.
  assert.deepStrictEqual(leaderboard(runs), board, "leaderboard replay");
  const extra = legit(`r${rng.int(1000)}`, "merge", "python", 20 + rng.int(5), 100);
  assert.strictEqual(leaderboard([...runs, extra]).accepted, board.accepted + 1, "a seeded extra run is accepted");
  console.log(`leaderboard: runs=${runs.length} accepted=${board.accepted} rejected=${board.rejected.length} `
    + `podium=${board.entries.map((e) => `${e.rank}:${e.code}/${e.language}@${e.medianMs}ms`).join(" ")}`);
  console.log("PASS");
}

module.exports = { makeRng, Counter, fnv1a, digestOf, countingSort, mergeSort, insertionSort, CODES,
  OPS_PER_MS, HARNESS_VERSION, TASKS, referenceRun, measuredMs, median, verifyRun, boardKey, leaderboard };
if (require.main === module) main();
