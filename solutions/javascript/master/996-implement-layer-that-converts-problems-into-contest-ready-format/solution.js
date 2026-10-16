/** Task 996 - Implement layer that converts problems into contest-ready format (time limits, scoring).
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Converts authored problems into a contest manifest: each problem gets a letter, a memory limit, a
 * time limit derived from a reference run (reference milliseconds times a safety factor, clamped into
 * the house range) and a points allocation over its subtests. Points use the largest-remainder method,
 * so they always add up to exactly 100 for a problem even when the naive percentage rounding would
 * not. The same manifest also drives the scoreboard, which ranks by score, then by penalty, then by id.
 *
 * What is simplified: no judge and no clock -- reference runtimes are inputs and a submission's score
 * and penalty are given. Verified: the time-limit formula at both clamp boundaries, exact 100-point
 * allocations for awkward weight splits, rejection of incomplete problems, letter assignment,
 * scoreboard ordering against an independent sort, and byte-stable manifests.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the file's only randomness source, so fixtures are byte-reproducible. */
function splitmix64(seed) {
  let s = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    s = BigInt.asUintN(64, s + 0x9e3779b97f4a7c15n);
    let z = BigInt.asUintN(64, (s ^ (s >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

const LIMITS = { minMs: 500, maxMs: 10000, factor: 3, memoryMb: 512 };
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const letterFor = (index) => (index < ALPHABET.length ? ALPHABET[index] : `P${index + 1}`);

/** Reference runtime times the safety factor, clamped so a flaky machine cannot set the limit. */
function timeLimitMs(refMs, { factor = LIMITS.factor, min = LIMITS.minMs, max = LIMITS.maxMs } = {}) {
  if (!(refMs > 0)) return null;
  return Math.min(max, Math.max(min, Math.ceil(refMs * factor)));
}

/**
 * Largest-remainder apportionment of 100 points: floors first, then the leftover points go to the
 * biggest fractional parts (ties by index), which is what makes the total exact.
 */
function allocatePoints(weights) {
  const total = weights.reduce((a, w) => a + w, 0);
  const exact = weights.map((w) => (w / total) * 100);
  const points = exact.map(Math.floor);
  let leftover = 100 - points.reduce((a, b) => a + b, 0);
  const order = exact.map((e, i) => [e - points[i], i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  for (let k = 0; k < leftover; k += 1) points[order[k % order.length][1]] += 1;
  return points;
}

/** Validation: a problem is contest-ready only with a title, constraints and calibrated runtimes. */
function validateProblem(problem) {
  const issues = [];
  if (!problem || !problem.title) issues.push({ id: problem && problem.id, reason: "missing-title" });
  if (!problem || !problem.constraints || Object.keys(problem.constraints).length === 0) {
    issues.push({ id: problem && problem.id, reason: "missing-constraints" });
  }
  if (!problem || !(problem.subtests || []).length) issues.push({ id: problem && problem.id, reason: "no-subtests" });
  if (!problem || !(problem.refMs > 0)) issues.push({ id: problem && problem.id, reason: "uncalibrated" });
  return { ok: issues.length === 0, issues };
}

/** Build the contest manifest. Problems are ordered by id before letters are assigned. */
function buildContest(problems, { mode = "partial" } = {}) {
  const rejected = [];
  const accepted = [];
  for (const problem of problems) {
    const check = validateProblem(problem);
    if (check.ok) accepted.push(problem); else rejected.push(...check.issues.map((i) => ({ ...i, reason: i.reason })));
  }
  const ordered = accepted.slice().sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const entries = ordered.map((problem, i) => {
    const weights = problem.subtests.map((s) => (s.weight > 0 ? s.weight : 1));
    const points = mode === "icpc" ? new Array(weights.length).fill(0) : allocatePoints(weights);
    return { id: problem.id, letter: letterFor(i), title: problem.title, mode,
      timeLimitMs: timeLimitMs(problem.refMs), memoryMb: LIMITS.memoryMb,
      subtests: problem.subtests.map((s, k) => ({ name: s.name, weight: weights[k], points: points[k] })),
      points: points.reduce((a, b) => a + b, 0), constraints: problem.constraints };
  });
  return { mode, entries, rejected, totalPoints: entries.reduce((a, e) => a + e.points, 0),
    digest: fnv1a(JSON.stringify(entries)) };
}

const PENALTY_PER_WRONG_MS = 20 * 60 * 1000;
/** Submission scoring: partial credit in partial mode, all-or-nothing with a 20-minute penalty in ICPC. */
function scoreSubmission(entry, submission) {
  let points = 0;
  entry.subtests.forEach((s, i) => {
    if (submission.passed[i]) points += entry.mode === "icpc" ? (s.points > 0 ? s.points : 0) : s.points;
  });
  const solved = submission.passed.every(Boolean);
  const icpcPoints = entry.mode === "icpc" ? (solved ? 1 : 0) : points;
  return { id: entry.id, score: icpcPoints, penaltyMs: submission.timeMs + submission.wrong * PENALTY_PER_WRONG_MS,
    solved };
}

/** Scoreboard ordering: score desc, penalty asc, id asc -- implemented without Array#sort. */
function insertionRank(results) {
  const better = (a, b) => (a.score !== b.score ? a.score > b.score : a.penaltyMs !== b.penaltyMs
    ? a.penaltyMs < b.penaltyMs : a.id < b.id);
  const out = [];
  for (const row of results) {
    let i = out.length;
    while (i > 0 && !better(out[i - 1], row)) { out[i] = out[i - 1]; i -= 1; }
    out[i] = row;
  }
  return out;
}
const rankScoreboard = (results) => results.slice()
  .sort((a, b) => b.score - a.score || a.penaltyMs - b.penaltyMs || (a.id < b.id ? -1 : 1));

const PROBLEMS = [
  { id: "p003", title: "Graph Walk", refMs: 60, constraints: { n: 1000 },
    subtests: [{ name: "s1", weight: 1 }, { name: "s2", weight: 1 }, { name: "s3", weight: 1 }] },
  { id: "p001", title: "Two Sum", refMs: 12, constraints: { n: 100 },
    subtests: [{ name: "s1", weight: 3 }, { name: "s2", weight: 1 }, { name: "s3", weight: 1 }] },
  { id: "p002", title: "Sorting", refMs: 9000, constraints: { n: 100000 },
    subtests: [{ name: "s1", weight: 5 }] },
];
/** Seeded problems with awkward weight splits so the apportionment remainder is always exercised. */
function seededProblems(seed, count) {
  const rng = splitmix64(seed);
  return Array.from({ length: count }, (_, i) => ({ id: `p${String(i).padStart(3, "0")}`,
    title: `Task ${i}`, refMs: 1 + rng.int(4000), constraints: { n: 10 + rng.int(100) },
    subtests: Array.from({ length: 1 + rng.int(6) }, (_, k) => ({ name: `s${k}`, weight: 1 + rng.int(7) })) }));
}

function main() {
  const contest = buildContest(PROBLEMS);
  const byId = new Map(contest.entries.map((e) => [e.id, e]));
  // 1. Happy path: ids sorted, letters assigned in order, limits from the formula, points summing to 100.
  assert.deepStrictEqual(contest.entries.map((e) => [e.letter, e.id]),
    [["A", "p001"], ["B", "p002"], ["C", "p003"]], "letters follow the id sort order");
  assert.deepStrictEqual(contest.entries.map((e) => e.timeLimitMs), [500, 10000, 500],
    "time limits clamp into the house range");
  assert.deepStrictEqual(contest.entries.map((e) => e.points), [100, 100, 100], "every problem is worth 100 points");
  // 2. Boundary: the clamp is inclusive at both ends, and a zero reference run is refused.
  assert.deepStrictEqual([timeLimitMs(LIMITS.minMs / LIMITS.factor), timeLimitMs(LIMITS.maxMs / LIMITS.factor),
    timeLimitMs(0), timeLimitMs(-5)], [500, 10000, null, null], "boundaries and invalid references");
  // 3. Apportionment: awkward splits still total 100, and a heavier subtest never earns fewer points.
  const awkward = allocatePoints([1, 1, 1]);
  assert.deepStrictEqual([awkward.reduce((a, b) => a + b, 0), Math.max(...awkward) - Math.min(...awkward)],
    [100, 1], "three equal subtests split 100 within one point");
  const lopsided = allocatePoints([7, 1, 1, 1, 1, 1]);
  assert.strictEqual(lopsided.reduce((a, b) => a + b, 0), 100, "lopsided split still totals 100");
  assert.ok(lopsided[0] > lopsided[1] && lopsided.slice(1).every((p) => p >= 1), "weights order the points");
  assert.notStrictEqual([1, 1, 1].map(() => Math.round(100 / 3)).reduce((a, b) => a + b, 0), 100,
    "naive rounding would miss the total, which is why the remainder method is used");
  // 4. Negative: problems without constraints, without subtests or without calibration are refused.
  const partial = buildContest([...PROBLEMS, { id: "bad1", title: "No Constraints", refMs: 10, subtests: [] },
    { id: "bad2", title: "Uncalibrated", refMs: 0, constraints: { n: 5 }, subtests: [{ name: "s", weight: 1 }] }]);
  assert.deepStrictEqual(partial.rejected.map((r) => r.reason).sort(),
    ["missing-constraints", "no-subtests", "uncalibrated"], "rejection reasons are itemised");
  assert.strictEqual(partial.entries.length, PROBLEMS.length, "the good problems still make it into the contest");
  // 5. Scoring: partial credit adds subtest points, ICPC scores a problem only when fully solved.
  const graph = byId.get("p003");
  assert.deepStrictEqual([scoreSubmission(graph, { passed: [true, false, true], timeMs: 1000, wrong: 0 }).score,
    scoreSubmission(graph, { passed: [true, true, true], timeMs: 1000, wrong: 2 }).penaltyMs],
    [67, 1000 + 2 * PENALTY_PER_WRONG_MS], "partial credit and wrong-attempt penalty");
  const icpc = buildContest(PROBLEMS, { mode: "icpc" });
  assert.deepStrictEqual([icpc.entries[0].points, scoreSubmission(icpc.entries[0], { passed: [true, false, false], timeMs: 5, wrong: 0 }).score,
    scoreSubmission(icpc.entries[0], { passed: [true, true, true], timeMs: 5, wrong: 0 }).score], [0, 0, 1],
    "ICPC mode awards one point only for a full solve");
  // 6. Scoreboard: ordered by score, then penalty, then id, and the oracle sort agrees.
  const results = [{ id: "p001", score: 100, penaltyMs: 500 }, { id: "p002", score: 100, penaltyMs: 400 },
    { id: "p003", score: 100, penaltyMs: 400 }, { id: "p004", score: 50, penaltyMs: 10 }];
  assert.deepStrictEqual(rankScoreboard(results).map((r) => r.id), ["p002", "p003", "p001", "p004"], "tie-breaking order");
  assert.deepStrictEqual(insertionRank(results).map((r) => r.id), rankScoreboard(results).map((r) => r.id),
    "independent insertion sort agrees with the scoreboard");
  // 7. Boundary: an empty problem list and a problem with a single subtest.
  assert.deepStrictEqual([buildContest([]).entries, buildContest([]).totalPoints], [[], 0], "empty contest");
  assert.deepStrictEqual([buildContest([PROBLEMS[2]]).entries[0].subtests[0].points, allocatePoints([5])], [100, [100]],
    "a single subtest takes all 100 points");
  // 8. Seeded batch: 12 problems stay valid, letters are unique and sequential, and manifests reproduce.
  const fixture = seededProblems(0x996, 12);
  const built = buildContest(fixture);
  assert.deepStrictEqual([built.rejected.length, built.entries.length, built.totalPoints], [0, 12, 1200],
    "seeded problems all convert");
  assert.deepStrictEqual(built.entries.map((e) => e.letter), ALPHABET.slice(0, 12).split(""), "letters are sequential");
  assert.ok(built.entries.every((e) => e.subtests.reduce((a, s) => a + s.points, 0) === 100), "each fixture problem sums to 100");
  assert.strictEqual(buildContest(fixture.slice().reverse()).digest, built.digest, "manifest is input-order independent");
  console.log(`problems=${built.entries.length} points=${built.totalPoints} letters=${built.entries[0].letter}-${built.entries[11].letter} digest=${built.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, letterFor, timeLimitMs, allocatePoints, validateProblem, buildContest,
  scoreSubmission, rankScoreboard, insertionRank, seededProblems, LIMITS, PENALTY_PER_WRONG_MS };
