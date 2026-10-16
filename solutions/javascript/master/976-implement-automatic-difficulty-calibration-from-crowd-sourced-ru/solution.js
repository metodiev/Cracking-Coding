/** Task 976 - Implement automatic difficulty calibration from crowd-sourced runtimes.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Every problem accumulates crowd submissions: raw runtimes (milliseconds) plus a pass count. The
 * calibrator turns each batch into robust statistics - the log-normal geometric mean (the maximum
 * likelihood median), the dispersion sigma on the log scale and the arithmetic median - then folds
 * them into a difficulty score that grows with runtime and falls with the pass rate, and finally
 * maps the score onto an ordinal band. A Rasch-style offset is fitted by bisection so that the pass
 * probability predicted for a fixed skill population reproduces the observed pass rate.
 * Simplified: crowds are simulated as log-normal samples (no per-submission language/profile
 * weights) and one skill population is used for every problem. Verified: the statistics against a
 * two-pass oracle, monotone ordering against the latent truth, exact band thresholds, fitted
 * residuals cross-checked with a grid-search oracle, the extreme-pass-rate clamps, the single/empty
 * batch boundaries and determinism per seed.
 */
"use strict";
const assert = require("node:assert");

const REFERENCE_MS = 100;
const BANDS = [{ band: "easy", max: 2.2 }, { band: "medium", max: 4.2 }, { band: "hard", max: 6.2 },
  { band: "expert", max: Infinity }];
const SKILLS = [-2, -1, 0, 1, 2];
const TRUTH = [
  { id: "two-sum", medianMs: 120, passRate: 0.82 },
  { id: "lru-cache", medianMs: 260, passRate: 0.55 },
  { id: "median-streams", medianMs: 430, passRate: 0.38 },
  { id: "regex-dp", medianMs: 900, passRate: 0.16 },
];

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

/** Crowd submissions: log-normal runtimes around the latent median plus an exact pass count. */
function makeSubmissions(seed, problems = TRUTH, perProblem = 200) {
  const rng = splitmix64(seed);
  return problems.map((problem) => ({
    id: problem.id,
    total: perProblem,
    passed: Math.round(problem.passRate * perProblem),
    runtimes: Array.from({ length: perProblem },
      () => Math.max(1, Math.round(problem.medianMs * Math.exp((rng.unit() - 0.5) * 0.5)))),
    truth: problem,
  }));
}

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/** Log-normal fit: geometric mean (MLE of the median) plus the log-scale dispersion. */
function runtimeStats(runtimes) {
  assert(runtimes.length > 0, "a calibration needs at least one runtime");
  const logs = runtimes.map((ms) => Math.log(ms));
  const meanLog = logs.reduce((sum, value) => sum + value, 0) / logs.length;
  const variance = logs.reduce((sum, value) => sum + (value - meanLog) ** 2, 0) / logs.length;
  return { n: runtimes.length, geometricMeanMs: Math.exp(meanLog), medianMs: median(runtimes),
    sigma: Math.sqrt(variance), maxMs: Math.max(...runtimes) };
}

/** Difficulty score: 1.5x the log-2 runtime ratio plus three times the failure rate. */
const difficultyScore = ({ medianMs, passRate }) =>
  1.5 * Math.log2(medianMs / REFERENCE_MS) + 3 * (1 - passRate);
const bandFor = (score) => (BANDS.find((row) => score < row.max) || BANDS.at(-1)).band;

/** Rasch pass probability for the fixed skill population; decreasing in difficulty. */
const predictedPass = (difficulty) =>
  SKILLS.reduce((sum, skill) => sum + 1 / (1 + Math.exp(difficulty - skill)), 0) / SKILLS.length;

/** Bisection: the difficulty whose predicted pass rate equals the observed one. */
function fitDifficulty(targetPass, { low = -8, high = 8, tolerance = 1e-10, maxIterations = 100 } = {}) {
  assert(targetPass >= 0 && targetPass <= 1, `pass rate ${targetPass} must lie in [0,1]`);
  let lo = low;
  let hi = high;
  for (let i = 0; i < maxIterations && hi - lo > tolerance; i += 1) {
    const mid = (lo + hi) / 2;
    if (predictedPass(mid) > targetPass) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** Independent oracle: brute-force scan of the same equation on a coarse grid. */
function gridFit(targetPass, step = 0.001) {
  let best = { difficulty: 0, error: Infinity };
  for (let difficulty = -8; difficulty <= 8; difficulty += step) {
    const error = Math.abs(predictedPass(difficulty) - targetPass);
    if (error < best.error) best = { difficulty, error };
  }
  return best;
}

function calibrate(submissions, options = {}) {
  return submissions.map((batch) => {
    const stats = runtimeStats(batch.runtimes);
    const passRate = batch.passed / batch.total;
    const score = difficultyScore({ medianMs: stats.geometricMeanMs, passRate });
    return { id: batch.id, n: stats.n, medianMs: stats.medianMs, geometricMeanMs: stats.geometricMeanMs,
      sigma: stats.sigma, passRate, score, band: bandFor(score),
      rasch: fitDifficulty(passRate, options), truth: batch.truth };
  });
}

function main() {
  const batches = makeSubmissions(976);
  const table = calibrate(batches);

  // 1. Happy path: the four problems land in four different bands, hardest last.
  assert.strictEqual(table.length, TRUTH.length, "every problem is calibrated");
  assert.deepStrictEqual(table.map((row) => row.band), ["easy", "medium", "hard", "expert"],
    `bands: ${table.map((row) => row.band)}`);
  assert(table.every((row) => row.n === 200 && row.sigma > 0), "each batch has 200 noisy runtimes");
  assert(Math.abs(table[0].passRate - 0.82) < 0.01, "pass rates come straight from the counts");

  // 2. Oracle: the log-normal statistics recomputed with a naive two-pass implementation.
  for (const row of table) {
    const runtimes = batches.find((batch) => batch.id === row.id).runtimes;
    const logs = runtimes.map((ms) => Math.log(ms));
    const meanLog = logs.reduce((a, b) => a + b, 0) / logs.length;
    const spread = Math.sqrt(logs.reduce((sum, l) => sum + (l - meanLog) * (l - meanLog), 0) / logs.length);
    assert(Math.abs(row.geometricMeanMs - Math.exp(meanLog)) < 1e-9, `${row.id} geometric mean matches`);
    assert(Math.abs(row.sigma - spread) < 1e-9, `${row.id} sigma matches the two-pass spread`);
    assert.strictEqual(row.medianMs, median([...runtimes]), `${row.id} median matches the sort`);
    assert(row.sigma < 0.3, `${row.id} dispersion stays small: ${row.sigma.toFixed(4)}`);
  }

  // 3. Monotone ordering: score and band both track the latent difficulty.
  for (let i = 1; i < table.length; i += 1) {
    assert(table[i].score > table[i - 1].score, `score rises: ${table[i - 1].score} -> ${table[i].score}`);
    assert(table[i].medianMs > table[i - 1].medianMs, "runtimes rise with difficulty");
    assert(table[i].passRate < table[i - 1].passRate, "pass rates fall with difficulty");
  }
  const easy = difficultyScore({ medianMs: REFERENCE_MS, passRate: 0.9 });
  const hard = difficultyScore({ medianMs: REFERENCE_MS * 8, passRate: 0.6 });
  assert(hard > easy, "slower and less-solved is harder");

  // 4. Fitted offset: residual is tiny and agrees with the coarse grid-search oracle.
  for (const row of table) {
    assert(Math.abs(predictedPass(row.rasch) - row.passRate) < 1e-6, `${row.id} residual`);
    const grid = gridFit(row.passRate);
    assert(Math.abs(grid.difficulty - row.rasch) <= 0.002, `${row.id} matches the grid oracle: ${row.rasch}`);
    assert(grid.error < 0.01, "the grid oracle lands on the same root");
  }
  assert(table[0].rasch < table[3].rasch, "the offset ordering follows the pass rates");

  // 5. Boundaries: band thresholds are exclusive, extreme pass rates clamp at the search ends.
  assert.strictEqual(bandFor(0), "easy", "zero is the easiest band");
  assert.strictEqual(bandFor(2.2), "medium", "the threshold itself moves up a band");
  assert.strictEqual(bandFor(4.2), "hard", "second threshold is exclusive");
  assert.strictEqual(bandFor(6.2), "expert", "third threshold is exclusive");
  assert(bandFor(-5) === "easy" && bandFor(99) === "expert", "the extremes saturate");
  assert(fitDifficulty(1) <= -7.9, "a problem nobody fails saturates the negative end");
  assert(fitDifficulty(0) >= 7.9, "a problem nobody solves saturates the positive end");
  assert(Math.abs(fitDifficulty(0.5)) < 1e-9, "the median skill defines difficulty zero");
  assert.throws(() => fitDifficulty(1.2), /must lie in/, "impossible pass rates are rejected");

  // 6. Degenerate batches: one submission, identical runtimes, and the empty batch.
  const single = runtimeStats([250]);
  assert.deepStrictEqual([single.n, single.medianMs, single.sigma], [1, 250, 0], "a single submission is sigma-free");
  const flat = runtimeStats([300, 300, 300, 300]);
  assert.strictEqual(flat.sigma, 0, "identical runtimes have no dispersion");
  assert(Math.abs(flat.geometricMeanMs - 300) < 1e-9, "identical runtimes keep the geometric mean");
  assert.strictEqual(difficultyScore({ medianMs: flat.geometricMeanMs, passRate: 1 }).toFixed(4),
    (1.5 * Math.log2(3)).toFixed(4), "zero-failure score is the runtime term alone");
  assert.throws(() => runtimeStats([]), /at least one runtime/, "an empty batch is rejected");
  assert.throws(() => calibrate([{ id: "x", total: 10, passed: 5, runtimes: [] }]), /at least one runtime/,
    "calibration propagates the empty-batch error");

  // 7. Invariant: sorting by score yields non-decreasing band ranks, whatever the noise.
  const ranks = { easy: 0, medium: 1, hard: 2, expert: 3 };
  const sorted = [...table].sort((a, b) => a.score - b.score);
  for (let i = 1; i < sorted.length; i += 1) {
    assert(ranks[sorted[i].band] >= ranks[sorted[i - 1].band], "bands follow the score order");
  }
  assert(sorted.every((row) => bandFor(row.score) === row.band), "the band is a pure function of the score");

  // 8. Determinism: the seed fixes the crowd, so the whole table is reproducible.
  const again = calibrate(makeSubmissions(976));
  assert(JSON.stringify(again) === JSON.stringify(table), "the calibration table is reproducible");
  const other = calibrate(makeSubmissions(977));
  assert(other.map((row) => row.band).join() === table.map((row) => row.band).join(), "bands are seed-stable");
  assert(other.some((row, i) => row.sigma !== table[i].sigma), "the crowd itself changes with the seed");
  assert.strictEqual(makeSubmissions(976)[0].runtimes.length, 200, "batch size is fixed");

  console.log(`calibration: ${table.length} problems, bands ${table.map((row) => row.band).join("/")}, ` +
    `scores ${table.map((row) => row.score.toFixed(2)).join(" < ")}, offsets ` +
    `${table.map((row) => row.rasch.toFixed(2)).join(" < ")}`);
  console.log("PASS");
}
module.exports = { REFERENCE_MS, BANDS, SKILLS, TRUTH, splitmix64, makeSubmissions, median, runtimeStats,
  difficultyScore, bandFor, predictedPass, fitDifficulty, gridFit, calibrate };
if (require.main === module) main();
