/** Task 908 - Implement certified ML predictions with conformal prediction intervals.
 *  (Tier: Master, CTCI Beyond — Statistical Learning & Certification)
 *
 * Split conformal prediction: fit a least-squares line on the training split, score the calibration split
 * with the nonconformity residual |y - y_hat|, and take the order statistic q = s_(ceil((n+1)(1-alpha))) of
 * those scores. Under exchangeability the new point falls in [y_hat - q, y_hat + q] with probability at
 * least 1-alpha, and for continuous scores the *exact* coverage is ceil((n+1)(1-alpha))/(n+1) — a closed
 * form the checks use as an independent oracle instead of merely bounding the simulation.
 *
 * Also implemented: the Mondrian (group-conditional) variant, which conformalises each variance group
 * separately so a heteroscedastic model still covers the quiet group after grouping; the infinite-interval
 * regime alpha < 1/(n+1); efficiency against a model-free interval; monotonicity of q in alpha; and
 * reproducibility. Everything is seeded, so the coverage numbers are stable run to run.
 */
"use strict";

const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  float() { return this.next() / 4294967296; }
  /** Box-Muller standard normal. */
  normal() {
    const u = Math.max(this.float(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * this.float());
  }
}
/** Least-squares line y = intercept + slope * x. */
function fitLine(xs, ys) {
  const n = xs.length;
  assert.ok(n >= 2, "the training split needs at least two points");
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const denominator = n * sxx - sx * sx;
  assert.ok(Math.abs(denominator) > 1e-9, "the design matrix must be non-degenerate");
  const slope = (n * sxy - sx * sy) / denominator;
  const intercept = (sy - slope * sx) / n;
  return { intercept, slope, predict: (x) => intercept + slope * x };
}
/** Heteroscedastic sample: y = 1 + 2x + sigma(group) * N(0,1) with x uniform on [-2, 2]. */
function sample(prng, count, groups = null) {
  const points = [];
  for (let i = 0; i < count; i++) {
    const group = groups ? groups[i % groups.length] : { name: "all", sigma: 0.5 };
    points.push({ x: prng.float() * 4 - 2, y: 0, group: group.name, sigma: group.sigma });
  }
  for (const point of points) point.y = 1 + 2 * point.x + point.sigma * prng.normal();
  return points;
}
/** Split conformal: returns q (the radius) and the model. */
function conformalise(train, calibration, alpha) {
  const model = fitLine(train.map((p) => p.x), train.map((p) => p.y));
  const scores = calibration.map((p) => Math.abs(p.y - model.predict(p.x))).sort((a, b) => a - b);
  return { model, scores, q: conformalQuantile(scores, alpha), alpha };
}
/** The usual order statistic: index ceil((n+1)(1-alpha)); beyond the end the interval is unbounded. */
function conformalQuantile(sortedScores, alpha) {
  assert.ok(alpha > 0 && alpha < 1, `alpha must lie in (0,1), got ${alpha}`);
  const index = Math.ceil((sortedScores.length + 1) * (1 - alpha));
  if (index > sortedScores.length) return Infinity;
  return sortedScores[index - 1];
}
/** Mondrian conformal: one radius per group, so coverage is conditional on the group too. */
function conformaliseByGroup(train, calibration, alpha) {
  const model = fitLine(train.map((p) => p.x), train.map((p) => p.y));
  const scores = new Map();
  for (const point of calibration) {
    if (!scores.has(point.group)) scores.set(point.group, []);
    scores.get(point.group).push(Math.abs(point.y - model.predict(point.x)));
  }
  const radii = new Map();
  for (const [group, list] of scores) radii.set(group, conformalQuantile(list.sort((a, b) => a - b), alpha));
  return { model, radii, alpha };
}

const exactCoverage = (n, alpha) => Math.min(1, Math.ceil((n + 1) * (1 - alpha)) / (n + 1));
const within = (prediction, q, truth) => Math.abs(truth - prediction) <= q;
const slackFor = (trials, rate) => 3 * Math.sqrt((rate * (1 - rate)) / trials);

function main() {
  const alpha = 0.1;
  const calibrationSize = 99;
  const trials = 2000;
  // 1. coverage — the empirical rate matches the exact exchangeability value, not just the bound.
  {
    let covered = 0;
    for (let trial = 0; trial < trials; trial++) {
      const prng = new Prng(0x5eed0908 + trial * 7919);
      const train = sample(prng, 60);
      const calibration = sample(prng, calibrationSize);
      const test = sample(prng, 1)[0];
      const fitted = conformalise(train, calibration, alpha);
      if (within(fitted.model.predict(test.x), fitted.q, test.y)) covered += 1;
    }
    const empirical = covered / trials;
    const exact = exactCoverage(calibrationSize, alpha);
    assert.ok(exact >= 1 - alpha, `the exact coverage ${exact} must meet the nominal level`);
    assert.ok(Math.abs(empirical - exact) <= slackFor(trials, exact),
      `empirical coverage ${empirical.toFixed(4)} vs exact ${exact.toFixed(4)}`);
    assert.ok(empirical >= 1 - alpha - slackFor(trials, 1 - alpha), "the guarantee must hold empirically");
  }
  // 2. the guarantee holds across the whole alpha grid (with the matching q each time).
  {
    for (const level of [0.05, 0.1, 0.2, 0.3]) {
      let covered = 0;
      let width = 0;
      for (let trial = 0; trial < 400; trial++) {
        const prng = new Prng(0xabc + trial * 104729 + Math.round(level * 1000));
        const train = sample(prng, 60);
        const calibration = sample(prng, 199);
        const test = sample(prng, 1)[0];
        const fitted = conformalise(train, calibration, level);
        if (within(fitted.model.predict(test.x), fitted.q, test.y)) covered += 1;
        width += 2 * fitted.q;
      }
      const empirical = covered / 400;
      const exact = exactCoverage(199, level);
      assert.ok(empirical >= 1 - level - slackFor(400, 1 - level),
        `alpha=${level}: coverage ${empirical.toFixed(3)} below nominal`);
      assert.ok(Math.abs(empirical - exact) <= slackFor(400, exact), `alpha=${level}: matches the oracle`);
      assert.ok(Number.isFinite(width), "finite intervals on this grid");
    }
  }
  // 3. efficiency — conformalisation beats the model-free interval of the same nominal level.
  {
    const prng = new Prng(4242);
    const train = sample(prng, 200);
    const calibration = sample(prng, 399);
    const test = sample(prng, 2000);
    const fitted = conformalise(train, calibration, alpha);
    let covered = 0;
    let width = 0;
    for (const point of test) {
      if (within(fitted.model.predict(point.x), fitted.q, point.y)) covered += 1;
      width += 2 * fitted.q;
    }
    const ys = calibration.map((p) => p.y).sort((a, b) => a - b);
    const modelFree = 2 * conformalQuantile(ys, alpha);
    assert.ok(covered / test.length >= 1 - alpha - 0.03, "coverage must hold on the held-out sample");
    assert.ok(width / test.length < modelFree / 2, "the interval must be far tighter than the marginal one");
    assert.ok(fitted.q > 0 && fitted.q < 3, `radius ${fitted.q} must be physically plausible`);
  }
  // 4. Mondrian (group-conditional) coverage on heteroscedastic data.
  {
    const groups = [{ name: "quiet", sigma: 0.2 }, { name: "noisy", sigma: 1.0 }];
    const level = 0.1;
    let quietCovered = 0;
    let noisyCovered = 0;
    let quietWidth = 0;
    let pooledQuietWidth = 0;
    const repetitions = 400;
    for (let repetition = 0; repetition < repetitions; repetition++) {
      const prng = new Prng(0x77 + repetition * 31);
      const train = sample(prng, 120, groups);
      const calibration = sample(prng, 200, groups);
      const quiet = sample(prng, 1, [groups[0]])[0];
      const noisy = sample(prng, 1, [groups[1]])[0];
      const mondrian = conformaliseByGroup(train, calibration, level);
      const pooled = conformalise(train, calibration, level);
      if (within(mondrian.model.predict(quiet.x), mondrian.radii.get("quiet"), quiet.y)) quietCovered += 1;
      if (within(mondrian.model.predict(noisy.x), mondrian.radii.get("noisy"), noisy.y)) noisyCovered += 1;
      quietWidth += 2 * mondrian.radii.get("quiet");
      pooledQuietWidth += 2 * pooled.q;
    }
    const exact = exactCoverage(100, level);
    assert.ok(quietCovered / repetitions >= 1 - level - slackFor(repetitions, 1 - level),
      `quiet group coverage ${(quietCovered / repetitions).toFixed(3)} below nominal`);
    assert.ok(noisyCovered / repetitions >= 1 - level - slackFor(repetitions, 1 - level),
      `noisy group coverage ${(noisyCovered / repetitions).toFixed(3)} below nominal`);
    assert.ok(Math.abs(noisyCovered / repetitions - exact) <= slackFor(repetitions, exact) + 0.01,
      "the noisy group matches the exchangeability oracle");
    assert.ok(quietWidth < 0.6 * pooledQuietWidth,
      "grouping must shrink the quiet group's interval instead of pooling its radius");
  }
  // 5. monotonicity — q is non-increasing in alpha, so intervals shrink as the level loosens.
  {
    const prng = new Prng(31337);
    const train = sample(prng, 100);
    const calibration = sample(prng, 299);
    const radii = [0.01, 0.05, 0.1, 0.25, 0.5].map((level) => conformalise(train, calibration, level).q);
    for (let i = 1; i < radii.length; i++) {
      assert.ok(radii[i] <= radii[i - 1] + 1e-12, `q must not grow with alpha: ${radii}`);
    }
    assert.ok(radii[4] < radii[0], "the loosest level must be strictly narrower");
    const sortedScores = conformalise(train, calibration, 0.1).scores;
    for (let i = 1; i < sortedScores.length; i++) {
      assert.ok(sortedScores[i] >= sortedScores[i - 1], "scores must come back sorted");
    }
  }
  // 6. boundary — tiny calibration sets, the infinite-interval regime, and a perfect model.
  {
    const prng = new Prng(5);
    const train = sample(prng, 50);
    const one = sample(prng, 1);
    const tiny = conformalise(train, one, 0.5);
    const loneScore = Math.abs(one[0].y - tiny.model.predict(one[0].x));
    assert.ok(Math.abs(tiny.q - loneScore) < 1e-12, "with one calibration point q is that point's score");
    assert.strictEqual(exactCoverage(1, 0.5), 0.5, "one calibration point at alpha=0.5 guarantees only 1/2");
    assert.strictEqual(conformalQuantile([1, 2, 3], 0.001), Infinity, "beyond the last order statistic");
    const calm = sample(prng, 40).map((point) => ({ ...point, sigma: 0 }));
    const exact = calm.map((point) => ({ ...point, y: 1 + 2 * point.x }));
    const perfect = conformalise(exact.slice(0, 20), exact.slice(20), 0.1);
    assert.strictEqual(perfect.q, 0, "a perfect model gets a zero-radius interval");
    assert.ok(within(perfect.model.predict(0.7), perfect.q, 1 + 2 * 0.7), "and still covers");
    assert.throws(() => conformalQuantile([1], 0), /alpha must lie/);
    assert.throws(() => fitLine([1], [1]), /at least two points/);
    assert.throws(() => fitLine([1, 1, 1], [1, 2, 3]), /non-degenerate/);
    assert.strictEqual(exactCoverage(99, 0.005), 1, "alpha below 1/(n+1) forces the trivial interval");
  }
  // 7. determinism — identical seeds give identical intervals and coverage counts.
  {
    const run = () => {
      const prng = new Prng(0x5eed0908);
      const train = sample(prng, 60);
      const calibration = sample(prng, calibrationSize);
      const fitted = conformalise(train, calibration, alpha);
      return [fitted.model.intercept, fitted.model.slope, fitted.q];
    };
    assert.deepStrictEqual(run(), run());
    const other = new Prng(0x5eed0909);
    const otherFit = conformalise(sample(other, 60), sample(other, calibrationSize), alpha);
    assert.notDeepStrictEqual([otherFit.model.intercept, otherFit.model.slope, otherFit.q], run());
  }
  console.log(`conformal: n_cal=${calibrationSize}, alpha=${alpha}, exact coverage ` +
    `${exactCoverage(calibrationSize, alpha).toFixed(4)}, ${trials} trials matched the oracle ` +
    `within Monte-Carlo error; Mondrian radii split quiet/noisy groups`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  Prng, fitLine, sample, conformalise, conformalQuantile, conformaliseByGroup, exactCoverage,
};
