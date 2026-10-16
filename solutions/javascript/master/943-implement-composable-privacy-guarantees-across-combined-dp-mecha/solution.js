/** Task 943 - Implement composable privacy guarantees across combined DP mechanisms.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A privacy accountant. Each mechanism contributes a guarantee in its own form: the Laplace
 * mechanism a pure epsilon (noise scale = sensitivity / epsilon), randomized response an epsilon
 * derived from its flip probability, and the Gaussian mechanism a zCDP/Renyi parameter rho which is
 * converted to (epsilon, delta) by minimising alpha * rho + log(1/delta) / (alpha - 1) over integer
 * Renyi orders. Composition then differs by structure: sequential mechanisms add their epsilons,
 * parallel mechanisms over disjoint partitions only add them for the individual present in one
 * partition, and advanced composition trades a delta for a much smaller total.
 *
 * Simplified: three canonical mechanisms, no exponential mechanism and no subsampling, and the
 * empirical side samples Laplace noise instead of running a real query engine. Verified: the exact
 * likelihood-ratio bound of a mechanism in closed form and against sampled outputs, sequential,
 * parallel and advanced composition, monotonicity, the Renyi-to-(epsilon, delta) conversion against
 * the classical Gaussian bound, empty and invalid inputs, and determinism.
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

function validate(epsilon, delta) {
  if (!(epsilon >= 0) || !(delta >= 0) || delta >= 1) throw new Error(`invalid privacy parameters (${epsilon}, ${delta})`);
}

const laplaceScale = (sensitivity, epsilon) => sensitivity / epsilon;
/** Log density ratio of a Laplace mechanism on neighbouring databases whose means differ by `shift`. */
const laplaceLogRatio = (x, shift, scale) => (Math.abs(x - shift) - Math.abs(x)) / scale;
/** Randomized response reports the truth with probability p, so epsilon = ln(p / (1 - p)). */
const randomizedResponseEpsilon = (p) => Math.log(p / (1 - p));
/** zCDP of a Gaussian mechanism, and the sigma that realises a given rho. */
const gaussianRho = (sensitivity, sigma) => (sensitivity * sensitivity) / (2 * sigma * sigma);
const gaussianSigma = (sensitivity, rho) => sensitivity / Math.sqrt(2 * rho);
/** Classical Gaussian mechanism bound, used as the independent oracle for the Renyi conversion. */
const classicalGaussianEpsilon = (sensitivity, sigma, delta) => Math.sqrt(2 * Math.log(1.25 / delta)) * (sensitivity / sigma);

/** Renyi DP to (epsilon, delta): minimise alpha * rho + log(1 / delta) / (alpha - 1). */
function rdpToDp(rho, delta, maxOrder = 256) {
  if (!(rho >= 0) || !(delta > 0) || delta >= 1) throw new Error("invalid Renyi parameters");
  let epsilon = Infinity;
  let order = 2;
  for (let alpha = 2; alpha <= maxOrder; alpha += 1) {
    const value = alpha * rho + Math.log(1 / delta) / (alpha - 1);
    if (value < epsilon) { epsilon = value; order = alpha; }
  }
  return { epsilon, order };
}

/** Sequential composition: epsilons add, deltas add. */
function composeSequential(mechanisms) {
  return mechanisms.reduce((acc, m) => {
    validate(m.epsilon, m.delta ?? 0);
    return { epsilon: acc.epsilon + m.epsilon, delta: acc.delta + (m.delta ?? 0) };
  }, { epsilon: 0, delta: 0 });
}

/** Parallel composition over disjoint partitions: the individual sits in at most one of them. */
function composeParallel(mechanisms) {
  return mechanisms.reduce((acc, m) => {
    validate(m.epsilon, m.delta ?? 0);
    return { epsilon: Math.max(acc.epsilon, m.epsilon), delta: Math.max(acc.delta, m.delta ?? 0) };
  }, { epsilon: 0, delta: 0 });
}

/** Advanced composition of pure mechanisms: sum e*(e^e - 1) plus a Gaussian tail term. */
function composeAdvanced(mechanisms, delta) {
  if (!(delta > 0) || delta >= 1) throw new Error("advanced composition needs 0 < delta < 1");
  const epsilons = mechanisms.map((m) => {
    validate(m.epsilon, m.delta ?? 0);
    return m.epsilon;
  });
  const expansion = epsilons.reduce((a, e) => a + e * (Math.exp(Math.min(50, e)) - 1), 0);
  const squares = epsilons.reduce((a, e) => a + e * e, 0);
  const tail = Math.sqrt(2 * Math.log(1 / delta) * squares);
  return { epsilon: expansion + tail, delta, tail, expansion };
}

/** Laplace sampling by inverse transform of the CDF, and the CDF itself for the oracle. */
function sampleLaplace(rng, mean, scale) {
  const u = unit(rng) - 0.5;
  return mean - scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
}
const laplaceCdf = (x, mean, scale) => (x < mean ? 0.5 * Math.exp((x - mean) / scale) : 1 - 0.5 * Math.exp(-(x - mean) / scale));

function main() {
  // 1. Pure epsilon: the noise scale is sensitivity / epsilon and the law is centred.
  const scale = laplaceScale(1, 0.5);
  assert(scale === 2 && laplaceScale(3, 1) === 3, "Laplace scale is sensitivity / epsilon");
  assert(Math.abs(laplaceCdf(0, 0, 2) - 0.5) < 1e-12, "the Laplace CDF is centred on the query result");

  // 2. The definition of epsilon-DP: the likelihood ratio over an output grid never exceeds e^eps.
  const grid = Array.from({ length: 401 }, (_, i) => -20 + i * 0.1);
  const worst = Math.max(...grid.map((x) => laplaceLogRatio(x, 1, scale)));
  assert(worst <= 0.5 + 1e-9, `the ratio bound must respect epsilon, got ${worst}`);
  assert(Math.abs(worst - 0.5) < 0.02, `the bound is tight close to the mean, got ${worst}`);
  assert(Math.max(...grid.map((x) => laplaceLogRatio(x, -1, scale))) <= 0.5 + 1e-9, "the bound holds for both directions");

  // 3. Oracle: sampled outputs follow the analytic Laplace law the ratio bound came from.
  const rng = splitmix64(943);
  const draws = Array.from({ length: 40000 }, () => sampleLaplace(rng, 0, scale));
  const quantiles = [0.25, 0.5, 0.75];
  quantiles.forEach((q) => {
    const empirical = draws.filter((d) => d <= q).length / draws.length;
    const analytic = laplaceCdf(q, 0, scale);
    assert(Math.abs(empirical - analytic) < 0.02, `empirical CDF ${empirical} vs analytic ${analytic}`);
  });

  // 4. Randomized response is a pure epsilon mechanism with an exact reporting ratio.
  const p = 0.75;
  const rrEpsilon = randomizedResponseEpsilon(p);
  assert(Math.abs(rrEpsilon - Math.log(3)) < 1e-12, `randomized response epsilon ${rrEpsilon}`);
  assert(Math.abs(p / (1 - p) - Math.exp(rrEpsilon)) < 1e-12, "the reporting ratio matches its epsilon");
  assert(p / (1 - p) <= Math.exp(rrEpsilon) * (1 + 1e-12) && (1 - p) / p <= Math.exp(rrEpsilon), "both reporting directions respect the bound");

  // 5. Sequential adds, parallel takes the maximum, and both are monotone in the mechanism set.
  const three = [{ epsilon: 0.4 }, { epsilon: 0.3 }, { epsilon: 0.2 }];
  const sequential = composeSequential(three);
  assert(Math.abs(sequential.epsilon - 0.9) < 1e-12, `sequential epsilon ${sequential.epsilon}`);
  assert(Math.abs(composeParallel(three).epsilon - 0.4) < 1e-12, "parallel epsilon is the maximum");
  assert(composeSequential([...three, { epsilon: 0.1 }]).epsilon > sequential.epsilon, "adding a mechanism never lowers epsilon");
  assert(composeSequential([]).epsilon === 0 && composeParallel([]).epsilon === 0, "an empty composition leaks nothing");
  assert(composeParallel([{ epsilon: 0.2 }, { epsilon: 0.5 }]).epsilon < composeSequential([{ epsilon: 0.2 }, { epsilon: 0.5 }]).epsilon, "partitioning is cheaper than repeating");

  // 6. Advanced composition: tighter than the basic bound once a delta is allowed.
  const many = Array.from({ length: 40 }, () => ({ epsilon: 0.05 }));
  const basic = composeSequential(many).epsilon;
  const advanced = composeAdvanced(many, 1e-6);
  assert(advanced.epsilon < basic, `advanced ${advanced.epsilon} should beat basic ${basic}`);
  assert(Math.abs(advanced.epsilon - (advanced.expansion + advanced.tail)) < 1e-12, "the advanced bound is the expansion plus the tail");
  assert(composeAdvanced(many, 1e-9).epsilon > advanced.epsilon, "a smaller delta costs more epsilon");

  // 7. Gaussian mechanism: rho and sigma are inverses, and the Renyi conversion matches the
  //    classical Gaussian analysis within a few percent.
  const sigma = 4;
  const rho = gaussianRho(1, sigma);
  assert(Math.abs(rho - 1 / 32) < 1e-12 && Math.abs(gaussianSigma(1, rho) - sigma) < 1e-12, "rho and sigma are inverses");
  const dp = rdpToDp(rho, 1e-5);
  const classical = classicalGaussianEpsilon(1, sigma, 1e-5);
  assert(Math.abs(dp.epsilon - classical) / classical < 0.1, `Renyi ${dp.epsilon} vs classical ${classical}`);
  assert(dp.order >= 2 && dp.epsilon > 0, `converted epsilon ${dp.epsilon} at order ${dp.order}`);
  assert(rdpToDp(rho, 1e-9).epsilon > dp.epsilon, "a smaller delta costs a larger epsilon");
  assert(rdpToDp(4 * rho, 1e-5).epsilon > dp.epsilon, "more Renyi divergence costs more epsilon");

  // 8. Composed Gaussians stay between the single mechanism and twice its epsilon.
  const half = gaussianRho(1, 8);
  const single = rdpToDp(half, 1e-5).epsilon;
  const together = rdpToDp(2 * half, 1e-5).epsilon;
  assert(together > single && together < 2 * single, `composed ${together} vs single ${single}`);
  assert(composeSequential([{ epsilon: single }, { epsilon: 0 }]).epsilon === single, "a zero-epsilon mechanism is a no-op");

  // 9. Boundary: invalid parameters are rejected instead of being silently accounted for.
  assert.throws(() => validate(-0.1, 0), /invalid privacy parameters/);
  assert.throws(() => validate(1, 1), /invalid privacy parameters/);
  assert.throws(() => rdpToDp(1, 0), /invalid Renyi parameters/);
  assert.throws(() => composeAdvanced(three, 0), /advanced composition/);
  assert.throws(() => composeAdvanced(three, 2), /advanced composition/);

  // 10. Determinism: the same seed draws the same samples and composition is a pure function.
  const rerun = splitmix64(943);
  assert(JSON.stringify(Array.from({ length: 40000 }, () => sampleLaplace(rerun, 0, scale))) === JSON.stringify(draws), "sampling is deterministic");
  assert(composeSequential(three).epsilon === sequential.epsilon && JSON.stringify(composeAdvanced(many, 1e-6)) === JSON.stringify(advanced), "composition is a pure function");

  console.log(`privacy: laplaceEps=0.5 rrEps=${rrEpsilon.toFixed(3)} basic=${basic.toFixed(3)} advanced=${advanced.epsilon.toFixed(3)} gaussian=${dp.epsilon.toFixed(3)}@${dp.order}`);
  console.log("PASS");
}

module.exports = { splitmix64, laplaceScale, laplaceLogRatio, laplaceCdf, sampleLaplace, randomizedResponseEpsilon,
  gaussianRho, gaussianSigma, rdpToDp, classicalGaussianEpsilon, composeSequential, composeParallel, composeAdvanced };
if (require.main === module) main();
