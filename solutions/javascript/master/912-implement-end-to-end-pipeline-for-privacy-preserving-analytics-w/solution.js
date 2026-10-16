/** Task 912 - Implement end-to-end pipeline for privacy-preserving analytics with DP budgeting.
 *  (Tier: Master, CTCI Beyond — Differential Privacy & Analytics)
 *
 * A small analytics API over a private count table. Every query is declared with its L1 sensitivity, the
 * accountant prices it, and a privacy filter refuses anything that would exceed the total budget — so the
 * *spend*, not the caller's intent, bounds the disclosure. Released answers use the Laplace mechanism
 * (scale = sensitivity/eps) or the Gaussian mechanism (sigma = sensitivity*sqrt(2 ln(1.25/delta))/eps),
 * and composition is tracked by three standard accountants: basic (sum of eps), advanced
 * (eps' = sum eps_i (e^{eps_i} - 1) + sqrt(2 ln(1/delta) sum eps_i^2)) and zCDP
 * (rho = sum eps_i^2 / 2, converted back with eps = rho + 2 sqrt(rho ln(1/delta))).
 *
 * The checks verify the definition rather than the prose: the Laplace density ratio over a fine output
 * grid never exceeds e^eps (that *is* eps-DP for neighbouring inputs), the Gaussian mechanism's violating
 * region carries at most delta probability, the sampled noise matches its declared scale, the filter
 * never overspends, and the accountants are ordered and monotone. Everything is seeded and deterministic.
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
  /** Uniform on (-0.5, 0.5), never exactly 0 so the logarithm stays finite. */
  signed() {
    const value = (this.next() + 0.5) / 4294967296 - 0.5;
    return Math.abs(value) < 1e-12 ? 1e-12 : value;
  }
  normal() {
    const u = Math.max((this.next() + 0.5) / 4294967296, 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * ((this.next() + 0.5) / 4294967296));
  }
}

const laplace = (scale, prng) => {
  const u = prng.signed();
  return -scale * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
};
const gaussian = (sigma, prng) => sigma * prng.normal();
const laplaceDensity = (x, location, scale) => Math.exp(-Math.abs(x - location) / scale) / (2 * scale);
const gaussianDensity = (x, location, sigma) => Math.exp(-((x - location) ** 2) / (2 * sigma ** 2)) /
  (sigma * Math.sqrt(2 * Math.PI));
/** sigma for (eps, delta)-DP at sensitivity 1 (the analytic Gaussian mechanism calibration). */
const gaussianSigma = (eps, delta) => Math.sqrt(2 * Math.log(1.25 / delta)) / eps;
/** Three composition accountants. */
const basicComposition = (epsilons) => epsilons.reduce((sum, value) => sum + value, 0);
const advancedComposition = (epsilons, delta) => {
  const sum = epsilons.reduce((total, value) => total + value, 0);
  const curvature = epsilons.reduce((total, value) => total + value * (Math.exp(value) - 1), 0);
  const squares = epsilons.reduce((total, value) => total + value * value, 0);
  return curvature + Math.sqrt(2 * Math.log(1 / delta) * squares);
};
const zcdpComposition = (epsilons, delta) => {
  const rho = epsilons.reduce((total, value) => total + (value * value) / 2, 0);
  return rho + 2 * Math.sqrt(rho * Math.log(1 / delta));
};
/** Mass of the region where the likelihood ratio of two Gaussians exceeds e^eps (numeric integral). */
function violatingMass(eps, sigma, separation, reach = 40, steps = 200000) {
  const half = sigma * reach;
  const step = (2 * half) / steps;
  let mass = 0;
  for (let i = 0; i < steps; i++) {
    const x = -half + (i + 0.5) * step;
    const ratio = Math.exp(-((x - separation) ** 2 - x ** 2) / (2 * sigma * sigma));
    if (ratio > Math.exp(eps)) mass += gaussianDensity(x, 0, sigma) * step;
  }
  return mass;
}
/** Worst-case likelihood ratio of the Laplace mechanism over a grid of outputs. */
function worstLaplaceRatio(scale, separation, reach = 60, steps = 200000) {
  const half = scale * reach;
  const step = (2 * half) / steps;
  let worst = 0;
  for (let i = 0; i < steps; i++) {
    const x = -half + (i + 0.5) * step;
    worst = Math.max(worst, laplaceDensity(x, separation, scale) / laplaceDensity(x, 0, scale));
  }
  return worst;
}
/** Budgeted analytics session over a private table. */
class PrivacyBudget {
  constructor(epsilon, delta, accountant = "advanced") {
    assert.ok(epsilon > 0, `budget must be positive, got ${epsilon}`);
    assert.ok(accountant === "basic" || accountant === "advanced" || accountant === "zcdp",
      `unknown accountant ${accountant}`);
    this.total = epsilon;
    this.delta = delta;
    this.accountant = accountant;
    this.spent = [];
    this.refused = 0;
  }
  get remaining() {
    return this.total - this.composition(this.spent.concat([]));
  }
  composition(epsilons) {
    if (epsilons.length === 0) return 0;
    if (this.accountant === "basic") return basicComposition(epsilons);
    if (this.accountant === "zcdp") return zcdpComposition(epsilons, this.delta);
    return advancedComposition(epsilons, this.delta);
  }
  /** Price a query and either admit it or refuse it; never leaves the budget overdrawn. */
  request(epsilon) {
    assert.ok(epsilon > 0, "a query must declare a positive epsilon");
    if (this.composition(this.spent.concat([epsilon])) > this.total) {
      this.refused += 1;
      return { admitted: false, remaining: this.remaining };
    }
    this.spent.push(epsilon);
    return { admitted: true, remaining: this.remaining, spent: this.composition(this.spent) };
  }
}

function main() {
  const prng = new Prng(0x5eed0912);
  const eps = 0.7;
  const delta = 1e-5;
  const scale = 1 / eps;
  const sigma = gaussianSigma(eps, delta);
  // 1. eps-DP of the Laplace mechanism, checked on the density ratio itself.
  {
    const worst = worstLaplaceRatio(scale, 1);
    assert.ok(worst <= Math.exp(eps) + 1e-9, `worst ratio ${worst} must not exceed e^eps=${Math.exp(eps)}`);
    assert.ok(Math.abs(worst - Math.exp(eps)) < 1e-3, "the bound is tight, so the scale is right");
    assert.ok(worstLaplaceRatio(2 * scale, 1) < Math.exp(eps), "more noise only helps");
  }
  // 2. (eps, delta)-DP of the Gaussian mechanism: the violating region carries at most delta mass.
  {
    const mass = violatingMass(eps, sigma, 1);
    assert.ok(mass <= delta, `violating mass ${mass} must stay under delta ${delta}`);
    const tightMass = violatingMass(eps, sigma * 0.5, 1);
    assert.ok(tightMass > delta, "halving sigma really does break the guarantee");
    assert.ok(violatingMass(eps, sigma * 2, 1) <= violatingMass(eps, sigma, 1), "more noise, less violation");
  }
  // 3. sampled noise matches the declared scale (mean absolute value and variance).
  {
    const trials = 40000;
    const samples = Array.from({ length: trials }, () => laplace(scale, prng));
    const meanAbsolute = samples.reduce((sum, x) => sum + Math.abs(x), 0) / trials;
    const variance = samples.reduce((sum, x) => sum + x * x, 0) / trials;
    const standardError = scale / Math.sqrt(trials);
    assert.ok(Math.abs(meanAbsolute - scale) < 5 * standardError, `mean |noise| ${meanAbsolute} vs ${scale}`);
    assert.ok(Math.abs(variance - 2 * scale * scale) < 0.1 * 2 * scale * scale, "variance must be 2 b^2");
    const gaussianSamples = Array.from({ length: trials }, () => gaussian(sigma, prng));
    const gaussianVariance = gaussianSamples.reduce((sum, x) => sum + x * x, 0) / trials;
    assert.ok(Math.abs(gaussianVariance - sigma * sigma) < 0.1 * sigma * sigma, "Gaussian variance matches");
  }
  // 4. released answers are accurate: the tail bound |noise| <= b ln(1/beta) holds with probability 1-beta.
  {
    const beta = 0.05;
    const bound = scale * Math.log(1 / beta);
    let violations = 0;
    let total = 0;
    for (let trial = 0; trial < 4000; trial++) {
      const truth = trial % 250;
      const released = truth + laplace(scale, prng);
      total += released - truth;
      if (Math.abs(released - truth) > bound) violations += 1;
    }
    assert.ok(violations / 4000 <= beta + 3 * Math.sqrt((beta * (1 - beta)) / 4000),
      `${violations}/4000 releases exceeded the tail bound`);
    assert.ok(Math.abs(total / 4000) < 0.05, "the mechanism is unbiased in expectation");
  }
  // 5. the privacy filter — spend is exact, refusals keep the session inside the budget.
  {
    const budget = new PrivacyBudget(1.0, delta, "basic");
    assert.deepStrictEqual(budget.request(0.4), { admitted: true, remaining: 0.6, spent: 0.4 });
    assert.strictEqual(budget.request(0.4).admitted, true);
    const refused = budget.request(0.4);
    assert.strictEqual(refused.admitted, false, "the third query would overspend");
    assert.strictEqual(budget.refused, 1);
    assert.ok(Math.abs(budget.remaining - 0.2) < 1e-12, "the refused query changed nothing");
    assert.ok(budget.composition(budget.spent) <= budget.total, "the spend never exceeds the budget");
    const generous = new PrivacyBudget(1.5, delta, "advanced");
    let admitted = 0;
    for (let i = 0; i < 50; i++) if (generous.request(0.05).admitted) admitted += 1;
    assert.ok(admitted > 0 && admitted < 50, `the accountant admitted ${admitted} of 50 small queries`);
    assert.ok(generous.spent.length === admitted, "every admitted query is on the ledger");
    assert.throws(() => new PrivacyBudget(0, delta), /budget must be positive/);
    assert.throws(() => new PrivacyBudget(1, delta, "vibes"), /unknown accountant/);
    assert.throws(() => budget.request(0), /positive epsilon/);
  }
  // 6. composition — the three accountants are ordered and monotone in the usual directions.
  {
    const many = new Array(60).fill(0.02);
    const basic = basicComposition(many);
    const advanced = advancedComposition(many, delta);
    const zcdp = zcdpComposition(many, delta);
    assert.ok(advanced < basic, `advanced ${advanced} must beat basic ${basic} for many small queries`);
    assert.ok(zcdp < basic, `zcdp ${zcdp} must beat basic ${basic}`);
    assert.ok(advancedComposition(many, 1e-9) > advancedComposition(many, 1e-3), "smaller delta costs more");
    const fewer = many.slice(0, 30);
    assert.ok(advancedComposition(fewer, delta) < advancedComposition(many, delta), "fewer queries cost less");
    const cheaper = new Array(60).fill(0.01);
    assert.ok(advancedComposition(cheaper, delta) < advancedComposition(many, delta), "smaller eps costs less");
    assert.strictEqual(basicComposition([]), 0, "no queries cost nothing");
    assert.strictEqual(zcdpComposition([], delta), 0);
    assert.strictEqual(advancedComposition([0.3], 1e-5) >= 0.3, true, "one query cannot cost less than its eps");
  }
  // 7. boundary — a zero-noise query, an empty table and a degenerate budget.
  {
    const zeroSensitivity = laplace(0, prng);
    assert.ok(zeroSensitivity === 0, "sensitivity zero needs no noise");
    const emptyTable = [];
    const countQuery = (table, predicate) => table.filter(predicate).length;
    assert.strictEqual(countQuery(emptyTable, () => true), 0, "an empty table answers zero");
    assert.strictEqual(countQuery(emptyTable, () => true) + laplace(0, prng), 0, "and releases zero noiselessly");
    const broke = new PrivacyBudget(0.001, delta, "basic");
    assert.strictEqual(broke.request(0.001).admitted, true, "the exact budget admits one query");
    assert.strictEqual(broke.request(0.001).admitted, false, "and then refuses");
    assert.ok(zcdpComposition([1e-6, 1e-6], delta) < 1e-3, "tiny queries cost almost nothing");
  }
  // 8. determinism — the mechanism and the session replay exactly, a different seed diverges.
  {
    const first = laplace(scale, new Prng(9));
    const second = laplace(scale, new Prng(9));
    assert.strictEqual(first, second);
    assert.notStrictEqual(laplace(scale, new Prng(10)), first);
    const session = () => {
      const budget = new PrivacyBudget(1, delta, "advanced");
      const prng = new Prng(77);
      const released = [];
      for (let i = 0; i < 20; i++) {
        const answer = budget.request(0.05);
        if (answer.admitted) released.push(10 + laplace(1 / 0.05, prng));
      }
      return { released, spent: budget.spent, refused: budget.refused };
    };
    assert.deepStrictEqual(session(), session());
    assert.deepStrictEqual(gaussianSigma(eps, delta), sigma);
  }
  console.log(`dp-analytics: eps=${eps} delta=${delta}, Laplace scale ${scale} (worst ratio ` +
    `${worstLaplaceRatio(scale, 1).toFixed(4)} <= e^eps ${Math.exp(eps).toFixed(4)}), Gaussian sigma ` +
    `${sigma.toFixed(3)} (violating mass ${violatingMass(eps, sigma, 1).toExponential(2)} <= ${delta}); ` +
    `advanced composition of 60 x 0.02 queries: ${advancedComposition(new Array(60).fill(0.02), delta).toFixed(3)} ` +
    `vs basic ${basicComposition(new Array(60).fill(0.02)).toFixed(3)}`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  Prng, laplace, gaussian, laplaceDensity, gaussianDensity, gaussianSigma, basicComposition,
  advancedComposition, zcdpComposition, violatingMass, worstLaplaceRatio, PrivacyBudget,
};
