/** Task 906 - Implement multi-fidelity simulation orchestration with surrogate models.
 *  (Tier: Master, CTCI Beyond — Scientific Computing & Resource Orchestration)
 *
 * A cheap biased simulator (fidelity 1) plus an expensive accurate one (fidelity 10) are combined into
 * the difference/control-variate estimator
 *
 *     mu_hat = mean_Nl( f_low )  +  mean_Nh( f_high - f_low )
 *
 * which is unbiased for E[f_high] because the correction term estimates the low-fidelity bias exactly,
 * while its variance sigma_low^2/N_l + sigma_diff^2/N_h is minimised by sizing the fidelities against
 * their costs: N_l/N_h = (sigma_low/sqrt(c_low)) / (sigma_diff/sqrt(c_high)), with the sigmas estimated
 * from a small pilot. The orchestrator therefore spends most of the budget on the surrogate and only
 * buys high-fidelity samples for the bias (a scalar quantity, so very few are needed).
 *
 * Checks: the estimator is unbiased where the surrogate-only estimator is measurably biased (compared
 * against a dense quadrature oracle), the analytic allocation beats naive splits and pure high-fidelity
 * sampling at the same budget, the cost ledger never exceeds the budget, degraded configurations fall
 * back safely, and results reproduce from a fixed seed. Runtime is a few hundred thousand exp() calls.
 */
"use strict";

const assert = require("node:assert");

const COST = { low: 1, high: 10, correction: 11 };   // a correction sample needs both fidelities
const FIDELITY = { low: 1, high: 10 };

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  float() { return this.next() / 4294967296; }
}

/** Expensive truth: f(x) = 2x + 0.5 sin(3 pi x) on [0,1). */
const highFidelity = (x) => 2 * x + 0.5 * Math.sin(3 * Math.PI * x);
/** Cheap surrogate: correct shape plus a smooth modelling error, whose mean is the bias to estimate. */
const lowFidelity = (x) => highFidelity(x) - 0.2 * Math.cos(4 * x);
const biasOf = (x) => lowFidelity(x) - highFidelity(x);
/** The correction term the estimator averages: minus the surrogate bias, so the sum is unbiased. */
const correctionOf = (x) => highFidelity(x) - lowFidelity(x);

/** Independent oracle: Simpson quadrature on a dense grid (not the estimator's sampling path). */
function quadrature(f, panels) {
  const h = 1 / panels;
  let sum = f(0) + f(1);
  for (let i = 1; i < panels; i++) sum += f(i * h) * (i % 2 === 1 ? 4 : 2);
  return (sum * h) / 3;
}

/** Pilot run: estimate sigma_low and sigma_diff (and the low-fidelity mean) with a few samples per level. */
function pilotComplexity(prng, samples) {
  const lows = [];
  const diffs = [];
  for (let i = 0; i < samples; i++) {
    const x = prng.float();
    lows.push(lowFidelity(x));
    diffs.push(correctionOf(x));
  }
  const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
  const variance = (values) => mean(values.map((value) => (value - mean(values)) ** 2));
  return { sigmaLow: Math.sqrt(variance(lows)) || 1e-9, sigmaDiff: Math.sqrt(variance(diffs)) + 1e-9, pilotLow: mean(lows) };
}

/** Analytic allocation: minimise sigma_low^2/N_l + sigma_diff^2/N_h subject to the budget.
 *  Lagrange multipliers give N_l ∝ sigma_low/sqrt(c_low) and N_h ∝ sigma_diff/sqrt(c_correction);
 *  both are then rounded into the affordable range. */
function allocate(budget, sigmaLow, sigmaDiff, cost = COST) {
  assert.ok(Number.isInteger(budget) && budget > 0, `budget must be a positive integer, got ${budget}`);
  const affordableCorrections = Math.floor((budget - cost.low) / cost.correction);
  if (affordableCorrections < 1) {
    const low = Math.floor(budget / cost.low);      // degraded: only the surrogate is affordable
    return { low, high: 0, cost: low * cost.low };
  }
  const weightLow = sigmaLow * Math.sqrt(cost.low);
  const weightDiff = sigmaDiff * Math.sqrt(cost.correction);
  const scale = budget / (weightLow + weightDiff);
  const high = Math.max(1, Math.min(affordableCorrections,
    Math.round((scale * sigmaDiff) / Math.sqrt(cost.correction))));
  const low = Math.floor((budget - high * cost.correction) / cost.low);
  return { low, high, cost: low * cost.low + high * cost.correction };
}

/** One multi-fidelity estimate: N_l surrogate samples plus N_h independent correction samples. */
function estimate(plan, prng) {
  const spend = plan.low * COST.low + plan.high * COST.correction;
  assert.ok(spend <= plan.budget, `the plan spends ${spend} of a ${plan.budget} budget`);
  assert.ok(plan.low > 0, "the surrogate mean needs at least one cheap sample");
  let lowSum = 0;
  for (let i = 0; i < plan.low; i++) lowSum += lowFidelity(prng.float());
  let diffSum = 0;
  for (let i = 0; i < plan.high; i++) {
    const x = prng.float();
    diffSum += highFidelity(x) - lowFidelity(x);
  }
  const lowMean = lowSum / plan.low;
  return { value: lowMean + (plan.high > 0 ? diffSum / plan.high : 0), lowOnly: lowMean,
    highSamples: plan.high, spend };
}

/** Pure high-fidelity Monte Carlo with the same budget, the baseline to beat. */
function estimateHighOnly(budget, prng) {
  const count = Math.floor(budget / COST.high);
  let sum = 0;
  for (let i = 0; i < count; i++) sum += highFidelity(prng.float());
  return { value: sum / count, samples: count };
}

const rmse = (values, truth) => Math.sqrt(values.reduce((sum, value) => sum + (value - truth) ** 2, 0) / values.length);
const meanOf = (values) => values.reduce((a, b) => a + b, 0) / values.length;

function main() {
  const truth = quadrature(highFidelity, 20000);
  const biasTruth = quadrature(biasOf, 20000);
  const budget = 2000;
  const repetitions = 120;

  // 1. happy path — the corrected estimator is unbiased, the surrogate-only estimator is not.
  const corrected = [];
  const lowOnly = [];
  for (let run = 0; run < repetitions; run++) {
    const prng = new Prng(0x5eed0906 + run * 7919);
    const plan = { ...allocate(budget, 0.5, 0.1), budget };
    const result = estimate(plan, prng);
    corrected.push(result.value);
    lowOnly.push(result.lowOnly);
  }
  {
    const correctedBias = meanOf(corrected) - truth;
    assert.ok(Math.abs(correctedBias) < 0.01, `corrected bias ${correctedBias.toFixed(4)} must vanish`);
    assert.ok(Math.abs(biasTruth) > 0.03, `the surrogate really is biased by ${biasTruth.toFixed(4)}`);
    assert.ok(Math.abs(meanOf(lowOnly) - truth) > 3 * Math.abs(correctedBias),
      "surrogate-only estimation keeps its bias while the corrected estimator does not");
    assert.ok(Math.abs(biasTruth - (meanOf(lowOnly) - truth)) < 0.02, "the measured bias matches the oracle");
  }

  // 2. the estimator beats pure high-fidelity Monte Carlo at the same budget.
  const highOnly = [];
  const pilot = pilotComplexity(new Prng(42), 200);
  const plan = allocate(budget, pilot.sigmaLow, pilot.sigmaDiff);
  for (let run = 0; run < repetitions; run++) {
    highOnly.push(estimateHighOnly(budget, new Prng(0xbeef + run * 104729)).value);
  }
  {
    const multifidelityError = rmse(corrected, truth);
    const highOnlyError = rmse(highOnly, truth);
    assert.ok(multifidelityError <= 0.75 * highOnlyError,
      `multi-fidelity RMSE ${multifidelityError.toFixed(4)} must beat ${highOnlyError.toFixed(4)}`);
    assert.ok(plan.high < plan.low / 4, "most of the budget must go to the cheap surrogate");
  }

  // 3. orchestration — the analytic allocation beats naive splits and matches a grid-search oracle.
  {
    const errorFor = (candidate) => {
      const estimates = [];
      for (let run = 0; run < repetitions; run++) {
        estimates.push(estimate({ ...candidate, budget }, new Prng(0xabc + run * 31)).value);
      }
      return rmse(estimates, truth);
    };
    const analytic = errorFor(plan);
    const starved = errorFor({ low: budget - COST.correction, high: 1 });
    const correctionHeavy = errorFor({ low: 200, high: 160 });
    let gridBest = Infinity;
    let gridPlan = null;
    for (const high of [1, 5, 15, 30, 50, 75, 100, 125, 150]) {
      const low = budget - high * COST.correction;
      if (low < 1) continue;
      const error = errorFor({ low, high });
      if (error < gridBest) { gridBest = error; gridPlan = { low, high }; }
    }
    assert.ok(analytic < starved, `analytic ${analytic.toFixed(4)} vs one-shot ${starved.toFixed(4)}`);
    assert.ok(analytic < correctionHeavy,
      `analytic ${analytic.toFixed(4)} vs correction-heavy ${correctionHeavy.toFixed(4)}`);
    assert.ok(analytic <= gridBest * 1.15,
      `analytic ${analytic.toFixed(4)} must match the grid oracle ${gridBest.toFixed(4)} at ` +
      `N_low=${gridPlan.low} N_high=${gridPlan.high}`);
  }

  // 4. cost ledger — a plan never spends more than the budget, and every sample is accounted for.
  {
    for (const candidate of [plan, { low: 10, high: 3 }, { low: 5, high: 0 }]) {
      const spend = candidate.low * COST.low + candidate.high * COST.correction;
      assert.ok(spend <= budget, `plan ${JSON.stringify(candidate)} spends ${spend} of ${budget}`);
    }
    assert.strictEqual(allocate(budget, 0.5, 0.1).cost, plan.cost);
    const saturated = allocate(budget, 0.5, 0.1, { low: 1, high: 10, correction: 11 });
    assert.ok(saturated.cost <= budget && saturated.cost > budget - COST.correction,
      "the budget is nearly saturated");
    const result = estimate({ ...plan, budget }, new Prng(1));
    assert.strictEqual(result.highSamples, plan.high, "the correction count is exactly as planned");
    assert.strictEqual(result.spend, plan.cost, "the ledger matches the plan");
  }

  // 5. boundary — a budget that cannot buy any correction sample degrades to the surrogate.
  {
    const tiny = allocate(COST.correction - 1, 0.5, 0.1);
    assert.strictEqual(tiny.high, 0, "no correction sample is affordable");
    assert.strictEqual(tiny.low, COST.correction - 1, "the remaining budget buys cheap samples");
    assert.strictEqual(estimate({ ...tiny, budget: COST.correction - 1 }, new Prng(3)).highSamples, 0);
    const single = allocate(COST.correction + COST.low, 0.5, 0.1);
    assert.strictEqual(single.high, 1, "exactly one correction sample is affordable");
    assert.ok(allocate(10 * COST.correction, 0.5, 0.1).high >= 1, "a bigger budget buys corrections");
    assert.throws(() => allocate(0, 0.5, 0.1), /positive integer/);
    assert.throws(() => estimate({ low: 10, high: 1, budget: 5 }, new Prng(3)), /spends 21/);
  }

  // 6. oracle cross-check — Simpson on a coarser grid agrees, so the truth is not an artefact.
  {
    for (const panels of [200, 2000]) {
      assert.ok(Math.abs(quadrature(highFidelity, panels) - truth) < 1e-4,
        `Simpson with ${panels} panels must agree with the dense grid`);
      assert.ok(Math.abs(quadrature(biasOf, panels) - biasTruth) < 1e-4, "the bias oracle agrees too");
    }
    const closedForm = 1 + (0.5 * (1 - Math.cos(3 * Math.PI))) / (3 * Math.PI);
    assert.ok(Math.abs(closedForm - truth) < 1e-6, `closed form ${closedForm} vs quadrature ${truth}`);
  }

  // 7. determinism — the same seed replays the estimate, and the plan is a function of its inputs.
  {
    const first = estimate({ ...plan, budget }, new Prng(0x5eed));
    const second = estimate({ ...plan, budget }, new Prng(0x5eed));
    assert.deepStrictEqual(first, second);
    assert.notDeepStrictEqual(estimate({ ...plan, budget }, new Prng(0x5eee)), first);
    assert.deepStrictEqual(pilotComplexity(new Prng(42), 200), pilot);
  }

  console.log(`multi-fidelity: budget ${budget}, plan N_low=${plan.low} N_high=${plan.high} ` +
    `(spend ${plan.cost}), sigma_low ${pilot.sigmaLow.toFixed(3)}, sigma_diff ${pilot.sigmaDiff.toFixed(4)}; ` +
    `RMSE ${rmse(corrected, truth).toFixed(4)} vs high-only ${rmse(highOnly, truth).toFixed(4)} ` +
    `(surrogate bias ${biasTruth.toFixed(4)})`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  COST, FIDELITY, Prng, highFidelity, lowFidelity, biasOf, quadrature, pilotComplexity, allocate,
  estimate, estimateHighOnly, rmse,
};
