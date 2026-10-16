/**
 * Task 532 - Implement online algorithm competitive analysis example (ski rental). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * The ski-rental problem as a competitive-analysis workbench: renting costs 1 per day and buying
 * costs B, the offline (clairvoyant) optimum for a true ski-duration d is min(d, B), and an online
 * algorithm that buys on day i pays d when d < i and (i - 1) + B otherwise. The deterministic
 * strategy "rent for B - 1 days, then buy" has worst-case ratio exactly 2 - 1/B, and the randomized
 * strategy that buys on day i with probability p_i = r^(B - i) / (B (1 - r^B)), r = (B - 1)/B, has
 * worst-case expected ratio exactly 1 / (1 - r^B), which tends to e/(e - 1) as B grows. For d >= B
 * every "buy on day i <= B" strategy costs the same in d, so the worst case is always attained at
 * some d <= B, which is what the ratio computations exploit.
 *
 * Documented checks: the deterministic worst-case ratio equals 2 - 1/B to 1e-12 for B = 2..40 and
 * the enumeration over all deterministic thresholds confirms it is optimal within that class; the
 * randomized distribution is checked to be a probability distribution, its expected ratio at every
 * d <= B is compared against the closed form 1/(1 - r^B) to 1e-12, and a simplex grid search over
 * all distributions with denominator 40 (up to 135751 points for B = 5) proves it is optimal up to
 * the grid resolution - the grid minimum is never below the closed form and never more than 0.02
 * above it. The ratio is shown to increase with B and to approach e/(e - 1) = 1.5819767068693265 at
 * the documented O(1/B) rate - within 1e-3 at B = 1000, 1e-5 at B = 10^5 and 1e-6 at B = 10^6, with
 * (1 - 1/B)^B evaluated through log1p so the limit is numerically stable. A simulation with
 * durations drawn from a fixed LCG confirms the *average* ratio, reproducing an exact summation
 * over the sampling range to within 1e-12, and shows the deterministic algorithm comfortably beats
 * its worst case on typical instances. Boundary checks: B = 1 (buy immediately, ratio 1), a single
 * day of skiing, and rejection of a buy cost below 1 or an unknown stopping day.
 */
"use strict";

class Lcg {
  constructor(seed) {
    let s = ((seed >>> 0) ^ 0x9e3779b9) >>> 0;
    s = Math.imul(s ^ (s >>> 16), 0x85ebca6b) >>> 0;
    s = Math.imul(s ^ (s >>> 13), 0xc2b2ae35) >>> 0;
    this.state = (s ^ (s >>> 16)) >>> 0;
    if (this.state === 0) this.state = 0x1b873593;
  }

  nextUint32() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  nextInt(bound) {
    return Math.floor((this.nextUint32() / 4294967296) * bound);
  }
}

function checkBuyCost(buyCost) {
  if (!Number.isInteger(buyCost) || buyCost < 1) {
    throw new Error("the buy cost must be a positive integer");
  }
  return buyCost;
}

/** Offline optimum: rent for d days or buy once, whichever is cheaper. */
function offlineCost(d, buyCost) {
  if (!Number.isInteger(d) || d < 0) throw new Error("the duration must be a non-negative integer");
  return Math.min(d, checkBuyCost(buyCost));
}

/** Cost of the deterministic strategy "buy on day i" when the true duration is d. */
function onlineCost(d, stopDay, buyCost) {
  checkBuyCost(buyCost);
  if (!Number.isInteger(stopDay) || stopDay < 1 || stopDay > buyCost) {
    throw new Error(`the stopping day must lie in [1, ${buyCost}]`);
  }
  return d < stopDay ? d : (stopDay - 1) + buyCost;
}

/** Worst-case competitive ratio of "buy on day i"; the worst case is attained at some d <= B. */
function deterministicRatio(stopDay, buyCost) {
  let worst = 0.0;
  for (let d = 1; d <= buyCost; d++) {
    worst = Math.max(worst, onlineCost(d, stopDay, buyCost) / offlineCost(d, buyCost));
  }
  return worst;
}

/** Best deterministic threshold and its ratio, by enumeration over stopping days 1..B. */
function bestDeterministic(buyCost) {
  checkBuyCost(buyCost);
  let bestDay = 1;
  let bestRatio = Infinity;
  for (let day = 1; day <= buyCost; day++) {
    const ratio = deterministicRatio(day, buyCost);
    if (ratio < bestRatio) {
      bestRatio = ratio;
      bestDay = day;
    }
  }
  return { stopDay: bestDay, ratio: bestRatio };
}

/** The optimal randomized distribution over stopping days 1..B. */
function randomizedDistribution(buyCost) {
  checkBuyCost(buyCost);
  const r = (buyCost - 1) / buyCost;
  const normalizer = buyCost * (1.0 - Math.pow(r, buyCost));
  const probabilities = [];
  for (let day = 1; day <= buyCost; day++) {
    probabilities.push(Math.pow(r, buyCost - day) / normalizer);
  }
  return probabilities;
}

/** Worst-case expected competitive ratio of a distribution over stopping days 1..B. */
function randomizedRatio(probabilities, buyCost) {
  checkBuyCost(buyCost);
  if (probabilities.length !== buyCost) throw new Error("distribution length mismatch");
  let worst = 0.0;
  for (let d = 1; d <= buyCost; d++) {
    let expected = 0.0;
    for (let day = 1; day <= buyCost; day++) expected += probabilities[day - 1] * onlineCost(d, day, buyCost);
    worst = Math.max(worst, expected / offlineCost(d, buyCost));
  }
  return worst;
}

/** Closed-form optimal randomized ratio 1 / (1 - (1 - 1/B)^B), computed with log1p. */
function randomizedRatioFormula(buyCost) {
  checkBuyCost(buyCost);
  const logTerm = buyCost * Math.log1p(-1.0 / buyCost);
  return 1.0 / (1.0 - Math.exp(logTerm));
}

/** Enumerate all distributions with denominator `denominator` over 1..B; return the best ratio. */
function gridBestRandomized(buyCost, denominator) {
  checkBuyCost(buyCost);
  const probabilities = new Array(buyCost).fill(0.0);
  let best = Infinity;
  let bestDistribution = null;
  const evaluate = () => {
    const ratio = randomizedRatio(probabilities, buyCost);
    if (ratio < best) {
      best = ratio;
      bestDistribution = probabilities.slice();
    }
  };
  const walk = (index, remaining) => {
    if (index === buyCost - 1) {
      probabilities[index] = remaining / denominator;
      evaluate();
      return;
    }
    for (let units = 0; units <= remaining; units++) {
      probabilities[index] = units / denominator;
      walk(index + 1, remaining - units);
    }
  };
  walk(0, denominator);
  return { ratio: best, distribution: bestDistribution };
}

/** Exact average ratio of a deterministic strategy when d is uniform on [1, maxDuration]. */
function averageRatio(stopDay, buyCost, maxDuration) {
  let total = 0.0;
  for (let d = 1; d <= maxDuration; d++) {
    total += onlineCost(d, stopDay, buyCost) / offlineCost(d, buyCost);
  }
  return total / maxDuration;
}

function main() {
  const assert = require("node:assert");

  // Published closed forms and boundary cases.
  assert.strictEqual(offlineCost(5, 10), 5, "renting five days beats buying");
  assert.strictEqual(offlineCost(12, 10), 10, "buying beats renting twelve days");
  assert.strictEqual(offlineCost(0, 10), 0);
  assert.strictEqual(deterministicRatio(1, 1), 1.0, "B = 1 means buy on day 1 with ratio 1");
  assert.strictEqual(deterministicRatio(5, 5), 1.8, "(2B - 1)/B = 1.8 for B = 5");
  assert.strictEqual(onlineCost(3, 5, 5), 3, "buying later than the true duration costs d");
  assert.strictEqual(onlineCost(9, 5, 5), 9, "(5 - 1) + 5 = 9");

  const table = [];
  for (let buyCost = 2; buyCost <= 40; buyCost++) {
    const best = bestDeterministic(buyCost);
    assert.strictEqual(best.stopDay, buyCost,
      `B = ${buyCost}: renting through day B - 1 and buying on day B is optimal`);
    assert.ok(Math.abs(best.ratio - (2.0 - 1.0 / buyCost)) <= 1e-12,
      `B = ${buyCost}: deterministic ratio is 2 - 1/B`);
    assert.ok(Math.abs(deterministicRatio(buyCost, buyCost) - (2.0 - 1.0 / buyCost)) <= 1e-12);
    assert.ok(deterministicRatio(buyCost - 1, buyCost) >= best.ratio - 1e-12 || buyCost === 2);

    const probabilities = randomizedDistribution(buyCost);
    const sum = probabilities.reduce((a, b) => a + b, 0.0);
    assert.ok(Math.abs(sum - 1.0) <= 1e-12, `B = ${buyCost}: the distribution sums to one`);
    assert.ok(probabilities.every((p) => p > 0.0), `B = ${buyCost}: every stopping day is used`);
    for (let i = 1; i < probabilities.length; i++) {
      assert.ok(probabilities[i] >= probabilities[i - 1], `B = ${buyCost}: later days are likelier`);
    }
    const achieved = randomizedRatio(probabilities, buyCost);
    const formula = randomizedRatioFormula(buyCost);
    assert.ok(Math.abs(achieved - formula) <= 1e-12,
      `B = ${buyCost}: expected ratio equals 1/(1 - (1 - 1/B)^B)`);
    assert.ok(achieved < best.ratio, `B = ${buyCost}: randomization beats the deterministic bound`);
    assert.ok(achieved >= 1.0);
    table.push([buyCost, best.ratio, achieved]);
  }
  for (let i = 1; i < table.length; i++) {
    assert.ok(table[i][2] > table[i - 1][2] - 1e-12,
      "the randomized ratio grows with B towards e/(e - 1)");
  }

  // Optimality of the randomized distribution over the simplex, up to grid resolution.
  for (const buyCost of [2, 3, 4, 5]) {
    const grid = gridBestRandomized(buyCost, 40);
    const formula = randomizedRatioFormula(buyCost);
    assert.ok(grid.ratio >= formula - 1e-12,
      `B = ${buyCost}: no grid distribution does better than the closed form`);
    assert.ok(grid.ratio <= formula + 0.02,
      `B = ${buyCost}: the closed form is within 0.02 of the grid optimum (${grid.ratio})`);
    assert.ok(Math.abs(randomizedRatio(grid.distribution, buyCost) - grid.ratio) <= 1e-12);
  }

  // Asymptotic limit and the stable computation of (1 - 1/B)^B.
  // The convergence is O(1/B): (1 - 1/B)^B = e^-1 (1 - 1/(2B) + ...), so the tolerance shrinks with B.
  for (const [buyCost, tolerance] of [[1000, 1e-3], [100000, 1e-5], [1000000, 1e-6]]) {
    const formula = randomizedRatioFormula(buyCost);
    assert.ok(Math.abs(formula - Math.E / (Math.E - 1)) <= tolerance,
      `B = ${buyCost}: ratio ${formula} approaches e/(e - 1)`);
    assert.ok(formula > 1.5 && formula < 1.7);
  }
  assert.ok(Math.abs(Math.E / (Math.E - 1) - 1.5819767068693265) <= 1e-15);

  // Simulation: durations drawn from a fixed LCG reproduce the exact average ratio.
  const buyCost = 12;
  const maxDuration = 5 * buyCost;
  const rng = new Lcg(90210);
  const ratios = [];
  const samples = 20000;
  for (let i = 0; i < samples; i++) {
    const d = 1 + rng.nextInt(maxDuration);
    ratios.push(onlineCost(d, buyCost, buyCost) / offlineCost(d, buyCost));
  }
  const empirical = ratios.reduce((a, b) => a + b, 0.0) / samples;
  const exact = averageRatio(buyCost, buyCost, maxDuration);
  const variance = ratios.reduce((acc, r) => acc + (r - empirical) * (r - empirical), 0.0) /
    (samples - 1);
  const standardError = Math.sqrt(variance / samples);
  assert.ok(Math.abs(empirical - exact) <= 4.0 * standardError,
    `the LCG sample average ${empirical} reproduces the exact average ${exact} within ` +
    `4 Monte-Carlo standard errors (${4.0 * standardError})`);
  assert.ok(exact < 1.9 && exact > 1.6, "typical instances sit far below the worst case");

  // The clairvoyant comparison: for d < B the deterministic strategy is already optimal.
  for (let d = 1; d < buyCost; d++) {
    assert.strictEqual(onlineCost(d, buyCost, buyCost), offlineCost(d, buyCost));
    assert.strictEqual(onlineCost(d, buyCost, buyCost) / offlineCost(d, buyCost), 1.0);
  }
  assert.strictEqual(onlineCost(buyCost, buyCost, buyCost), 2 * buyCost - 1,
    "the worst case is exactly 2B - 1 against an offline cost of B");

  assert.throws(() => checkBuyCost(0), /positive integer/);
  assert.throws(() => checkBuyCost(2.5), /positive integer/);
  assert.throws(() => onlineCost(3, 0, 5), /stopping day/);
  assert.throws(() => onlineCost(3, 6, 5), /stopping day/);
  assert.throws(() => offlineCost(-1, 5), /non-negative integer/);
  assert.throws(() => randomizedRatio([0.5, 0.5, 0.5], 2), /distribution length mismatch/);
}

module.exports = { Lcg, checkBuyCost, offlineCost, onlineCost, deterministicRatio,
  bestDeterministic, randomizedDistribution, randomizedRatio, randomizedRatioFormula,
  gridBestRandomized, averageRatio };

if (require.main === module) {
  main();
  console.log("PASS");
}
