/** Task 769 - Implement knapsack approximation PTAS idea demonstration. (Tier: Expert, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * 0/1 knapsack: pick a subset of items maximising total profit with total weight <= capacity.
 * Three solvers are implemented and cross-checked:
 *
 *   exactKnapsack     capacity-indexed DP over weights -> the exact optimum (oracle for small
 *                     instances, together with exhaustive subset enumeration).
 *   fptas             Ibarra-Kim profit rounding: replace profit p_i by m_i = floor(p_i / K)
 *                     and solve the scaled instance exactly with a profit-indexed DP. Every
 *                     rounded profit loses < K, so for the optimal set S*:  OPT - value(S) <
 *                     |S*| * K <= nK. With K = eps * p_max / n the classic FPTAS guarantee
 *                     value >= (1 - eps) * OPT follows (p_max <= OPT because S* is non-empty).
 *                     K = 1 rounds nothing, so the FPTAS then reproduces the exact optimum -
 *                     that is the boundary check that ties the two solvers together.
 *   sahniPtas         the "PTAS idea" in its combinatorial form: let k = ceil(1/eps); enumerate
 *                     every subset A with |A| <= k and fill the rest greedily by profit/weight
 *                     ratio. If the filler is replaced by an *optimal* one the scheme is exact
 *                     whenever k >= |S*| (A = the k largest-profit items of S* is enumerated and
 *                     the optimal filler dominates S* \ A); with the greedy filler the guarantee
 *                     degrades to (1 - eps). Both claims are measured below, and the greedy
 *                     filler is never allowed to exceed capacity.
 *
 * What is simplified: the FPTAS profit DP is a dense array indexed by total rounded profit
 * (fine for the small demonstrations here) instead of the sparse list-of-Pareto-states
 * implementation, the Sahni enumeration is capped at k <= 12 (the width is what makes the scheme
 * exponential in 1/eps, so the cap is exactly the point of the demonstration), and profits are
 * integer. What is verified: exactness of
 * the capacity DP against exhaustive enumeration, the (1 - eps) ratio for both PTAS variants,
 * the analytic bound value >= OPT - nK, feasibility (distinct indices, weight <= capacity) of
 * every returned set, and the degenerate instances below.
 */
"use strict";

function validate(items, capacity) {
  if (!Array.isArray(items)) throw new TypeError("items must be an array");
  if (!Number.isInteger(capacity) || capacity < 0) throw new RangeError("capacity must be a non-negative integer");
  for (const it of items) {
    if (!it || !Number.isInteger(it.weight) || !Number.isInteger(it.profit)) throw new TypeError("items need integer weight/profit");
    if (it.weight < 0 || it.profit < 0) throw new RangeError("negative weight/profit is not allowed");
  }
}

/** Item indices chosen by an index list; returns { profit, weight, indices }. */
function describe(items, indices) {
  let profit = 0;
  let weight = 0;
  for (const i of indices) {
    profit += items[i].profit;
    weight += items[i].weight;
  }
  return { profit, weight, indices: indices.slice().sort((x, y) => x - y) };
}

/** Exact DP over capacity: best[s] = max profit using items processed so far with weight <= s. */
function exactKnapsack(items, capacity) {
  validate(items, capacity);
  const n = items.length;
  const best = new Int32Array(capacity + 1);
  const take = [];
  for (let i = 0; i < n; i++) take.push(new Uint8Array(capacity + 1));
  for (let i = 0; i < n; i++) {
    const { weight, profit } = items[i];
    for (let s = capacity; s >= weight; s--) {
      const candidate = best[s - weight] + profit;
      if (candidate > best[s]) {
        best[s] = candidate;
        take[i][s] = 1;
      }
    }
  }
  const indices = [];
  let s = capacity;
  for (let i = n - 1; i >= 0; i--) {
    if (take[i][s]) { indices.push(i); s -= items[i].weight; }
  }
  return describe(items, indices);
}

/** Exhaustive subset enumeration — independent oracle, only usable for n <= ~20. */
function bruteForceKnapsack(items, capacity) {
  validate(items, capacity);
  let bestProfit = 0;
  let bestMask = 0;
  for (let mask = 0; mask < (1 << items.length); mask++) {
    let weight = 0;
    let profit = 0;
    for (let i = 0; i < items.length; i++) {
      if (mask & (1 << i)) { weight += items[i].weight; profit += items[i].profit; }
    }
    if (weight <= capacity && profit > bestProfit) { bestProfit = profit; bestMask = mask; }
  }
  return describe(items, decodeMask(bestMask, items.length));
}

function decodeMask(mask, n) {
  const out = [];
  for (let i = 0; i < n; i++) if (mask & (1 << i)) out.push(i);
  return out;
}

/**
 * Exact DP on rounded profits: profile[s] = minimum weight of a subset whose rounded profit
 * total is exactly s; masks[s] records one such subset for reconstruction.
 */
function roundedProfitDP(items, capacity, K) {
  validate(items, capacity);
  if (!(K > 0)) throw new RangeError("scaling factor must be positive");
  const rounded = items.map((it) => Math.floor(it.profit / K));
  const maxScaled = rounded.reduce((a, b) => a + b, 0);
  const weight = new Float64Array(maxScaled + 1).fill(Infinity);
  const masks = new Float64Array(maxScaled + 1).fill(-1);
  weight[0] = 0;
  masks[0] = 0;
  for (let i = 0; i < items.length; i++) {
    const m = rounded[i];
    if (m === 0) continue; // profit so small it contributes nothing rounded: adding it never helps
    for (let s = maxScaled - m; s >= 0; s--) {
      if (weight[s] === Infinity) continue;
      const nextWeight = weight[s] + items[i].weight;
      if (nextWeight <= capacity && nextWeight < weight[s + m]) {
        weight[s + m] = nextWeight;
        masks[s + m] = masks[s] + Math.pow(2, i);
      }
    }
  }
  let bestScaled = 0;
  let bestMask = -1;
  for (let s = maxScaled; s >= 0; s--) {
    if (weight[s] !== Infinity) { bestScaled = s; bestMask = masks[s]; break; }
  }
  if (bestMask < 0) return { profit: 0, weight: 0, indices: [], scaledProfit: 0 };
  const indices = decodeMask(bestMask, items.length);
  return { ...describe(items, indices), scaledProfit: bestScaled };
}

/** FPTAS entry point: K = eps * p_max / n, so nK = eps * p_max <= eps * OPT. */function fptas(items, capacity, eps) {
  validate(items, capacity);
  if (!(eps > 0 && eps < 1)) throw new RangeError("eps must lie in (0, 1)");
  if (items.length === 0) return { profit: 0, weight: 0, indices: [], K: 0 };
  const pMax = Math.max(...items.map((it) => it.profit));
  const K = (eps * pMax) / items.length;
  return { ...roundedProfitDP(items, capacity, K), K };
}

/** Greedy filler: best profit/weight ratio first, skipping items that no longer fit. */
function greedyFill(items, capacity, banned) {
  const order = items
    .map((it, i) => ({ i, ratio: it.weight === 0 ? Infinity : it.profit / it.weight }))
    .filter((e) => !banned.has(e.i))
    .sort((x, y) => (y.ratio - x.ratio) || (items[y.i].profit - items[x.i].profit) || (x.i - y.i));
  const indices = [];
  let weight = 0;
  let profit = 0;
  for (const e of order) {
    if (weight + items[e.i].weight <= capacity) {
      weight += items[e.i].weight;
      profit += items[e.i].profit;
      indices.push(e.i);
    }
  }
  return describe(items, indices);
}

/**
 * Sahni's PTAS shape: enumerate subsets A with |A| <= k, fill the remainder greedily.
 * greedyFiller=false swaps in the exact capacity DP as the filler, which makes the scheme
 * exact as soon as k >= |S*| (documented above and checked below).
 */
function sahniPtas(items, capacity, eps, greedyFiller = true, kOverride = null) {
  validate(items, capacity);
  if (!(eps > 0 && eps <= 1)) throw new RangeError("eps must lie in (0, 1]");
  const k = kOverride === null ? Math.ceil(1 / eps) : kOverride;
  if (!Number.isInteger(k) || k < 0 || k > 12) throw new RangeError("enumeration width k must be an integer in [0, 12]");
  let best = { profit: 0, weight: 0, indices: [] };
  const n = items.length;
  const chosen = [];
  const consider = (A) => {
    const banned = new Set(A);
    let baseWeight = 0;
    let baseProfit = 0;
    for (const i of A) { baseWeight += items[i].weight; baseProfit += items[i].profit; }
    if (baseWeight > capacity) return;
    const remaining = items.map((_, i) => i).filter((i) => !banned.has(i));
    let picked;
    if (greedyFiller) {
      picked = greedyFill(items, capacity - baseWeight, banned).indices;
    } else {
      // exact filler: DP on the surviving items, then map positions back to original indices
      const sub = exactKnapsack(remaining.map((i) => items[i]), capacity - baseWeight);
      picked = sub.indices.map((pos) => remaining[pos]);
    }
    const total = baseProfit + describe(items, picked).profit;
    if (total > best.profit) best = describe(items, [...A, ...picked]);
  };
  const recurse = (start) => {
    consider(chosen);
    if (chosen.length === k) return;
    for (let i = start; i < n; i++) {
      chosen.push(i);
      recurse(i + 1);
      chosen.pop();
    }
  };
  recurse(0);
  return { ...best, k };
}

function makeRng(seed) {
  let state = seed >>> 0 || 0x1b873593;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

function randomInstance(rng, n, maxW, maxP) {
  return {
    items: Array.from({ length: n }, () => ({ weight: rng(maxW), profit: rng(maxP) })),
    capacity: 5 + rng(6 * maxW),
  };
}

module.exports = {
  exactKnapsack, bruteForceKnapsack, roundedProfitDP, fptas, sahniPtas, greedyFill,
};

if (require.main === module) {
  const assert = require("node:assert");
  const { deepStrictEqual, strictEqual, ok, throws } = assert;
  const rng = makeRng(769769);
  let ratioMin = Infinity;
  let sahniRatioMin = Infinity;

  // -- degenerate instances ---------------------------------------------------------
  deepStrictEqual(exactKnapsack([], 10), { profit: 0, weight: 0, indices: [] }, "no items -> zero");
  deepStrictEqual(exactKnapsack([{ weight: 1, profit: 5 }], 0), { profit: 0, weight: 0, indices: [] }, "capacity 0 -> nothing fits");
  deepStrictEqual(exactKnapsack([{ weight: 4, profit: 9 }], 3), { profit: 0, weight: 0, indices: [] }, "single item too heavy");
  deepStrictEqual(exactKnapsack([{ weight: 2, profit: 0 }], 5), { profit: 0, weight: 0, indices: [] }, "zero-profit items never enter");
  const allFit = exactKnapsack([{ weight: 1, profit: 3 }, { weight: 2, profit: 4 }], 99);
  deepStrictEqual(allFit, { profit: 7, weight: 3, indices: [0, 1] }, "everything fits -> take all");
  strictEqual(exactKnapsack([{ weight: 0, profit: 7 }], 0).profit, 7, "zero-weight item is free");
  throws(() => exactKnapsack([{ weight: -1, profit: 1 }], 5), RangeError, "negative weight rejected");
  throws(() => exactKnapsack([{ weight: 1, profit: 1 }], -2), RangeError, "negative capacity rejected");
  throws(() => fptas([{ weight: 1, profit: 1 }], 5, 0), RangeError, "eps must be positive");
  throws(() => sahniPtas([{ weight: 1, profit: 1 }], 5, 0.5, true, 13), RangeError, "k above the enumeration cap rejected");

  // -- exact DP vs exhaustive enumeration (oracle) ----------------------------------
  for (let trial = 0; trial < 60; trial++) {
    const n = 1 + rng(12);
    const { items, capacity } = randomInstance(rng, n, 9, 15);
    const exact = exactKnapsack(items, capacity);
    const brute = bruteForceKnapsack(items, capacity);
    strictEqual(exact.profit, brute.profit, `exact DP disagrees with brute force on trial ${trial}`);
    strictEqual(exact.weight, describe(items, exact.indices).weight, "reported weight must match the indices");
    ok(exact.weight <= capacity, "the exact solution must be feasible");
    strictEqual(new Set(exact.indices).size, exact.indices.length, "indices must be distinct");
  }

  // -- FPTAS: analytic bound value >= OPT - n*K and the (1 - eps) ratio ---------------
  for (let trial = 0; trial < 40; trial++) {
    const n = 2 + rng(12);
    const { items, capacity } = randomInstance(rng, n, 12, 40);
    const opt = bruteForceKnapsack(items, capacity).profit;
    for (const eps of [0.5, 0.2, 0.1, 0.02]) {
      const approx = fptas(items, capacity, eps);
      ok(approx.weight <= capacity, "FPTAS solution must be feasible");
      ok(approx.profit <= opt, "no approximation may beat the optimum");
      ok(approx.profit >= opt - items.length * approx.K - 1e-9,
        `analytic bound violated: ${approx.profit} < ${opt} - ${items.length * approx.K}`);
      if (opt > 0) ratioMin = Math.min(ratioMin, approx.profit / opt);
      ok(approx.profit >= (1 - eps) * opt - 1e-9,
        `(1 - eps) guarantee violated for eps=${eps}: ${approx.profit} vs ${opt}`);
    }
  }

  // -- boundary: K = 1 rounds nothing, so the scaled DP is exact --------------------
  const sample = randomInstance(rng, 11, 8, 30);
  const exactSample = exactKnapsack(sample.items, sample.capacity);
  const unscaled = roundedProfitDP(sample.items, sample.capacity, 1);
  strictEqual(unscaled.profit, exactSample.profit, "K = 1 must reproduce the exact optimum");
  ok(unscaled.weight <= exactSample.weight,
    "the profit DP keeps the minimum-weight optimiser, so it cannot weigh more than the capacity DP");
  strictEqual(roundedProfitDP(sample.items, sample.capacity, 1e9).profit, 0, "huge K flattens every profit to zero");

  // -- Sahni enumeration: exact with an optimal filler once k >= |S*| ----------------
  deepStrictEqual(
    sahniPtas(sample.items, sample.capacity, 1, false, sample.items.length).profit,
    exactSample.profit,
    "enumerating every subset with an exact filler is exact",
  );
  let sahniTightRatio = 2;
  for (let trial = 0; trial < 25; trial++) {
    const { items, capacity } = randomInstance(rng, 8 + rng(7), 10, 25);
    const opt = bruteForceKnapsack(items, capacity).profit;
    if (opt === 0) continue;
    for (const eps of [0.5, 0.25, 0.2]) {
      // k = ceil(1/eps) stays well below n here, so the enumeration is genuinely partial
      const k = Math.min(6, Math.ceil(1 / eps));
      const approx = sahniPtas(items, capacity, eps, true, k);
      ok(approx.weight <= capacity, "Sahni solution must be feasible");
      ok(approx.profit <= opt, "no approximation may beat the optimum");
      const ratio = approx.profit / opt;
      sahniRatioMin = Math.min(sahniRatioMin, ratio);
      sahniTightRatio = Math.min(sahniTightRatio, ratio);
      ok(ratio >= 1 - eps, `Sahni (1 - eps) guarantee violated for eps=${eps} (k=${k}): ratio ${ratio}`);
    }
  }
  // A greedy filler is optimal when every remaining item still fits; the (1-eps) claim is
  // measured over the trials above, and the enumerator must dominate plain greedy.
  const greedyOnly = greedyFill(sample.items, sample.capacity, new Set());
  ok(sahniPtas(sample.items, sample.capacity, 0.5, true, 0).profit >= greedyOnly.profit,
    "the A = {} candidate is exactly the greedy solution, so Sahni can never be worse");
  // Hand-built instance where pure greedy (k = 0) is strictly suboptimal and enumerating one
  // extra item already recovers the optimum: this is the "PTAS idea" in miniature.
  const craft = {
    items: [{ weight: 2, profit: 9 }, { weight: 2, profit: 9 }, { weight: 3, profit: 10 }],
    capacity: 3,
  };
  const craftOpt = bruteForceKnapsack(craft.items, craft.capacity).profit;
  strictEqual(craftOpt, 10, "the crafted instance must have a known optimum");
  const craftGreedy = sahniPtas(craft.items, craft.capacity, 1, true, 0);
  const craftK1 = sahniPtas(craft.items, craft.capacity, 0.5, true, 1);
  strictEqual(craftGreedy.profit, 9, "pure greedy takes the two ratio-winners and wastes capacity");
  strictEqual(craftK1.profit, craftOpt, "enumerating singletons lets the greedy filler reach the optimum");
  ok(sahniRatioMin >= 0.5, `Sahni (1 - eps) claim with eps <= 0.5 held at ${sahniRatioMin}`);

  console.log(`fptas min ratio=${ratioMin.toFixed(4)} sahni min ratio=${sahniTightRatio.toFixed(4)}`);
  console.log("PASS");
}
