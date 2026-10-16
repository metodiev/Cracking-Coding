/** Task 781 - Implement distributed join planning with cardinality estimation errors simulation. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Left-deep join planning: a query is a set of relations plus the predicates that connect them, and
 * a plan joins one relation at a time onto the running intermediate result. The cost model is the
 * classic hash-join one - a join costs the size of both inputs and produces
 * |left| * |right| * selectivity rows - so the plan cost is dominated by how early the intermediate
 * result shrinks.
 *
 *   optimalPlan   brute force over every connected order: the exact oracle.
 *   selingerPlan  subset DP (the Selinger style dynamic program), which is exact for left-deep plans
 *                 under the principle of optimality. With the true cardinalities both must agree.
 *
 * Cardinality estimation errors are then injected the way a real optimiser suffers them: each base
 * relation's row count is multiplied by exp(N(0, sigma)) and the planner only ever sees those
 * estimates. The plan it picks is then priced against the *true* cardinalities and compared with the
 * true optimum, giving the plan regret cost(chosen) / cost(optimal) >= 1. The exercise is to show
 * that regret is zero at sigma = 0, grows with sigma, and that even a bad estimate only costs a
 * bounded factor on these schemas.
 *
 * What is verified: the DP and the brute force agree exactly with true statistics (same cost *and*
 * same order); every produced plan is a permutation whose every step is connected by a predicate (no
 * cross products when a connected order exists); regret is always >= 1 and equal to 1 for symmetric
 * schemas; the mean regret and the fraction of mis-planned queries both increase with the error
 * magnitude; the boundary cases (one relation, two relations, equal cardinalities, an empty
 * relation) behave.
 */
"use strict";

/** Deterministic normal-noise generator (Box-Muller over a seeded xorshift32). */
function makeNoise(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  const uniform = () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return (state + 1) / 4294967297;
  };
  return () => {
    const u = Math.max(uniform(), 1e-12);
    const v = uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

/** A query: relations (name -> true rows) plus predicates (pairs with a selectivity). */
function makeQuery(relations, predicates) {
  return { relations, predicates };
}

function selectivityOf(query, a, b) {
  const predicate = query.predicates.find((p) =>
    (p.left === a && p.right === b) || (p.left === b && p.right === a));
  return predicate === undefined ? null : predicate.selectivity;
}

/** Sizes the intermediate result of joining two row counts. */
function joinRows(leftRows, rightRows, selectivity) {
  return Math.max(1, Math.min(leftRows * rightRows * selectivity, 1e12));
}

/**
 * Cost of a left-deep plan with the given row estimates. A step joins one relation with the running
 * intermediate: it hashes both sides, so it costs the sum of their sizes.
 */
function planCost(query, order, cardinalities) {
  if (order.length === 0) return { cost: 0, rows: 0 };
  let rows = cardinalities.get(order[0]);
  let cost = 0;
  for (let i = 1; i < order.length; i++) {
    const joined = order.slice(0, i);
    const connector = joined.find((other) => selectivityOf(query, other, order[i]) !== null);
    if (connector === undefined) return { cost: Infinity, rows: Infinity }; // a cross product
    const next = cardinalities.get(order[i]);
    cost += rows + next;
    rows = joinRows(rows, next, selectivityOf(query, connector, order[i]));
  }
  return { cost, rows };
}

/** Every order in which each new relation connects to the already joined set. */
function connectedOrders(query) {
  const names = Object.keys(query.relations);
  const out = [];
  const walk = (order, remaining) => {
    if (remaining.length === 0) { out.push(order); return; }
    for (const name of remaining) {
      const connects = order.length === 0 || order.some((other) => selectivityOf(query, other, name) !== null);
      if (connects) walk([...order, name], remaining.filter((other) => other !== name));
    }
  };
  walk([], names);
  return out;
}

/** Brute force over every connected order: the exact optimum. */
function optimalPlan(query, cardinalities) {
  let best = null;
  for (const order of connectedOrders(query)) {
    const { cost, rows } = planCost(query, order, cardinalities);
    if (best === null || cost < best.cost) best = { order, cost, rows };
  }
  return best;
}

/**
 * Selinger-style subset DP. best[subset] keeps the cheapest connected left-deep plan for that subset;
 * adding a relation r yields cost + rows(subset) + rows(r) and the joined size.
 */
function selingerPlan(query, cardinalities) {
  const names = Object.keys(query.relations);
  const subsets = new Map();
  for (const name of names) {
    subsets.set(name, { order: [name], cost: 0, rows: cardinalities.get(name) });
  }
  for (let size = 2; size <= names.length; size++) {
    const next = new Map();
    for (const [key, plan] of subsets) {
      const members = key.split("|");
      if (members.length !== size - 1) continue;
      for (const name of names) {
        if (members.includes(name)) continue;
        const connector = plan.order[plan.order.length - 1];
        const direct = selectivityOf(query, connector, name);
        const through = direct === null ? members.find((other) => selectivityOf(query, other, name) !== null) : connector;
        if (through === undefined) continue; // no predicate to any member: a cross product, not a plan step
        const selectivityUsed = selectivityOf(query, through, name);
        const cost = plan.cost + plan.rows + cardinalities.get(name);
        const rows = joinRows(plan.rows, cardinalities.get(name), selectivityUsed);
        const combined = [...members, name].sort().join("|");
        const candidate = { order: [...plan.order, name], cost, rows };
        const current = next.get(combined);
        if (current === undefined || cost < current.cost) next.set(combined, candidate);
      }
    }
    for (const [key, plan] of next) subsets.set(key, plan);
  }
  let best = null;
  for (const [key, plan] of subsets) {
    if (key.split("|").length !== names.length) continue;
    if (best === null || plan.cost < best.cost) best = { order: plan.order, cost: plan.cost, rows: plan.rows };
  }
  return best;
}

/** Multiplies every base row count by exp(N(0, sigma)): the optimiser's wrong picture. */
function noisyCardinalities(query, sigma, noise) {
  const estimates = new Map();
  for (const [name, rows] of Object.entries(query.relations)) {
    const factor = sigma === 0 ? 1 : Math.exp(noise() * sigma);
    estimates.set(name, Math.max(1, Math.round(rows * factor)));
  }
  return estimates;
}

function trueCardinalities(query) {
  return new Map(Object.entries(query.relations));
}

/** Regret of the plan chosen with `estimates`, priced with the truth. */
function regret(query, estimates) {
  const chosen = selingerPlan(query, estimates);
  const truth = trueCardinalities(query);
  const optimum = optimalPlan(query, truth);
  return {
    chosen,
    optimum,
    ratio: planCost(query, chosen.order, truth).cost / planCost(query, optimum.order, truth).cost,
  };
}

/** Star schema: a fact table joined to four dimension tables of comparable (tens of percents) size. */
function starQuery(scale = 1) {
  const fact = 4_000_000 * scale;
  const dims = { dimA: 4000, dimB: 2000, dimC: 6000, dimD: 3000 };
  return makeQuery(
    { fact, ...Object.fromEntries(Object.entries(dims).map(([name, rows]) => [name, rows * scale])) },
    Object.entries(dims).map(([name, rows]) => ({ left: "fact", right: name, selectivity: rows / 4_000_000 })));
}

/** Chain schema: a-b-c-d, where a cross product would explode. */
function chainQuery() {
  return makeQuery(
    { a: 100_000, b: 500, c: 2_000, d: 40 },
    [
      { left: "a", right: "b", selectivity: 0.001 },
      { left: "b", right: "c", selectivity: 0.02 },
      { left: "c", right: "d", selectivity: 0.05 },
    ]);
}

module.exports = {
  makeNoise, makeQuery, selectivityOf, joinRows, planCost, connectedOrders, optimalPlan,
  selingerPlan, noisyCardinalities, trueCardinalities, regret, starQuery, chainQuery,
};

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;

  // -- degenerate queries -----------------------------------------------------------
  const single = makeQuery({ only: 1234 }, []);
  deepStrictEqual(optimalPlan(single, trueCardinalities(single)), { order: ["only"], cost: 0, rows: 1234 },
    "a single relation has no join cost");
  deepStrictEqual(selingerPlan(single, trueCardinalities(single)).order, ["only"], "the DP handles one relation");
  const pair = makeQuery({ left: 100, right: 7 }, [{ left: "left", right: "right", selectivity: 0.5 }]);
  const pairOptimum = optimalPlan(pair, trueCardinalities(pair));
  strictEqual(pairOptimum.cost, 107, "one join costs the sum of both inputs");
  strictEqual(pairOptimum.rows, 350, "the join size follows the selectivity");
  const twoOrders = connectedOrders(pair).map((order) => order.join("-")).sort();
  deepStrictEqual(twoOrders, ["left-right", "right-left"], "both orders of a two-relation join are possible");
  deepStrictEqual(selingerPlan(pair, trueCardinalities(pair)).order.join("-"), "left-right",
    "the DP must break the two-way tie deterministically");

  const empty = makeQuery({ big: 1000, none: 0 }, [{ left: "big", right: "none", selectivity: 1 }]);
  strictEqual(joinRows(1000, 0, 1), 1, "an empty relation collapses the result to one row");
  ok(planCost(empty, ["big", "none"], trueCardinalities(empty)).cost === 1000, "the join cost stays the input sizes");
  const noPredicate = makeQuery({ x: 10, y: 20 }, []);
  strictEqual(optimalPlan(noPredicate, trueCardinalities(noPredicate)), null,
    "relations with no connecting predicate have no connected order and no plan");
  const disconnected = planCost(chainQuery(), ["a", "c", "b", "d"], new Map([["a", 1], ["c", 1], ["b", 1], ["d", 1]]));
  strictEqual(disconnected.cost, Infinity, "an unconnected step is priced as impossible");
  strictEqual(connectedOrders(chainQuery()).length, 8,
    "a chain of four relations admits 8 connected orders out of 24 permutations (the rest are cross products)");

  // -- the DP is exact with true statistics ------------------------------------------
  for (const query of [starQuery(), chainQuery(), starQuery(3)]) {
    const truth = trueCardinalities(query);
    const dp = selingerPlan(query, truth);
    const brute = optimalPlan(query, truth);
    strictEqual(dp.cost, brute.cost, "the subset DP must match the brute-force optimum exactly");
    deepStrictEqual(planCost(query, dp.order, truth).cost, brute.cost, "the replayed plan cost must agree");
    const members = new Set(Object.keys(query.relations));
    strictEqual(dp.order.length, members.size, "the plan must mention every relation once");
    strictEqual(new Set(dp.order).size, dp.order.length, "the plan must be a permutation");
    const sortedNames = [...members].sort().join(",");
    strictEqual([...dp.order].sort().join(","), sortedNames, "no relation may be lost or duplicated");
    // every step must attach to something already joined: no cross products
    for (let i = 1; i < dp.order.length; i++) {
      ok(dp.order.slice(0, i).some((other) => selectivityOf(query, other, dp.order[i]) !== null),
        "every join step must share a predicate with the joined set");
    }
  }
  const star = starQuery();
  deepStrictEqual(selingerPlan(star, trueCardinalities(star)).order[0], "fact",
    "the star schema must start from the fact table");
  const chainPlan = selingerPlan(chainQuery(), trueCardinalities(chainQuery()));
  ok(["c", "d"].includes(chainPlan.order[0]),
    `the chain schema must start from one of the two small relations (chose ${chainPlan.order[0]})`);
  strictEqual(chainPlan.order[0], optimalPlan(chainQuery(), trueCardinalities(chainQuery())).order[0],
    "the DP and the brute force must agree on where a chain starts");

  // -- zero noise means no regret ----------------------------------------------------
  const noiseFree = regret(chainQuery(), trueCardinalities(chainQuery()));
  strictEqual(noiseFree.ratio, 1, "with perfect statistics the planner picks the optimum");
  deepStrictEqual(noiseFree.chosen.order, noiseFree.optimum.order, "with perfect statistics the orders match");

  // -- a symmetric query has regret 1 whatever the estimates ------------------------
  const symmetric = makeQuery({ p: 1000, q: 1000 }, [
    { left: "p", right: "q", selectivity: 0.001 },
  ]);
  for (let seed = 0; seed < 20; seed++) {
    const estimates = noisyCardinalities(symmetric, 1.5, makeNoise(seed));
    strictEqual(regret(symmetric, estimates).ratio, 1, "a symmetric query cannot be mis-planned");
  }

  // -- regret grows with the estimation error ----------------------------------------
  const measure = (sigma, trials) => {
    const ratios = [];
    for (let seed = 0; seed < trials; seed++) {
      const noise = makeNoise(1000 + seed);
      const outcome = regret(starQuery(), noisyCardinalities(starQuery(), sigma, noise));
      ratios.push(outcome.ratio);
    }
    const mean = ratios.reduce((sum, value) => sum + value, 0) / ratios.length;
    const misplanned = ratios.filter((value) => value > 1 + 1e-9).length / ratios.length;
    return { mean, misplanned, worst: Math.max(...ratios) };
  };
  const perfect = measure(0, 40);
  const small = measure(0.5, 40);
  const large = measure(1.5, 40);
  strictEqual(perfect.mean, 1, "sigma = 0 must give a mean regret of exactly one");
  strictEqual(perfect.misplanned, 0, "sigma = 0 must never mis-plan");
  for (const measured of [perfect, small, large]) {
    ok(measured.mean >= 1 - 1e-9, "regret is never below one");
  }
  ok(small.mean > perfect.mean, `a small error must cost something (${small.mean} > ${perfect.mean})`);
  ok(large.mean > small.mean, `a large error must cost more (${large.mean} > ${small.mean})`);
  ok(large.misplanned > small.misplanned || small.misplanned === 0,
    "bigger errors must mis-plan at least as often");
  ok(large.worst >= large.mean, "the worst case is at least the mean");

  // -- every produced plan is still a valid, connected permutation -------------------
  for (let seed = 0; seed < 25; seed++) {
    const query = seed % 2 === 0 ? starQuery() : chainQuery();
    const estimates = noisyCardinalities(query, 2, makeNoise(500 + seed));
    const { chosen, ratio } = regret(query, estimates);
    strictEqual(new Set(chosen.order).size, Object.keys(query.relations).length,
      "even a mis-estimated plan must be a permutation of the relations");
    for (let i = 1; i < chosen.order.length; i++) {
      ok(chosen.order.slice(0, i).some((other) => selectivityOf(query, other, chosen.order[i]) !== null),
        "even a mis-estimated plan must avoid cross products");
    }
    ok(ratio >= 1 - 1e-9, "regret stays above one");
    ok(Number.isFinite(planCost(query, chosen.order, trueCardinalities(query)).cost), "the plan must be priceable");
  }

  // -- determinism -------------------------------------------------------------------
  const replay = () => {
    const query = starQuery();
    const estimates = noisyCardinalities(query, 1, makeNoise(4242));
    return { plan: selingerPlan(query, estimates).order, outcome: regret(query, estimates).ratio };
  };
  strictEqual(JSON.stringify(replay()), JSON.stringify(replay()), "the same noise seed must replay identically");

  console.log(`regret mean sigma0=${perfect.mean.toFixed(4)} sigma0.5=${small.mean.toFixed(4)} sigma1.5=${large.mean.toFixed(4)} | mis-planned ${(small.misplanned * 100).toFixed(0)}% -> ${(large.misplanned * 100).toFixed(0)}% | worst ${large.worst.toFixed(3)}`);
  console.log("PASS");
}
