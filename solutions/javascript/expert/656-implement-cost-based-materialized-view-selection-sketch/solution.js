/** Task 656 - Implement cost-based materialized view selection sketch. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Choosing which materialised views to build: given candidate views (each with a storage size
 * and a build/maintenance cost) and a workload of queries with frequencies and base costs,
 * pick a subset fitting a storage budget that minimises total execution cost.
 *
 * SKETCH — what is deliberately simplified, and what is still really verified:
 *   * cost is a linear "what-if" model: a query answered by a view costs `viewReadCost * freq`
 *     (instead of the full base-table cost), the maintenance term is `viewWriteCost * freq`
 *     while a view exists, and building a view costs `buildCost` once. Real optimisers also
 *     estimate join orders, index interaction and cache effects — none of that is modelled.
 *   * a view "covers" a query by declaration (a real optimiser would test logical equivalence
 *     of the query against the view definition).
 *   What is verified for real: the storage budget and the cost accounting are exact, every
 *   candidate's marginal value is recomputed from scratch and cross-checked against the
 *   benefit-per-space greedy's incremental arithmetic, and the greedy result is compared with
 *   an exhaustive optimum over all subsets within the same budget.
 *
 * Greedy here is the size-aware rule an advisor actually uses: repeatedly add the view with the
 * best cost reduction *per KiB consumed*, stopping when nothing fits or nothing helps.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

function makeView(name, sizeKiB, buildCost, covers, options = {}) {
  if (!(sizeKiB > 0)) throw new RangeError(`view ${name} needs a positive size`);
  if (!(buildCost >= 0)) throw new RangeError(`view ${name} has a negative build cost`);
  if (!Array.isArray(covers) || covers.length === 0) throw new RangeError(`view ${name} must cover something`);
  return { name, sizeKiB, buildCost, covers: covers.slice(), readCost: options.readCost ?? 1 };
}

function makeQuery(id, frequency, baseReadCost) {
  if (!(frequency > 0)) throw new RangeError(`query ${id} needs a positive frequency`);
  if (!(baseReadCost > 0)) throw new RangeError(`query ${id} needs a positive cost`);
  return { id, frequency, baseReadCost };
}

class ViewSelectionModel {
  constructor(views, workload, options = {}) {
    this.views = views.slice();
    this.workload = workload.slice();
    this.writeCostPerUse = options.writeCostPerUse ?? 0.05;
    this.queryIndex = new Map(this.workload.map((q, i) => [q.id, i]));
    for (const view of this.views) {
      for (const id of view.covers) {
        if (!this.queryIndex.has(id)) throw new RangeError(`view ${view.name} covers unknown query ${id}`);
      }
    }
  }

  /** Cost of answering one query when `selected` views are available. */
  queryCost(selected, query) {
    let best = query.baseReadCost;
    for (const view of selected) {
      if (!view.covers.includes(query.id)) continue;
      if (view.readCost < best) best = view.readCost;
    }
    return best;
  }

  /** Total cost: workload reads + per-use view maintenance + the one-off build costs. */
  cost(selected) {
    let total = 0;
    for (const query of this.workload) {
      total += query.frequency * this.queryCost(selected, query);
    }
    for (const view of selected) {
      total += view.buildCost;
      const uses = this.workload.filter((q) => view.covers.includes(q.id))
        .reduce((sum, q) => sum + q.frequency, 0);
      total += this.writeCostPerUse * uses;
    }
    return total;
  }

  baselineCost() {
    return this.workload.reduce((sum, q) => sum + q.frequency * q.baseReadCost, 0);
  }

  sizeOf(selected) {
    return selected.reduce((sum, view) => sum + view.sizeKiB, 0);
  }
}

/** Size-aware greedy: best (cost reduction / KiB) first, until nothing fits or helps. */
function greedySelectViews(model, budgetKiB) {
  const chosen = [];
  const remaining = new Set(model.views.keys());
  const trace = [{ chosen: [], cost: model.cost([]), marginal: null }];
  let used = 0;
  for (;;) {
    const before = model.cost(chosen);
    let best = null;
    let bestScore = 0;
    for (const index of remaining) {
      const view = model.views[index];
      if (used + view.sizeKiB > budgetKiB) continue;
      const saving = before - model.cost(chosen.concat([view]));
      const score = saving / view.sizeKiB;
      if (saving > 0 && score > bestScore) {
        bestScore = score;
        best = index;
      }
    }
    if (best === null) break;
    const picked = model.views[best];
    chosen.push(picked);
    used += picked.sizeKiB;
    remaining.delete(best);
    trace.push({ chosen: chosen.map((v) => v.name), cost: model.cost(chosen), marginal: bestScore });
  }
  return { chosen: chosen.map((v) => v.name), views: chosen, cost: model.cost(chosen), trace };
}

/** Exhaustive optimum over all subsets that fit the budget (small instances only). */
function exhaustiveSelectViews(model, budgetKiB) {
  const n = model.views.length;
  if (n > 18) throw new RangeError("exhaustive search is limited to 18 candidate views");
  let best = { views: [], cost: model.cost([]) };
  let feasible = 1;
  for (let mask = 1; mask < 1 << n; mask++) {
    const subset = [];
    let size = 0;
    for (let b = 0; b < n; b++) {
      if (mask & (1 << b)) {
        subset.push(model.views[b]);
        size += model.views[b].sizeKiB;
      }
    }
    if (size > budgetKiB) continue;
    feasible++;
    const cost = model.cost(subset);
    if (cost < best.cost) best = { views: subset, cost };
  }
  return { chosen: best.views.map((v) => v.name), views: best.views, cost: best.cost, feasible };
}

module.exports = {
  makeView, makeQuery, ViewSelectionModel, greedySelectViews, exhaustiveSelectViews,
};

if (require.main === module) {
  // -- validation ---------------------------------------------------------------
  throws(() => makeView("bad", 0, 1, ["q"]), RangeError, "a view needs a positive size");
  throws(() => makeView("bad", 1, -1, ["q"]), RangeError, "a view cannot have a negative build cost");
  throws(() => makeView("bad", 1, 1, []), RangeError, "a view must cover at least one query");
  throws(() => makeQuery("q", 0, 1), RangeError, "a query needs a positive frequency");
  throws(() => makeQuery("q", 1, 0), RangeError, "a query needs a positive cost");
  throws(() => new ViewSelectionModel([makeView("v", 1, 0, ["nope"])], [makeQuery("q", 1, 10)]),
    RangeError, "a view covering an unknown query must be rejected");

  // -- baseline is the hand-computable sum of base read costs --------------------
  const workload = [
    makeQuery("q_scan", 100, 400),
    makeQuery("q_daily", 50, 900),
    makeQuery("q_join", 20, 5000),
    makeQuery("q_rare", 2, 800),
  ];
  const views = [
    makeView("v_daily", 200, 500, ["q_daily"]),
    makeView("v_join", 900, 4000, ["q_join"]),
    makeView("v_both", 1100, 6000, ["q_daily", "q_join"], { readCost: 40 }),
    makeView("v_all", 1400, 9000, ["q_scan", "q_daily", "q_join", "q_rare"], { readCost: 60 }),
    makeView("v_rare", 300, 100, ["q_rare"]),
    makeView("v_pig", 500, 50_000, ["q_scan"]), // build cost exceeds any possible saving
  ];
  const model = new ViewSelectionModel(views, workload);
  strictEqual(model.baselineCost(), 100 * 400 + 50 * 900 + 20 * 5000 + 2 * 800);
  strictEqual(model.cost([]), model.baselineCost(), "no views means the baseline cost");
  strictEqual(model.sizeOf([]), 0);

  // -- budget discipline ---------------------------------------------------------
  const budget = 1600;
  const greedy = greedySelectViews(model, budget);
  ok(model.sizeOf(greedy.views) <= budget, `greedy must fit the budget, used ${model.sizeOf(greedy.views)} KiB`);
  ok(!greedy.chosen.includes("v_pig"), "a view whose build cost outweighs its total benefit is never chosen");
  ok(model.cost([model.views[5]]) > model.baselineCost(), "that view alone would make the workload worse");
  ok(greedy.cost < model.baselineCost(), "the chosen views must beat the baseline");
  for (const step of greedy.trace) {
    const subset = step.chosen.map((name) => model.views.find((v) => v.name === name));
    strictEqual(step.cost, model.cost(subset), "trace cost must equal a fresh from-scratch evaluation");
  }
  deepStrictEqual(greedySelectViews(model, budget), greedy, "selection must be deterministic");

  // -- greedy vs exhaustive ------------------------------------------------------
  const optimum = exhaustiveSelectViews(model, budget);
  ok(greedy.cost >= optimum.cost, "an exhaustive optimum can never be worse than greedy");
  const captured = (model.baselineCost() - greedy.cost) / (model.baselineCost() - optimum.cost);
  // greedy is a heuristic: this is a deterministic regression guard, not a theorem.
  ok(captured >= 0.8, `greedy captured only ${(captured * 100).toFixed(1)}% of the optimal benefit`);
  const relaxed = exhaustiveSelectViews(model, budget * 2);
  ok(relaxed.cost <= optimum.cost, "a larger budget can only lower the optimum");
  const constrained = exhaustiveSelectViews(model, 250);
  ok(constrained.cost >= optimum.cost, "a smaller budget can only raise the optimum");
  ok(optimum.feasible > 4 && optimum.feasible < 64, "the enumeration must skip the infeasible subsets");

  // -- budget edge cases ---------------------------------------------------------
  const exact = greedySelectViews(model, model.views[0].sizeKiB);
  ok(exact.views.every((v) => v.sizeKiB <= model.views[0].sizeKiB), "only views that fit exactly are used");
  ok(greedySelectViews(model, 1).chosen.length === 0, "a 1 KiB budget admits nothing");
  strictEqual(greedySelectViews(model, 0).cost, model.baselineCost(), "a zero budget changes nothing");
  const emptyModel = new ViewSelectionModel([], []);
  strictEqual(emptyModel.baselineCost(), 0);
  deepStrictEqual(greedySelectViews(emptyModel, budget).chosen, [], "no candidates means nothing to build");
  const useless = new ViewSelectionModel([makeView("v_useless", 1, 5, ["q"], { readCost: 50 })],
    [makeQuery("q", 1, 50)]);
  deepStrictEqual(greedySelectViews(useless, 1000).chosen, [],
    "a view that cannot beat the base read cost is refused even when the budget is huge");
  deepStrictEqual(exhaustiveSelectViews(useless, 1000).chosen, [], "and the optimum agrees");

  // -- a view that covers nothing is rejected, but a duplicate view is legal ------
  const dupViews = [makeView("a", 100, 0, ["q1"]), makeView("b", 100, 0, ["q1"]), makeView("c", 100, 0, ["q1"])];
  const dupModel = new ViewSelectionModel(dupViews, [makeQuery("q1", 10, 1000)]);
  const dupSelection = greedySelectViews(dupModel, 1000);
  strictEqual(dupSelection.chosen.length, 1, "the second identical view has zero marginal benefit");
  deepStrictEqual(exhaustiveSelectViews(dupModel, 1000).chosen, dupSelection.chosen);

  // -- read cost ordering: a cheaper covering view always wins -------------------
  const cheap = new ViewSelectionModel([
    makeView("slow", 10, 0, ["q"], { readCost: 100 }),
    makeView("fast", 10, 0, ["q"], { readCost: 5 }),
  ], [makeQuery("q", 10, 1000)]);
  deepStrictEqual(greedySelectViews(cheap, 10).chosen, ["fast"], "the cheaper read path is preferred");
  deepStrictEqual(greedySelectViews(cheap, 20).chosen, ["fast"],
    "a redundant second view only adds maintenance cost, so it is refused");
  ok(cheap.cost([cheap.views[0], cheap.views[1]]) > cheap.cost([cheap.views[1]]),
    "covering the same query twice must cost more");

  // -- a wider candidate set: budget-limited greedy on 12 candidates -------------
  const wideViews = Array.from({ length: 12 }, (_, i) =>
    makeView(`w${i}`, 100 + 50 * i, 200 + 100 * i, [`q${i}`], { readCost: 50 + i }));
  const wideWorkload = Array.from({ length: 12 }, (_, i) => makeQuery(`q${i}`, 10 + 5 * i, 1000 + 100 * i));
  const wideModel = new ViewSelectionModel(wideViews, wideWorkload);
  const wideGreedy = greedySelectViews(wideModel, 900);
  ok(wideModel.sizeOf(wideGreedy.views) <= 900, "the wide instance must respect its budget too");
  ok(wideGreedy.cost < wideModel.baselineCost());
  strictEqual(wideModel.cost(wideGreedy.views), wideGreedy.cost, "the returned cost is recomputed identically");

  console.log(`views=${views.length} budget=${budget}KiB greedy=${greedy.cost.toFixed(0)} ` +
    `optimum=${optimum.cost.toFixed(0)} baseline=${model.baselineCost()} captured=${(captured * 100).toFixed(1)}%`);
  console.log("PASS");
}
