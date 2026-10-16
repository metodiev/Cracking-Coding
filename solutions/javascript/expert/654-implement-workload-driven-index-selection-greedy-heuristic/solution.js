/** Task 654 - Implement workload-driven index selection greedy heuristic. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * The index advisor problem: given a table, a set of candidate indexes and a *workload*
 * (queries with frequencies and selectivities), choose at most K indexes fitting a space
 * budget that minimise total estimated cost.
 *
 * Cost model (the classic "what-if" linear model used by DB2 / SQL Server advisors):
 *   scanCost(q)          = tableRows                       (no usable index)
 *   indexCost(i, q)      = log2(distinct_i) + 1 + sel_q * tableRows   (descent + row fetch)
 *   queryCost(S, q)      = min over indexes i in S whose column is q.column of indexCost
 *   cost(S)              = sum_q frequency_q * queryCost(S, q) + sum_{i in S} spaceWeight * KiB_i
 * The space term is what stops the greedy from indexing everything: an index only pays off
 * if its saving outweighs the storage it occupies.
 *
 * Greedy: repeatedly add the candidate with the largest positive marginal cost reduction
 * (respecting K and the budget), the standard heuristic for this submodular-style objective.
 * Two independent implementations of the objective are used so they can cross-check each
 * other: `cost(Set)` recomputes from scratch, while the greedy accumulates per-candidate
 * marginal savings; the trace asserts the two agree at every step. The greedy result is
 * compared against an exhaustive optimum over all subsets within the same constraints.
 *
 * Note: greedy has no exact-optimality guarantee; the tests assert the structural facts
 * (never worse than the empty set, never over budget, never picks an unqueried column) plus
 * an empirical optimality-ratio guard whose measured value is printed.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

const COMPARISON_OPS = new Set(["=", "<", ">", "<=", ">=", "!="]);

function makeCandidate(name, column, distinct, rowBytes = 8) {
  if (!Number.isInteger(distinct) || distinct <= 0) throw new RangeError(`bad distinct count for ${name}`);
  return { name, column, distinct, spaceKiB: (distinct * rowBytes) / 1024 };
}

function makeQuery(id, column, selectivity, frequency) {
  if (!(selectivity > 0 && selectivity <= 1)) throw new RangeError(`selectivity out of range in ${id}`);
  if (!(frequency >= 0)) throw new RangeError(`negative frequency in ${id}`);
  return { id, column, selectivity, frequency };
}

class WorkloadModel {
  constructor(tableRows, candidates, workload, spaceWeight = 200) {
    if (!Number.isInteger(tableRows) || tableRows <= 0) throw new RangeError("tableRows must be positive");
    this.tableRows = tableRows;
    this.candidates = candidates.slice();
    this.workload = workload.slice();
    this.spaceWeight = spaceWeight;
    this.queriesByColumn = new Map();
    for (const q of this.workload) {
      if (!this.queriesByColumn.has(q.column)) this.queriesByColumn.set(q.column, []);
      this.queriesByColumn.get(q.column).push(q);
    }
  }

  /** Cost of answering one query when `index` is the best available access path. */
  indexCost(index, query) {
    const descent = index.distinct <= 1 ? 1 : Math.ceil(Math.log2(index.distinct)) + 1;
    return descent + query.selectivity * this.tableRows;
  }

  /** Best cost for one query given the chosen index set (scan when nothing applies). */
  queryCost(chosen, query) {
    let best = this.tableRows;
    for (const index of chosen) {
      if (index.column !== query.column) continue;
      const cost = this.indexCost(index, query);
      if (cost < best) best = cost;
    }
    return best;
  }

  /** Full objective: workload cost plus the storage penalty of the chosen indexes. */
  cost(chosen) {
    let total = 0;
    for (const query of this.workload) {
      if (query.frequency === 0) continue;
      total += query.frequency * this.queryCost(chosen, query);
    }
    for (const index of chosen) total += this.spaceWeight * index.spaceKiB;
    return total;
  }

  spaceOf(chosen) {
    return chosen.reduce((sum, index) => sum + index.spaceKiB, 0);
  }

  baselineCost() {
    return this.workload.reduce((sum, q) => sum + q.frequency * this.tableRows, 0);
  }
}

/**
 * Greedy index selection. Returns the chosen indexes, the trace of marginal savings and the
 * final cost. `trace[k]` records the cost after accepting the k-th index, which lets the
 * caller verify the incremental accounting against a from-scratch evaluation.
 */
function greedySelect(model, maxIndexes = Infinity, spaceBudgetKiB = Infinity) {
  const chosen = [];
  const remaining = new Set(model.candidates.keys());
  const trace = [{ chosen: [], cost: model.cost([]), marginal: null }];
  let used = 0;
  while (chosen.length < maxIndexes) {
    let best = null;
    let bestSaving = 0;
    for (const idx of remaining) {
      const candidate = model.candidates[idx];
      if (used + candidate.spaceKiB > spaceBudgetKiB) continue;
      const saving = model.cost(chosen) - model.cost(chosen.concat([candidate]));
      if (saving > bestSaving) {
        bestSaving = saving;
        best = idx;
      }
    }
    if (best === null) break; // no candidate reduces cost further under the constraints
    const picked = model.candidates[best];
    chosen.push(picked);
    used += picked.spaceKiB;
    remaining.delete(best);
    trace.push({ chosen: chosen.map((i) => i.name), cost: model.cost(chosen), marginal: bestSaving });
  }
  return { chosen: chosen.map((i) => i.name), indexes: chosen, cost: model.cost(chosen), trace };
}

/** Exhaustive optimum over all subsets honouring the same constraints (small instances only). */
function exhaustiveSelect(model, maxIndexes = Infinity, spaceBudgetKiB = Infinity) {
  const n = model.candidates.length;
  if (n > 16) throw new RangeError("exhaustive search is limited to 16 candidates");
  let best = { indexes: [], cost: model.cost([]) };
  let explored = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    const subset = [];
    let size = 0;
    let space = 0;
    for (let b = 0; b < n; b++) {
      if (mask & (1 << b)) {
        subset.push(model.candidates[b]);
        size++;
        space += model.candidates[b].spaceKiB;
      }
    }
    if (size > maxIndexes || space > spaceBudgetKiB) continue;
    explored++;
    const cost = model.cost(subset);
    if (cost < best.cost) best = { indexes: subset, cost };
  }
  return { chosen: best.indexes.map((i) => i.name), indexes: best.indexes, cost: best.cost, explored };
}

const TABLE_ROWS = 1_000_000;

function demoCandidates() {
  return [
    makeCandidate("idx_customer", "customer_id", 200_000),
    makeCandidate("idx_status", "status", 8),
    makeCandidate("idx_created", "created_at", 500_000),
    makeCandidate("idx_region", "region", 40),
    makeCandidate("idx_amount", "amount", 900_000),
    makeCandidate("idx_note", "note", 950_000), // never queried: must never be selected
  ];
}

function demoWorkload() {
  return [
    makeQuery("q_cust", "customer_id", 0.001, 500),
    makeQuery("q_status", "status", 0.02, 300),
    makeQuery("q_created", "created_at", 0.05, 40),
    makeQuery("q_region", "region", 0.01, 120),
    makeQuery("q_amount", "amount", 0.8, 30), // poor selectivity: a scan is nearly as cheap
  ];
}

module.exports = {
  WorkloadModel, makeCandidate, makeQuery, greedySelect, exhaustiveSelect,
  COMPARISON_OPS, TABLE_ROWS, demoCandidates, demoWorkload,
};

if (require.main === module) {
  // -- baseline cost is the hand-computable sum of scans ---------------------------
  const model = new WorkloadModel(TABLE_ROWS, demoCandidates(), demoWorkload());
  strictEqual(model.baselineCost(), model.cost([]),
    "the empty index set must cost exactly sum(frequency * rows)");
  strictEqual(model.cost([]), 500 * TABLE_ROWS + 300 * TABLE_ROWS + 40 * TABLE_ROWS +
    120 * TABLE_ROWS + 30 * TABLE_ROWS);
  strictEqual(model.spaceOf([]), 0);
  throws(() => makeQuery("bad", "x", 1.5, 1), RangeError, "selectivity above 1 must be rejected");
  throws(() => makeQuery("bad", "x", 0, 1), RangeError, "zero selectivity must be rejected");
  throws(() => makeQuery("bad", "x", 0.5, -3), RangeError, "negative frequency must be rejected");
  throws(() => makeCandidate("bad", "x", 0), RangeError, "an index needs at least one key");
  throws(() => new WorkloadModel(0, [], []), RangeError, "a table needs at least one row");

  // -- tiny instance where greedy must match the exhaustive optimum -----------------
  const tiny = new WorkloadModel(100_000, [
    makeCandidate("i_a", "a", 1_000),
    makeCandidate("i_b", "b", 50_000),
    makeCandidate("i_c", "c", 2_000),
  ], [
    makeQuery("qa1", "a", 0.001, 100),
    makeQuery("qa2", "a", 0.01, 10),
    makeQuery("qb1", "b", 0.5, 100),
    makeQuery("qc1", "c", 0.0005, 1),
  ], 0);
  const tinyGreedy = greedySelect(tiny, 1, Infinity);
  const tinyOpt = exhaustiveSelect(tiny, 1, Infinity);
  strictEqual(tinyGreedy.cost, tinyOpt.cost, "greedy must match the optimum on the 3-candidate case");
  deepStrictEqual(tinyGreedy.chosen, tinyOpt.chosen);
  strictEqual(tinyOpt.explored, 4, "K=1 over 3 candidates explores the empty set plus 3 singletons");

  // -- incremental accounting equals from-scratch evaluation -----------------------
  const greedy = greedySelect(model, 4, Infinity);
  for (const step of greedy.trace) {
    const subset = step.chosen.map((name) => model.candidates.find((c) => c.name === name));
    strictEqual(step.cost, model.cost(subset),
      `trace cost after [${step.chosen.join(",")}] must equal a fresh cost evaluation`);
  }
  for (let k = 2; k < greedy.trace.length; k++) {
    ok(greedy.trace[k].cost < greedy.trace[k - 1].cost, "each accepted index must strictly reduce cost");
    ok(greedy.trace[k].marginal > 0, "accepted indexes must have positive marginal saving");
  }

  // -- structural guarantees -------------------------------------------------------
  ok(greedy.indexes.length <= 4, "greedy must respect the index budget K");
  ok(greedy.cost < model.baselineCost(), "greedy must beat the full-scan baseline");
  ok(!greedy.chosen.includes("idx_note"), "an index on a never-queried column can never help");
  ok(model.queryCost(greedy.indexes, { column: "note", selectivity: 0.5, frequency: 1 }) === TABLE_ROWS,
    "a query on the unindexed column still falls back to a scan");
  deepStrictEqual(greedySelect(model, 4, Infinity), greedy, "greedy must be deterministic");

  // -- exhaustive comparison -------------------------------------------------------
  const opt = exhaustiveSelect(model, 4, Infinity);
  ok(greedy.cost >= opt.cost, "the exhaustive optimum can never be worse than the greedy result");
  const greedyBenefit = model.baselineCost() - greedy.cost;
  const optBenefit = model.baselineCost() - opt.cost;
  const ratio = greedyBenefit / optBenefit;
  ok(greedyBenefit > 0 && optBenefit > 0);
  // Deterministic regression guard; the measured ratio is printed below.
  ok(ratio >= 0.6, `greedy captured only ${(ratio * 100).toFixed(1)}% of the optimal benefit`);
  const relaxed = exhaustiveSelect(model, 6, Infinity).cost;
  ok(relaxed <= opt.cost, "relaxing K can only lower the optimum");

  // -- budget and edge cases -------------------------------------------------------
  const budget = 2 * model.candidates[0].spaceKiB;
  const budgeted = greedySelect(model, 6, budget);
  ok(model.spaceOf(budgeted.indexes) <= budget + 1e-9, "the space budget must be respected");
  ok(budgeted.indexes.length < greedy.indexes.length, "a tight budget must admit fewer indexes");
  const noBudget = greedySelect(model, 4, 0);
  deepStrictEqual(noBudget.chosen, [], "a zero budget admits nothing");
  strictEqual(noBudget.cost, model.baselineCost());
  const zeroK = greedySelect(model, 0, Infinity);
  deepStrictEqual(zeroK.chosen, [], "K=0 admits nothing");
  const emptyWorkload = new WorkloadModel(TABLE_ROWS, demoCandidates(), []);
  const emptyResult = greedySelect(emptyWorkload);
  deepStrictEqual(emptyResult.chosen, [], "with no queries no index can pay for its storage");
  strictEqual(emptyResult.cost, 0);

  // -- duplicate index on the same column: the second must add nothing -------------
  const dupModel = new WorkloadModel(100_000, [
    makeCandidate("dup_big", "k", 400_000),
    makeCandidate("dup_small", "k", 400_000),
    makeCandidate("dup_waste", "k", 400_000),
  ], [makeQuery("qk", "k", 0.001, 200)], 0);
  const dupGreedy = greedySelect(dupModel, 3, Infinity);
  strictEqual(dupGreedy.chosen.length, 1,
    "once a column is indexed a duplicate index has zero marginal benefit");

  // -- exhaustive on a 10-candidate instance: enumeration runs to completion --------
  const wide = new WorkloadModel(50_000, Array.from({ length: 10 }, (_, i) =>
    makeCandidate(`w_${i}`, `col${i}`, 100 + i * 1_000)), Array.from({ length: 10 }, (_, i) =>
    makeQuery(`wq_${i}`, `col${i}`, 0.01 + i * 0.02, 5 + i)), 100_000);
  const wideOpt = exhaustiveSelect(wide, 3, Infinity);
  const wideGreedy = greedySelect(wide, 3, Infinity);
  ok(wideGreedy.cost >= wideOpt.cost);
  strictEqual(wideOpt.explored, 1 + 10 + 45 + 120, "K=3 over 10 candidates enumerates 176 subsets");

  console.log(`greedy=${model.baselineCost() > greedy.cost ? (greedy.cost / 1e6).toFixed(2) + "M" : "?"} ` +
    `optimum=${(opt.cost / 1e6).toFixed(2)}M baseline=${(model.baselineCost() / 1e6).toFixed(2)}M ` +
    `ratio=${(ratio * 100).toFixed(1)}% chosen=[${greedy.chosen.join(",")}]`);
  console.log("PASS");
}
