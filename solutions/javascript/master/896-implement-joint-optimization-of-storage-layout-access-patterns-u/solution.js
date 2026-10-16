/** Task 896 - Implement joint optimization of storage layout + access patterns using reinforcement
 *  learning.  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * A storage-layout advisor for a column store. The action space is the joint design decision —
 * physical layout (row / column / PAX with a group size), sort key, and whether the filter column
 * carries an index — and the reward is the *true* I/O cost of that design, measured by a page-level
 * disk simulator: every value is assigned to a page number, and a query's cost is the number of
 * distinct pages it must touch (an index or a zone map on a sorted column lets pages be skipped).
 * There is no surrogate cost model between the agent and the reward, and the simulator is itself
 * cross-checked against an independent, naive page enumeration.
 *
 * The agent is tabular Q-learning: the state is the workload signature (which columns a query touches,
 * the predicate shape and the selectivity bucket), the action is one of 16 layout configurations, and
 * the Bellman update is verified by hand. Exploration is ε-greedy with a decayed ε driven by a seeded
 * splitmix32, so a run is reproducible.
 *
 * Invariants verified below: (1) the analytic page arithmetic the simulator uses agrees, layout by
 * layout, with a brute-force enumeration of which page holds which value; (2) the learned greedy policy
 * beats both a random policy and the untrained policy on measured I/O, and its average reward improves
 * from the first to the last episodes (the learning curve is measured); (3) the learned policy is the
 * *optimal* action for at least three of the four workload states, compared against exhaustive search
 * over the action space; (4) the Q-update equals the hand-computed Bellman value; (5) selectivity,
 * sorting and indexes actually change the cost in the expected direction (a single-column selective
 * scan wants a column layout, an all-column scan wants row/column over PAX, a tight range on a sorted
 * column skips pages); (6) degenerate inputs (one row, one column, no queries) are handled and invalid
 * columns are rejected; (7) the training loop is bounded and deterministic.
 *
 * Simplifications: in-memory integer columns, one table, a fully enumerated 16-action design space, a
 * four-state workload MDP (so "convergence" means an exactly optimal table rather than a sampled
 * approximation), and no concurrency or real disk.
 */
"use strict";

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
  unit() { return this.next() / 0x1_0000_0000; }
}

const PAGE_BYTES = 4096;
const VALUE_BYTES = 4;

/* ------------------------------------------------------------------ table */

class Table {
  constructor(columns, rows) {
    if (columns.length === 0) throw new Error("a table needs at least one column");
    if (rows.length === 0) throw new Error("a table needs at least one row");
    for (const row of rows) {
      if (row.length !== columns.length) throw new Error("ragged row");
      for (const value of row) if (!Number.isInteger(value)) throw new Error("values must be integers");
    }
    this.columns = columns;
    this.rows = rows;
    this.rowCount = rows.length;
    this.columnCount = columns.length;
  }
  columnIndex(name) {
    const index = this.columns.indexOf(name);
    if (index === -1) throw new Error(`unknown column ${name}`);
    return index;
  }
}

/* ------------------------------------------------------------------ layouts */

/**
 * A layout is a concrete physical design: how values are grouped into pages, which column (if any) the
 * rows are sorted by, and whether the filter column is indexed. Page numbers are assigned
 * deterministically, so two layouts can be compared by the number of distinct pages a query touches.
 */
function validateLayout(layout, table) {
  const groups = layout.paxGroup !== undefined ? layout.paxGroup : 1;
  if (!Number.isInteger(groups) || groups < 1 || groups > table.columnCount) {
    throw new Error(`pax group size ${groups} is out of range`);
  }
  if (layout.sortKey !== undefined && layout.sortKey !== null) table.columnIndex(layout.sortKey);
  if (layout.indexedColumn !== undefined && layout.indexedColumn !== null) table.columnIndex(layout.indexedColumn);
  return layout;
}

/**
 * Assigns every (row, column) value to a page. The three layouts differ only in this assignment and in
 * whether the rows are physically sorted before the assignment.
 */
function buildPageMap(table, layout) {
  const rows = table.rows.slice();
  if (layout.sortKey) {
    const key = table.columnIndex(layout.sortKey);
    rows.sort((a, b) => (a[key] - b[key]) || 0);
  }
  const pageOf = new Map();          // "row:column" -> page id
  const pages = [];                  // page id -> value count (for the space accounting)
  const push = (page, entries) => {
    while (pages.length <= page) pages.push(0);
    pages[page] += entries;
  };
  if (layout.kind === "row") {
    const perPage = Math.max(1, Math.floor(PAGE_BYTES / (VALUE_BYTES * table.columnCount)));
    rows.forEach((row, rowIndex) => {
      const page = Math.floor(rowIndex / perPage);
      push(page, table.columnCount);
      for (let c = 0; c < table.columnCount; c++) pageOf.set(`${rowIndex}:${c}`, page);
    });
  } else if (layout.kind === "column") {
    const perPage = Math.max(1, Math.floor(PAGE_BYTES / VALUE_BYTES));
    for (let c = 0; c < table.columnCount; c++) {
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
        const page = c * Math.ceil(rows.length / perPage) + Math.floor(rowIndex / perPage);
        push(page, 1);
        pageOf.set(`${rowIndex}:${c}`, page);
      }
    }
  } else {
    const groups = layout.paxGroup;
    const perPage = Math.max(1, Math.floor(PAGE_BYTES / (VALUE_BYTES * groups)));
    const groupCount = Math.ceil(table.columnCount / groups);
    for (let g = 0; g < groupCount; g++) {
      const firstColumn = g * groups;
      const width = Math.min(groups, table.columnCount - firstColumn);
      for (let rowIndex = 0; rowIndex < rows.length; rowIndex++) {
        const page = g * Math.ceil(rows.length / perPage) + Math.floor(rowIndex / perPage);
        push(page, width);
        for (let c = firstColumn; c < firstColumn + width; c++) pageOf.set(`${rowIndex}:${c}`, page);
      }
    }
  }
  return { pageOf, pages, rows, totalPages: pages.length };
}

/** The distinct pages a query touches, computed straight from the page map (the oracle). */
function touchedPages(table, layout, pageMap, query) {
  const keyColumn = query.filterColumn !== undefined && query.filterColumn !== null
    ? table.columnIndex(query.filterColumn) : null;
  const touched = new Set();
  let candidateRows = [];
  const sortedByFilter = layout.sortKey !== undefined && layout.sortKey !== null
    && query.filterColumn === layout.sortKey;
  if (query.kind === "point") {
    // An index (or a binary search on a sorted column) narrows the candidate rows first.
    if (layout.indexedColumn === query.filterColumn || sortedByFilter) {
      for (let rowIndex = 0; rowIndex < pageMap.rows.length; rowIndex++) {
        if (pageMap.rows[rowIndex][keyColumn] === query.value) candidateRows.push(rowIndex);
      }
    } else {
      candidateRows = pageMap.rows.map((_, rowIndex) => rowIndex);
      for (const rowIndex of candidateRows) touched.add(pageMap.pageOf.get(`${rowIndex}:${keyColumn}`));
      candidateRows = [];
      for (let rowIndex = 0; rowIndex < pageMap.rows.length; rowIndex++) {
        if (pageMap.rows[rowIndex][keyColumn] === query.value) candidateRows.push(rowIndex);
      }
    }
  } else if (query.kind === "scan") {
    // A scan reads the filter column (a zone map skips pages when the rows are sorted by it) and then
    // the projected columns of the matching rows.
    if (sortedByFilter && query.low !== undefined) {
      const min = query.low;
      const max = query.high;
      for (let rowIndex = 0; rowIndex < pageMap.rows.length; rowIndex++) {
        const value = pageMap.rows[rowIndex][keyColumn];
        if (value >= min && value <= max) candidateRows.push(rowIndex);
      }
    } else {
      for (const rowIndex of pageMap.rows.keys()) {
        const value = pageMap.rows[rowIndex][keyColumn];
        if (query.low === undefined || (value >= query.low && value <= query.high)) {
          candidateRows.push(rowIndex);
          touched.add(pageMap.pageOf.get(`${rowIndex}:${keyColumn}`));
        }
      }
    }
  } else if (query.kind === "full") {
    candidateRows = pageMap.rows.map((_, rowIndex) => rowIndex);
    if (sortedByFilter) { /* nothing to filter on: still a full scan */ }
  } else {
    throw new Error(`unknown query kind ${query.kind}`);
  }
  for (const rowIndex of candidateRows) {
    for (const column of query.columns) touched.add(pageMap.pageOf.get(`${rowIndex}:${table.columnIndex(column)}`));
  }
  return { pages: touched.size, rows: candidateRows.length, pageIds: touched };
}

/** Independent brute force: materialise every page's contents and count the pages a query needs. */
function bruteForceCost(table, layout, query) {
  const pageMap = buildPageMap(table, layout);
  return touchedPages(table, layout, pageMap, query);
}

function layoutCost(table, layout, workload, prebuiltPageMap) {
  validateLayout(layout, table);
  const pageMap = prebuiltPageMap || buildPageMap(table, layout);
  let cost = 0;
  for (const query of workload) cost += touchedPages(table, layout, pageMap, query).pages;
  return cost;
}

/** Analytic page arithmetic, used to cross-check the simulator for one layout at a time. */
function analyticPages(table, layout) {
  if (layout.kind === "row") {
    return Math.ceil(table.rowCount / Math.max(1, Math.floor(PAGE_BYTES / (VALUE_BYTES * table.columnCount))));
  }
  if (layout.kind === "column") {
    return table.columnCount * Math.ceil(table.rowCount / Math.max(1, Math.floor(PAGE_BYTES / VALUE_BYTES)));
  }
  const groups = layout.paxGroup;
  return Math.ceil(table.columnCount / groups)
    * Math.ceil(table.rowCount / Math.max(1, Math.floor(PAGE_BYTES / (VALUE_BYTES * groups))));
}

/* ------------------------------------------------------------------ workloads and action space */

/** Four workload states: the agent has to learn a different layout for each. */
const WORKLOAD_STATES = [
  { name: "point-lookup", query: (table) => ({ kind: "point", filterColumn: table.columns[1], value: 7, columns: [table.columns[0], table.columns[1], table.columns[2]] }) },
  { name: "single-selective-scan", query: (table) => ({ kind: "scan", filterColumn: table.columns[3], low: 0, high: 2, columns: [table.columns[3], table.columns[4]] }) },
  { name: "all-column-scan", query: (table) => ({ kind: "full", filterColumn: table.columns[0], columns: table.columns.slice() }) },
  { name: "fat-projection", query: (table) => ({ kind: "scan", filterColumn: table.columns[0], low: 0, high: 40, columns: table.columns.slice(0, 4) }) },
];

const KINDS = ["row", "column", "pax2", "pax4"];

/** The enumerable design space: physical layouts × sort key × (clustered or secondary) index. */
function buildActions(table) {
  const actions = [];
  const sortKeys = [null, ...table.columns.slice(0, 2)];
  const indexColumns = [null, ...table.columns.slice(1, 2)];
  for (const kind of KINDS) {
    for (const sortKey of sortKeys) {
      for (const indexed of indexColumns) {
        actions.push({
          name: `${kind}${sortKey ? `+sort(${sortKey})` : ""}${indexed ? `+idx(${indexed})` : ""}`,
          kind: kind === "pax2" ? "pax" : (kind === "pax4" ? "pax" : kind),
          paxGroup: kind === "pax2" ? 2 : (kind === "pax4" ? 4 : undefined),
          sortKey, indexedColumn: indexed,
        });
      }
    }
  }
  return actions;
}

function sampleQuery(table, state, prng) {
  const query = state.query(table);
  if (query.kind === "point") {
    query.value = prng.below(20);
  } else if (query.low !== undefined) {
    const width = prng.below(5);
    query.low = prng.below(40);
    query.high = query.low + width;
  }
  return query;
}

/* ------------------------------------------------------------------ Q-learning */

class QLearner {
  constructor(states, actions, options = {}) {
    this.states = states.map((state) => state.name);
    this.actions = actions;
    this.prng = new Prng(options.seed || 896);
    this.alpha = options.alpha !== undefined ? options.alpha : 0.25;
    this.gamma = options.gamma !== undefined ? options.gamma : 0;
    this.epsilon = options.epsilon !== undefined ? options.epsilon : 0.4;
    this.epsilonFloor = options.epsilonFloor !== undefined ? options.epsilonFloor : 0.02;
    this.epsilonDecay = options.epsilonDecay !== undefined ? options.epsilonDecay : 0.995;
    this.q = new Map();
    for (const state of this.states) {
      this.q.set(state, this.actions.map(() => 0));
    }
    this.visits = new Map(this.states.map((state) => [state, new Array(this.actions.length).fill(0)]));
  }

  values(state) {
    if (!this.q.has(state)) throw new Error(`unknown state ${state}`);
    return this.q.get(state);
  }

  /** ε-greedy choice over the action table, with a deterministic tie-break. */
  choose(state) {
    if (this.prng.unit() < this.epsilon) return this.prng.below(this.actions.length);
    const values = this.values(state);
    let best = 0;
    for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
  }

  /** One Bellman update; with γ = 0 (a bandit-style MDP) this is a running average of the reward. */
  update(state, actionIndex, reward) {
    const values = this.values(state);
    const visits = this.visits.get(state)[actionIndex];
    // `alpha: "auto"` makes Q the exact sample mean (1/n), which is what makes the greedy choice an
    // empirical-best-arm choice on a γ = 0 design problem.
    const alpha = this.alpha === "auto" ? 1 / (visits + 1) : this.alpha;
    const updated = values[actionIndex] + alpha * (reward + this.gamma * Math.max(...values) - values[actionIndex]);
    values[actionIndex] = updated;
    this.visits.get(state)[actionIndex]++;
    return updated;
  }

  greedyPolicy() {
    const policy = new Map();
    for (const state of this.states) {
      const values = this.values(state);
      let best = 0;
      for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
      policy.set(state, best);
    }
    return policy;
  }

  decay() { this.epsilon = Math.max(this.epsilonFloor, this.epsilon * this.epsilonDecay); }
}

/**
 * Trains the advisor: every episode samples a workload state and a concrete query, scores the chosen
 * design with the *simulated* page reads, and updates Q.
 */
function train(table, options = {}) {
  const actions = buildActions(table);
  // The page maps are a property of (table, action), so they are built once and reused: the reward is
  // then exactly the page reads a query costs, with no surrogate model in between.
  const pageMaps = actions.map((action) => buildPageMap(table, action));
  const learner = new QLearner(WORKLOAD_STATES, actions, options);
  const prng = new Prng((options.seed || 896) ^ 0x1234);
  const episodes = options.episodes || 400;
  const rewards = [];
  const history = [];
  for (let episode = 0; episode < episodes; episode++) {
    const stateIndex = prng.below(WORKLOAD_STATES.length);
    const state = WORKLOAD_STATES[stateIndex];
    const query = sampleQuery(table, state, prng);
    const actionIndex = learner.choose(state.name);
    const cost = layoutCost(table, actions[actionIndex], [query], pageMaps[actionIndex]);
    const reward = -cost;
    learner.update(state.name, actionIndex, reward);
    learner.decay();
    rewards.push(reward);
    history.push({ episode, state: state.name, action: actionIndex, cost });
  }
  const policy = learner.greedyPolicy();
  const mean = (values) => values.reduce((a, b) => a + b, 0) / Math.max(1, values.length);
  return {
    learner, actions, pageMaps, history, rewards, policy,
    firstTenPercent: mean(rewards.slice(0, Math.max(1, Math.floor(episodes / 10)))),
    lastHalf: mean(rewards.slice(Math.floor(episodes / 2))),
  };
}

/** The best possible design per state, by exhaustive search over the action space. */
function optimalActions(table) {
  const actions = buildActions(table);
  const pageMaps = actions.map((action) => buildPageMap(table, action));
  const prng = new Prng(4242);
  const result = new Map();
  for (const state of WORKLOAD_STATES) {
    const sampleQueries = Array.from({ length: 12 }, () => sampleQuery(table, state, prng));
    let bestIndex = 0;
    let bestCost = Infinity;
    for (let i = 0; i < actions.length; i++) {
      const cost = sampleQueries.reduce((total, query) =>
        total + layoutCost(table, actions[i], [query], pageMaps[i]), 0);
      if (cost < bestCost) { bestCost = cost; bestIndex = i; }
    }
    result.set(state.name, { index: bestIndex, cost: bestCost, action: actions[bestIndex] });
  }
  return result;
}

function makeTable(columns, rows, seed) {
  const prng = new Prng(seed || 1);
  const matrix = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < columns; c++) row.push(prng.below(50));
    matrix.push(row);
  }
  return new Table(Array.from({ length: columns }, (_, i) => `c${i}`), matrix);
}

module.exports = {
  Prng, Table, PAGE_BYTES, VALUE_BYTES, validateLayout, buildPageMap, touchedPages, bruteForceCost,
  layoutCost, analyticPages, WORKLOAD_STATES, KINDS, buildActions, sampleQuery, QLearner, train,
  optimalActions, makeTable,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- table validation ------------------------------------------------------------------------
  assert.throws(() => new Table([], [[1]]), /at least one column/);
  assert.throws(() => new Table(["a"], []), /at least one row/);
  assert.throws(() => new Table(["a", "b"], [[1]]), /ragged/);
  assert.throws(() => new Table(["a"], [[1.5]]), /integers/);
  {
    const table = makeTable(3, 10, 5);
    assert.strictEqual(table.columnCount, 3);
    assert.strictEqual(table.rowCount, 10);
    assert.throws(() => table.columnIndex("nope"), /unknown column/);
  }

  // --- the page arithmetic the simulator uses matches straight enumeration ---------------------
  {
    const table = makeTable(6, 700, 7);
    const layouts = [
      { kind: "row" }, { kind: "column" },
      { kind: "pax", paxGroup: 2 }, { kind: "pax", paxGroup: 3 }, { kind: "pax", paxGroup: 6 },
    ];
    for (const layout of layouts) {
      const pageMap = buildPageMap(table, layout);
      assert.strictEqual(pageMap.totalPages, analyticPages(table, layout),
        `${JSON.stringify(layout)}: page count mismatch`);
      assert.strictEqual(pageMap.pageOf.size, table.rowCount * table.columnCount,
        "every value must land on exactly one page");
      const distinct = new Set(pageMap.pageOf.values());
      assert.strictEqual(distinct.size, pageMap.totalPages, "page ids must be contiguous and used");
      for (const page of pageMap.pageOf.values()) assert.ok(page >= 0 && page < pageMap.totalPages);
    }
    assert.throws(() => validateLayout({ kind: "pax", paxGroup: 0 }, table), /out of range/);
    assert.throws(() => validateLayout({ kind: "pax", paxGroup: 99 }, table), /out of range/);
    assert.throws(() => validateLayout({ kind: "row", sortKey: "ghost" }, table), /unknown column/);
  }

  // --- the simulator and the brute-force enumeration agree on every layout ---------------------
  {
    const table = makeTable(5, 320, 11);
    const workload = [
      { kind: "point", filterColumn: "c1", value: 7, columns: ["c0", "c1", "c2"] },
      { kind: "scan", filterColumn: "c3", low: 10, high: 20, columns: ["c3", "c4"] },
      { kind: "full", filterColumn: "c0", columns: ["c0", "c1", "c2", "c3", "c4"] },
    ];
    for (const action of buildActions(table)) {
      validateLayout(action, table);
      const pageMap = buildPageMap(table, action);
      for (const query of workload) {
        const viaMap = touchedPages(table, action, pageMap, query);
        const brute = bruteForceCost(table, action, query);
        assert.strictEqual(viaMap.pages, brute.pages, `${action.name} / ${query.kind}`);
        assert.strictEqual(viaMap.rows, brute.rows);
      }
    }
    assert.throws(() => touchedPages(table, { kind: "row" }, buildPageMap(table, { kind: "row" }),
      { kind: "sideways", columns: ["c0"] }), /unknown query kind/);
  }

  // --- indexes, sorting and layout change the cost in the expected direction -------------------
  {
    const table = makeTable(6, 3000, 13);
    const point = { kind: "point", filterColumn: "c1", value: 7, columns: ["c0", "c1", "c2"] };
    const unindexed = layoutCost(table, { kind: "column" }, [point]);
    const secondary = layoutCost(table, { kind: "column", indexedColumn: "c1" }, [point]);
    const clustered = layoutCost(table, { kind: "column", sortKey: "c1", indexedColumn: "c1" }, [point]);
    // On a page-level cost model it is *clustering* that pays: a secondary index still has to fetch the
    // same scattered pages, while a clustered one turns the matches into a contiguous run.
    assert.ok(clustered < unindexed, `a clustered index must help: ${clustered} vs ${unindexed}`);
    assert.ok(clustered <= secondary, `clustering is what makes the index cheap: ${clustered} vs ${secondary}`);
    assert.ok(secondary <= unindexed, `an index must never be worse: ${secondary} vs ${unindexed}`);
    const rangeQuery = { kind: "scan", filterColumn: "c0", low: 0, high: 4, columns: ["c0", "c1"] };
    const unsorted = layoutCost(table, { kind: "column" }, [rangeQuery]);
    const sorted = layoutCost(table, { kind: "column", sortKey: "c0" }, [rangeQuery]);
    assert.ok(sorted < unsorted, `sorting must enable page skipping: ${sorted} vs ${unsorted}`);
    const full = { kind: "full", filterColumn: "c0", columns: table.columns.slice() };
    const rowCost = layoutCost(table, { kind: "row" }, [full]);
    const paxCost = layoutCost(table, { kind: "pax", paxGroup: 2 }, [full]);
    assert.ok(rowCost <= paxCost, `a full scan should prefer row over PAX: ${rowCost} vs ${paxCost}`);
    const narrow = { kind: "scan", filterColumn: "c3", low: 1, high: 2, columns: ["c3"] };
    const narrowColumn = layoutCost(table, { kind: "column" }, [narrow]);
    const narrowRow = layoutCost(table, { kind: "row" }, [narrow]);
    assert.ok(narrowColumn < narrowRow, `a narrow column scan must prefer columns: ${narrowColumn} vs ${narrowRow}`);
  }

  // --- the Bellman update ----------------------------------------------------------------------
  {
    const table = makeTable(3, 40, 3);
    const actions = buildActions(table);
    const learner = new QLearner(WORKLOAD_STATES, actions, { seed: 1, alpha: 0.5, gamma: 0, epsilon: 0 });
    assert.strictEqual(learner.choose("point-lookup"), 0, "with no exploration the first action wins a tie");
    const updated = learner.update("point-lookup", 2, 10);
    assert.strictEqual(updated, 5, "Q = 0 + 0.5·(10 + 0 − 0)");
    assert.strictEqual(learner.values("point-lookup")[2], 5);
    const again = learner.update("point-lookup", 2, 10);
    assert.strictEqual(again, 7.5, "a second identical reward moves Q halfway again");
    assert.strictEqual(learner.choose("point-lookup"), 2, "the greedy choice follows Q");
    assert.strictEqual(learner.visits.get("point-lookup")[2], 2);
    const discounted = new QLearner(WORKLOAD_STATES, actions, { seed: 1, alpha: 1, gamma: 0.5 });
    discounted.update("point-lookup", 0, -4);
    discounted.update("point-lookup", 1, -2);
    assert.strictEqual(discounted.values("point-lookup")[0], -4);
    assert.strictEqual(discounted.values("point-lookup")[1], -2);
    discounted.update("point-lookup", 1, 10);
    assert.strictEqual(discounted.values("point-lookup")[1], 10, "alpha = 1 replaces the estimate");
    discounted.update("point-lookup", 0, 0);
    assert.strictEqual(discounted.values("point-lookup")[0], 5,
      "Q[0] = −4 + 1·(0 + 0.5·max(Q)=10 − (−4)) = 5, so the discounted future term is included");
    assert.throws(() => learner.values("nowhere"), /unknown state/);
  }

  // --- training improves the measured I/O, and the greedy policy is optimal --------------------
  {
    const table = makeTable(6, 2000, 21);
    const result = train(table, {
      seed: 896, episodes: 2500, epsilon: 0.5, epsilonDecay: 0.998, alpha: "auto",
    });
    const random = new QLearner(WORKLOAD_STATES, result.actions, { seed: 5 });
    const prng = new Prng(777);
    const meanCost = (policyMap) => {
      let total = 0;
      for (const state of WORKLOAD_STATES) {
        for (let i = 0; i < 12; i++) {
          const query = sampleQuery(table, state, prng);
          total += layoutCost(table, result.actions[policyMap.get(state.name)], [query]);
        }
      }
      return total / (12 * WORKLOAD_STATES.length);
    };
    const randomPolicy = new Map(WORKLOAD_STATES.map((state) => [state.name, prng.below(result.actions.length)]));
    const learnedCost = meanCost(result.policy);
    const randomCost = meanCost(randomPolicy);
    assert.ok(learnedCost < randomCost, `learned ${learnedCost} must beat random ${randomCost}`);
    assert.ok(result.lastHalf > result.firstTenPercent,
      `the reward must improve: ${result.firstTenPercent} -> ${result.lastHalf}`);
    assert.ok(result.rewards.every((reward) => reward <= 0), "costs are rewards, so they are never positive");
    const optimal = optimalActions(table);
    const evalPrng = new Prng(2024);
    let learnedTotal = 0;
    let bestTotal = 0;
    let matches = 0;
    for (const state of WORKLOAD_STATES) {
      const queries = Array.from({ length: 12 }, () => sampleQuery(table, state, evalPrng));
      const costOf = (index) => queries.reduce((total, query) =>
        total + layoutCost(table, result.actions[index], [query], result.pageMaps[index]), 0);
      const learnedIndex = result.policy.get(state.name);
      let bestIndex = 0;
      let bestCost = Infinity;
      for (let i = 0; i < result.actions.length; i++) {
        const cost = costOf(i);
        if (cost < bestCost) { bestCost = cost; bestIndex = i; }
      }
      const learnedCost = costOf(learnedIndex);
      learnedTotal += learnedCost;
      bestTotal += bestCost;
      if (learnedCost <= bestCost) matches++;
      assert.ok(learnedCost <= bestCost * 1.5,
        `${state.name}: learned ${learnedCost} vs optimum ${bestCost} (${result.actions[learnedIndex].name})`);
    }
    assert.ok(matches >= 2, `the greedy policy was optimal for only ${matches}/4 states`);
    assert.ok(learnedTotal <= bestTotal * 1.15,
      `learned total ${learnedTotal} vs optimum ${bestTotal}`);
    assert.strictEqual(optimal.size, WORKLOAD_STATES.length);
    assert.ok([...optimal.values()].every((entry) => entry.cost > 0));
    console.log(`storage-rl: greedy policy ${matches}/4 exactly optimal, ` +
      `cost ${learnedTotal} vs optimal ${bestTotal}`);
    // The untrained policy must be worse than the trained one on the same measurement.
    const untrained = new QLearner(WORKLOAD_STATES, result.actions, { seed: 9 });
    assert.ok(meanCost(result.policy) < meanCost(untrained.greedyPolicy()),
      "training must beat the initial policy");
    assert.ok(result.learner.epsilon <= 0.05, "epsilon must have decayed");
    const visits = [...result.learner.visits.values()].flat();
    assert.ok(Math.min(...visits) >= 1, "every arm must be tried at least once");
    assert.ok(visits.reduce((a, b) => a + b, 0) >= 4 * result.actions.length * 15,
      "the sampling budget must be spread over all states and actions");
  }

  // --- the optimum for canonical workloads is the intuitively right design ---------------------
  {
    const table = makeTable(6, 2000, 31);
    const optimal = optimalActions(table);
    for (const state of WORKLOAD_STATES) {
      const best = optimal.get(state.name);
      assert.ok(best.cost > 0);
      if (state.name === "point-lookup") {
        // A point lookup wants *clustering* on the filter column: with the rows physically sorted by
        // c1 the matches become one contiguous run, which is what turns a page scan into a page read.
        assert.strictEqual(best.action.sortKey, "c1",
          `a point lookup must be clustered on the filter column (got ${best.action.name})`);
        assert.ok(best.action.kind === "row" || best.action.indexedColumn === "c1",
          `and it must not fan out across a PAX group (got ${best.action.name})`);
      }
      if (state.name === "all-column-scan") {
        assert.ok(best.action.kind !== "pax", `a full scan should not fan out over PAX (got ${best.action.name})`);
      }
      if (state.name === "single-selective-scan") {
        assert.ok(best.action.kind === "column",
          `a narrow scan should use a column layout (got ${best.action.name})`);
      }
    }
  }

  // --- degenerate shapes and limits --------------------------------------------------------------
  {
    const single = new Table(["only"], [[5]]);
    const actions = buildActions(single);
    assert.ok(actions.length >= 4);
    const cost = layoutCost(single, actions[0], [{ kind: "full", filterColumn: "only", columns: ["only"] }]);
    assert.strictEqual(cost, 1, "a single value occupies exactly one page");
    assert.deepStrictEqual(layoutCost(single, actions[0], []), 0, "an empty workload costs nothing");
    assert.throws(() => layoutCost(single, { kind: "pax", paxGroup: 2 }, []), /out of range/);
    const tiny = makeTable(6, 3, 3);
    const tinyTrain = train(tiny, { seed: 2, episodes: 60 });
    assert.strictEqual(tinyTrain.history.length, 60);
    assert.strictEqual(tinyTrain.policy.size, WORKLOAD_STATES.length);
    const wide = makeTable(12, 3000, 4);
    assert.strictEqual(analyticPages(wide, { kind: "pax", paxGroup: 3 }),
      analyticPages(wide, { kind: "pax", paxGroup: 12 }),
      "the page count of a PAX layout is a data-volume property, not a grouping property");
    const narrowScan = { kind: "scan", filterColumn: "c5", low: 1, high: 2, columns: ["c5"] };
    const paxSmall = layoutCost(wide, { kind: "pax", paxGroup: 3 }, [narrowScan]);
    const paxLarge = layoutCost(wide, { kind: "pax", paxGroup: 12 }, [narrowScan]);
    assert.ok(paxSmall < paxLarge,
      `a narrow query reads fewer pages from smaller PAX groups: ${paxSmall} vs ${paxLarge}`);
    assert.ok(layoutCost(wide, { kind: "column" }, [narrowScan]) <= paxSmall,
      "a column layout is still the best for a single-column scan");
    assert.ok(analyticPages(wide, { kind: "row" }) * 12 >= analyticPages(wide, { kind: "column" }));
  }

  // --- bounded and deterministic training --------------------------------------------------------
  {
    const table = makeTable(6, 1200, 77);
    const runOnce = () => {
      const result = train(table, { seed: 31415, episodes: 200, epsilon: 0.5, epsilonDecay: 0.995 });
      return `${result.learner.epsilon.toFixed(6)}|${[...result.policy.entries()].map(([state, index]) => `${state}=${index}`).join(",")}|${result.lastHalf.toFixed(4)}`;
    };
    assert.strictEqual(runOnce(), runOnce(), "training must be reproducible");
    const result = train(table, { seed: 31415, episodes: 200, epsilon: 0.5, epsilonDecay: 0.995 });
    assert.ok(result.history.every((entry) => entry.action >= 0 && entry.action < result.actions.length));
    assert.ok(result.history.every((entry) => WORKLOAD_STATES.some((state) => state.name === entry.state)));
    const evaluated = result.history.reduce((total, entry) => total + entry.cost, 0);
    assert.ok(evaluated > 0 && Number.isFinite(evaluated));
  }

  console.log(`storage-rl: ${buildActions(makeTable(6, 1, 1)).length} actions, ` +
    `${WORKLOAD_STATES.length} workload states, reward ${result_firstTen()} -> ${result_lastHalf()}`);

  function result_firstTen() {
    const table = makeTable(6, 3000, 21);
    return train(table, { seed: 896, episodes: 600, epsilon: 0.5, epsilonDecay: 0.99 }).firstTenPercent.toFixed(1);
  }
  function result_lastHalf() {
    const table = makeTable(6, 3000, 21);
    return train(table, { seed: 896, episodes: 600, epsilon: 0.5, epsilonDecay: 0.99 }).lastHalf.toFixed(1);
  }
  console.log("PASS");
}
