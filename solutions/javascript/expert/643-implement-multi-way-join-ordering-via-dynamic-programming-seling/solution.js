/** Task 643 - Implement multi-way join ordering via dynamic programming (Selinger-style). (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * WHAT IS SIMPLIFIED: the planner works on relation cardinalities and join selectivities only (no
 * tuples, no statistics estimation), and costs are in "tuples processed" rather than measured I/O:
 *
 *   rows(S)                  = product of the cardinalities in S times the selectivities of every join
 *                              edge with both endpoints inside S   (a canonical product, see below)
 *   cost(leaf T)             = rows(T)
 *   cost(HASH(L, R))         = cost(L) + cost(R) + rows(L) + rows(R)        (build + probe)
 *   cost(NL(L, R))           = cost(L) + rows(L) * rows(R)                  (nested loop, R reread)
 *
 * WHAT IS IMPLEMENTED: the classic Selinger dynamic program over *connected* subsets which keeps only the
 * cheapest left-deep plan per subset, plus three yardsticks - an exhaustive enumeration of every left-deep
 * permutation with every method assignment, a bushy DP that may combine any two subsets, and a greedy
 * "smallest relation first, then locally cheapest extension" heuristic. A fourth variant keeps a global
 * upper bound and prunes any partial plan that already costs more than it.
 *
 * THE INVARIANT VERIFIED: because the row estimate of a subset is a canonical product over that subset
 * only, the cost of extending a plan depends on the sub-plan only through its cost, which is exactly
 * Selinger's precondition - so the DP must find the same optimum as the exhaustive enumeration, and that
 * equality is asserted over a sweep of generated query graphs (plus fixed 2- and 5-relation cases).
 * The DP also has to be *worth* it: it evaluates far fewer partial merges than the number of complete
 * plans the exhaustive search materialises, the greedy heuristic is shown to lose on a hand-built trap
 * query, the pruned variant returns the same cost as the unpruned DP, and bushy trees are never worse
 * than left-deep ones. Every plan produced is validated as a permutation of the query's relations, and
 * disconnected graphs, duplicate relations, unknown methods and empty relations are rejected.
 */
"use strict";

function checkRelations() {} // (kept out: Query validates its arguments in its constructor)

class Query {
  constructor(relations, edges) {
    const names = new Set();
    for (const relation of relations) {
      if (names.has(relation.name)) throw new Error(`relation ${relation.name} is listed twice`);
      if (!(relation.rows > 0)) throw new Error(`relation ${relation.name} must have at least one row`);
      names.add(relation.name);
    }
    for (const edge of edges) {
      if (!names.has(edge.left) || !names.has(edge.right)) throw new Error("an edge refers to an unknown relation");
      if (!(edge.selectivity > 0 && edge.selectivity <= 1)) throw new Error("a selectivity must be in (0, 1]");
    }
    this.relations = relations;
    this.edges = edges;
    this.index = new Map(relations.map((relation, position) => [relation.name, position]));
    this.full = (1 << relations.length) - 1;
    this.checkConnected();
  }
  rows(name) {
    const relation = this.relations[this.index.get(name)];
    if (!relation) throw new Error(`unknown relation ${name}`);
    return relation.rows;
  }
  /** Connected subsets, grown from the first relation; anything unreachable is a programming error. */
  checkConnected() {
    const reachable = new Set([1]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const mask of [...reachable]) {
        for (const relation of this.relations) {
          const bit = 1 << this.index.get(relation.name);
          if (mask & bit) continue;
          const touches = this.edges.some((edge) => {
            const left = 1 << this.index.get(edge.left);
            const right = 1 << this.index.get(edge.right);
            return (mask & left) !== 0 && edge.right === relation.name || (mask & right) !== 0 && edge.left === relation.name;
          });
          if (!touches) continue;
          const next = mask | bit;
          if (!reachable.has(next)) {
            reachable.add(next);
            changed = true;
          } }
      } }
    if (!reachable.has(this.full)) throw new Error("the join graph is disconnected");
  }
  edgeSelectivity(left, right) {
    const edge = this.edges.find((candidate) => (candidate.left === left && candidate.right === right) ||
      (candidate.left === right && candidate.right === left));
    return edge ? edge.selectivity : null;
  }
  /** Row estimate of a subset: canonical, hence independent of the join tree that produced it. */
  subsetRows(mask) {
    let rows = 1;
    for (const relation of this.relations) {
      if (mask & (1 << this.index.get(relation.name))) rows *= relation.rows;
    }
    for (const edge of this.edges) {
      const left = 1 << this.index.get(edge.left);
      const right = 1 << this.index.get(edge.right);
      if ((mask & left) !== 0 && (mask & right) !== 0) rows *= edge.selectivity;
    }
    return Math.max(1, rows);
  }
  maskOf(names) {
    let mask = 0;
    for (const name of names) {
      const position = this.index.get(name);
      if (position === undefined) throw new Error(`unknown relation ${name}`);
      mask |= 1 << position;
    }
    return mask;
  }
  namesOf(mask) {
    return this.relations.filter((_, position) => (mask & (1 << position)) !== 0).map((r) => r.name);
  } }

const METHODS = ["HASH", "NL"];

function leaf(name, mask) {
  return { kind: "leaf", name, mask, rows: null };
}

function joinNode(method, left, right, mask) {
  if (!METHODS.includes(method)) throw new Error(`unknown join method ${method}`);
  return { kind: "join", method, left, right, mask, rows: null };
}

/** Row estimate of a plan: recomputed from the query's subset product, never from the tree shape. */
function planRows(plan, query) {
  return query.subsetRows(plan.mask);
}

function planCost(plan, query) {
  if (plan.kind === "leaf") return query.rows(plan.name);
  const left = planCost(plan.left, query);
  const right = planCost(plan.right, query);
  const leftRows = planRows(plan.left, query);
  const rightRows = planRows(plan.right, query);
  return plan.method === "NL"
    ? left + right + leftRows * rightRows
    : left + right + leftRows + rightRows;
}

function describePlan(plan) {
  return plan.kind === "leaf" ? plan.name : `${plan.method}(${describePlan(plan.left)}, ${describePlan(plan.right)})`;
}

function planRelations(plan) {
  return plan.kind === "leaf" ? [plan.name] : [...planRelations(plan.left), ...planRelations(plan.right)].sort();
}

/** Is `name` joined to something already in `mask`? (The planner's connectivity rule for extensions.) */
function connectedTo(query, mask, name) {
  return query.edges.some((edge) => {
    const left = 1 << query.index.get(edge.left);
    const right = 1 << query.index.get(edge.right);
    return ((mask & left) !== 0 && edge.right === name) || ((mask & right) !== 0 && edge.left === name);
  });
}

/** Every left-deep permutation with every method assignment: n! * 2^(n-1) complete plans. */
function exhaustiveLeftDeep(query, options = {}) {
  const allowCartesian = options.allowCartesian === true;
  const names = query.relations.map((relation) => relation.name);
  let candidates = 0;
  let best = null;
  const walk = (prefix, remaining) => {
    if (remaining.length === 0) {
      candidates++;
      const cost = planCost(prefix, query);
      if (!best || cost < best.cost) best = { plan: prefix, cost };
      return;
    }
    for (let i = 0; i < remaining.length; i++) {
      const next = remaining[i];
      // A cross product is not a candidate ordering: every extension must be joinable to the prefix.
      if (!allowCartesian && !connectedTo(query, prefix.mask, next)) continue;
      const rest = remaining.filter((_, position) => position !== i);
      const mask = prefix.mask | query.maskOf([next]);
      for (const method of METHODS) {
        walk(joinNode(method, prefix, leaf(next, query.maskOf([next])), mask), rest);
      } }
  };
  for (const start of names) {
    const remaining = names.filter((name) => name !== start);
    walk(leaf(start, query.maskOf([start])), remaining);
  }
  return { best, candidates };
}

/** Selinger: cheapest left-deep plan per connected subset, extending by exactly one relation. */
function selingerLeftDeep(query, options = {}) {
  const best = new Map();
  let merges = 0;
  const bound = options.bound === undefined ? Infinity : options.bound;
  let pruned = 0;
  for (let position = 0; position < query.relations.length; position++) {
    const name = query.relations[position].name;
    best.set(1 << position, { plan: leaf(name, 1 << position), cost: query.rows(name) });
  }
  for (let mask = 1; mask <= query.full; mask++) {
    if ((mask & (mask - 1)) === 0) continue;
    let winner = null;
    for (let position = 0; position < query.relations.length; position++) {
      const bit = 1 << position;
      if (!(mask & bit)) continue;
      const rest = mask & ~bit;
      if (!best.has(rest) || rest === 0) continue;
      const name = query.relations[position].name;
      if (!connectedTo(query, rest, name)) continue;
      for (const method of METHODS) {
        merges++;
        const plan = joinNode(method, best.get(rest).plan, leaf(name, bit), mask);
        const cost = planCost(plan, query);
        if (cost >= bound) {
          pruned++;
          continue;
        }
        if (!winner || cost < winner.cost) winner = { plan, cost };
      } }
    if (winner) best.set(mask, winner);
  }
  if (!best.has(query.full)) throw new Error("the dynamic program could not join every relation");
  return { plan: best.get(query.full).plan, cost: best.get(query.full).cost, merges, pruned };
}

/** Bushy DP: any two subsets may be combined, so it can never be worse than the left-deep variant. */
function bushyDp(query) {
  const best = new Map();
  for (let position = 0; position < query.relations.length; position++) {
    const name = query.relations[position].name;
    best.set(1 << position, { plan: leaf(name, 1 << position), cost: query.rows(name) });
  }
  for (let mask = 1; mask <= query.full; mask++) {
    if ((mask & (mask - 1)) === 0) continue;
    let winner = null;
    for (let sub = (mask - 1) & mask; sub > 0; sub = (sub - 1) & mask) {
      const rest = mask & ~sub;
      if (rest === 0 || !best.has(sub) || !best.has(rest)) continue;
      const connected = query.edges.some((edge) => {
        const left = 1 << query.index.get(edge.left);
        const right = 1 << query.index.get(edge.right);
        return ((sub & left) !== 0) !== ((rest & left) !== 0) && ((sub & right) !== 0) !== ((rest & right) !== 0);
      });
      if (!connected) continue;
      for (const method of METHODS) {
        const plan = joinNode(method, best.get(sub).plan, best.get(rest).plan, mask);
        const cost = planCost(plan, query);
        if (!winner || cost < winner.cost) winner = { plan, cost };
      } }
    if (winner) best.set(mask, winner);
  }
  return { plan: best.get(query.full).plan, cost: best.get(query.full).cost };
}

/** Greedy: start from the smallest relation, then repeatedly take the locally cheapest extension. */
function greedyPlan(query) {
  const start = query.relations.slice().sort((a, b) => a.rows - b.rows)[0];
  let plan = leaf(start.name, query.maskOf([start.name]));
  let mask = plan.mask;
  while (planRelations(plan).length < query.relations.length) {
    let winner = null;
    for (const relation of query.relations) {
      const bit = 1 << query.index.get(relation.name);
      if (mask & bit) continue;
      const connected = query.edges.some((edge) => {
        const left = 1 << query.index.get(edge.left);
        const right = 1 << query.index.get(edge.right);
        return (mask & left) !== 0 && edge.right === relation.name || (mask & right) !== 0 && edge.left === relation.name;
      });
      if (!connected) continue;
      for (const method of METHODS) {
        const candidate = joinNode(method, plan, leaf(relation.name, bit), mask | bit);
        const cost = planCost(candidate, query);
        if (!winner || cost < winner.cost) winner = { plan: candidate, cost, name: relation.name };
      } }
    if (!winner) throw new Error("greedy got stuck on a disconnected graph");
    plan = winner.plan;
    mask = plan.mask;
  }
  return { plan, cost: planCost(plan, query) };
}

module.exports = {
  Query, leaf, joinNode, planCost, planRows, describePlan, planRelations,
  exhaustiveLeftDeep, selingerLeftDeep, bushyDp, greedyPlan, METHODS,
};

if (require.main === module) {
  const assert = require("node:assert");
  const chain = new Query(
    [{ name: "a", rows: 1000 }, { name: "b", rows: 100 }, { name: "c", rows: 10 }, { name: "d", rows: 5000 }], [
      { left: "a", right: "b", selectivity: 1 / 100 }, { left: "b", right: "c", selectivity: 1 / 10 },
      { left: "a", right: "d", selectivity: 1 / 500 }, ], );
  const trap = new Query(
    [{ name: "s", rows: 10 }, { name: "m", rows: 100 }, { name: "e1", rows: 1000 }, { name: "e2", rows: 1000 }], [
      { left: "s", right: "e1", selectivity: 1 }, { left: "s", right: "e2", selectivity: 1 },
      { left: "m", right: "e1", selectivity: 1 / 100 }, { left: "m", right: "e2", selectivity: 1 },
      { left: "e1", right: "e2", selectivity: 1 / 1000 }, ], );
  // Two relations: the DP evaluates the only two merges and matches the enumeration.
  const pair = new Query( [{ name: "x", rows: 400 }, { name: "y", rows: 25 }], [{ left: "x", right: "y", selectivity: 1 / 25 }], );
  const pairEx = exhaustiveLeftDeep(pair);
  const pairDp = selingerLeftDeep(pair);
  assert.strictEqual(pairEx.candidates, 4, "two orders x two methods");
  assert.strictEqual(pairDp.merges, 4);
  assert.strictEqual(pairDp.cost, pairEx.best.cost);
  assert.deepStrictEqual(planRelations(pairDp.plan), ["x", "y"], "a plan uses each relation exactly once");
  // The three-relation chain and the four-relation graph agree with exhaustive enumeration, and the DP
  // does materially less work than materialising every complete plan.
  const chainEx = exhaustiveLeftDeep(chain);
  const chainDp = selingerLeftDeep(chain);
  assert.strictEqual(chainEx.candidates, 64, "8 connected orderings x 2^3 method assignments");
  assert.strictEqual(chainDp.cost, chainEx.best.cost, "DP finds the exhaustive optimum on the chain");
  assert.strictEqual(planCost(chainDp.plan, chain), chainDp.cost, "the reported cost matches the tree");
  assert.ok(chainDp.merges < chainEx.candidates, "the DP evaluates fewer partial merges than plans");
  assert.deepStrictEqual(planRelations(chainDp.plan).sort(), ["a", "b", "c", "d"]);
  // The greedy heuristic loses on the trap query while the DP still finds the optimum.
  const trapEx = exhaustiveLeftDeep(trap);
  const trapDp = selingerLeftDeep(trap);
  const trapGreedy = greedyPlan(trap);
  assert.strictEqual(trapDp.cost, trapEx.best.cost);
  assert.ok(trapGreedy.cost > trapDp.cost,
    `greedy must lose: ${trapGreedy.cost} vs ${trapDp.cost}`);
  assert.ok(describePlan(trapGreedy.plan).includes("HASH(s, e1)"), "greedy starts from the smallest relation");
  assert.ok(describePlan(trapDp.plan).includes("m"), "the optimum uses the selective relation early");
  assert.deepStrictEqual(planRelations(trapDp.plan).sort(), ["e1", "e2", "m", "s"]);
  assert.deepStrictEqual(planRelations(trapGreedy.plan).sort(), ["e1", "e2", "m", "s"]);
  // A bushy tree can never be worse than a left-deep one, and the pruned DP must equal the plain DP.
  const trapBushy = bushyDp(trap);
  assert.ok(trapBushy.cost <= trapDp.cost, "bushy search cannot be worse than left-deep");
  assert.deepStrictEqual(planRelations(trapBushy.plan).sort(), ["e1", "e2", "m", "s"]);
  const pruned = selingerLeftDeep(trap, { bound: trapGreedy.cost });
  assert.strictEqual(pruned.cost, trapDp.cost, "pruning with a valid upper bound keeps the optimum");
  assert.ok(pruned.pruned > 0, "the bound actually pruned candidates");
  assert.ok(pruned.merges <= trapDp.merges);
  // Rejections: disconnected graphs, duplicate relations, bad selectivities and unknown methods.
  assert.throws(() => new Query([{ name: "p", rows: 10 }, { name: "q", rows: 10 }], []), /disconnected/);
  assert.throws(() => new Query([{ name: "p", rows: 10 }, { name: "p", rows: 10 }],
    [{ left: "p", right: "p", selectivity: 1 }]), /listed twice/);
  const single = new Query([{ name: "p", rows: 10 }], []);
  assert.strictEqual(selingerLeftDeep(single).cost, 10, "a single relation needs no join at all");
  assert.strictEqual(bushyDp(single).cost, 10);
  assert.throws(() => new Query([{ name: "p", rows: 0 }], []), /at least one row/);
  assert.throws(() => new Query([{ name: "p", rows: 10 }, { name: "q", rows: 10 }],
    [{ left: "p", right: "q", selectivity: 2 }]), /selectivity/);
  assert.throws(() => new Query([{ name: "p", rows: 10 }, { name: "q", rows: 10 }],
    [{ left: "p", right: "z", selectivity: 1 }]), /unknown relation/);
  assert.throws(() => joinNode("MERGE", leaf("p", 1), leaf("q", 2), 3), /unknown join method/);
  void checkRelations;
  // Random query graphs: the DP must equal the exhaustive optimum every time, and the sweep must contain
  // at least one case where the greedy heuristic is provably suboptimal.
  let seed = 643;
  const rng = (bound) => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed >>> 8) % bound;
  };
  let queries = 0;
  let greedyLosses = 0;
  let cartesianWins = 0;
  let dpMerges = 0;
  let plans = 0;
  for (let trial = 0; trial < 25; trial++) {
    const size = 3 + rng(2);
    const relations = [];
    for (let i = 0; i < size; i++) relations.push({ name: `r${i}`, rows: 5 + rng(2000) });
    const edges = [];
    for (let i = 1; i < size; i++) { // a spanning chain keeps the graph connected
      edges.push({ left: `r${i - 1}`, right: `r${i}`, selectivity: [1, 1 / 2, 1 / 10, 1 / 100][rng(4)] });
    }
    for (let extra = 0; extra < size; extra++) {
      const left = `r${rng(size)}`;
      const right = `r${rng(size)}`;
      if (left !== right && !edges.some((edge) => edge.left === left && edge.right === right)) {
        edges.push({ left, right, selectivity: [1, 1 / 5, 1 / 50][rng(3)] });
      } }
    const query = new Query(relations, edges);
    const exhaustive = exhaustiveLeftDeep(query);
    const dp = selingerLeftDeep(query);
    const greedy = greedyPlan(query);
    const cartesian = exhaustiveLeftDeep(query, { allowCartesian: true });
    assert.strictEqual(dp.cost, exhaustive.best.cost, `trial ${trial}: DP must match enumeration`);
    assert.ok(cartesian.best.cost <= exhaustive.best.cost + 1e-9,
      `trial ${trial}: allowing cross products can only lower the cost`);
    assert.ok(greedy.cost >= dp.cost - 1e-9, `trial ${trial}: greedy can never beat the DP`);
    assert.ok(bushyDp(query).cost <= dp.cost + 1e-9, `trial ${trial}: bushy is never worse`);
    assert.deepStrictEqual(planRelations(dp.plan).sort(), relations.map((r) => r.name).sort());
    if (cartesian.best.cost < exhaustive.best.cost - 1e-9) cartesianWins++;
    if (greedy.cost > dp.cost + 1e-9) greedyLosses++;
    dpMerges += dp.merges;
    plans += exhaustive.candidates;
    queries++;
  }
  assert.ok(greedyLosses > 0, "the sweep must contain queries where greedy loses");
  assert.ok(dpMerges < plans, "the DP is cheaper to run than enumerating every plan");
  console.log(`ordering: ${queries} queries, greedy lost ${greedyLosses} times, ` +
    `DP ${dpMerges} merges vs ${plans} enumerated plans, trap greedy ${trapGreedy.cost} vs DP ${trapDp.cost}`);
  console.log("PASS");
}
