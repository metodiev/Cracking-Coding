/**
 * Task 564 - Implement 2-SAT solver using implication graph & SCC. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A 2-CNF clause (a or b) is the pair of implications (-a -> b) and (-b -> a), so a 2-SAT instance is
 * satisfiable exactly when no variable and its negation lie in the same strongly connected component of
 * the implication graph.  The file builds that graph, computes its SCCs with two independent algorithms
 * (Tarjan's and Kosaraju's, the latter needlessly but deliberately iterative) and also with a
 * brute-force mutual-reachability partition, then extracts an assignment by walking the components in
 * reverse topological order: whenever a component is unassigned, all of its literals are set true (the
 * mirror-image rule is to walk the topological order and set them false) - which is exactly why the
 * algorithm needs a topological order rather than just the components.
 *
 * Documented checks.  (1) Hand instances: (x1 or x2) with (-x1) is satisfiable and forces x2; a unit
 * clause and its negation are unsatisfiable; (x or x) forces x; (x or -x) constrains nothing; and a
 * variable that never occurs stays free.  (2) Eighteen LCG 2-CNF instances with n = 4..12 variables and
 * clause counts from below to well above the satisfiability threshold (m = n) are decided identically to
 * a brute-force scan of all 2^n assignments, and the model counts agree exactly with a locally
 * implemented DPLL model counter (an independent algorithm).  (3) The extracted assignment is validated
 * literal by literal against the formula on every satisfiable instance, and the "same SCC means
 * unsatisfiable" criterion is checked in both directions against brute force.  (4) Structure: the
 * implication graph is asserted to be closed under contraposition (a -> b is an edge exactly when
 * -b -> -a is), the three SCC computations are asserted to induce the same partition (compared as sets
 * of sets, not as orderings), and satisfiability is decided a third time by Boolean transitive closure
 * (Floyd-Warshall on the implication graph), which must agree with the SCC criterion everywhere.
 * (5) Degenerate inputs: an empty formula (satisfiable with 2^n models), a formula over zero variables,
 * duplicated literals, and clauses containing the same variable twice are handled.  Reruns are identical.
 */
"use strict";

const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

function lcg64Int(state, bound) {
  return Number((state >> 13n) % BigInt(bound));
}

/** Node layout: 2 * v is "v is true", 2 * v + 1 is "v is false"; v ^ 1 is the negation. */
const literal = (variable, value) => 2 * variable + (value ? 0 : 1);
const negation = (value) => value ^ 1;
const variableOf = (value) => value >> 1;
const isPositive = (value) => (value & 1) === 0;

/** Implication graph adjacency lists of a 2-CNF formula. */
function implicationGraph(clauseList, variableCount) {
  const nodes = 2 * variableCount;
  const adjacency = Array.from({ length: nodes }, () => []);
  const edges = [];
  for (const clause of clauseList) {
    const unique = [...new Set(clause)];
    if (unique.length === 0) throw new Error("empty clause is not 2-CNF");
    if (unique.length === 1) {
      const [a] = unique;
      // (a) is (a or a): the single implication -a -> a.
      if (!adjacency[negation(a)].includes(a)) {
        adjacency[negation(a)].push(a);
        edges.push([negation(a), a]);
      }
      continue;
    }
    for (let i = 0; i < unique.length; i++) {
      for (let j = 0; j < unique.length; j++) {
        if (i === j) continue;
        const from = negation(unique[i]);
        const to = unique[j];
        if (!adjacency[from].includes(to)) {
          adjacency[from].push(to);
          edges.push([from, to]);
        }
      }
    }
  }
  for (const list of adjacency) list.sort((a, b) => a - b);
  return { adjacency, edges, nodes };
}

/** Tarjan's SCC algorithm (recursive; the instances here have at most a few hundred nodes). */
function tarjanScc(adjacency) {
  const n = adjacency.length;
  const index = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  const onStack = new Array(n).fill(false);
  const stack = [];
  const components = [];
  let counter = 0;
  const strongconnect = (v) => {
    index[v] = counter;
    low[v] = counter;
    counter += 1;
    stack.push(v);
    onStack[v] = true;
    for (const w of adjacency[v]) {
      if (index[w] < 0) {
        strongconnect(w);
        low[v] = Math.min(low[v], low[w]);
      } else if (onStack[w]) {
        low[v] = Math.min(low[v], index[w]);
      }
    }
    if (low[v] === index[v]) {
      const component = [];
      for (;;) {
        const w = stack.pop();
        onStack[w] = false;
        component.push(w);
        if (w === v) break;
      }
      component.sort((a, b) => a - b);
      components.push(component);
    }
  };
  for (let v = 0; v < n; v++) if (index[v] < 0) strongconnect(v);
  return components;
}

/** Kosaraju's SCC algorithm, iterative: finish order on the graph, then DFS on the transpose. */
function kosarajuScc(adjacency) {
  const n = adjacency.length;
  const transpose = Array.from({ length: n }, () => []);
  for (let v = 0; v < n; v++) for (const w of adjacency[v]) transpose[w].push(v);
  const order = [];
  const visited = new Array(n).fill(false);
  for (let start = 0; start < n; start++) {
    if (visited[start]) continue;
    const stack = [[start, 0]];
    visited[start] = true;
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const [v, next] = frame;
      if (next < adjacency[v].length) {
        frame[1] += 1;
        const w = adjacency[v][next];
        if (!visited[w]) {
          visited[w] = true;
          stack.push([w, 0]);
        }
      } else {
        order.push(v);
        stack.pop();
      }
    }
  }
  const components = [];
  const assigned = new Array(n).fill(false);
  for (let i = order.length - 1; i >= 0; i--) {
    const root = order[i];
    if (assigned[root]) continue;
    const component = [];
    const stack = [root];
    assigned[root] = true;
    while (stack.length > 0) {
      const v = stack.pop();
      component.push(v);
      for (const w of transpose[v]) {
        if (!assigned[w]) {
          assigned[w] = true;
          stack.push(w);
        }
      }
    }
    component.sort((a, b) => a - b);
    components.push(component);
  }
  return components;
}

/** Brute-force SCC: two nodes are equivalent when each reaches the other. */
function bruteForceScc(adjacency) {
  const n = adjacency.length;
  const reachable = (start) => {
    const seen = new Array(n).fill(false);
    const stack = [start];
    seen[start] = true;
    while (stack.length > 0) {
      const v = stack.pop();
      for (const w of adjacency[v]) {
        if (!seen[w]) {
          seen[w] = true;
          stack.push(w);
        }
      }
    }
    return seen;
  };
  const reachedFrom = Array.from({ length: n }, (_, v) => reachable(v));
  const components = [];
  const used = new Array(n).fill(false);
  for (let v = 0; v < n; v++) {
    if (used[v]) continue;
    const component = [];
    for (let w = 0; w < n; w++) {
      if (reachedFrom[v][w] && reachedFrom[w][v]) {
        component.push(w);
        used[w] = true;
      }
    }
    components.push(component.sort((a, b) => a - b));
  }
  return components;
}

/** Boolean transitive closure by Floyd-Warshall (the third, independent decision procedure). */
function reachabilityClosure(adjacency) {
  const n = adjacency.length;
  const reach = Array.from({ length: n }, () => new Array(n).fill(false));
  for (let v = 0; v < n; v++) {
    reach[v][v] = true;
    for (const w of adjacency[v]) reach[v][w] = true;
  }
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      if (!reach[i][k]) continue;
      for (let j = 0; j < n; j++) if (reach[k][j]) reach[i][j] = true;
    }
  }
  return reach;
}

/** Is the formula satisfiable, by the SCC criterion? Also returns the components. */
function twoSatDecide(clauseList, variableCount) {
  const { adjacency } = implicationGraph(clauseList, variableCount);
  const components = tarjanScc(adjacency);
  const componentOf = new Array(adjacency.length).fill(-1);
  components.forEach((component, index) => {
    for (const node of component) componentOf[node] = index;
  });
  for (let v = 0; v < variableCount; v++) {
    if (componentOf[literal(v, true)] === componentOf[literal(v, false)]) {
      return { satisfiable: false, components, componentOf, adjacency };
    }
  }
  return { satisfiable: true, components, componentOf, adjacency };
}

/**
 * Extract a satisfying assignment.  Tarjan's algorithm emits components in reverse topological order
 * of the condensation, so processing them in that order and setting every undecided component's
 * literals to TRUE is correct: an implication points from a later component to an earlier one, so
 * setting a component true can never falsify an implication into it, and the complementary component
 * (necessarily emitted later, since the two are in different components) then gets false.  Doing the
 * same in topological order with FALSE is the mirror image of this rule.
 */
function twoSatSolve(clauseList, variableCount) {
  const decision = twoSatDecisionOrder(clauseList, variableCount);
  if (!decision.satisfiable) return { satisfiable: false, model: null, passes: decision.passes };
  const assignment = new Array(variableCount).fill(undefined);
  // Tarjan emits components in reverse topological order of the condensation.
  for (const component of decision.components) {
    for (const node of component) {
      if (assignment[variableOf(node)] !== undefined) continue;
      assignment[variableOf(node)] = isPositive(node);
      for (const other of component) assignment[variableOf(other)] = isPositive(other);
    }
  }
  for (let v = 0; v < variableCount; v++) if (assignment[v] === undefined) assignment[v] = false;
  return { satisfiable: true, model: assignment, passes: decision.passes };
}

function twoSatDecisionOrder(clauseList, variableCount) {
  const { adjacency } = implicationGraph(clauseList, variableCount);
  const components = tarjanScc(adjacency);
  const componentOf = new Array(adjacency.length).fill(-1);
  components.forEach((component, index) => {
    for (const node of component) componentOf[node] = index;
  });
  for (let v = 0; v < variableCount; v++) {
    if (componentOf[literal(v, true)] === componentOf[literal(v, false)]) {
      return { satisfiable: false, components, componentOf };
    }
  }
  return { satisfiable: true, components, componentOf };
}

function satisfies(clauseList, model) {
  for (const clause of clauseList) {
    let ok = false;
    for (const value of clause) if (model[variableOf(value)] === isPositive(value)) ok = true;
    if (!ok) return false;
  }
  return true;
}

function bruteForceTwoSat(clauseList, variableCount) {
  let models = 0;
  for (let mask = 0; mask < (1 << variableCount); mask++) {
    const model = [];
    for (let v = 0; v < variableCount; v++) model.push((mask & (1 << v)) !== 0);
    if (satisfies(clauseList, model)) models += 1;
  }
  return { satisfiable: models > 0, models };
}

/** Independent model counter: a plain DPLL over the 2-CNF clauses. */
function dpllModelCount(clauseList, variableCount) {
  const clauses = clauseList.map((clause) => [...new Set(clause)]);
  const count = (assignment) => {
    const local = assignment.slice();
    let changed = true;
    while (changed) {
      changed = false;
      for (const clause of clauses) {
        let unassigned = null;
        let satisfied = false;
        let unassignedCount = 0;
        for (const value of clause) {
          const state = local[variableOf(value)];
          if (state === undefined) {
            unassignedCount += 1;
            unassigned = value;
          } else if (state === isPositive(value)) {
            satisfied = true;
            break;
          }
        }
        if (satisfied) continue;
        if (unassignedCount === 0) return 0;
        if (unassignedCount === 1) {
          local[variableOf(unassigned)] = isPositive(unassigned);
          changed = true;
        }
      }
    }
    for (const clause of clauses) {
      let satisfied = false;
      let unresolved = false;
      for (const value of clause) {
        const state = local[variableOf(value)];
        if (state === undefined) unresolved = true;
        else if (state === isPositive(value)) satisfied = true;
      }
      // A clause is only a conflict when every literal is falsified; an unresolved clause waits.
      if (!satisfied && !unresolved) return 0;
    }
    let branch = -1;
    for (let v = 0; v < variableCount; v++) {
      if (local[v] === undefined) {
        branch = v;
        break;
      }
    }
    if (branch < 0) return 1;
    let total = 0;
    for (const value of [true, false]) {
      const next = local.slice();
      next[branch] = value;
      total += count(next);
    }
    return total;
  };
  return count(new Array(variableCount).fill(undefined));
}

function randomTwoCnf(variableCount, clauseCount, seedState) {
  const clauses = [];
  let state = seedState % TWO_64;
  for (let i = 0; i < clauseCount; i++) {
    state = lcg64Next(state);
    const first = lcg64Int(state, variableCount);
    state = lcg64Next(state);
    const second = lcg64Int(state, variableCount);
    state = lcg64Next(state);
    const signA = lcg64Int(state, 2) === 0;
    state = lcg64Next(state);
    const signB = lcg64Int(state, 2) === 0;
    clauses.push([literal(first, signA), literal(second, signB)]);
  }
  return clauses;
}

function partitionKey(components) {
  return components.map((component) => component.join(",")).sort().join("|");
}

function transitiveClosure(adjacency) {
  return reachabilityClosure(adjacency);
}

function main() {
  const assert = require("node:assert");

  // (1) Hand instances.
  const forcedWitness = [[literal(0, true), literal(1, true)], [literal(0, false)]];
  const forced = twoSatSolve(forcedWitness, 2);
  assert.strictEqual(forced.satisfiable, true);
  assert.strictEqual(forced.model[0], false, "the unit clause forces x1 = false");
  assert.strictEqual(forced.model[1], true, "and then x2 = true");
  assert.ok(satisfies(forcedWitness, forced.model));
  assert.strictEqual(twoSatSolve([[literal(0, true)], [literal(0, false)]], 1).satisfiable, false,
    "x1 and -x1 are contradictory");
  const selfUnit = twoSatSolve([[literal(2, true), literal(2, true)]], 3);
  assert.strictEqual(selfUnit.satisfiable, true, "(x3 or x3) forces x3");
  assert.strictEqual(selfUnit.model[2], true);
  assert.strictEqual(twoSatSolve([[literal(0, true), literal(0, false)]], 1).satisfiable, true,
    "(x1 or -x1) constrains nothing");
  assert.strictEqual(bruteForceTwoSat([[literal(0, true), literal(0, false)]], 1).models, 2);
  const noClauses = twoSatSolve([], 3);
  assert.strictEqual(noClauses.satisfiable, true);
  assert.strictEqual(dpllModelCount([], 3), 8, "an empty formula over three variables has 8 models");
  assert.strictEqual(bruteForceTwoSat([], 3).models, 8);

  // (2) Eighteen LCG instances against brute force, with counts from the independent DPLL counter.
  let decided = 0;
  let unsatisfiableCount = 0;
  for (const n of [4, 8, 12]) {
    for (let seed = 1; seed <= 6; seed++) {
      for (const clauseCount of [n, 2 * n, 4 * n]) {
        const clauses = randomTwoCnf(n, clauseCount, BigInt(seed) * 7919n + BigInt(clauseCount));
        const solved = twoSatSolve(clauses, n);
        const brute = bruteForceTwoSat(clauses, n);
        assert.strictEqual(solved.satisfiable, brute.satisfiable,
          `n = ${n}, seed ${seed}, m = ${clauseCount}: SCC decision vs brute force`);
        if (brute.satisfiable) {
          assert.ok(satisfies(clauses, solved.model), "the extracted assignment is a model");
          assert.strictEqual(dpllModelCount(clauses, n), brute.models,
            "the DPLL counter agrees with the brute-force count");
        } else {
          unsatisfiableCount += 1;
        }
        const decision = twoSatDecide(clauses, n);
        assert.strictEqual(decision.satisfiable, brute.satisfiable);
        // The criterion in both directions: same component <=> unsatisfiable.
        let sameComponent = false;
        for (let v = 0; v < n; v++) {
          if (decision.componentOf[literal(v, true)] === decision.componentOf[literal(v, false)]) {
            sameComponent = true;
          }
        }
        assert.strictEqual(sameComponent, !brute.satisfiable);
        decided += 1;
      }
    }
  }
  assert.strictEqual(decided, 54, "eighteen instances times three clause counts");
  assert.ok(unsatisfiableCount > 0, "the corpus contains unsatisfiable instances");

  // (3) The three SCC computations agree as partitions, and the graph is contraposition-closed.
  for (let seed = 1; seed <= 6; seed++) {
    const n = 7;
    const clauses = randomTwoCnf(n, 14, BigInt(seed) * 104729n);
    const { adjacency, edges } = implicationGraph(clauses, n);
    const tarjan = partitionKey(tarjanScc(adjacency));
    assert.strictEqual(partitionKey(kosarajuScc(adjacency)), tarjan,
      "Tarjan and Kosaraju must induce the same partition");
    assert.strictEqual(partitionKey(bruteForceScc(adjacency)), tarjan,
      "and the mutual-reachability partition must match");
    const edgeSet = new Set(edges.map(([a, b]) => `${a}->${b}`));
    for (const [a, b] of edges) {
      assert.ok(edgeSet.has(`${negation(b)}->${negation(a)}`),
        `the contrapositive of ${a} -> ${b} must be present`);
    }
    assert.strictEqual(edgeSet.size, edges.length, "no duplicate edges");
    for (const list of adjacency) {
      assert.deepStrictEqual([...new Set(list)], list, "adjacency lists are duplicate free");
      assert.ok(list.every((value, i) => i === 0 || list[i - 1] < value), "and sorted");
    }
  }

  // (4) The transitive-closure decision agrees with the SCC criterion everywhere.
  for (let seed = 1; seed <= 8; seed++) {
    const n = 6 + (seed % 4);
    const clauses = randomTwoCnf(n, 2 * n + seed, BigInt(seed) * 6151n);
    const { adjacency } = implicationGraph(clauses, n);
    const reach = transitiveClosure(adjacency);
    let closureUnsat = false;
    for (let v = 0; v < n; v++) {
      const positive = literal(v, true);
      const negative = literal(v, false);
      if (reach[positive][negative] && reach[negative][positive]) closureUnsat = true;
      assert.ok(reach[positive][positive] && reach[negative][negative],
        "reachability is reflexive");
    }
    assert.strictEqual(closureUnsat, !bruteForceTwoSat(clauses, n).satisfiable,
      "the Floyd-Warshall closure must decide the same way as the SCCs");
  }

  // (5) The topological-order rule is what makes the model valid, so it must satisfy the formula even
  //     for instances with many interacting clauses.
  {
    const clauses = [];
    for (let v = 0; v + 1 < 10; v++) {
      clauses.push([literal(v, true), literal(v + 1, true)]);
      clauses.push([literal(v, false), literal(v + 1, false)]);
    }
    const solved = twoSatSolve(clauses, 10);
    assert.strictEqual(solved.satisfiable, true, "an alternating chain is satisfiable");
    assert.ok(satisfies(clauses, solved.model));
    assert.strictEqual(dpllModelCount(clauses, 10), bruteForceTwoSat(clauses, 10).models);
  }

  // (6) Degenerate inputs: duplicated literals, a single variable, zero variables.
  assert.strictEqual(twoSatSolve([[literal(0, true), literal(0, true)]], 2).model[0], true);
  assert.strictEqual(twoSatSolve([[literal(0, false), literal(0, false)]], 2).model[0], false);
  assert.strictEqual(twoSatSolve([[literal(0, true), literal(0, false)], []].slice(0, 1), 1)
    .satisfiable, true, "a tautological clause over one variable is fine");
  assert.strictEqual(twoSatSolve([], 1).satisfiable, true);
  assert.deepStrictEqual(twoSatSolve([], 1).model, [true],
    "with no clauses the extraction rule sets the single component true");
  assert.strictEqual(twoSatSolve([], 0).satisfiable, true, "no variables, no constraints");
  assert.deepStrictEqual(twoSatSolve([], 0).model, []);
  assert.strictEqual(bruteForceTwoSat([], 0).models, 1, "the empty assignment is the only model");
  assert.throws(() => implicationGraph([[]], 1), /empty clause is not 2-CNF/);
  assert.deepStrictEqual(implicationGraph([[literal(0, true), literal(1, true)]], 2).edges,
    [[literal(0, false), literal(1, true)], [literal(1, false), literal(0, true)]].sort((a, b) =>
      (a[0] * 4 + a[1]) - (b[0] * 4 + b[1])),
  "one clause gives exactly the two contrapositive implications");
  assert.strictEqual(twoSatDecide([[literal(0, true), literal(0, true)]], 1).satisfiable, true,
    "(x or x) is satisfiable: -x -> x leaves x = true");
  assert.strictEqual(twoSatSolve([[literal(0, true), literal(0, true)]], 1).model[0], true);

  // (7) Determinism.
  const repeat = randomTwoCnf(9, 22, 20240607n);
  assert.deepStrictEqual(twoSatSolve(repeat, 9), twoSatSolve(repeat, 9));
  assert.deepStrictEqual(partitionKey(tarjanScc(implicationGraph(repeat, 9).adjacency)),
    partitionKey(tarjanScc(implicationGraph(repeat, 9).adjacency)));
}

module.exports = { lcg64Next, lcg64Int, literal, negation, variableOf, isPositive,
  implicationGraph, tarjanScc, kosarajuScc, bruteForceScc, reachabilityClosure, twoSatDecide,
  twoSatSolve, twoSatDecisionOrder, satisfies, bruteForceTwoSat, dpllModelCount, randomTwoCnf,
  partitionKey, transitiveClosure };

if (require.main === module) {
  main();
  console.log("PASS");
}
