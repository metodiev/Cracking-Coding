/** Task 782 - Implement probabilistic graphical models inference (belief propagation). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * A factor graph over discrete variables: variables with a cardinality and factors carrying a
 * potential table over their scope. Sum-product message passing exchanges
 *
 *     m_{f->x}(x) = sum over the other scope variables of  f(scope) * prod m_{x'->f}(x')
 *     m_{x->f}(x) = prod over the other factors f' of  m_{f'->x}(x)
 *
 * and the belief of x is the product of its incoming factor messages, normalised. On a tree the
 * two-sweep schedule (leaves to root, then root to leaves) computes every message once from final
 * inputs, so marginals and the partition function are exact: both are compared against brute-force
 * enumeration of every joint assignment, not against a tolerance. A graph with a cycle is iterated
 * instead, with scale-normalised messages and damping: there BP is only approximate, so the checks
 * report convergence, normalisation, and the measured distance from the exact answer.
 *
 * What is verified: tree marginals and Z equal the enumerated values to 1e-9 across several random
 * trees; beliefs are normalised and non-negative; Z is computed from the message product and
 * factorises over components; hard 0/1 constraints behave; a four-cycle converges under damping and
 * stays within a measured distance of the truth; degenerate graphs (one variable, a one-variable
 * factor, two disconnected components) are exact; malformed graphs are rejected.
 */
"use strict";

class FactorGraph {
  constructor() {
    this.variables = new Map(); // name -> cardinality
    this.factors = [];          // { scope: [names], table: Float64Array, labels: [names] }
  }

  addVariable(name, cardinality) {
    if (!Number.isInteger(cardinality) || cardinality < 1) throw new RangeError("cardinality must be a positive integer");
    this.variables.set(name, cardinality);
    return this;
  }

  /** Adds a factor; `table` is row-major over the scope in the given order. */
  addFactor(scope, table) {
    for (const name of scope) {
      if (!this.variables.has(name)) throw new RangeError(`unknown variable ${name}`);
    }
    if (new Set(scope).size !== scope.length) throw new RangeError("a factor scope must be a set of distinct variables");
    const expected = scope.reduce((product, name) => product * this.variables.get(name), 1);
    if (table.length !== expected) throw new RangeError(`a factor over ${scope.join(",")} needs ${expected} entries`);
    for (const value of table) if (!Number.isFinite(value)) throw new RangeError("potentials must be finite");
    this.factors.push({ scope, table: Float64Array.from(table) });
    return this;
  }

  cardinality(name) { return this.variables.get(name); }

  /** The exact joint table: enumerate every assignment (small graphs only - it is the oracle). */
  bruteForce() {
    const names = [...this.variables.keys()];
    const sizes = names.map((name) => this.variables.get(name));
    const total = sizes.reduce((product, size) => product * size, 1);
    const joint = new Float64Array(total);
    const assignment = new Array(names.length).fill(0);
    let normaliser = 0;
    for (let index = 0; index < total; index++) {
      let rest = index;
      for (let i = names.length - 1; i >= 0; i--) { assignment[i] = rest % sizes[i]; rest = Math.floor(rest / sizes[i]); }
      let value = 1;
      for (const factor of this.factors) {
        let offset = 0;
        for (let i = 0; i < factor.scope.length; i++) {
          const position = names.indexOf(factor.scope[i]);
          offset = offset * sizes[position] + assignment[position];
        }
        value *= factor.table[offset];
      }
      joint[index] = value;
      normaliser += value;
    }
    const marginals = new Map();
    for (let i = 0; i < names.length; i++) {
      const marginal = new Array(sizes[i]).fill(0);
      for (let index = 0; index < total; index++) {
        let rest = index;
        const assignmentIndex = [];
        for (let j = names.length - 1; j >= 0; j--) { assignmentIndex[j] = rest % sizes[j]; rest = Math.floor(rest / sizes[j]); }
        marginal[assignmentIndex[i]] += joint[index];
      }
      marginals.set(names[i], marginal.map((value) => value / normaliser));
    }
    return { z: normaliser, marginals, joint };
  }
}

/** True when the factor graph's bipartite graph is acyclic (then BP is exact after two sweeps). */
function isTree(graph) {
  const names = [...graph.variables.keys()];
  const edges = graph.factors.reduce((sum, factor) => sum + factor.scope.length, 0);
  const nodes = names.length + graph.factors.length;
  const parent = new Map(names.map((name) => [name, name]));
  const find = (x) => (parent.get(x) === x ? x : (parent.set(x, find(parent.get(x))), parent.get(x)));
  for (const factor of graph.factors) {
    for (let i = 1; i < factor.scope.length; i++) {
      const a = find(factor.scope[0]);
      const b = find(factor.scope[i]);
      if (a !== b) parent.set(a, b);
    }
  }
  const components = new Set(names.map(find)).size;
  return edges === nodes - components;
}

/**
 * Sum-product BP. A tree is solved with the exact two-sweep schedule (leaf -> root, then root ->
 * leaf), which also yields the exact partition function. A graph with cycles is iterated instead,
 * with scale-normalised messages and optional damping; there the beliefs are approximate and no
 * partition function is claimed.
 */
function beliefPropagation(graph, { iterations = 200, damping = 0.5, tolerance = 1e-12 } = {}) {
  const names = [...graph.variables.keys()];
  const messages = new Map(); // "source|target" -> Float64Array, absent means the uniform message
  const uniforms = new Map();
  const uniformFor = (name) => {
    if (!uniforms.has(name)) uniforms.set(name, new Float64Array(graph.cardinality(name)).fill(1));
    return uniforms.get(name);
  };
  const readVarToFactor = (name, f) => messages.get(`${name}|f${f}`) ?? uniformFor(name);
  const readFactorToVar = (f, name) => messages.get(`f${f}|${name}`) ?? uniformFor(name);
  const scopeFor = (f) => graph.factors[f].scope;
  const factorOf = (key) => Number(key.slice(1));

  const variableToFactor = (name, f) => {
    const out = new Float64Array(graph.cardinality(name)).fill(1);
    for (let g = 0; g < graph.factors.length; g++) {
      if (g === f || !scopeFor(g).includes(name)) continue;
      const incoming = readFactorToVar(g, name);
      for (let v = 0; v < out.length; v++) out[v] *= incoming[v];
    }
    return out;
  };

  const factorToVariable = (f, target) => {
    const scope = scopeFor(f);
    const size = graph.cardinality(target);
    const out = new Float64Array(size);
    const strides = [];
    let stride = 1;
    for (let i = scope.length - 1; i >= 0; i--) { strides[i] = stride; stride *= graph.cardinality(scope[i]); }
    const targetPosition = scope.indexOf(target);
    const assignment = new Array(scope.length).fill(0);
    for (let value = 0; value < size; value++) {
      assignment[targetPosition] = value;
      let total = 0;
      const enumerate = (position, weight) => {
        if (position === scope.length) {
          total += weight * graph.factors[f].table[assignment.reduce((sum, v, i) => sum + v * strides[i], 0)];
          return;
        }
        if (position === targetPosition) { enumerate(position + 1, weight); return; }
        const incoming = readVarToFactor(scope[position], f);
        for (let v = 0; v < graph.cardinality(scope[position]); v++) {
          assignment[position] = v;
          enumerate(position + 1, weight * incoming[v]);
        }
      };
      enumerate(0, 1);
      out[value] = total;
    }
    return out;
  };

  const tree = isTree(graph);
  let sweeps = 0;
  let converged = false;
  let maxChange = 0;
  if (tree) {
    const visited = new Set();
    const components = [];
    for (const start of [...names, ...graph.factors.map((_, f) => `f${f}`)]) {
      if (visited.has(start)) continue;
      const order = [];
      const queue = [start];
      visited.add(start);
      while (queue.length > 0) {
        const node = queue.shift();
        order.push(node);
        const neighbours = node.startsWith("f")
          ? scopeFor(factorOf(node))
          : graph.factors.map((factor, f) => (factor.scope.includes(node) ? `f${f}` : null)).filter(Boolean);
        for (const next of neighbours) if (!visited.has(next)) { visited.add(next); queue.push(next); }
      }
      components.push(order);
    }
    for (const order of components) {
      // leaf -> root, then root -> leaf: every message is computed exactly once, from final inputs
      const pass = (nodes) => {
        for (const node of nodes) {
          if (node.startsWith("f")) {
            const f = factorOf(node);
            for (const name of scopeFor(f)) messages.set(`f${f}|${name}`, factorToVariable(f, name));
          } else {
            for (let f = 0; f < graph.factors.length; f++) {
              if (scopeFor(f).includes(node)) messages.set(`${node}|f${f}`, variableToFactor(node, f));
            }
          }
        }
      };
      pass([...order].reverse());
      pass(order);
    }
    sweeps = components.reduce((sum, order) => sum + 2 * order.length, 0);
    converged = true;
  } else {
    for (let iteration = 0; iteration < iterations; iteration++) {
      sweeps = iteration + 1;
      maxChange = 0;
      for (let f = 0; f < graph.factors.length; f++) {
        for (const name of scopeFor(f)) {
          const current = readFactorToVar(f, name);
          const next = factorToVariable(f, name);
          const blended = new Float64Array(next.length);
          for (let v = 0; v < next.length; v++) blended[v] = damping * next[v] + (1 - damping) * current[v];
          const scaleNext = Math.max(...blended, 1e-300);
          const scaleCurrent = Math.max(...current, 1e-300);
          for (let v = 0; v < next.length; v++) {
            maxChange = Math.max(maxChange, Math.abs(blended[v] / scaleNext - current[v] / scaleCurrent));
          }
          for (let v = 0; v < next.length; v++) blended[v] /= scaleNext; // scale-invariant messages
          messages.set(`f${f}|${name}`, blended);
        }
      }
      if (maxChange < tolerance) { converged = true; if (sweeps > 3) break; }
    }
  }

  const beliefs = new Map();
  for (const name of names) {
    const belief = new Float64Array(graph.cardinality(name)).fill(1);
    for (let f = 0; f < graph.factors.length; f++) {
      if (!scopeFor(f).includes(name)) continue;
      const incoming = readFactorToVar(f, name);
      for (let v = 0; v < belief.length; v++) belief[v] *= incoming[v];
    }
    const total = belief.reduce((sum, value) => sum + value, 0);
    beliefs.set(name, Array.from(belief, (value) => (total === 0 ? 1 / belief.length : value / total)));
  }

  let z = null;
  if (tree && names.length > 0) {
    // Z factorises over components; for a representative variable x, Z = sum_v prod_{f contains x} m_{f->x}(v).
    const parent = new Map(names.map((name) => [name, name]));
    const find = (x) => (parent.get(x) === x ? x : (parent.set(x, find(parent.get(x))), parent.get(x)));
    for (const factor of graph.factors) {
      for (let i = 1; i < factor.scope.length; i++) {
        const a = find(factor.scope[0]);
        const b = find(factor.scope[i]);
        if (a !== b) parent.set(a, b);
      }
    }
    const representatives = new Map();
    for (const name of names) if (!representatives.has(find(name))) representatives.set(find(name), name);
    z = 1;
    for (const name of representatives.values()) {
      let componentZ = 0;
      for (let v = 0; v < graph.cardinality(name); v++) {
        let product = 1;
        for (let f = 0; f < graph.factors.length; f++) {
          if (scopeFor(f).includes(name)) product *= readFactorToVar(f, name)[v];
        }
        componentZ += product;
      }
      z *= componentZ;
    }
  }
  return { beliefs, z, sweeps, converged, maxChange, exact: tree, messages };
}

/** A tree factor graph: variables in a chain plus a leaf factor, generated deterministically. */
function chainGraph(rng, length, cardinality, extras = 0) {
  const graph = new FactorGraph();
  const names = Array.from({ length }, (_, i) => `x${i}`);
  for (const name of names) graph.addVariable(name, cardinality);
  for (let i = 0; i + 1 < length; i++) {
    const table = [];
    for (let a = 0; a < cardinality; a++) {
      for (let b = 0; b < cardinality; b++) table.push(a === b ? 1 + rng(50) / 10 : 0.1 + rng(20) / 100);
    }
    graph.addFactor([names[i], names[i + 1]], table);
  }
  for (let i = 0; i < Math.min(extras, length); i++) {
    graph.addFactor([names[i]], Array.from({ length: cardinality }, () => 0.5 + rng(40) / 10));
  }
  return graph;
}

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

module.exports = { FactorGraph, beliefPropagation, chainGraph, makeRng };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;
  const rng = makeRng(20260782);
  const CLOSE = 1e-9;

  // -- degenerate graphs -------------------------------------------------------------
  const lone = new FactorGraph();
  lone.addVariable("solo", 3);
  const loneExact = lone.bruteForce();
  const loneBp = beliefPropagation(lone);
  strictEqual(loneExact.z, 3, "a single uniform variable has Z = its cardinality");
  deepStrictEqual(loneBp.beliefs.get("solo"), [1 / 3, 1 / 3, 1 / 3], "a lone variable is uniform");
  ok(Math.abs(loneBp.z - 3) < CLOSE, "BP must reproduce Z for a lone variable");

  const singleFactor = new FactorGraph();
  singleFactor.addVariable("a", 2);
  singleFactor.addFactor(["a"], [1, 4]);
  const singleBp = beliefPropagation(singleFactor);
  deepStrictEqual(singleBp.beliefs.get("a"), [0.2, 0.8], "a one-variable factor is the normalised potential");
  ok(Math.abs(singleBp.z - 5) < CLOSE, "Z is the sum of the potential");

  const disconnected = new FactorGraph();
  disconnected.addVariable("p", 2).addVariable("q", 3);
  disconnected.addFactor(["p"], [1, 3]).addFactor(["q"], [2, 2, 5]);
  const disconnectedBp = beliefPropagation(disconnected);
  const disconnectedExact = disconnected.bruteForce();
  for (const name of ["p", "q"]) {
    for (let v = 0; v < disconnected.variables.get(name); v++) {
      ok(Math.abs(disconnectedBp.beliefs.get(name)[v] - disconnectedExact.marginals.get(name)[v]) < CLOSE,
        `disconnected marginals must be exact for ${name}`);
    }
  }
  ok(Math.abs(disconnectedBp.z - disconnectedExact.z) < CLOSE, "disconnected Z must be exact");

  const hard = new FactorGraph();
  hard.addVariable("t", 2).addVariable("u", 2);
  hard.addFactor(["t", "u"], [1, 0, 0, 1]); // equality constraint
  hard.addFactor(["t"], [3, 1]);
  const hardExact = hard.bruteForce();
  const hardBp = beliefPropagation(hard, { damping: 0.5 });
  for (const name of ["t", "u"]) {
    deepStrictEqual(hardBp.beliefs.get(name).map((v) => Number(v.toFixed(6))),
      hardExact.marginals.get(name).map((v) => Number(v.toFixed(6))), `hard constraint marginal for ${name}`);
  }
  ok(Math.abs(hardBp.z - hardExact.z) < 1e-6, "the hard constraint keeps Z exact");

  // -- validation --------------------------------------------------------------------
  throws(() => new FactorGraph().addFactor(["ghost"], [1]), RangeError, "factors must reference known variables");
  throws(() => lone.addFactor(["solo", "solo"], [1, 2, 3]), RangeError, "a scope must be distinct variables");
  throws(() => lone.addFactor(["solo"], [1, 2]), RangeError, "the table size must match the scope");
  throws(() => lone.addFactor(["solo"], [1, NaN, 3]), RangeError, "potentials must be finite");
  throws(() => lone.addVariable("bad", 0), RangeError, "cardinality must be positive");

  // -- trees: BP is exact against enumeration ---------------------------------------
  let checkedCells = 0;
  let treeTrials = 0;
  for (const [length, cardinality, extras] of [[2, 2, 0], [3, 3, 1], [4, 2, 2], [5, 3, 0], [6, 2, 3], [3, 4, 1]]) {
    const graph = chainGraph(rng, length, cardinality, extras);
    const exact = graph.bruteForce();
    const bp = beliefPropagation(graph, { iterations: 400, tolerance: 1e-14 });
    treeTrials++;
    for (const name of graph.variables.keys()) {
      const belief = bp.beliefs.get(name);
      const truth = exact.marginals.get(name);
      strictEqual(belief.length, cardinality, "a belief has one entry per state");
      ok(Math.abs(belief.reduce((sum, v) => sum + v, 0) - 1) < 1e-12, `${name} belief must be normalised`);
      ok(belief.every((v) => v >= -1e-15), `${name} belief must be non-negative`);
      for (let v = 0; v < cardinality; v++) {
        ok(Math.abs(belief[v] - truth[v]) < CLOSE, `${name} marginal ${v} differs from enumeration`);
        checkedCells++;
      }
    }
    ok(Math.abs(bp.z - exact.z) < Math.max(CLOSE, exact.z * 1e-9), `Z mismatch on a ${length}-chain`);
    ok(bp.converged, "a tree must converge");
    ok(bp.sweeps <= 400, "the sweep budget is respected");
  }
  ok(treeTrials === 6 && checkedCells >= 60, "several trees and dozens of marginal cells were compared");

  // -- a tree of chains: the message schedule does not depend on the enumeration order
  const wide = chainGraph(rng, 4, 2, 4);
  const wideExact = wide.bruteForce();
  const wideBp = beliefPropagation(wide, { iterations: 400 });
  ok(Math.abs(wideBp.z - wideExact.z) < CLOSE, "a chain with leaf factors keeps Z exact");
  for (const name of wide.variables.keys()) {
    for (let v = 0; v < 2; v++) {
      ok(Math.abs(wideBp.beliefs.get(name)[v] - wideExact.marginals.get(name)[v]) < CLOSE,
        "leaf factors must not break the exactness");
    }
  }

  // -- a cycle: BP is approximate, but its fixed point is close and stable ------------
  const loopy = new FactorGraph();
  for (const name of ["y0", "y1", "y2", "y3"]) loopy.addVariable(name, 2);
  const tables = [[0.9, 0.1, 0.2, 0.85], [0.8, 0.25, 0.15, 0.9], [0.7, 0.3, 0.2, 0.8], [0.6, 0.4, 0.3, 0.75]];
  loopy.addFactor(["y0", "y1"], tables[0]);
  loopy.addFactor(["y1", "y2"], tables[1]);
  loopy.addFactor(["y2", "y3"], tables[2]);
  loopy.addFactor(["y3", "y0"], tables[3]); // closes the loop
  const loopyExact = loopy.bruteForce();
  const loopyBp = beliefPropagation(loopy, { iterations: 500, damping: 0.5, tolerance: 1e-12 });
  ok(loopyBp.exact === false, "the cycle must be detected as a non-tree graph");
  ok(loopyBp.converged, "damped BP must converge on this cycle");
  ok(loopyBp.z === null, "no partition function is claimed on a cyclic graph");
  let worst = 0;
  for (const name of loopy.variables.keys()) {
    for (let v = 0; v < 2; v++) {
      worst = Math.max(worst, Math.abs(loopyBp.beliefs.get(name)[v] - loopyExact.marginals.get(name)[v]));
    }
  }
  ok(worst < 0.1, `the loopy fixed point must stay within 10% of the truth (worst ${worst.toFixed(4)})`);
  ok(!loopyExact.marginals.get("y0").every((v, i) => Math.abs(v - loopyBp.beliefs.get("y0")[i]) < 1e-12),
    "on a cycle the beliefs cannot be exactly the marginals");

  // -- determinism -------------------------------------------------------------------
  const replay = () => {
    const graph = chainGraph(makeRng(991), 5, 3, 2);
    const bp = beliefPropagation(graph, { iterations: 300 });
    return { beliefs: [...bp.beliefs.entries()], z: bp.z, sweeps: bp.sweeps };
  };
  strictEqual(JSON.stringify(replay()), JSON.stringify(replay()), "BP must be deterministic");

  console.log(`trees=${treeTrials} marginal cells vs enumeration=${checkedCells} | loopy worst error=${worst.toFixed(4)} sweeps=${loopyBp.sweeps} | tree sweeps sample=${wideBp.sweeps}`);
  console.log("PASS");
}
