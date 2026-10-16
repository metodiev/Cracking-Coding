/** Task 933 - Implement runtime that compiles high-level ops into fused kernels with autotuning.
 *  (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)
 *
 * A tiny tensor runtime for element-wise programs. A node's args are either `{constant: v}` or the
 * id of an earlier node, so the graph is always in topological order and validation rejects forward
 * references outright.
 *   interpreter  the reference: one kernel launch per node, every intermediate materialised, hence
 *                n * 8 * sum(arity + 1) bytes of traffic and |nodes| launches.
 *   compiler     fusion groups are built by walking back from every element-wise sink through
 *                producers with exactly one consumer, so a chain of k element-wise ops becomes ONE
 *                kernel that reads only the group's external values - the k-1 intermediates vanish.
 *                A shared (diamond) value has two consumers and is therefore materialised once.
 *   autotuner    a deterministic cost model (launch + traffic + per-tile overhead + a spill penalty
 *                that grows with the tile) is minimised by hill climbing over the candidate tiles and
 *                the winner is cross-checked against an exhaustive scan of the same candidates.
 * Simplification: rank-1 tensors, closures instead of generated code, synthetic cost model rather
 * than wall-clock timing. Verified: fused and interpreted results agree exactly, fusion cuts launches
 * and traffic, the tuner returns the exhaustive optimum, tile remainders stay correct, boundary and
 * invalid-input cases, and determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    return (this.state = x >>> 0);
  }
  int(bound) { return this.next() % bound; }
}

const BINARY = { add: (a, b) => a + b, mul: (a, b) => a * b, sub: (a, b) => a - b };
const UNARY = { relu: (a) => Math.max(0, a), neg: (a) => -a, half: (a) => a / 2 };
const ELEMENTWISE = new Set([...Object.keys(BINARY), ...Object.keys(UNARY)]);
const SOURCE = "input";
const REDUCTION = "sum";
const isConstant = (arg) => typeof arg === "object" && arg !== null && "constant" in arg;

/** Nodes must be in topological order, so a reference can only point backwards. */
function validate(graph) {
  assert.ok(graph && Array.isArray(graph.nodes) && graph.nodes.length > 0, "the graph needs nodes");
  const ids = new Set();
  for (const node of graph.nodes) {
    assert.ok(typeof node.id === "string" && !ids.has(node.id), `duplicate or invalid id ${node.id}`);
    const known = ELEMENTWISE.has(node.op) || node.op === REDUCTION || node.op === SOURCE;
    assert.ok(known, `unknown op ${node.op}`);
    const arity = node.op === SOURCE ? 0 : (node.op === REDUCTION ? 1 : (BINARY[node.op] ? 2 : 1));
    assert.strictEqual(node.args.length, arity, `op ${node.op} takes ${arity} argument(s)`);
    for (const arg of node.args) {
      if (isConstant(arg)) continue;
      assert.ok(ids.has(arg), `node ${node.id} reads an undefined value`);
    }
    ids.add(node.id);
  }
  const outputs = graph.outputs ?? [graph.nodes[graph.nodes.length - 1].id];
  for (const output of outputs) assert.ok(ids.has(output), `unknown output ${output}`);
  return { ...graph, outputs };
}

/* ------------------------------------------------------------------ interpreter (reference) */

function interpret(graph, inputs) {
  const g = validate(graph);
  const env = new Map();
  let launches = 0, bytes = 0;
  for (const node of g.nodes) {
    const args = node.args.map((arg) => (isConstant(arg) ? arg.constant : env.get(arg)));
    let result;
    if (node.op === SOURCE) {
      result = inputs.slice();
    } else if (node.op === REDUCTION) {
      result = [args[0].reduce((a, v) => a + v, 0)];
    } else {
      const at = (value, index) => (Array.isArray(value) ? value[index] : value); // constants broadcast
      result = inputs.map((_, i) => (BINARY[node.op]
        ? BINARY[node.op](at(args[0], i), at(args[1], i))
        : UNARY[node.op](at(args[0], i), at(args[1], i))));
    }
    env.set(node.id, result);
    launches++;
    bytes += inputs.length * 8 * (node.args.length + 1);
  }
  return { env, launches, bytes, outputs: g.outputs.map((id) => env.get(id)) };
}

/* ------------------------------------------------------------------ fusion compiler */

function consumerCounts(nodes) {
  const counts = new Map();
  for (const node of nodes) {
    for (const arg of node.args) if (!isConstant(arg)) counts.set(arg, (counts.get(arg) ?? 0) + 1);
  }
  return counts;
}

/** One group per element-wise sink, extended backwards through single-consumer producers. */
function buildGroups(graph) {
  const g = validate(graph);
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const consumers = consumerCounts(g.nodes);
  const assigned = new Set();
  const groups = [];
  for (const node of g.nodes) {
    if (!ELEMENTWISE.has(node.op) || assigned.has(node.id)) continue;
    const hasElementwiseConsumer = g.nodes.some((other) =>
      ELEMENTWISE.has(other.op) && other.args.some((arg) => !isConstant(arg) && arg === node.id));
    if (hasElementwiseConsumer) continue; // not a sink: its consumer will pull it in
    const members = [];
    let current = node;
    for (;;) {
      members.unshift(current.id);
      assigned.add(current.id);
      const producer = current.args.find((arg) => !isConstant(arg)
        && ELEMENTWISE.has(byId.get(arg).op) && consumers.get(arg) === 1);
      if (!producer) break;
      current = byId.get(producer);
    }
    const external = [];
    for (const id of members) {
      for (const arg of byId.get(id).args) {
        if (!isConstant(arg) && members.includes(arg)) continue;
        external.push(arg);
      }
    }
    groups.push({ sink: node.id, members, external });
  }
  for (const node of g.nodes) {
    if (assigned.has(node.id)) continue;
    groups.push({ sink: node.id, members: [node.id], external: node.args });
  }
  return orderGroups(groups);
}

/** Execute the groups in dependency order: a group runs after every group it reads from. */
function orderGroups(groups) {
  const owner = new Map();
  groups.forEach((group, index) => group.members.forEach((id) => owner.set(id, index)));
  const pending = groups.map((_, index) => index);
  const done = new Set();
  const ordered = [];
  while (pending.length > 0) {
    const position = pending.findIndex((index) => groups[index].external
      .every((arg) => isConstant(arg) || done.has(owner.get(arg))));
    assert.ok(position >= 0, "the fusion groups must be acyclic");
    const index = pending.splice(position, 1)[0];
    done.add(index);
    ordered.push(groups[index]);
  }
  return ordered;
}

/** Fused execution: one kernel per group, materialising only the group outputs. */
function execute(graph, inputs, options = {}) {
  const g = validate(graph);
  const groups = buildGroups(g);
  const tile = options.tile ?? inputs.length;
  assert.ok(Number.isInteger(tile) && tile >= 1, "the tile must be a positive integer");
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const env = new Map();
  let bytes = 0, launches = 0;
  for (const group of groups) {
    const sink = byId.get(group.sink);
    const member = new Set(group.members);
    // Each argument becomes an index accessor: constants broadcast, group members are recomputed.
    const accessor = (arg) => {
      if (isConstant(arg)) return () => arg.constant;
      if (!member.has(arg)) { const value = env.get(arg); return (i) => value[i]; }
      const node = byId.get(arg);
      const inner = node.args.map(accessor);
      return (i) => (BINARY[node.op] ? BINARY[node.op](inner[0](i), inner[1](i))
        : UNARY[node.op](inner[0](i), inner[1] ? inner[1](i) : undefined));
    };
    if (sink.op === SOURCE) {
      env.set(sink.id, inputs.slice());
      bytes += inputs.length * 8;
    } else if (sink.op === REDUCTION) {
      const source = accessor(sink.args[0]);
      let total = 0;
      for (let i = 0; i < inputs.length; i++) total += source(i);
      env.set(sink.id, [total]);
      bytes += 8 * (group.external.length + 1);
    } else {
      const evaluate = accessor(sink.id);
      const out = new Array(inputs.length);
      for (let base = 0; base < inputs.length; base += tile) {
        for (let i = base; i < Math.min(base + tile, inputs.length); i++) out[i] = evaluate(i);
      }
      env.set(sink.id, out);
      bytes += inputs.length * 8 * (group.external.length + 1);
    }
    launches++;
  }
  return { env, groups, bytes, launches, outputs: g.outputs.map((id) => env.get(id)) };
}

/* ------------------------------------------------------------------ autotuner */

/** Deterministic cost model: launches + traffic + per-tile overhead + a spill penalty. */
function costModel(plan, options = {}) {
  const launchCost = options.launchCost ?? 12;
  const byteCost = options.byteCost ?? 1 / 1000;
  const tileCost = options.tileCost ?? 0.4;
  const cacheLine = options.cacheLine ?? 8;
  const spillCost = options.spillCost ?? 0.15;
  const tiles = Math.ceil(plan.size / plan.tile);
  return plan.launches * launchCost + plan.bytes * byteCost + tiles * tileCost
    + Math.max(0, plan.tile - cacheLine) * spillCost * tiles;
}

function autoTune(size, launches, bytes, options = {}) {
  const candidates = options.candidates ?? [1, 2, 4, 8, 16, 32];
  assert.ok(candidates.length > 0, "there must be at least one candidate tile");
  const score = (tile) => costModel({ size, tile, launches, bytes }, options);
  let current = candidates[0];
  for (let step = 0; step < candidates.length * 2; step++) {
    const position = candidates.indexOf(current);
    const better = [candidates[position - 1], candidates[position + 1]]
      .filter((tile) => tile !== undefined && score(tile) < score(current) - 1e-12);
    if (better.length === 0) break;
    current = better.reduce((best, tile) => (score(tile) < score(best) ? tile : best), current);
  }
  const exhaustive = candidates.reduce((best, tile) => (score(tile) < score(best) ? tile : best), candidates[0]);
  return { best: current, exhaustive, scores: candidates.map((tile) => [tile, score(tile)]) };
}

/* ------------------------------------------------------------------ checks */

function buildChain(length) {
  const nodes = [{ id: "src", op: SOURCE, args: [] }];
  let previous = "src";
  for (let k = 0; k < length; k++) {
    const id = `m${k}`;
    nodes.push({ id, op: k % 2 === 0 ? "mul" : "add", args: [previous, { constant: 2 + k }] });
    previous = id;
  }
  return { nodes, outputs: [previous] };
}

function main() {
  const prng = new Prng(20250933);
  const inputs = Array.from({ length: 10 }, () => prng.int(21) - 10);
  const graph = buildChain(6);

  // 1. Equivalence: the fused kernels reproduce the interpreter exactly on integer data.
  const interpreted = interpret(graph, inputs);
  const compiled = execute(graph, inputs, { tile: 4 });
  assert.deepStrictEqual(compiled.outputs, interpreted.outputs, "fusion changed the result");
  assert.strictEqual(compiled.outputs.length, 1);
  assert.strictEqual(graph.nodes.length, 7);

  // 2. Fusion economics: fewer launches and less traffic, and both match the group formula.
  assert.ok(interpreted.launches > compiled.launches, "fusion must reduce kernel launches");
  assert.ok(compiled.bytes < interpreted.bytes / 2,
    `traffic only fell from ${interpreted.bytes} to ${compiled.bytes}`);
  assert.strictEqual(compiled.launches, compiled.groups.length);
  assert.strictEqual(compiled.groups.length, 2, "the input source plus one fused chain");
  const formula = compiled.groups.reduce((acc, g) => acc + inputs.length * 8 * (g.external.length + 1), 0);
  assert.strictEqual(compiled.bytes, formula, "the traffic accounting must match the group formula");
  assert.strictEqual(interpreted.bytes,
    graph.nodes.reduce((acc, n) => acc + inputs.length * 8 * (n.args.length + 1), 0));

  // 3. Autotuning: hill climbing must land on the exhaustive optimum, and the model must have an
  //    interior optimum so the search is non-trivial.
  const tuned = autoTune(inputs.length, compiled.launches, compiled.bytes);
  assert.strictEqual(tuned.best, tuned.exhaustive, "the tuner missed the exhaustive optimum");
  const scores = tuned.scores.map(([, score]) => score);
  const bestIndex = scores.indexOf(Math.min(...scores));
  assert.ok(Math.min(...scores) < Math.max(...scores), "the cost model must not be constant");
  assert.ok(bestIndex > 0 && bestIndex < scores.length - 1,
    "the optimum must be an interior tile for the spill penalty to matter");
  assert.ok(costModel({ size: 100, tile: 8, launches: 1, bytes: 800 })
    < costModel({ size: 100, tile: 32, launches: 1, bytes: 800 }), "the spill penalty must bite");
  assert.ok(autoTune(inputs.length, 1, 80, { candidates: [4] }).best === 4, "a single candidate wins");

  // 4. Boundary: a one-op program saves nothing; a reduction splits the schedule.
  const single = { nodes: [{ id: "src", op: SOURCE, args: [] },
    { id: "y", op: "relu", args: ["src"] }], outputs: ["y"] };
  const singleCompiled = execute(single, inputs);
  assert.deepStrictEqual(singleCompiled.outputs, interpret(single, inputs).outputs);
  assert.strictEqual(singleCompiled.launches, 2, "the source and the single op are separate kernels");
  assert.strictEqual(singleCompiled.bytes, interpret(single, inputs).bytes);
  const mixed = { nodes: [
    { id: "src", op: SOURCE, args: [] },
    { id: "a", op: "mul", args: ["src", { constant: 2 }] },
    { id: "b", op: REDUCTION, args: ["a"] },
    { id: "c", op: "relu", args: ["b"] },
  ], outputs: ["c"] };
  const mixedCompiled = execute(mixed, inputs);
  assert.strictEqual(mixedCompiled.launches, 4, "the source, the mul, the reduction and the relu");
  assert.deepStrictEqual(mixedCompiled.outputs, interpret(mixed, inputs).outputs);

  // 5. Remainder handling: sizes that are not multiples of the tile stay correct.
  const odd = inputs.slice(0, 7);
  for (const tile of [1, 2, 3, 4, 8, 16]) {
    assert.deepStrictEqual(execute(graph, odd, { tile }).outputs, interpret(graph, odd).outputs,
      `tile ${tile} mishandled the remainder`);
  }

  // 6. Semantics: relu clamps, and a shared subexpression is materialised once and reused.
  const diamond = { nodes: [
    { id: "src", op: SOURCE, args: [] },
    { id: "s", op: "sub", args: ["src", { constant: 4 }] },
    { id: "left", op: "relu", args: ["s"] },
    { id: "right", op: "neg", args: ["s"] },
    { id: "out", op: "add", args: ["left", "right"] },
  ], outputs: ["out"] };
  const diamondInterpreted = interpret(diamond, inputs);
  const diamondCompiled = execute(diamond, inputs);
  assert.deepStrictEqual(diamondCompiled.outputs, diamondInterpreted.outputs);
  assert.strictEqual(diamondInterpreted.launches, 5);
  assert.ok(diamondCompiled.groups.some((group) => group.members.length === 1 && group.sink === "s"),
    "the shared value must not be fused away");
  assert.ok(diamondInterpreted.outputs[0].every((v, i) =>
    v === Math.max(0, inputs[i] - 4) - (inputs[i] - 4)));

  // 7. Invalid input is rejected: unknown ops, wrong arity, forward references, unknown outputs.
  assert.throws(() => validate({ nodes: [{ id: "a", op: "pow", args: ["src"] }] }), /unknown op/);
  assert.throws(() => validate({ nodes: [{ id: "a", op: "add", args: ["src"] }] }), /takes 2 argument/);
  assert.throws(() => validate({ nodes: [{ id: "a", op: "relu", args: ["ghost"] }] }), /undefined value/);
  assert.throws(() => validate({ nodes: [
    { id: "a", op: "relu", args: ["b"] }, { id: "b", op: "relu", args: ["a"] },
  ] }), /undefined value/);
  assert.throws(() => validate({ nodes: [{ id: "src", op: SOURCE, args: [] },
    { id: "y", op: "relu", args: ["src"] }], outputs: ["nope"] }), /unknown output/);
  assert.throws(() => validate({ nodes: [] }), /needs nodes/);
  assert.throws(() => execute(graph, inputs, { tile: 0 }), /positive integer/);
  assert.throws(() => autoTune(10, 1, 8, { candidates: [] }), /at least one candidate/);

  // 8. Determinism: identical seeds rebuild the same program and the same schedule.
  const rerun = () => {
    const p = new Prng(20250933);
    const data = Array.from({ length: 10 }, () => p.int(21) - 10);
    const g = buildChain(6);
    const fused = execute(g, data, { tile: 4 });
    return { outputs: fused.outputs, groups: fused.groups,
      tile: autoTune(data.length, fused.launches, fused.bytes).best };
  };
  assert.deepStrictEqual(rerun(), rerun());
  assert.deepStrictEqual(interpret(graph, inputs).outputs, compiled.outputs);

  console.log(`runtime: ${graph.nodes.length} nodes -> ${compiled.launches} fused kernel(s), ` +
    `traffic ${interpreted.bytes} -> ${compiled.bytes} bytes, tuned tile ${tuned.best} ` +
    `(exhaustive ${tuned.exhaustive})`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, validate, interpret, buildGroups, execute, costModel, autoTune, buildChain,
  consumerCounts };
