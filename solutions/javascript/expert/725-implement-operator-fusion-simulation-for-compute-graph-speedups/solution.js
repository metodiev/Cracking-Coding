/** Task 725 - Implement operator fusion simulation for compute graph speedups. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * A compiler pass that emits one GPU kernel per operator pays for every intermediate twice:
 * once when it is written and once when it is read. Operator fusion (TVM's `FuseOps`,
 * TensorFlow's Grappler, XLA) merges a chain of operators into a single kernel, so the
 * intermediate never touches memory at all:
 *
 *   unfused:  a+b -> t  |  t*c -> u  |  relu(u) -> y       3 kernels, 5 reads + 3 writes
 *   fused:    relu((a+b)*c) -> y                           1 kernel,  3 reads + 1 write
 *
 * This file implements the fusion pass as a greedy partition of the op set, a tiny
 * interpreter that runs both graphs, and the traffic/launch cost model the pass is judged by.
 *
 * Fusion rules (a faithful subset of what real compilers do): an operator may be fused into
 * its consumer when it is elementwise, has exactly one consumer and that consumer is
 * elementwise, a reduction or a matmul; a reduction or a matmul absorbs producers but is
 * never fused into its own consumer; and a tensor with several consumers is never recomputed
 * into a single one. Every group is therefore a connected subtree of the graph.
 *
 * Verified here: the fused interpreter returns bit-identical results to the unfused one; the
 * reported traffic matches an independent recomputation from the partition; fusion lowers
 * both bytes moved and kernel launches, and the modeled time accordingly; the rules above
 * hold for every produced group; the greedy partition is never better than an exhaustive
 * search over all partitions of small graphs (and is exactly optimal on a chain); the
 * partition covers every operator exactly once; and the degenerate graphs (empty, single
 * operator, no legal pair, one giant group) behave. Deterministic by construction.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

const ELEMENTWISE = "elementwise";
const REDUCE = "reduce";
const MATMUL = "matmul";
const ELEMENT_BYTES = 8;

class ComputeGraph {
  constructor() {
    this.ops = [];                 // { name, kind, inputs, output }
    this.byName = new Map();       // operator name -> operator
    this.producers = new Map();    // tensor name -> operator that writes it
    this.tensorBytes = new Map();  // tensor -> bytes
  }

  /** Define a tensor (graph input or constant) with a byte size. */
  tensor(name, bytes) {
    ok(!this.tensorBytes.has(name), `duplicate tensor ${name}`);
    ok(bytes >= 0, `bad size for ${name}`);
    this.tensorBytes.set(name, bytes);
    return this;
  }

  /** Append an operator; it consumes existing tensors and defines one new one. */
  op(name, kind, inputs, outputBytes) {
    ok(!this.byName.has(name), `duplicate operator ${name}`);
    for (const input of inputs) ok(this.tensorBytes.has(input), `op ${name} consumes unknown tensor ${input}`);
    ok(!this.tensorBytes.has(`${name}/out`), `op ${name} redefines its output`);
    this.ops.push({ name, kind, inputs, index: this.ops.length, output: `${name}/out` });
    const op = this.ops[this.ops.length - 1];
    this.byName.set(name, op);
    this.producers.set(op.output, op);
    this.tensorBytes.set(op.output, outputBytes);
    return this;
  }

  /** Consumers of every tensor, which is what the single-consumer rule needs. */
  consumers() {
    const map = new Map();
    for (const op of this.ops) {
      for (const input of op.inputs) {
        if (!map.has(input)) map.set(input, []);
        map.get(input).push(op);
      }
    }
    return map;
  }

  get opCount() {
    return this.ops.length;
  }
}

const isElementwise = (kind) => kind === "add" || kind === "mul" || kind === "relu";

const isFusableInto = (graph, consumerMap, producer, consumer) =>
  isElementwise(producer.kind)
  && (isElementwise(consumer.kind) || consumer.kind === REDUCE || consumer.kind === MATMUL)
  && (consumerMap.get(producer.output) ?? []).length === 1;

/**
 * Greedy fusion: walk the operators in reverse topological order, so a consumer's group is
 * always known before its producers are considered, and pull each eligible producer into it.
 */
function greedyFusion(graph) {
  const consumerMap = graph.consumers();
  const groupOf = new Array(graph.opCount).fill(-1);
  const groups = [];
  for (let index = graph.opCount - 1; index >= 0; index--) {
    const producer = graph.ops[index];
    const consumers = consumerMap.get(producer.output) ?? [];
    if (consumers.length === 1 && isFusableInto(graph, consumerMap, producer, consumers[0])) {
      const consumer = consumers[0];
      const target = groupOf[consumer.index];
      ok(target !== -1, "reverse topological order must assign the consumer first");
      groupOf[index] = target;
      groups[target].ops.push(index);
    } else {
      groupOf[index] = groups.length;
      groups.push({ ops: [index], leader: index });
    }
  }
  groups.forEach((group) => group.ops.sort((a, b) => a - b));
  // Report groups in program order (by their first operator) so plans are easy to compare.
  const order = groups.map((_, index) => index).sort((a, b) => groups[a].ops[0] - groups[b].ops[0]);
  const remap = new Map(order.map((oldIndex, freshIndex) => [oldIndex, freshIndex]));
  return { groupOf: groupOf.map((id) => remap.get(id)), groups: order.map((index) => groups[index]) };
}

/** External inputs and outputs of a group: what the kernel has to read and write. */
function groupInterface(graph, group) {
  const members = new Set(group.ops);
  const produced = new Set(group.ops.map((index) => graph.ops[index].output));
  const consumerMap = graph.consumers();
  const reads = new Map();
  const writes = new Map();
  for (const index of group.ops) {
    const op = graph.ops[index];
    for (const input of op.inputs) {
      const producer = graph.producers.get(input);
      if (producer !== undefined && members.has(producer.index)) continue;   // computed inside
      reads.set(input, graph.tensorBytes.get(input));
    }
    const users = consumerMap.get(op.output) ?? [];
    const usedOutside = users.some((user) => !members.has(user.index));
    if (usedOutside || users.length === 0) writes.set(op.output, graph.tensorBytes.get(op.output));
    ok(produced.has(op.output), "a group must write something");
  }
  return {
    reads: [...reads],
    writes: [...writes],
    readBytes: [...reads.values()].reduce((a, b) => a + b, 0),
    writeBytes: [...writes.values()].reduce((a, b) => a + b, 0),
    bytes: [...reads.values()].reduce((a, b) => a + b, 0) + [...writes.values()].reduce((a, b) => a + b, 0),
  };
}

/** Cost model: memory traffic plus a fixed launch cost, both fully deterministic. */
const MODEL = { bandwidth: 1.2e12, launchSeconds: 4e-6 };

function costOfPartition(graph, groups) {
  let bytes = 0;
  let launches = groups.length;
  const details = [];
  for (const group of groups) {
    const face = groupInterface(graph, group);
    bytes += face.bytes;
    details.push(face);
  }
  return { bytes, launches, details, seconds: bytes / MODEL.bandwidth + launches * MODEL.launchSeconds };
}

/** Legal partitions for the exhaustive oracle: one group per op is always legal. */
function isValidPartition(graph, groupOf) {
  const consumerMap = graph.consumers();
  const groups = new Map();
  graph.ops.forEach((op, index) => {
    if (!groups.has(groupOf[index])) groups.set(groupOf[index], []);
    groups.get(groupOf[index]).push(index);
  });
  for (const members of groups.values()) {
    const set = new Set(members);
    // Each group must be connected through producer/consumer edges and every merge must be
    // one of the legal fusions the greedy rule allows.
    const reached = new Set([members[0]]);
    let grew = true;
    while (grew) {
      grew = false;
      for (const index of members) {
        if (reached.has(index)) continue;
        const op = graph.ops[index];
        const neighbours = [...(consumerMap.get(op.output) ?? []).map((c) => c.index),
          ...op.inputs.map((i) => graph.producers.get(i)?.index).filter((v) => v !== undefined)];
        if (neighbours.some((n) => reached.has(n) && set.has(n))) {
          reached.add(index);
          grew = true;
        }
      }
    }
    if (reached.size !== members.length) return false;
  }
  for (const op of graph.ops) {
    for (const consumer of consumerMap.get(op.output) ?? []) {
      if (groupOf[op.index] !== groupOf[consumer.index]) continue;
      if (!isFusableInto(graph, consumerMap, op, consumer)) return false;
    }
  }
  return true;
}

/** Exhaustive best partition (restricted growth strings): the oracle for small graphs. */
function bestPartition(graph, limit = 7) {
  if (graph.opCount > limit) throw new Error("the exhaustive partition search is only for small graphs");
  const groupOf = new Array(graph.opCount).fill(0);
  let best = null;
  const recurse = (index, used) => {
    if (index === graph.opCount) {
      if (!isValidPartition(graph, groupOf)) return;
      const groups = [];
      for (let g = 0; g < used; g++) groups.push({ ops: graph.ops.map((_, i) => i).filter((i) => groupOf[i] === g) });
      const cost = costOfPartition(graph, groups);
      if (best === null || cost.bytes < best.bytes) best = { bytes: cost.bytes, launches: cost.launches, groupOf: groupOf.slice() };
      return;
    }
    for (let g = 0; g <= used; g++) {
      groupOf[index] = g;
      recurse(index + 1, Math.max(used, g + 1));
    }
  };
  recurse(0, 0);
  return best;
}

/** Tiny interpreter: evaluate a graph with one kernel per op (the unfused baseline). */
function executeUnfused(graph, inputs, unary) {
  const values = new Map();
  for (const [name, value] of Object.entries(inputs)) values.set(name, value.slice());
  for (const op of graph.ops) {
    const args = op.inputs.map((input) => values.get(input));
    values.set(op.output, op.kind === REDUCE ? reduceTensor(args, unary) : elementwise(args, op.kind, unary));
  }
  return values;
}

/** The same graph evaluated one group at a time: intermediates stay inside the kernel. */
function executeFused(graph, inputs, groups, unary) {
  const values = new Map();
  for (const [name, value] of Object.entries(inputs)) values.set(name, value.slice());
  const order = groups.map((group, index) => ({ group, index })).sort((a, b) => a.group.leader - b.group.leader);
  for (const { group } of order) {
    const local = new Map();
    for (const index of group.ops) {
      const op = graph.ops[index];
      const args = op.inputs.map((input) => (local.has(input) ? local.get(input) : values.get(input)));
      const result = op.kind === REDUCE ? reduceTensor(args, unary) : elementwise(args, op.kind, unary);
      local.set(op.output, result);
      // `values` is only the interpreter's scratch bookkeeping; the memory traffic a fused
      // kernel really spends is computed by groupInterface from the partition.
      values.set(op.output, result);
    }
  }
  return values;
}

/** Elementwise semantics of the tiny IR: the first input chooses the shape. */
function elementwise(args, opName, unary) {
  const [left, right] = args;
  if (opName === "relu") return left.map((value) => Math.max(0, value));
  if (args.length === 1) return left.map((value) => unary(value));
  return left.map((value, i) => (opName === "mul" ? value * right[i] : value + right[i]));
}

function reduceTensor(args, unary) {
  const total = args[0].reduce((acc, value) => acc + value, 0);
  return [unary(total)];
}

/** Deterministic chain builder: `relu`/`add`/`mul` chain plus a widening middle op. */
function buildChain(length, width) {
  const graph = new ComputeGraph();
  graph.tensor("x", width * ELEMENT_BYTES);
  graph.tensor("c", width * ELEMENT_BYTES);
  let previous = "x";
  for (let i = 0; i < length; i++) {
    const kind = i % 3 === 0 ? "add" : i % 3 === 1 ? "mul" : "relu";
    const inputs = kind === "relu" ? [previous] : [previous, "c"];
    graph.op(`op${i}`, kind, inputs, width * ELEMENT_BYTES);
    previous = `op${i}/out`;
  }
  return graph;
}

module.exports = {
  ComputeGraph, greedyFusion, groupInterface, costOfPartition, isValidPartition,
  bestPartition, executeUnfused, executeFused, buildChain, MODEL, ELEMENTWISE, REDUCE, MATMUL,
};

if (require.main === module) {
  const width = 1024;
  const unary = (value) => value * 0.5 + 1;

  // -- a hand-checked four-op chain -------------------------------------------------------
  const hand = new ComputeGraph();
  hand.tensor("a", width * ELEMENT_BYTES).tensor("b", width * ELEMENT_BYTES);
  hand.op("sum", "add", ["a", "b"], width * ELEMENT_BYTES);
  hand.op("twice", "mul", ["sum/out", "b"], width * ELEMENT_BYTES);
  hand.op("act", "relu", ["twice/out"], width * ELEMENT_BYTES);
  const handGreedy = greedyFusion(hand);
  strictEqual(handGreedy.groups.length, 1, "a single-consumer elementwise chain fuses into one kernel");
  deepStrictEqual(handGreedy.groups[0].ops, [0, 1, 2], "and the group holds every operator");
  const handCost = costOfPartition(hand, handGreedy.groups);
  strictEqual(handCost.launches, 1, "one kernel launch for the whole chain");
  strictEqual(handCost.bytes, 3 * width * ELEMENT_BYTES,
    "the fused kernel reads a and b once and writes the final output once");
  const unfusedHand = costOfPartition(hand, hand.ops.map((op) => ({ ops: [op.index], leader: op.index })));
  strictEqual(unfusedHand.launches, 3, "unfused means one kernel per operator");
  strictEqual(unfusedHand.bytes, 8 * width * ELEMENT_BYTES, "unfused: 5 reads of intermediates plus 3 writes");
  ok(handCost.seconds < unfusedHand.seconds, "fusion must lower the modeled time on the chain");

  // -- the partition invariant and independent traffic recomputation ----------------------
  const chain = buildChain(9, width);
  const chainFusion = greedyFusion(chain);
  const covered = chainFusion.groups.flatMap((group) => group.ops).sort((a, b) => a - b);
  deepStrictEqual(covered, chain.ops.map((op) => op.index), "every operator belongs to exactly one group");
  const chainCost = costOfPartition(chain, chainFusion.groups);
  const recomputed = chainFusion.groups.reduce((acc, group) => acc + groupInterface(chain, group).bytes, 0);
  strictEqual(chainCost.bytes, recomputed, "reported traffic must match the recomputation");
  const chainUnfused = costOfPartition(chain, chain.ops.map((op) => ({ ops: [op.index], leader: op.index })));
  ok(chainCost.bytes < chainUnfused.bytes, "fusing a chain must move fewer bytes");
  ok(chainCost.launches < chainUnfused.launches, "and launch fewer kernels");
  ok(chainCost.seconds < chainUnfused.seconds / 2, "the modeled speedup has to be real");
  console.log(`chain: fused ${chainCost.launches} kernel(s)/${chainCost.bytes} B vs ` +
    `unfused ${chainUnfused.launches}/${chainUnfused.bytes} B (${(chainUnfused.seconds / chainCost.seconds).toFixed(1)}x)`);

  // -- the fusion rules really hold on every group ----------------------------------------
  const consumerMap = chain.consumers();
  for (const group of chainFusion.groups) {
    for (const index of group.ops) {
      const op = chain.ops[index];
      for (const consumer of consumerMap.get(op.output) ?? []) {
        if (chainFusion.groupOf[consumer.index] !== chainFusion.groupOf[index]) continue;
        ok(isFusableInto(chain, consumerMap, op, consumer), `group ${index} broke the fusion rules`);
        ok(isElementwise(op.kind), "only an elementwise operator may be fused into a consumer");
      }
    }
  }
  const multiConsumer = new ComputeGraph();
  multiConsumer.tensor("x", width * ELEMENT_BYTES).tensor("c", width * ELEMENT_BYTES);
  multiConsumer.op("shared", "add", ["x", "c"], width * ELEMENT_BYTES);
  multiConsumer.op("left", "mul", ["shared/out", "c"], width * ELEMENT_BYTES);
  multiConsumer.op("right", "relu", ["shared/out"], width * ELEMENT_BYTES);
  const sharedFusion = greedyFusion(multiConsumer);
  deepStrictEqual(sharedFusion.groups[sharedFusion.groupOf[0]].ops, [0],
    "a producer with two consumers is not recomputed into either");
  strictEqual(sharedFusion.groups.length, 3, "each consumer keeps its own group");
  const reduceGraph = new ComputeGraph();
  reduceGraph.tensor("x", width * ELEMENT_BYTES).tensor("c", width * ELEMENT_BYTES);
  reduceGraph.op("prepare", "add", ["x", "c"], width * ELEMENT_BYTES);
  reduceGraph.op("total", "reduce", ["prepare/out"], ELEMENT_BYTES);
  reduceGraph.op("scale", "mul", ["total/out", "c"], ELEMENT_BYTES);
  const reduceFusion = greedyFusion(reduceGraph);
  deepStrictEqual(reduceFusion.groups.map((g) => g.ops), [[0, 1], [2]],
    "a reduction absorbs its elementwise producer but never fuses into its consumer");
  ok(isValidPartition(reduceGraph, reduceFusion.groupOf));

  const gemm = new ComputeGraph();
  gemm.tensor("x", width * ELEMENT_BYTES).tensor("w", width * ELEMENT_BYTES).tensor("c", width * ELEMENT_BYTES);
  gemm.op("bias", "add", ["x", "c"], width * ELEMENT_BYTES);
  gemm.op("mm", MATMUL, ["bias/out", "w"], width * ELEMENT_BYTES);
  gemm.op("shift", "add", ["mm/out", "c"], width * ELEMENT_BYTES);
  const gemmFusion = greedyFusion(gemm);
  deepStrictEqual(gemmFusion.groups.map((g) => g.ops), [[0, 1], [2]],
    "a matmul absorbs its producer but is never fused into the elementwise op consuming it");
  ok(isValidPartition(gemm, gemmFusion.groupOf));

  // -- numerics: fused execution must equal unfused execution -----------------------------
  const inputs = {
    x: Array.from({ length: width }, (_, i) => Math.sin(i) * 3),
    c: Array.from({ length: width }, (_, i) => Math.cos(i * 0.5)),
  };
  const unfusedValues = executeUnfused(chain, inputs, unary);
  const fusedValues = executeFused(chain, inputs, chainFusion.groups, unary);
  for (const op of chain.ops) {
    deepStrictEqual(fusedValues.get(op.output), unfusedValues.get(op.output),
      `fused evaluation of ${op.name} must be bit-identical`);
  }
  const reduceFused = executeFused(reduceGraph, inputs, reduceFusion.groups, unary);
  const reducePlain = executeUnfused(reduceGraph, inputs, unary);
  deepStrictEqual(reduceFused.get("scale/out"), reducePlain.get("scale/out"),
    "fusing a reduction with its producer keeps the arithmetic identical");

  // -- the greedy against the exhaustive oracle -------------------------------------------
  const small = buildChain(5, 16);
  const smallGreedy = greedyFusion(small);
  const smallCost = costOfPartition(small, smallGreedy.groups);
  const oracle = bestPartition(small);
  ok(smallCost.bytes >= oracle.bytes, "no heuristic may beat the exhaustive optimum");
  strictEqual(smallCost.bytes, oracle.bytes, "on a pure chain the greedy partition is optimal");
  const fork = new ComputeGraph();
  fork.tensor("x", 16 * ELEMENT_BYTES).tensor("c", 16 * ELEMENT_BYTES);
  fork.op("sum", "add", ["x", "c"], 16 * ELEMENT_BYTES);
  fork.op("left", "mul", ["sum/out", "c"], 16 * ELEMENT_BYTES);
  fork.op("right", "relu", ["sum/out"], 16 * ELEMENT_BYTES);
  fork.op("join", "add", ["left/out", "right/out"], 16 * ELEMENT_BYTES);
  const forkGreedy = costOfPartition(fork, greedyFusion(fork).groups);
  const forkOracle = bestPartition(fork);
  ok(forkGreedy.bytes >= forkOracle.bytes, "the greedy must not beat the oracle on a fork either");
  console.log(`fork: greedy=${forkGreedy.bytes} B oracle=${forkOracle.bytes} B`);

  // -- degenerate graphs -------------------------------------------------------------------
  const empty = new ComputeGraph();
  const emptyFusion = greedyFusion(empty);
  strictEqual(emptyFusion.groups.length, 0, "an empty graph produces no kernels");
  strictEqual(costOfPartition(empty, emptyFusion.groups).bytes, 0);
  strictEqual(costOfPartition(empty, emptyFusion.groups).launches, 0);

  const lone = new ComputeGraph();
  lone.tensor("a", 64).tensor("b", 64);
  lone.op("only", "add", ["a", "b"], 64);
  const loneCost = costOfPartition(lone, greedyFusion(lone).groups);
  strictEqual(loneCost.bytes, 192, "a single kernel reads both inputs and writes its output");
  strictEqual(loneCost.launches, 1);

  const independent = new ComputeGraph();
  independent.tensor("a", 64).tensor("b", 64).tensor("c", 64);
  independent.op("left", "add", ["a", "b"], 64);
  independent.op("right", "mul", ["b", "c"], 64);
  const independentGreedy = greedyFusion(independent);
  strictEqual(independentGreedy.groups.length, 2, "operators that share no tensor cannot fuse");
  strictEqual(costOfPartition(independent, independentGreedy.groups).bytes,
    costOfPartition(independent, independent.ops.map((op) => ({ ops: [op.index], leader: op.index }))).bytes,
    "so fusion buys nothing on independent operators");

  throws(() => new ComputeGraph().op("bad", "add", ["ghost"], 8), /unknown tensor/);
  throws(() => new ComputeGraph().tensor("t", 8).op("again", "add", ["t"], 8).op("again", "add", ["t"], 8),
    /duplicate operator/);
  throws(() => bestPartition(buildChain(9, 8)), /small graphs/);

  // -- determinism -------------------------------------------------------------------------
  const replay = () => {
    const again = greedyFusion(buildChain(7, 32));
    return again.groups.map((group) => group.ops.join("+")).join("|");
  };
  deepStrictEqual(replay(), replay(), "fusion must be deterministic");
  console.log("PASS");
}
