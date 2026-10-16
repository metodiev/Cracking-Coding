/** Task 724 - Implement memory planner for computation graph scheduling (simple greedy). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Every tensor of a computation graph is live from the op that defines it until the last op
 * that consumes it, so a compiler can lay all lifetimes over one flat buffer and let tensors
 * whose lifetimes never meet share the same bytes (the "storage rewrite"/memory-planner pass
 * of TVM and XLA). This file implements the classic greedy planner and the checks that make
 * its output trustworthy.
 *
 *   * lifetime analysis: every tensor gets the inclusive op-range [definition, last consumer];
 *     a graph input starts at op 0, and a tensor nobody consumes is kept to the last op
 *     because it is a graph output.
 *   * placement: tensors are placed in a chosen order with a lowest-offset first fit that
 *     jumps past any already-placed block whose lifetime *meets* it -- two tensors may share
 *     bytes only when their lifetimes are disjoint, which is the single invariant that keeps
 *     the plan correct.
 *   * the floor is the peak live volume (the largest sum of simultaneously live sizes), the
 *     naive plan (one fresh region per tensor) is the ceiling, and the interesting question
 *     is how close the greedy gets to the floor.
 *
 * Verified here: every pair of tensors whose lifetimes meet ends up in disjoint byte ranges;
 * the pool never drops below the peak live volume and never exceeds the naive plan; an
 * execution replay that writes a unique marker into each tensor's bytes and checks it at
 * every consumer proves the plan preserves data; a depth-first chain is packed to the floor
 * by the exhaustive placement oracle and reasonably by the greedy; invalid graphs (unknown
 * input, duplicate tensor, use before definition) are rejected; degenerate graphs (empty,
 * single tensor, no dead tensor) behave; and planning is fully deterministic.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

class ComputationGraph {
  constructor() {
    this.tensors = new Map();     // name -> { size, definedBy }
    this.ops = [];                // { name, inputs, outputs, index } in the given schedule order
  }

  /** A graph input: defined by nobody and therefore live from the first op. */
  input(name, size) {
    ok(!this.tensors.has(name), `duplicate tensor ${name}`);
    ok(Number.isInteger(size) && size >= 0, `bad size for ${name}`);
    this.tensors.set(name, { size, definedBy: -1 });
    return this;
  }

  /** An op consuming existing tensors and defining new ones; call order is the schedule. */
  op(name, inputs, outputs) {
    for (const input of inputs) ok(this.tensors.has(input), `op ${name} consumes unknown tensor ${input}`);
    for (const output of outputs) {
      ok(!this.tensors.has(output.name), `op ${name} redefines tensor ${output.name}`);
      ok(Number.isInteger(output.size) && output.size >= 0, `bad size for ${output.name}`);
    }
    const index = this.ops.length;
    this.ops.push({ name, inputs, outputs: outputs.map((o) => o.name), index });
    for (const output of outputs) this.tensors.set(output.name, { size: output.size, definedBy: index });
    return this;
  }

  get opCount() {
    return this.ops.length;
  }

  /** A schedule that consumes something defined later is not topological (it is a cycle). */
  validateSchedule() {
    const defined = new Set();
    for (const [name, tensor] of this.tensors) if (tensor.definedBy === -1) defined.add(name);
    for (const op of this.ops) {
      for (const input of op.inputs) {
        ok(defined.has(input), `op ${op.name} consumes ${input} before it is defined (not topological)`);
      }
      for (const output of op.outputs) defined.add(output);
    }
    return true;
  }
}

/** Inclusive op-range every tensor is live on. */
function lifetimes(graph) {
  const intervals = new Map();
  const last = Math.max(0, graph.ops.length - 1);
  for (const [name, tensor] of graph.tensors) {
    intervals.set(name, {
      size: tensor.size,
      from: tensor.definedBy === -1 ? 0 : tensor.definedBy,
      to: null,
    });
  }
  for (const op of graph.ops) {
    for (const input of op.inputs) {
      const interval = intervals.get(input);
      interval.to = Math.max(interval.to ?? -1, op.index);
    }
  }
  for (const [name, tensor] of graph.tensors) {
    const interval = intervals.get(name);
    // An unconsumed tensor is a graph output: it has to survive until the schedule ends.
    interval.to = interval.to === null ? last : interval.to;
    interval.to = Math.max(interval.to, interval.from);
  }
  return intervals;
}

/** Peak live volume: the lower bound every valid plan has to respect. */
function peakLiveVolume(graph, intervals) {
  let peak = 0;
  for (let index = 0; index < Math.max(1, graph.ops.length); index++) {
    let live = 0;
    for (const interval of intervals.values()) if (interval.from <= index && index <= interval.to) live += interval.size;
    peak = Math.max(peak, live);
  }
  return peak;
}

const meets = (a, b) => a.from <= b.to && b.from <= a.to;

/**
 * Lowest-offset first fit in the given placement order. A candidate offset is bumped past
 * every already-placed block whose lifetime meets this tensor, and the scan restarts; tensors
 * with disjoint lifetimes may therefore share (or even nest inside) the same bytes.
 */
function placeOrder(intervals, order) {
  const offsets = new Map();
  let pool = 0;
  for (const name of order) {
    const self = intervals.get(name);
    let offset = 0;
    scan: for (;;) {
      for (const [other, placed] of offsets) {
        if (!meets(self, intervals.get(other))) continue;
        if (offset < placed.end && placed.offset < offset + self.size) {
          offset = placed.end;
          continue scan;
        }
      }
      break;
    }
    offsets.set(name, { offset, end: offset + self.size, size: self.size });
    pool = Math.max(pool, offset + self.size);
  }
  return { offsets, pool, order: order.slice() };
}

const byDefinition = (intervals) => [...intervals.keys()].sort((a, b) =>
  (intervals.get(a).from - intervals.get(b).from)
  || (intervals.get(b).size - intervals.get(a).size)
  || (a < b ? -1 : a > b ? 1 : 0));

const bySizeDescending = (intervals) => [...intervals.keys()].sort((a, b) =>
  (intervals.get(b).size - intervals.get(a).size)
  || (intervals.get(a).from - intervals.get(b).from)
  || (a < b ? -1 : a > b ? 1 : 0));

/** Naive baseline: no reuse at all, one fresh region per tensor. */
function planNaive(graph, intervals) {
  let offset = 0;
  const offsets = new Map();
  for (const name of [...intervals.keys()].sort()) {
    const interval = intervals.get(name);
    offsets.set(name, { offset, end: offset + interval.size, size: interval.size });
    offset += interval.size;
  }
  return { offsets, pool: offset, order: [...intervals.keys()].sort() };
}

/** Exhaustive oracle: the best pool any placement order of this family can reach. */
function planOptimalBySearch(graph, intervals, limit = 8) {
  const names = [...intervals.keys()].sort();
  if (names.length > limit) throw new Error("exhaustive placement search is only offered for small graphs");
  let best = null;
  const used = new Array(names.length).fill(false);
  const current = [];
  const recurse = () => {
    if (current.length === names.length) {
      const candidate = placeOrder(intervals, current);
      if (best === null || candidate.pool < best.pool) best = { ...candidate, order: current.slice() };
      return;
    }
    for (let i = 0; i < names.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      current.push(names[i]);
      recurse();
      current.pop();
      used[i] = false;
    }
  };
  recurse();
  return best;
}

/** The invariant: tensors whose lifetimes meet must occupy disjoint byte ranges. */
function validatePlan(intervals, plan) {
  const names = [...intervals.keys()];
  for (let i = 0; i < names.length; i++) {
    const mine = plan.offsets.get(names[i]);
    strictEqual(mine.size, intervals.get(names[i]).size, `${names[i]} changed size`);
    ok(mine.end <= plan.pool, `${names[i]} sticks out of the ${plan.pool}-byte pool`);
    for (let j = i + 1; j < names.length; j++) {
      if (!meets(intervals.get(names[i]), intervals.get(names[j]))) continue;
      const other = plan.offsets.get(names[j]);
      ok(mine.end <= other.offset || other.end <= mine.offset,
        `${names[i]} and ${names[j]} share bytes although their lifetimes meet`);
    }
  }
  return true;
}

/**
 * Replay the schedule over a buffer of the planned size: each output writes a unique marker
 * into its bytes, every consumer reads them back. Accidental aliasing shows up here at once.
 */
function replayExecution(graph, plan) {
  const markers = new Map();
  let next = 1;
  for (const name of graph.tensors.keys()) markers.set(name, next++);
  const memory = new Float64Array(plan.pool).fill(-1);
  // Graph inputs are produced before the schedule starts, so they are written up front.
  for (const [name, tensor] of graph.tensors) {
    if (tensor.definedBy !== -1) continue;
    const placed = plan.offsets.get(name);
    for (let i = placed.offset; i < placed.end; i++) memory[i] = markers.get(name);
  }
  for (const op of graph.ops) {
    for (const input of op.inputs) {
      const placed = plan.offsets.get(input);
      for (let i = placed.offset; i < placed.end; i++) {
        ok(memory[i] === markers.get(input),
          `op ${op.name} read clobbered bytes of ${input} at ${i}: ${memory[i]} != ${markers.get(input)}`);
      }
    }
    for (const output of op.outputs) {
      const placed = plan.offsets.get(output);
      for (let i = placed.offset; i < placed.end; i++) memory[i] = markers.get(output);
    }
  }
  return memory;
}

/** Deterministic transformer-ish chain: attention + MLP widths with a residual stream. */
function buildDeepGraph(blocks, width) {
  const graph = new ComputationGraph();
  let stream = "x";
  graph.input(stream, width);
  for (let block = 0; block < blocks; block++) {
    graph.op(`qkv${block}`, [stream], [{ name: `qkv_${block}`, size: width * 3 }]);
    graph.op(`scores${block}`, [`qkv_${block}`], [{ name: `scores_${block}`, size: width * 2 }]);
    graph.op(`attn${block}`, [`scores_${block}`], [{ name: `attn_${block}`, size: width }]);
    graph.op(`hidden${block}`, [`attn_${block}`], [{ name: `hidden_${block}`, size: width * 4 }]);
    graph.op(`mlp${block}`, [`hidden_${block}`], [{ name: `mlp_${block}`, size: width }]);
    const next = `stream${block}`;
    graph.op(`add${block}`, [stream, `mlp_${block}`], [{ name: next, size: width }]);
    stream = next;
  }
  return graph;
}

module.exports = {
  ComputationGraph, lifetimes, peakLiveVolume, placeOrder, byDefinition, bySizeDescending,
  planNaive, planOptimalBySearch, validatePlan, replayExecution, meets, buildDeepGraph,
};

if (require.main === module) {
  // -- lifetime analysis on a hand-checked graph -----------------------------------------
  const graph = new ComputationGraph();
  graph.input("a", 10);
  graph.input("b", 20);
  graph.op("sum", ["a", "b"], [{ name: "s", size: 10 }]);
  graph.op("scale", ["s"], [{ name: "t", size: 10 }]);
  graph.op("final", ["t", "b"], [{ name: "out", size: 5 }]);
  const intervals = lifetimes(graph);
  deepStrictEqual(intervals.get("a"), { size: 10, from: 0, to: 0 }, "a dies after the first op");
  deepStrictEqual(intervals.get("b"), { size: 20, from: 0, to: 2 }, "b is consumed once more at the end");
  deepStrictEqual(intervals.get("s"), { size: 10, from: 0, to: 1 }, "s is born in op 0 and dies in op 1");
  deepStrictEqual(intervals.get("t"), { size: 10, from: 1, to: 2 }, "t is born in op 1 and dies in op 2");
  deepStrictEqual(intervals.get("out"), { size: 5, from: 2, to: 2 }, "an unconsumed output survives");
  strictEqual(peakLiveVolume(graph, intervals), 40, "peak live volume is the floor for the pool");
  ok(graph.validateSchedule());

  const planned = placeOrder(intervals, byDefinition(intervals));
  validatePlan(intervals, planned);
  strictEqual(planned.pool, 40, "the greedy plan reaches the floor on this graph");
  replayExecution(graph, planned);
  const placedA = planned.offsets.get("a");
  const placedT = planned.offsets.get("t");
  ok(placedA.offset < placedT.end && placedT.offset < placedA.end,
    "a dies before t is born, so t reuses a's bytes");

  // -- a chain: how close does the greedy get to the floor? ------------------------------
  const sizes = [256, 128, 64, 128, 256, 512];
  const chain = new ComputationGraph();
  chain.input("x", 64);
  for (let i = 0; i < sizes.length; i++) {
    chain.op(`layer${i}`, [i === 0 ? "x" : `h${i - 1}`], [{ name: `h${i}`, size: sizes[i] }]);
  }
  const chainIntervals = lifetimes(chain);
  const chainFloor = peakLiveVolume(chain, chainIntervals);
  const chainGreedy = placeOrder(chainIntervals, byDefinition(chainIntervals));
  const chainNaive = planNaive(chain, chainIntervals);
  const chainOracle = planOptimalBySearch(chain, chainIntervals);
  validatePlan(chainIntervals, chainGreedy);
  validatePlan(chainIntervals, chainOracle);
  replayExecution(chain, chainGreedy);
  replayExecution(chain, chainOracle);
  strictEqual(chainFloor, 768, "the floor is the widest neighbouring pair (256 + 512)");
  strictEqual(chainOracle.pool, chainFloor,
    "some placement order packs the chain exactly to the floor, so the floor is achievable");
  ok(chainGreedy.pool < chainNaive.pool, "even the definition-order greedy beats no reuse");
  ok(chainGreedy.pool <= chainNaive.pool * 0.75,
    `the greedy must not waste much: ${chainGreedy.pool} vs naive ${chainNaive.pool}`);
  console.log(`chain: floor=${chainFloor} greedy=${chainGreedy.pool} oracle=${chainOracle.pool} naive=${chainNaive.pool}`);

  // -- a residual graph: two greedy orders, the oracle and the floor ---------------------
  const residual = new ComputationGraph();
  residual.input("x", 8);
  residual.op("up1", ["x"], [{ name: "u1", size: 32 }]);
  residual.op("down1", ["u1"], [{ name: "d1", size: 8 }]);
  residual.op("add1", ["x", "d1"], [{ name: "r1", size: 8 }]);
  residual.op("up2", ["r1"], [{ name: "u2", size: 32 }]);
  residual.op("down2", ["u2"], [{ name: "d2", size: 8 }]);
  residual.op("add2", ["r1", "d2"], [{ name: "out", size: 8 }]);
  const residualIntervals = lifetimes(residual);
  const residualFloor = peakLiveVolume(residual, residualIntervals);
  const definitions = placeOrder(residualIntervals, byDefinition(residualIntervals));
  const sizeOrder = placeOrder(residualIntervals, bySizeDescending(residualIntervals));
  const oracle = planOptimalBySearch(residual, residualIntervals);
  const naiveResidual = planNaive(residual, residualIntervals);
  for (const plan of [definitions, sizeOrder, oracle]) {
    validatePlan(residualIntervals, plan);
    replayExecution(residual, plan);
    ok(plan.pool >= residualFloor, `pool ${plan.pool} must not beat the floor ${residualFloor}`);
    ok(plan.pool <= naiveResidual.pool, "no plan may waste more than the naive layout");
  }
  ok(oracle.pool <= definitions.pool && oracle.pool <= sizeOrder.pool,
    "the exhaustive oracle cannot be worse than a fixed order");
  strictEqual(definitions.pool, residualFloor, "definition order packs this graph exactly to the floor");
  ok(sizeOrder.pool >= definitions.pool, "placement order matters: size-descending is worse here");
  ok(Math.min(definitions.pool, sizeOrder.pool) <= naiveResidual.pool * 0.75,
    `reuse must cut the pool below the naive ${naiveResidual.pool}`);
  console.log(`residual: floor=${residualFloor} definition=${definitions.pool}` +
    ` size-desc=${sizeOrder.pool} oracle=${oracle.pool} naive=${naiveResidual.pool}`);

  // -- a deep graph exercises both analysis and placement at scale ------------------------
  const deep = buildDeepGraph(12, 8);
  deep.validateSchedule();
  const deepIntervals = lifetimes(deep);
  const deepPlan = placeOrder(deepIntervals, bySizeDescending(deepIntervals));
  validatePlan(deepIntervals, deepPlan);
  replayExecution(deep, deepPlan);
  const deepNaive = planNaive(deep, deepIntervals);
  ok(deepPlan.pool > 0 && deepPlan.pool < deepNaive.pool / 2,
    `the deep plan ${deepPlan.pool} must be far below the naive ${deepNaive.pool}`);
  ok(deepPlan.pool >= peakLiveVolume(deep, deepIntervals), "and still above the floor");
  console.log(`deep graph: tensors=${deepIntervals.size} ops=${deep.opCount} pool=${deepPlan.pool} naive=${deepNaive.pool}`);

  // -- degenerate graphs -------------------------------------------------------------------
  const empty = new ComputationGraph();
  const emptyPlan = placeOrder(lifetimes(empty), []);
  strictEqual(emptyPlan.pool, 0, "an empty graph needs no memory");
  strictEqual(emptyPlan.offsets.size, 0);
  strictEqual(replayExecution(empty, emptyPlan).length, 0);

  const single = new ComputationGraph();
  single.input("only", 4);
  const singlePlan = placeOrder(lifetimes(single), ["only"]);
  strictEqual(singlePlan.pool, 4, "one tensor, one region");
  replayExecution(single, singlePlan);

  const crowded = new ComputationGraph();
  crowded.input("p", 4);
  crowded.input("q", 4);
  crowded.op("both", ["p", "q"], [{ name: "pq", size: 4 }]);
  const crowdedIntervals = lifetimes(crowded);
  const crowdedPlan = placeOrder(crowdedIntervals, byDefinition(crowdedIntervals));
  strictEqual(crowdedPlan.pool, 12, "three tensors alive at once cannot share anything");
  strictEqual(crowdedPlan.pool, planNaive(crowded, crowdedIntervals).pool,
    "with no dead tensor the naive plan is already optimal");
  strictEqual(crowdedPlan.pool, peakLiveVolume(crowded, crowdedIntervals), "and it is the floor");
  replayExecution(crowded, crowdedPlan);

  // -- invalid graphs and schedules --------------------------------------------------------
  throws(() => new ComputationGraph().op("bad", ["ghost"], [{ name: "g", size: 1 }]), /unknown tensor/);
  throws(() => new ComputationGraph().input("dup", 1).input("dup", 1), /duplicate tensor/);
  throws(() => new ComputationGraph().input("t", 1).op("again", ["t"], [{ name: "t", size: 1 }]), /redefines/);
  throws(() => new ComputationGraph().input("neg", -1), /bad size/);
  // A caller that assembles ops by hand (bypassing the builder) can still produce a
  // non-topological schedule; validateSchedule is the guard for that entry point.
  const scrambled = new ComputationGraph();
  scrambled.input("seed", 1);
  scrambled.tensors.set("later", { size: 1, definedBy: 1 });
  scrambled.tensors.set("made", { size: 1, definedBy: 0 });
  scrambled.ops.push({ name: "first", inputs: ["later"], outputs: ["made"], index: 0 });
  scrambled.ops.push({ name: "second", inputs: ["seed"], outputs: ["later"], index: 1 });
  throws(() => scrambled.validateSchedule(), /not topological/);
  throws(() => planOptimalBySearch(residual, residualIntervals, 3), /small graphs/);

  // -- determinism and layout sanity -------------------------------------------------------
  const replay = () => {
    const again = lifetimes(residual);
    const plan = placeOrder(again, bySizeDescending(again));
    return [...plan.offsets].map(([name, placed]) => `${name}@${placed.offset}+${placed.size}`).join(",") + `|${plan.pool}`;
  };
  deepStrictEqual(replay(), replay(), "planning must be deterministic");
  for (const [name, placed] of definitions.offsets) {
    ok(Number.isInteger(placed.offset) && placed.offset >= 0, `${name} must start at a non-negative offset`);
    ok(placed.end > placed.offset || placed.size === 0, `${name} must have a non-negative extent`);
  }
  deepStrictEqual(oracle.order.length, residualIntervals.size, "the oracle places every tensor exactly once");
  console.log("PASS");
}
