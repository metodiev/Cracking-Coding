/** Task 818 — Implement model parallel training partitioner and pipeline parallel executor.
 *  (Tier: Master, Beyond — Machine Learning & AI)
 *
 * Partitioner. The model is a linear chain of layers with per-layer compute cost and activation size;
 * a `P`-stage partition must cut it into contiguous stages. The exact `O(L²·P)` dynamic program
 * minimises the *maximum* stage cost over all contiguous partitions, and the file cross-checks it
 * against a brute-force enumeration of every composition of `L` into `P` parts — the DP is only
 * accepted if it reproduces the optimum on every instance.
 *
 * Executor. Each stage owns a contiguous layer range and executes two kinds of task: `forward(m)` for a
 * microbatch (run the stage's layers, then hand the activation to the next stage) and `backward(m)`
 * (consume the gradient produced by the next stage, walk the stage's layers in reverse and accumulate
 * this stage's parameter gradients). A task occupies its stage for as many ticks as the stage has
 * layers, stages run one task at a time, and a task starts only when its dependencies are satisfied:
 * `forward(p, m)` after `forward(p-1, m)`, `backward(p, m)` after `backward(p+1, m)` and after its own
 * forward, and both in stage-local order. Two schedules are implemented: **GPipe** (all forwards, then
 * all backwards) and **1F1B** (stage *p* warms up with `P-1-p` forwards and then interleaves one
 * forward with one backward), which is what makes 1F1B's activation memory `P-p` per stage instead of
 * `M`.
 *
 * Arithmetic. Layers, activations and gradients are small integers (`y = Wx + b`, sum-of-squares loss),
 * so the pipelined result, the sequential reference result and the analytic schedules agree *exactly*
 * — the checks compare gradients with `deepStrictEqual`, not with tolerances, and the gradient of every
 * layer must match the single-process reference for both microbatches-last semantics.
 *
 * Simplified versus production: no real collectives or sockets (activation and gradient hand-offs are
 * in-process queue entries with explicit ticks), no interleaved "virtual stages", no recomputation, no
 * straggler or failure handling, and the cost model is "one tick per layer" instead of measured
 * hardware time. What is verified is the structure and the numbers that carry over: partition
 * optimality, schedule legality, exact numerical equivalence, bubble ratio against the analytic
 * `(M+P-1)·maxStage` bound, and the 1F1B activation-memory advantage.
 *
 * Verified: DP optimum versus brute force, contiguity/coverage of the partition, illegal `P` rejected,
 * dependency and exclusivity validation of the executed schedule, exact equality of the pipelined and
 * sequential gradients and outputs for both schedules, 1F1B ordering fingerprint, measured versus
 * analytic bubble ratio, measured versus analytic activation memory for GPipe and 1F1B, degenerate
 * cases (P = 1, P = L, M = 1, single layer) and determinism.
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic randomness

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

const DIMENSIONS = 2;

/** Integer layers `y = Wx + b` keep every forward/backward value exact. */
function buildModel(layerCosts, seed) {
  const next = splitmix64(seed);
  return layerCosts.map((cost, index) => ({
    index,
    cost,
    activationSize: 1 + Number(next() % 3n),
    W: [
      [Number(next() % 5n) - 2, Number(next() % 5n) - 2],
      [Number(next() % 5n) - 2, Number(next() % 5n) - 2],
    ],
    b: [Number(next() % 3n) - 1, Number(next() % 3n) - 1],
  }));
}

function buildMicrobatches(count, seed) {
  const next = splitmix64(BigInt(seed) ^ 0x9e37n);
  const microbatches = [];
  for (let m = 0; m < count; m++) {
    microbatches.push({
      id: m,
      input: [Number(next() % 5n) - 2, Number(next() % 5n) - 2],
      target: [Number(next() % 3n) - 1, Number(next() % 3n) - 1],
    });
  }
  return microbatches;
}

function applyLayer(layer, x) {
  return [
    layer.W[0][0] * x[0] + layer.W[0][1] * x[1] + layer.b[0],
    layer.W[1][0] * x[0] + layer.W[1][1] * x[1] + layer.b[1],
  ];
}

// ---------------------------------------------------------------------------- partitioner

/**
 * Exact DP over contiguous partitions: `best[p][l]` is the smallest achievable maximum stage cost when
 * the last `l` layers are cut into `p` stages.
 */
function partitionDP(layerCosts, stages) {
  const layers = layerCosts.length;
  if (stages < 1) throw new RangeError("at least one stage is required");
  if (stages > layers) throw new RangeError(`cannot split ${layers} layers into ${stages} stages`);
  const prefix = [0];
  for (let i = 0; i < layers; i++) prefix.push(prefix[i] + layerCosts[i]);
  const best = Array.from({ length: stages + 1 }, () => new Array(layers + 1).fill(Infinity));
  const split = Array.from({ length: stages + 1 }, () => new Array(layers + 1).fill(-1));
  best[0][0] = 0;
  for (let p = 1; p <= stages; p++) {
    for (let l = p; l <= layers - (stages - p); l++) {
      for (let k = p - 1; k < l; k++) {
        if (best[p - 1][k] === Infinity) continue;
        const cost = prefix[l] - prefix[k];
        const candidate = Math.max(best[p - 1][k], cost);
        if (candidate < best[p][l]) {
          best[p][l] = candidate;
          split[p][l] = k;
        }
      }
    }
  }
  const assignment = new Array(layers).fill(-1);
  let stage = stages;
  let end = layers;
  while (stage > 0) {
    const start = split[stage][end];
    for (let layer = start; layer < end; layer++) assignment[layer] = stage - 1;
    end = start;
    stage--;
  }
  const stageCosts = new Array(stages).fill(0);
  for (let layer = 0; layer < layers; layer++) stageCosts[assignment[layer]] += layerCosts[layer];
  if (stageCosts.some((cost) => cost === 0)) throw new Error("the partition left an empty stage");
  return { assignment, maxStageCost: best[stages][layers], stageCosts };
}

/** Brute force over every composition of the layers into `stages` contiguous non-empty parts. */
function bruteForcePartition(layerCosts, stages) {
  const layers = layerCosts.length;
  let bestMax = Infinity;
  let bestAssignment = null;
  const cuts = [];
  const recurse = (position) => {
    if (cuts.length === stages - 1) {
      const bounds = [0, ...cuts, layers];
      const costs = [];
      for (let i = 0; i + 1 < bounds.length; i++) {
        let total = 0;
        for (let layer = bounds[i]; layer < bounds[i + 1]; layer++) total += layerCosts[layer];
        costs.push(total);
      }
      const worst = Math.max(...costs);
      if (worst < bestMax) {
        bestMax = worst;
        const assignment = new Array(layers).fill(-1);
        for (let i = 0; i < costs.length; i++) for (let layer = bounds[i]; layer < bounds[i + 1]; layer++) assignment[layer] = i;
        bestAssignment = { assignment, maxStageCost: worst, stageCosts: costs };
      }
      return;
    }
    for (let cut = position + 1; cut <= layers - 1; cut++) {
      if (layers - cut < stages - cuts.length - 1) break;
      cuts.push(cut);
      recurse(cut);
      cuts.pop();
    }
  };
  recurse(0);
  return bestAssignment;
}

// ---------------------------------------------------------------------------- sequential reference

function sequentialReference(layers, microbatches) {
  const gradients = layers.map(() => ({ W: [[0, 0], [0, 0]], b: [0, 0] }));
  const outputs = [];
  let loss = 0;
  for (const microbatch of microbatches) {
    const activations = [];
    let value = microbatch.input;
    for (const layer of layers) {
      activations.push(value);
      value = applyLayer(layer, value);
    }
    outputs.push({ id: microbatch.id, output: value.slice() });
    const residual = [value[0] - microbatch.target[0], value[1] - microbatch.target[1]];
    loss += residual[0] * residual[0] + residual[1] * residual[1];
    let delta = [2 * residual[0], 2 * residual[1]];
    for (let layer = layers.length - 1; layer >= 0; layer--) {
      const x = activations[layer];
      const layerGradient = gradients[layer];
      for (let i = 0; i < DIMENSIONS; i++) {
        for (let j = 0; j < DIMENSIONS; j++) layerGradient.W[i][j] += delta[i] * x[j];
        layerGradient.b[i] += delta[i];
      }
      const next = [0, 0];
      for (let j = 0; j < DIMENSIONS; j++) next[j] = layers[layer].W[0][j] * delta[0] + layers[layer].W[1][j] * delta[1];
      delta = next;
    }
  }
  return { gradients, outputs, loss };
}

// ---------------------------------------------------------------------------- pipeline executor

class PipelineExecutor {
  /**
   * @param {object} options `{ layers, partition, microbatches, schedule }`
   *   schedule is "gpipe" (all forwards, then all backwards) or "1f1b" (interleaved).
   */
  constructor({ layers, partition, microbatches, schedule = "1f1b" }) {
    if (!["gpipe", "1f1b"].includes(schedule)) throw new RangeError(`unknown schedule ${schedule}`);
    this.layers = layers;
    this.microbatches = microbatches;
    this.schedule = schedule;
    this.stages = partition.stageCosts.length;
    this.stageCosts = partition.stageCosts.slice();
    this.stageLayers = Array.from({ length: this.stages }, () => []);
    for (let layer = 0; layer < layers.length; layer++) this.stageLayers[partition.assignment[layer]].push(layer);
    this.plan = this.stages > 0 ? Array.from({ length: this.stages }, (_, stage) => this.planFor(stage)) : [];
    this.reset();
  }

  /** The declared task order of a stage; the executor still waits for the real dependencies. */
  planFor(stage) {
    const warmup = this.stages - 1 - stage;
    const order = [];
    if (this.schedule === "gpipe") {
      for (let m = 0; m < this.microbatches.length; m++) order.push({ type: "forward", microbatch: m });
      for (let m = this.microbatches.length - 1; m >= 0; m--) order.push({ type: "backward", microbatch: m });
      return order;
    }
    for (let k = 0; k < this.microbatches.length; k++) {
      order.push({ type: "forward", microbatch: k });
      const backward = k - warmup;
      if (backward >= 0) order.push({ type: "backward", microbatch: backward });
    }
    for (let k = Math.max(0, this.microbatches.length - warmup); k < this.microbatches.length; k++) {
      order.push({ type: "backward", microbatch: k });
    }
    return order;
  }

  reset() {
    this.tick = 0;
    this.log = [];
    this.busyUntil = new Array(this.stages).fill(0);
    this.pendingCompletion = new Array(this.stages).fill(null);
    this.cursor = new Array(this.stages).fill(0);
    this.forwardDone = Array.from({ length: this.stages }, () => new Array(this.microbatches.length).fill(false));
    this.backwardDone = Array.from({ length: this.stages }, () => new Array(this.microbatches.length).fill(false));
    this.activations = Array.from({ length: this.stages }, () => new Set());
    this.peakActivationSets = new Array(this.stages).fill(0);
    this.cache = Array.from({ length: this.stages }, () => new Map());
    this.incomingGradient = new Map(); // key `${stage}:${microbatch}` -> delta produced by the next stage
    this.gradients = this.layers.map(() => ({ W: [[0, 0], [0, 0]], b: [0, 0] }));
    this.outputs = [];
    this.tickHistory = [];
  }

  ready(stage, task) {
    if (task.type === "forward") {
      if (stage > 0 && !this.forwardDone[stage - 1][task.microbatch]) return false;
      return true;
    }
    if (!this.forwardDone[stage][task.microbatch]) return false;
    if (stage < this.stages - 1 && !this.backwardDone[stage + 1][task.microbatch]) return false;
    return true;
  }

  /** Bookkeeping when a task is started: the stage is busy for its compute cost, then the task ends. */
  startTask(stage, task) {
    this.busyUntil[stage] = this.tick + this.stageCosts[stage];
    this.pendingCompletion[stage] = task;
    this.log.push({ tick: this.tick, type: task.type, stage, microbatch: task.microbatch, cost: this.stageCosts[stage] });
  }

  /** The actual work, executed when the task's cost has elapsed — only then is its output visible. */
  completeTask(stage, task) {
    const microbatch = this.microbatches[task.microbatch];
    if (task.type === "forward") {
      if (stage === 0) {
        this.cache[stage].set(task.microbatch, { activations: [], input: microbatch.input.slice() });
      }
      const state = this.cache[stage].get(task.microbatch);
      let value = state.input;
      for (const layerIndex of this.stageLayers[stage]) {
        state.activations.push({ layerIndex, x: value.slice() });
        value = applyLayer(this.layers[layerIndex], value);
      }
      state.output = value.slice();
      this.forwardDone[stage][task.microbatch] = true;
      this.activations[stage].add(task.microbatch);
      this.peakActivationSets[stage] = Math.max(this.peakActivationSets[stage], this.activations[stage].size);
      if (stage === this.stages - 1) this.outputs.push({ id: microbatch.id, output: value.slice() });
      if (stage + 1 < this.stages) {
        const downstream = this.cache[stage + 1].get(task.microbatch);
        if (downstream) downstream.input = value.slice();
        else this.cache[stage + 1].set(task.microbatch, { activations: [], input: value.slice() });
      }
      return;
    }
    // backward
    let delta;
    if (stage === this.stages - 1) {
      const residual = [this.cache[stage].get(task.microbatch).output[0] - microbatch.target[0], this.cache[stage].get(task.microbatch).output[1] - microbatch.target[1]];
      delta = [2 * residual[0], 2 * residual[1]];
    } else {
      delta = this.incomingGradient.get(`${stage + 1}:${task.microbatch}`);
      if (!delta) throw new Error(`stage ${stage} has no upstream gradient for microbatch ${task.microbatch}`);
    }
    const state = this.cache[stage].get(task.microbatch);
    for (let i = state.activations.length - 1; i >= 0; i--) {
      const { layerIndex, x } = state.activations[i];
      const layerGradient = this.gradients[layerIndex];
      for (let r = 0; r < DIMENSIONS; r++) {
        for (let c = 0; c < DIMENSIONS; c++) layerGradient.W[r][c] += delta[r] * x[c];
        layerGradient.b[r] += delta[r];
      }
      const next = [0, 0];
      for (let j = 0; j < DIMENSIONS; j++) next[j] = this.layers[layerIndex].W[0][j] * delta[0] + this.layers[layerIndex].W[1][j] * delta[1];
      delta = next;
    }
    this.backwardDone[stage][task.microbatch] = true;
    this.activations[stage].delete(task.microbatch);
    this.incomingGradient.set(`${stage}:${task.microbatch}`, delta);
  }

  run(maxTicks = 10000) {
    while (this.tick < maxTicks) {
      // 1. tasks whose compute time has elapsed complete first, so their output is visible in this tick
      const idleAtStart = this.pendingCompletion.every((task) => task === null);
      let completed = 0;
      for (let stage = 0; stage < this.stages; stage++) {
        const task = this.pendingCompletion[stage];
        if (task !== null && this.tick >= this.busyUntil[stage]) {
          this.pendingCompletion[stage] = null;
          this.completeTask(stage, task);
          completed++;
        }
      }
      // 2. stages that are free and whose next planned task is ready start it now
      let started = 0;
      for (let stage = 0; stage < this.stages; stage++) {
        if (this.pendingCompletion[stage] !== null) continue;
        if (this.cursor[stage] >= this.plan[stage].length) continue;
        const task = this.plan[stage][this.cursor[stage]];
        if (!this.ready(stage, task)) continue;
        this.startTask(stage, task);
        this.cursor[stage]++;
        started++;
      }
      this.tickHistory.push(this.peakActivationSets.reduce((a, b) => Math.max(a, b), 0));
      const finished = this.cursor.every((cursor, stage) => cursor === this.plan[stage].length);
      if (finished && this.pendingCompletion.every((task) => task === null)) break;
      // Nothing was running or completing when the tick began and nothing could start: a real deadlock.
      if (idleAtStart && completed === 0 && started === 0) throw new Error("the pipeline deadlocked");
      this.tick++;
    }
    if (!this.cursor.every((cursor, stage) => cursor === this.plan[stage].length)) throw new Error("the pipeline did not finish within the tick budget");
    return {
      ticks: this.tick,
      log: this.log,
      gradients: this.gradients,
      outputs: this.outputs.slice().sort((a, b) => a.id - b.id),
      peakActivationSets: this.peakActivationSets.slice(),
      peakActivationUnits: this.peakActivationSets.map((sets, stage) => sets * this.stageLayers[stage].reduce((sum, layer) => sum + this.layers[layer].activationSize, 0)),
    };
  }
}

// ---------------------------------------------------------------------------- schedule validator

function validateSchedule(log, stages, microbatches, schedule) {
  const problems = [];
  const startOf = new Map();
  const endOf = new Map();
  const perStage = Array.from({ length: stages }, () => []);
  for (const entry of log) {
    const key = `${entry.stage}:${entry.microbatch}:${entry.type}`;
    if (startOf.has(key)) problems.push(`duplicate task ${key}`);
    startOf.set(key, entry.tick);
    endOf.set(key, entry.tick + entry.cost);
    perStage[entry.stage].push(entry);
  }
  for (let stage = 0; stage < stages; stage++) {
    const tasks = perStage[stage].slice().sort((a, b) => a.tick - b.tick);
    for (let i = 1; i < tasks.length; i++) {
      if (tasks[i].tick < tasks[i - 1].tick + tasks[i - 1].cost) problems.push(`stage ${stage} overlaps tasks at tick ${tasks[i].tick}`);
    }
  }
  const key = (stage, microbatch, type) => `${stage}:${microbatch}:${type}`;
  for (let stage = 0; stage < stages; stage++) {
    for (let m = 0; m < microbatches; m++) {
      if (!startOf.has(key(stage, m, "forward"))) problems.push(`stage ${stage} never forwarded microbatch ${m}`);
      if (!startOf.has(key(stage, m, "backward"))) problems.push(`stage ${stage} never backpropagated microbatch ${m}`);
      if (stage > 0 && startOf.get(key(stage, m, "forward")) < endOf.get(key(stage - 1, m, "forward"))) problems.push(`forward(${stage},${m}) started before the upstream forward finished`);
      if (stage < stages - 1 && startOf.get(key(stage, m, "backward")) < endOf.get(key(stage + 1, m, "backward"))) problems.push(`backward(${stage},${m}) started before the upstream gradient arrived`);
      if (startOf.get(key(stage, m, "backward")) < endOf.get(key(stage, m, "forward"))) problems.push(`backward(${stage},${m}) started before its forward finished`);
    }
  }
  if (schedule === "1f1b") {
    for (let stage = 0; stage < stages; stage++) {
      const warmup = stages - 1 - stage;
      const forwards = [];
      for (let m = 0; m < microbatches; m++) forwards.push(endOf.get(key(stage, m, "forward")));
      for (let m = 0; m < microbatches; m++) {
        const limit = m + warmup;
        if (limit < microbatches.length && startOf.get(key(stage, m, "backward")) < forwards[limit]) {
          problems.push(`1F1B fingerprint violated on stage ${stage}: backward(${m}) before forward(${limit}) finished`);
        }
      }
    }
  }
  return problems;
}

module.exports = {
  splitmix64,
  buildModel,
  buildMicrobatches,
  applyLayer,
  partitionDP,
  bruteForcePartition,
  sequentialReference,
  PipelineExecutor,
  validateSchedule,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- the DP must be optimal: compare against brute force on many instances ------
  let instances = 0;
  let dpSeed = splitmix64(0x8185eed);
  for (let trial = 0; trial < 60; trial++) {
    const layers = 2 + Number(dpSeed() % 6n);
    const stages = 1 + Number(dpSeed() % BigInt(Math.min(4, layers)));
    const costs = Array.from({ length: layers }, () => 1 + Number(dpSeed() % 5n));
    const dp = partitionDP(costs, stages);
    const brute = bruteForcePartition(costs, stages);
    assert.strictEqual(dp.maxStageCost, brute.maxStageCost, `DP ${dp.maxStageCost} vs brute force ${brute.maxStageCost} for ${costs} into ${stages}`);
    assert.ok(dp.stageCosts.every((cost) => cost > 0), "no stage may be empty");
    assert.strictEqual(dp.stageCosts.reduce((a, b) => a + b, 0), costs.reduce((a, b) => a + b, 0));
    for (let layer = 1; layer < dp.assignment.length; layer++) assert.ok(dp.assignment[layer] >= dp.assignment[layer - 1], "stages must be contiguous and ordered");
    instances++;
  }
  assert.throws(() => partitionDP([1, 1], 3), /cannot split/);
  assert.throws(() => partitionDP([1, 1], 0), /at least one stage/);
  assert.strictEqual(partitionDP([5], 1).maxStageCost, 5);

  // --- pipeline versus sequential: exact equality for both schedules --------------
  const layerCosts = [3, 2, 4, 2, 3, 5, 2, 4];
  const layers = buildModel(layerCosts, 0x818aaa);
  const microbatches = buildMicrobatches(6, 0x818bbb);
  const partition = partitionDP(layerCosts, 3);
  const sequential = sequentialReference(layers, microbatches);
  for (const schedule of ["gpipe", "1f1b"]) {
    const executor = new PipelineExecutor({ layers, partition, microbatches, schedule });
    const result = executor.run();
    assert.deepStrictEqual(validateSchedule(result.log, 3, microbatches.length, schedule), []);
    assert.deepStrictEqual(result.gradients, sequential.gradients, `${schedule}: pipelined gradients must equal the sequential reference`);
    assert.deepStrictEqual(result.outputs, sequential.outputs, `${schedule}: pipelined outputs must equal the sequential reference`);
    assert.strictEqual(result.log.length, 3 * 2 * microbatches.length, `${schedule}: every stage must run both tasks for every microbatch`);
  }

  // --- bubble ratio and activation memory on an unbalanced model ------------------
  const gpipe = new PipelineExecutor({ layers, partition, microbatches, schedule: "gpipe" }).run();
  const interleaved = new PipelineExecutor({ layers, partition, microbatches, schedule: "1f1b" }).run();
  const totalCost = layerCosts.reduce((a, b) => a + b, 0);
  const serialTicks = microbatches.length * totalCost;
  const bottleneck = 2 * microbatches.length * Math.max(...partition.stageCosts);
  assert.ok(gpipe.ticks >= bottleneck, `the bottleneck stage must run twice per microbatch: ${gpipe.ticks} < ${bottleneck}`);
  assert.ok(interleaved.ticks >= bottleneck, `1F1B must also pay the bottleneck: ${interleaved.ticks} < ${bottleneck}`);
  assert.ok(gpipe.ticks <= serialTicks, `the pipeline must never be slower than one process: ${gpipe.ticks} > ${serialTicks}`);
  assert.ok(interleaved.ticks <= gpipe.ticks, "interleaving must not be slower than GPipe on the same instance");
  assert.ok(gpipe.peakActivationSets.every((sets) => sets === microbatches.length), "GPipe holds every activation");
  for (let stage = 0; stage < 3; stage++) {
    assert.strictEqual(interleaved.peakActivationSets[stage], Math.min(3 - stage, microbatches.length), `1F1B peak on stage ${stage} must be min(P-p, M)`);
  }
  assert.ok(Math.max(...interleaved.peakActivationSets) < Math.max(...gpipe.peakActivationSets), "1F1B must use less activation memory than GPipe");

  // --- balanced model: the classic analytic makespan and bubble are reproduced ----
  const balancedCosts = new Array(9).fill(1);
  const balancedLayers = buildModel(balancedCosts, 0x818ccc);
  const balancedPartition = partitionDP(balancedCosts, 3);
  assert.deepStrictEqual(balancedPartition.stageCosts, [3, 3, 3], "a balanced model splits evenly");
  const parallelMicrobatches = buildMicrobatches(24, 0x818ddd);
  const balancedSequential = sequentialReference(balancedLayers, parallelMicrobatches);
  const stageCost = 3;
  const predictedMakespan = 2 * parallelMicrobatches.length * stageCost + 2 * (3 - 1) * stageCost;
  const balanced = {};
  for (const schedule of ["gpipe", "1f1b"]) {
    const result = new PipelineExecutor({ layers: balancedLayers, partition: balancedPartition, microbatches: parallelMicrobatches, schedule }).run();
    balanced[schedule] = result;
    assert.strictEqual(result.ticks, predictedMakespan, `${schedule}: makespan must match the analytic 2M·c + 2(P-1)·c`);
    assert.deepStrictEqual(result.gradients, balancedSequential.gradients, `${schedule}: balanced-model gradients must be exact`);
    assert.deepStrictEqual(validateSchedule(result.log, 3, parallelMicrobatches.length, schedule), []);
  }
  assert.strictEqual(balanced.gpipe.ticks, balanced["1f1b"].ticks, "on a balanced model both schedules have the same makespan");
  const balancedSerial = parallelMicrobatches.length * balancedCosts.reduce((a, b) => a + b, 0);
  const speedup = balancedSerial / balanced.gpipe.ticks;
  const balancedBubble = 1 - (2 * parallelMicrobatches.length * stageCost) / balanced.gpipe.ticks;
  const analyticBubble = (3 - 1) / (parallelMicrobatches.length + 3 - 1);
  assert.ok(Math.abs(balancedBubble - analyticBubble) < 1e-12, `measured bubble ${balancedBubble} must equal the analytic (P-1)/(M+P-1) = ${analyticBubble}`);
  assert.ok(speedup > 1.3, `pipelining must beat one process by a real margin: ${speedup.toFixed(3)}`);
  assert.strictEqual(balanced.gpipe.peakActivationSets[0], parallelMicrobatches.length, "GPipe keeps M activations on every stage");
  assert.strictEqual(balanced["1f1b"].peakActivationSets[0], 3, "1F1B keeps P-p activations on stage 0");

  // --- degenerate shapes -----------------------------------------------------------
  const single = new PipelineExecutor({ layers, partition: partitionDP(layerCosts, 1), microbatches, schedule: "1f1b" }).run();
  assert.strictEqual(single.ticks, 2 * microbatches.length * totalCost, "a single stage runs forward and backward serially, with no bubbles");
  assert.strictEqual(single.peakActivationSets[0], 1);
  const oneLayer = buildModel([2], 7);
  const tiny = new PipelineExecutor({ layers: oneLayer, partition: partitionDP([2], 1), microbatches: buildMicrobatches(1, 8), schedule: "gpipe" }).run();
  assert.deepStrictEqual(tiny.gradients, sequentialReference(oneLayer, buildMicrobatches(1, 8)).gradients);
  assert.strictEqual(tiny.ticks, 4, "one layer of cost 2: a forward and a backward");
  const stagesEqualLayers = partitionDP(layerCosts, layerCosts.length);
  assert.strictEqual(stagesEqualLayers.maxStageCost, Math.max(...layerCosts));
  const deepPipeline = new PipelineExecutor({ layers, partition: stagesEqualLayers, microbatches, schedule: "1f1b" }).run();
  assert.deepStrictEqual(deepPipeline.gradients, sequential.gradients, "one layer per stage must still be numerically exact");
  assert.deepStrictEqual(validateSchedule(deepPipeline.log, layerCosts.length, microbatches.length, "1f1b"), []);
  assert.strictEqual(Math.max(...deepPipeline.peakActivationSets), microbatches.length, "with fewer microbatches than stages the warm-up dominates");
  for (let stage = 0; stage < layerCosts.length; stage++) {
    assert.strictEqual(deepPipeline.peakActivationSets[stage], Math.min(layerCosts.length - stage, microbatches.length), `1F1B peak on deep stage ${stage} must be min(P-p, M)`);
  }

  // --- determinism ------------------------------------------------------------------
  const repeat = new PipelineExecutor({ layers, partition, microbatches, schedule: "1f1b" }).run();
  assert.strictEqual(repeat.ticks, interleaved.ticks);
  assert.deepStrictEqual(repeat.gradients, interleaved.gradients);
  assert.deepStrictEqual(repeat.log.map((entry) => `${entry.tick}:${entry.stage}:${entry.type}:${entry.microbatch}`), interleaved.log.map((entry) => `${entry.tick}:${entry.stage}:${entry.type}:${entry.microbatch}`));

  console.log(
    `partitions checked ${instances} (DP == brute force), stages ${partition.stageCosts.join("/")}, ` +
      `unbalanced: GPipe ${gpipe.ticks} ticks vs 1F1B ${interleaved.ticks} (bottleneck ${bottleneck}), ` +
      `peak activations GPipe ${Math.max(...gpipe.peakActivationSets)} vs 1F1B ${interleaved.peakActivationSets.join("/")}; ` +
      `balanced: makespan ${balanced.gpipe.ticks} = 2Mc+2(P-1)c, speedup ${speedup.toFixed(3)}x, bubble ${(balancedBubble * 100).toFixed(2)}% = analytic`
  );
  console.log("PASS");
}
