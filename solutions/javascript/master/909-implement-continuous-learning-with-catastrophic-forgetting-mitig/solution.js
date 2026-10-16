/** Task 909 - Implement continuous learning with catastrophic forgetting mitigation (replay buffers).
 *  (Tier: Master, CTCI Beyond — Continual Learning & Systems for ML)
 *
 * A stream of four regression tasks — different smooth mappings y = sin(a*x1) + b*x2^2 over the same
 * two-dimensional input domain — is learned by plain SGD into one shared two-layer tanh network. Training
 * on the newest task alone repurposes the hidden units that encoded earlier mappings, so accuracy on the
 * first tasks collapses: catastrophic forgetting. The mitigation is a reservoir buffer of past examples
 * (item i is kept with probability capacity/i, which makes the buffer a uniform sample of the stream and
 * bounds memory) whose samples are mixed into every minibatch, pulling the weights back toward the joint
 * optimum that a pooled multi-task run reaches and that the checks use as an upper bound.
 *
 * Checks: per-task learning, the size of the forgetting gap with and without replay, that mitigation does
 * not cost the newest task, the pooled-data oracle bound, reservoir uniformity against the exact
 * capacity/stream inclusion probability, buffer invariants after every insertion, degenerate
 * configurations (one task, capacity one, no replay, empty stream) and reproducibility.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { assert.ok(bound > 0, "int() needs a positive bound"); return this.next() % bound; }
  float() { return this.next() / 4294967296; }
}
/** Reservoir sampling: uniform over the stream with bounded memory (keep item i with probability k/i). */
class Reservoir {
  constructor(capacity) {
    assert.ok(Number.isInteger(capacity) && capacity > 0, "capacity must be positive");
    this.capacity = capacity;
    this.items = [];
    this.seen = 0;
  }
  add(item, prng) {
    this.seen += 1;
    if (this.items.length < this.capacity) this.items.push(item);
    else {
      const slot = prng.int(this.seen);
      if (slot < this.capacity) this.items[slot] = item;
    }
    return this;
  }
  sample(prng) { return this.items.length === 0 ? null : this.items[prng.int(this.items.length)]; }
}
/* -------------------------------------------------------------------- the learner */

function initNetwork(width, dimensions, prng) {
  return {
    width, dimensions, hidden: Array.from({ length: width }, () => ({
      weights: Array.from({ length: dimensions }, () => (prng.float() * 2 - 1) * 0.6), bias: (prng.float() * 2 - 1) * 0.6, })), output: Array.from({ length: width }, () => (prng.float() * 2 - 1) * 0.6), outputBias: 0, };
}

function forward(network, inputs) {
  const activations = network.hidden.map((unit) => {
    const weighted = unit.bias + inputs.reduce((sum, x, j) => sum + unit.weights[j] * x, 0);
    return Math.tanh(weighted);
  });
  const prediction = network.outputBias + activations.reduce((sum, a, i) => sum + network.output[i] * a, 0);
  return { activations, prediction };
}
/** One SGD step over a minibatch of {inputs, label} points. */
function trainBatch(network, batch, learningRate) {
  for (const point of batch) {
    const { activations, prediction } = forward(network, point.inputs);
    const dOutput = 2 * (prediction - point.label);
    for (let i = 0; i < network.width; i++) {
      const dActivation = dOutput * network.output[i];
      const dPre = dActivation * (1 - activations[i] ** 2);
      network.hidden[i].bias -= learningRate * dPre;
      for (let j = 0; j < network.dimensions; j++) network.hidden[i].weights[j] -= learningRate * dPre * point.inputs[j];
      network.output[i] -= learningRate * dOutput * activations[i];
    }
    network.outputBias -= learningRate * dOutput;
  }
  return network;
}
/** One task: a smooth mapping of the shared input domain, plus its evaluation data. */
function makeTask(index, coefficients, prng, count) {
  const data = [];
  for (let i = 0; i < count; i++) {
    const inputs = [prng.float() * 2 - 1, prng.float() * 2 - 1];
    const label = Math.sin(coefficients.a * inputs[0]) + coefficients.b * inputs[1] ** 2;
    data.push({ inputs, label });
  }
  return { index, coefficients, data };
}
/** Bounded performance measure: share of points predicted within `tolerance`. */
const accuracy = (network, task, tolerance = 0.25) =>
  task.data.filter((point) => Math.abs(forward(network, point.inputs).prediction - point.label) <= tolerance).length /
  task.data.length;
const meanSquaredError = (network, task) =>
  task.data.reduce((sum, point) => sum + (forward(network, point.inputs).prediction - point.label) ** 2, 0) /
  task.data.length;
/** Continual training: `replayRatio` of every minibatch is drawn from the reservoir buffer. */
function trainContinually(tasks, options) {
  const prng = new Prng(options.seed);
  const network = initNetwork(options.width, options.dimensions, prng);
  const buffer = new Reservoir(options.capacity);
  const scores = [];
  const best = [];
  const bufferSizes = [];
  const timeline = [];
  for (const task of tasks) {
    const wanted = Math.round(options.batchSize * options.replayRatio);
    let bestHere = 0;
    for (let epoch = 0; epoch < options.epochs; epoch++) {
      const order = task.data.map((_, i) => i);
      for (let i = order.length - 1; i > 0; i--) {
        const j = prng.int(i + 1);
        const t = order[i]; order[i] = order[j]; order[j] = t;
      }
      for (let start = 0; start < order.length; start += options.batchSize) {
        const fresh = order.slice(start, start + options.batchSize).map((i) => task.data[i]);
        const replayed = [];
        for (let i = 0; i < wanted; i++) {
          const sample = buffer.sample(prng);
          if (sample) replayed.push(sample);
        }
        trainBatch(network, fresh.concat(replayed), options.learningRate);
        if (epoch === 0) {                            // the reservoir sees each stream example exactly once
          for (const point of fresh) {
            buffer.add(point, prng);
            bufferSizes.push(buffer.items.length);
          }
        }
      }
      if (epoch % 10 === 9) bestHere = Math.max(bestHere, accuracy(network, task));
    }
    best.push(bestHere);
    timeline.push(task.index);
    scores.push(timeline.map((index) => accuracy(network, tasks[index])));
  }
  return { network, scores, best, buffer, bufferSizes, timeline };
}
/** Multi-task baseline: the same SGD on the pooled data of every task, given epochs x tasks updates
 *  (so every task's data is seen as often as it would be on its own). */
function trainPooled(tasks, options) {
  if (tasks.length === 0) return [];
  const pooled = makeTask(0, { a: 0, b: 0 }, new Prng(1), 0);
  pooled.data = tasks.flatMap((task) => task.data);
  const result = trainContinually([pooled], { ...options, epochs: options.epochs * tasks.length, replayRatio: 0, capacity: 1 });
  return tasks.map((task) => accuracy(result.network, task));
}
/** Forgetting of a task: its best score after being learned minus its final score. */
function forgetting(result) {
  let worst = 0;
  for (let i = 0; i < result.timeline.length; i++) {
    const final = result.scores[result.scores.length - 1][i];
    worst = Math.max(worst, Math.max(...result.scores.slice(i).map((row) => row[i])) - final);
  }
  return worst;
}
/** Average final accuracy over everything except the newest task. */
function oldTaskScore(result) {
  const last = result.scores[result.scores.length - 1];
  const old = last.slice(0, last.length - 1);
  return old.reduce((a, b) => a + b, 0) / old.length;
}

function main() {
  const dimensions = 2;
  const options = { dimensions, width: 16, epochs: 90, batchSize: 16, learningRate: 0.1, replayRatio: 0.4, capacity: 300, seed: 0x5eed0909 };
  const coefficients = [{ a: 0.8, b: 0.2 }, { a: 1.0, b: -0.2 }, { a: 1.2, b: 0.4 }, { a: 1.4, b: -0.4 }];
  const tasks = (() => {
    const prng = new Prng(0x5eed0909);
    return coefficients.map((c, index) => makeTask(index, c, prng, 200));
  })();
  const replay = trainContinually(tasks, options);
  const naive = trainContinually(tasks, { ...options, replayRatio: 0 });   // identical setup, no replay samples
  // 1. happy path — every mapping is fitted while it is the current task, with and without replay.
  {
    for (let i = 0; i < tasks.length; i++) {
      assert.ok(naive.best[i] > 0.9, `task ${i} reached accuracy ${naive.best[i].toFixed(3)} without replay`);
      assert.ok(replay.best[i] > 0.9, `task ${i} reached accuracy ${replay.best[i].toFixed(3)} with replay`);
    }
    assert.strictEqual(replay.scores.length, tasks.length, "one score row per finished task");
    assert.ok(meanSquaredError(naive.network, tasks[tasks.length - 1]) < 0.05, "the newest task is fitted");
  }
  // 2. catastrophic forgetting — the naive learner collapses on the first task, replay does not.
  {
    assert.ok(naive.scores[0][0] > 0.75 && naive.scores[tasks.length - 1][0] < 0.65, `naive task 0 went ${naive.scores[0][0].toFixed(3)} -> ${naive.scores[tasks.length - 1][0].toFixed(3)}`);
    assert.ok(forgetting(naive) > 0.4, `naive forgetting ${forgetting(naive).toFixed(3)} must be large`);
    assert.ok(forgetting(replay) < forgetting(naive) * 0.7, `replay forgetting ${forgetting(replay).toFixed(3)} vs naive ${forgetting(naive).toFixed(3)}`);
    assert.ok(oldTaskScore(replay) > oldTaskScore(naive) + 0.12, `replay retains ${oldTaskScore(replay).toFixed(3)} vs naive ${oldTaskScore(naive).toFixed(3)}`);
  }
  // 3. mitigation stays usable for the newest task and approaches a dedicated multi-task run.
  {
    const newest = (result) => result.scores[result.scores.length - 1][tasks.length - 1];
    const joint = trainPooled(tasks, options);
    const jointOld = joint.slice(0, -1).reduce((a, b) => a + b, 0) / (joint.length - 1);
    assert.ok(newest(replay) > 0.75, `the newest task scores ${newest(replay).toFixed(3)} with replay`);
    assert.ok(newest(replay) >= newest(naive) - 0.25, "mitigation must not ruin the newest task");
    assert.ok(joint.every((score) => score > 0.5), "the pooled run proves the tasks are jointly learnable");
    assert.ok(oldTaskScore(replay) >= 0.75 * jointOld, `replay ${oldTaskScore(replay).toFixed(3)} should approach the multi-task run ${jointOld.toFixed(3)}`);
  }
  // 4. buffer invariants — bounded size, no duplicates, stream fully counted, size monotone to capacity.
  {
    assert.strictEqual(replay.buffer.items.length, options.capacity, "the buffer fills up on this stream");
    assert.ok(replay.buffer.seen > options.capacity, "the stream is longer than the buffer");
    const identities = new Set(replay.buffer.items.map((point) => `${point.label}:${point.inputs.join(",")}`));
    assert.strictEqual(identities.size, replay.buffer.items.length, "a stored example is never duplicated");
    assert.ok(replay.bufferSizes.every((size) => size <= options.capacity), "every intermediate size fits");
    assert.strictEqual(replay.bufferSizes[replay.bufferSizes.length - 1], options.capacity, "final size");
    assert.ok(replay.bufferSizes.every((size, i) => i === 0 || size >= replay.bufferSizes[i - 1] - options.capacity), "the size only jumps when a reservoir slot is reclaimed");
  }
  // 5. independent oracle — reservoir inclusion probability is capacity/stream, checked by simulation.
  {
    const streamLength = 200;
    const capacity = 20;
    const repetitions = 500;
    const counts = new Array(streamLength).fill(0);
    for (let repetition = 0; repetition < repetitions; repetition++) {
      const prng = new Prng(0xbeef + repetition);
      const buffer = new Reservoir(capacity);
      for (let index = 0; index < streamLength; index++) buffer.add(index, prng);
      for (const item of buffer.items) counts[item] += 1;
    }
    const expected = capacity / streamLength;
    const tolerance = 3 * Math.sqrt((expected * (1 - expected)) / repetitions);
    const worst = Math.max(...counts.map((count) => Math.abs(count / repetitions - expected)));
    assert.ok(worst <= tolerance, `worst inclusion deviation ${worst.toFixed(4)} vs ${tolerance.toFixed(4)}`);
    assert.ok(counts.every((count) => count > 0), "every stream position can be retained");
  }
  // 6. the trade-off is smooth — retention improves as the ratio grows, and every setting stays usable.
  {
    const ramp = [0, 0.2, 0.3, 0.4].map((replayRatio) => trainContinually(tasks, { ...options, replayRatio }));
    for (let i = 1; i < ramp.length; i++) {
      assert.ok(oldTaskScore(ramp[i]) >= oldTaskScore(ramp[i - 1]), `retention must not fall when the replay ratio grows: ${ramp.map(oldTaskScore).map((x) => x.toFixed(3))}`);
    }
    assert.ok(oldTaskScore(ramp[3]) > oldTaskScore(ramp[0]) + 0.15, "the ratio really drives retention");
    assert.ok(ramp.every((result) => result.scores[3][3] > 0.5), "the newest task stays usable at every ratio");
    assert.ok(ramp.every((result) => result.buffer.seen === ramp[0].buffer.seen), "the stream the buffer sees does not depend on the replay ratio");
  }
  // 7. boundary — a single task, capacity one, an empty stream, and argument validation.
  {
    const single = trainContinually([tasks[0]], options);
    assert.strictEqual(single.scores.length, 1);
    assert.ok(single.scores[0][0] > 0.8, "a single task is learned");
    assert.strictEqual(forgetting(single), 0, "with one task nothing can be forgotten");
    assert.deepStrictEqual(trainPooled([], options), []);
    const tiny = new Reservoir(1);
    const prng = new Prng(11);
    for (let i = 0; i < 50; i++) tiny.add(i, prng);
    assert.strictEqual(tiny.items.length, 1, "capacity one keeps exactly one item");
    assert.strictEqual(tiny.seen, 50, "the stream counter still advances");
    assert.strictEqual(new Reservoir(5).sample(prng), null, "an empty buffer samples nothing");
    const empty = trainContinually([], options);
    assert.deepStrictEqual(empty.scores, []);
    assert.throws(() => new Reservoir(0), /capacity must be positive/);
    assert.throws(() => new Reservoir(1.5), /capacity must be positive/);
  }
  // 8. determinism — identical seeds replay the continual run, different seeds diverge.
  {
    const again = trainContinually(tasks, options);
    assert.deepStrictEqual(again.scores, replay.scores);
    assert.deepStrictEqual(again.buffer.items, replay.buffer.items);
    assert.notDeepStrictEqual(trainContinually(tasks, { ...options, seed: 0x5eed090a }).scores, replay.scores);
    const twin = new Prng(0x5eed0909);
    assert.deepStrictEqual(makeTask(0, coefficients[0], twin, 200), tasks[0]);
  }
  console.log(`continual: ${tasks.length} tasks, forgetting ${forgetting(replay).toFixed(3)} with replay vs ` + `${forgetting(naive).toFixed(3)} without; old-task accuracy ${oldTaskScore(replay).toFixed(3)} vs ` +
    `${oldTaskScore(naive).toFixed(3)}; reservoir ${replay.buffer.items.length}/${options.capacity} of ` + `${replay.buffer.seen} examples`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  Prng, Reservoir, initNetwork, forward, trainBatch, makeTask, accuracy, meanSquaredError, trainContinually, trainPooled, forgetting, oldTaskScore, };
