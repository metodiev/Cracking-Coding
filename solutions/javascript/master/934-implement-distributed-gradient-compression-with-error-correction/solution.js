/** Task 934 - Implement distributed gradient compression with error correction under packet loss.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A parameter-server round with W workers, d-dimensional gradients, top-k sparsification and
 * duplicated shards so a lost packet does not lose the whole update:
 *   compression    each worker keeps a residual r_i. It sparsifies (g_i + r_i) down to the k largest
 *                  magnitudes, sends those (index, value) pairs, and carries r_i <- (g_i + r_i) - sent.
 *                  With error feedback the *accumulated* sent vector tracks the accumulated true
 *                  gradient, so a high cosine similarity survives arbitrary rounds, while without it
 *                  the alignment decays - the checks assert both directions;
 *   erasure coding  the payload is cut into shards and every shard is sent `replication` times (an
 *                  MDS-style (r,1) code); the server rebuilds a shard when at least one copy arrives
 *                  and drops the worker's update only when every copy of some shard is lost, which is
 *                  exactly what the checks count against the closed-form per-shard survivability;
 *   accounting     bytes on the wire are compared with the dense baseline (d * 8 bytes per worker).
 * Simplification: seeded Bernoulli loss (no bursts or reordering), integer shard counts, no
 * quantisation of the values. Verified: error-feedback alignment versus no-feedback decay, exactness
 * when k = d, shard survivability versus the formula, traffic savings, the no-loss and total-loss
 * boundaries, a converging training loop, invalid input and determinism.
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
  float() { return this.next() / 4294967296; }
  normal(scale) { return (this.float() + this.float() + this.float() - 1.5) * 2 * scale; }
}

const norm = (v) => Math.sqrt(v.reduce((a, x) => a + x * x, 0));
const dot = (a, b) => a.reduce((acc, x, i) => acc + x * b[i], 0);
const cosine = (a, b) => (norm(a) === 0 || norm(b) === 0 ? 0 : dot(a, b) / (norm(a) * norm(b)));

/** Keep the k entries with the largest magnitude, ties broken by index for determinism. */
function topK(vector, k) {
  assert.ok(Number.isInteger(k) && k >= 1, "k must be a positive integer");
  assert.ok(k <= vector.length, "k cannot exceed the dimension");
  const order = vector.map((value, index) => ({ value, index }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value) || a.index - b.index)
    .slice(0, k);
  const sent = new Array(vector.length).fill(0);
  for (const entry of order) sent[entry.index] = entry.value;
  return sent;
}

/** One worker: compress the residual-corrected gradient and return the new residual. */
function compress(workerGradient, residual, k, model = {}) {
  const target = workerGradient.map((v, i) => v + residual[i]);
  const sent = topK(target, k);
  const nextResidual = target.map((v, i) => v - sent[i]);
  assert.ok(norm(nextResidual) <= norm(target) + 1e-12, "the residual cannot grow on compression");
  return { sent, residual: nextResidual, delta: model.noFeedback ? nextResidual : nextResidual };
}

/** Cut a sparse vector into `shards` equal parts (the payload travels as one shard per packet). */
function shardPayload(vector, shards) {
  assert.ok(Number.isInteger(shards) && shards >= 1, "the shard count must be positive");
  const per = Math.ceil(vector.length / shards);
  return Array.from({ length: shards }, (_, s) => vector.slice(s * per, Math.min((s + 1) * per, vector.length))
    .map((value, offset) => ({ index: s * per + offset, value })));
}

/** Send every shard `replication` times over a lossy channel; returns what the server can rebuild. */
function transmit(shards, prng, options = {}) {
  const lossRate = options.lossRate ?? 0;
  const replication = options.replication ?? 1;
  assert.ok(lossRate >= 0 && lossRate <= 1, "the loss rate must be a probability");
  assert.ok(Number.isInteger(replication) && replication >= 1, "replication must be at least one");
  let packets = 0, lost = 0;
  const received = [];
  const survivor = [];
  for (const shard of shards) {
    let arrived = false;
    for (let copy = 0; copy < replication; copy++) {
      packets++;
      if (prng.float() < lossRate) { lost++; continue; }
      arrived = true;
    }
    survivor.push(arrived ? 1 : 0);
    if (arrived) received.push(...shard);
  }
  const update = new Array(shards.flat().length).fill(0);
  for (const entry of received) update[entry.index] = entry.value;
  return { update, packets, lost, survivor, complete: received.length === shards.flat().length };
}

function makeGradients(prng, workers, dimension, scale) {
  return Array.from({ length: workers },
    () => Array.from({ length: dimension }, () => prng.normal(scale)));
}

/* ------------------------------------------------------------------ checks */

function main() {
  const prng = new Prng(20250934);
  const workers = 4, dimension = 64, k = 8;

  // 1. Error feedback: the accumulated sent vector tracks the accumulated gradient closely, while
  //    the same loop without feedback drifts away (the residual must not be discarded).
  const simulate = (useFeedback, rounds = 60) => {
    const local = new Prng(20250934);
    const residuals = Array.from({ length: workers }, () => new Array(dimension).fill(0));
    const sentTotal = new Array(dimension).fill(0);
    const trueTotal = new Array(dimension).fill(0);
    let byteTotal = 0;
    for (let round = 0; round < rounds; round++) {
      for (let w = 0; w < workers; w++) {
        const gradient = Array.from({ length: dimension }, () => local.normal(1));
        for (let i = 0; i < dimension; i++) trueTotal[i] += gradient[i];
        const target = gradient.map((v, i) => v + (useFeedback ? residuals[w][i] : 0));
        const sent = topK(target, k);
        if (useFeedback) residuals[w] = target.map((v, i) => v - sent[i]);
        for (let i = 0; i < dimension; i++) sentTotal[i] += sent[i];
        byteTotal += k * 8;
      }
    }
    return { similarity: cosine(sentTotal, trueTotal), byteTotal };
  };
  const withFeedback = simulate(true);
  const withoutFeedback = simulate(false);
  assert.ok(withFeedback.similarity > 0.99, `error feedback only reached ${withFeedback.similarity}`);
  assert.ok(withoutFeedback.similarity < 0.8, `without feedback the similarity was ${withoutFeedback.similarity}`);
  assert.ok(withFeedback.similarity > withoutFeedback.similarity + 0.2);
  assert.strictEqual(withoutFeedback.byteTotal, withFeedback.byteTotal, "the wire cost is identical");

  // 2. Boundary: k = d compresses nothing, so the reconstruction is exact.
  {
    const gradient = makeGradients(prng, 1, 16, 1)[0];
    const exact = topK(gradient, 16);
    assert.deepStrictEqual(exact, gradient, "k = d must be the identity");
    assert.strictEqual(norm(gradient.map((v, i) => v - exact[i])), 0);
    const residual = compress(gradient, new Array(16).fill(0), 16).residual;
    assert.ok(residual.every((v) => v === 0), "an exact send leaves no residual");
  }

  // 3. Traffic accounting: sparsification cuts the bytes, replication costs a factor r.
  {
    const denseBytes = workers * dimension * 8;
    const sparseBytes = workers * k * 8;
    assert.ok(sparseBytes < denseBytes / 6, `sparse ${sparseBytes} vs dense ${denseBytes}`);
    assert.strictEqual(sparseBytes * 2, denseBytes / 4, "duplicating every packet must double the bytes");
    const shards = shardPayload(makeGradients(prng, 1, dimension, 1)[0], 4);
    assert.strictEqual(shards.length, 4);
    assert.strictEqual(shards.flat().length, dimension, "sharding must preserve every entry");
  }

  // 4. Erasure coding: with replication 2 a shard survives unless both copies are lost, which the
  //    closed form (1 - p)^2 per shard predicts; replication 1 must lose more.
  const lossRate = 0.3;
  const survivorRate = (replication, trials = 400) => {
    const local = new Prng(99);
    let survived = 0, total = 0, lostPackets = 0;
    for (let trial = 0; trial < trials; trial++) {
      const payload = { length: 8 };
      const shards = [Array.from({ length: 8 }, (_, index) => ({ index, value: 1 }))];
      const result = transmit(shards, local, { lossRate, replication });
      survived += result.complete ? 1 : 0;
      lostPackets += result.lost;
      total += result.packets;
      assert.ok(payload.length === 8);
    }
    return { rate: survived / trials, lostPackets, total };
  };
  const single = survivorRate(1);
  const double = survivorRate(2);
  assert.ok(double.rate > single.rate + 0.15, `replication barely helped: ${single.rate} -> ${double.rate}`);
  assert.ok(Math.abs(double.rate - (1 - lossRate * lossRate)) < 0.05,
    `replication 2 survived ${double.rate}, expected ~${1 - lossRate * lossRate}`);
  assert.ok(Math.abs(single.rate - 0.7) < 0.05, `replication 1 survived ${single.rate}, expected ~0.7`);
  assert.ok(double.lostPackets > single.lostPackets, "more copies means more packets at risk");

  // 5. The reconstruction is exact whenever the payload arrives, and empty when it does not.
  {
    const vector = makeGradients(prng, 1, 32, 1)[0];
    const shards = shardPayload(vector, 4);
    const clean = transmit(shards, new Prng(1), { lossRate: 0, replication: 1 });
    assert.ok(clean.update.every((v, i) => Math.abs(v - vector[i]) < 1e-12), "a clean channel is exact");
    assert.strictEqual(clean.lost, 0);
    const dead = transmit(shards, new Prng(1), { lossRate: 1, replication: 3 });
    assert.ok(dead.update.every((v) => v === 0), "total loss delivers nothing");
    assert.strictEqual(dead.complete, false);
    assert.strictEqual(dead.packets, 12);
  }

  // 6. End to end: compressed, lossy, error-corrected training still converges on a quadratic.
  {
    const run = (options) => {
      const local = new Prng(2024);
      const target = Array.from({ length: 16 }, (_, i) => (i % 5) - 2);
      let weights = new Array(16).fill(0);
      const residual = new Array(16).fill(0);
      let bytes = 0, incomplete = 0;
      const history = [];
      for (let round = 0; round < 40; round++) {
        const gradient = weights.map((w, i) => w - target[i]);
        const target_ = gradient.map((v, i) => v + residual[i]);
        const sent = topK(target_, options.k);
        const nextResidual = target_.map((v, i) => v - sent[i]);
        for (let i = 0; i < 16; i++) residual[i] = nextResidual[i];
        const shards = shardPayload(sent, 2);
        const channel = transmit(shards, local, { lossRate: options.lossRate, replication: options.replication });
        bytes += options.k * 8;
        if (!channel.complete) incomplete++;
        weights = weights.map((w, i) => w - 0.5 * channel.update[i]);
        history.push(norm(weights.map((w, i) => w - target[i])));
      }
      return { loss: history[history.length - 1], bytes, incomplete, history };
    };
    const perfect = run({ k: 16, lossRate: 0, replication: 1 });
    const compressed = run({ k: 4, lossRate: 0, replication: 1 });
    const lossy = run({ k: 4, lossRate: 0.25, replication: 2 });
    assert.ok(perfect.loss < 1e-9, "an uncompressed run must converge to the target");
    assert.ok(compressed.loss < 1e-6, `compression should still converge, got ${compressed.loss}`);
    assert.ok(lossy.loss < 1e-4, `lossy compression should still converge, got ${lossy.loss}`);
    assert.ok(lossy.bytes < 16 * 8 * 40 / 2, "the compressed run must move far fewer bytes");
    assert.ok(lossy.incomplete >= 0 && lossy.incomplete < 40, "some updates should still arrive");
    assert.ok(compressed.history.every((value, index) => index === 0 || value <= compressed.history[index - 1] + 1e-9),
      "the compressed run must descend monotonically");
  }

  // 7. Invalid input is rejected.
  assert.throws(() => topK([1, 2, 3], 4), /cannot exceed the dimension/);
  assert.throws(() => topK([1, 2, 3], 0), /positive integer/);
  assert.throws(() => shardPayload([1, 2], 0), /positive/);
  assert.throws(() => transmit([[]], new Prng(1), { lossRate: 1.5 }), /probability/);
  assert.throws(() => transmit([[]], new Prng(1), { replication: 0 }), /at least one/);

  // 8. Determinism: identical seeds replay identical channels, residuals and losses.
  const rerun = () => {
    const local = new Prng(7);
    const vector = Array.from({ length: 12 }, () => local.normal(1));
    const sent = topK(vector, 3);
    const residual = compress(vector, new Array(12).fill(0), 3).residual;
    const channel = transmit(shardPayload(sent.map((value, index) => ({ index, value })), 3),
      local, { lossRate: 0.4, replication: 2 });
    return { sent, residual, survivor: channel.survivor, lost: channel.lost };
  };
  assert.deepStrictEqual(rerun(), rerun());
  assert.ok(rerun().survivor.length === 3);

  console.log(`compression: k=${k}/${dimension} -> ${workers * k * 8} bytes vs ` +
    `${workers * dimension * 8} dense; feedback similarity ${withFeedback.similarity.toFixed(4)} ` +
    `vs ${withoutFeedback.similarity.toFixed(4)} without; replication-2 survivability ${double.rate.toFixed(3)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, topK, compress, shardPayload, transmit, makeGradients, cosine, norm, dot };
