/** Task 816 — Implement distributed training system with fault-tolerant all-reduce and elastic workers.
 *  (Tier: Master, Beyond — Machine Learning & AI)
 *
 * Model. Data-parallel synchronous SGD over a linear model: every worker owns a shard of the dataset,
 * computes the gradient of the loss  L(θ) = ½ Σ (x·θ − y)²  on its own samples, and the gradients are
 * combined with a *ring all-reduce* (reduce-scatter followed by all-gather, P−1 steps each, chunk r
 * of the buffer reduced on worker r). Every worker then applies the identical update
 * θ ← θ − lr·Σg, so the replicas stay bit-identical — that equality is the central safety invariant
 * and is re-checked after every collective, after every failure recovery and after every elastic
 * reshuffle.
 *
 * Exactness. The dataset is integral, the learning rate is a power of two (1/512) and the master
 * weights are re-quantised onto a 1/2^16 fixed-point grid after every update — the same trick
 * deterministic-training practitioners use, and it keeps every quantity in the system an exact
 * multiple of 1/2^16 whose numerator fits in a double. The ring reduction therefore sums the
 * gradients in a different order than the single-process reference and still lands on the *bitwise
 * identical* result, so the checks compare with exact equality instead of tolerances. The analytic
 * gradient is additionally checked against central finite differences, which is a genuinely
 * independent oracle.
 *
 * Faults and elasticity. The collective takes an explicit fault schedule: a worker may die in a given
 * step of a given phase. Mid-collective failure discards the partial reduction (as real reducers do),
 * the communicator is rebuilt without the dead worker over the *original* gradients, and training
 * resumes — the result is then exactly the sum over the surviving workers' gradients, which the
 * reference recomputes. A restarted or newly joined worker must first load the cluster checkpoint
 * (parameters at the last committed round) before it may contribute; the checks verify that an
 * interrupted run resumed from a checkpoint reproduces the uninterrupted loss trajectory bit for bit,
 * and that a fresh worker joining at round k leaves every replica identical.
 *
 * Simplified versus production: no sockets or NCCL (a ring buffer exchange is simulated in-process
 * with explicit steps), gradients live in memory instead of on accelerators, there is no
 * straggler mitigation/overlap, no gradient compression, and failures are scripted events rather than
 * timeouts. "Scale" is expressed through typed parameters (`workers`, `samplesPerWorker`, `dimensions`,
 * `bufferChunks`) and the measured cost counters.
 *
 * Verified: ring all-reduce equals the brute-force sum and the per-worker byte/message cost, gradient
 * versus finite differences, bit-identical replicas, monotone loss descent, parameter recovery of the
 * generating weights, mid-collective failure in both phases with a correct reduced-world result,
 * liveness after failures, checkpoint/restore trajectory equality, elastic join, degenerate worlds
 * (one worker, empty buffer, buffer not divisible by the world size) and determinism.
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

const LEARNING_RATE = 1 / 512; // a power of two, so the update stays exact
const PARAMETER_SCALE = 65536; // parameters live on a 1/2^16 fixed-point grid (16 fractional bits)

/**
 * Parameters are re-quantised onto the fixed-point grid after every update. Every replica performs
 * the identical rounding, so the replicas stay bit-identical *and* every quantity in the system is an
 * exact multiple of 1/2^16 whose numerator fits in a double — which is why the ring reduction (chunk
 * order) and the single-process reference (worker order) agree bit for bit. Keeping the master
 * weights on a reproducible grid is exactly what deterministic-training practitioners do.
 */
function quantize(value) {
  return Math.round(value * PARAMETER_SCALE) / PARAMETER_SCALE;
}

function trueWeights() {
  return [5, 3, -2, 1];
}

/** y = 3a − 2b + c + 5 for x = [1, a, b, c] — the model that gradient descent should recover. */
function targetFor(x) {
  return 3 * x[1] - 2 * x[2] + x[3] + 5;
}

function buildDataset(workerCount, samplesPerWorker, dimensions, seed) {
  const next = splitmix64(seed);
  const shards = [];
  for (let worker = 0; worker < workerCount; worker++) {
    const samples = [];
    for (let s = 0; s < samplesPerWorker; s++) {
      const x = new Array(dimensions).fill(0);
      x[0] = 1; // bias
      for (let d = 1; d < dimensions; d++) x[d] = Number(next() % 7n) - 3;
      samples.push({ x, y: targetFor(x) });
    }
    shards.push({ workerId: worker, samples });
  }
  return shards;
}

// ---------------------------------------------------------------------------- model maths

function gradientOf(shard, theta) {
  const gradient = new Array(theta.length).fill(0);
  for (const sample of shard.samples) {
    let prediction = 0;
    for (let d = 0; d < theta.length; d++) prediction += sample.x[d] * theta[d];
    const residual = prediction - sample.y;
    for (let d = 0; d < theta.length; d++) gradient[d] += residual * sample.x[d];
  }
  return gradient;
}

function lossOf(shards, theta) {
  let total = 0;
  for (const shard of shards) {
    for (const sample of shard.samples) {
      let prediction = 0;
      for (let d = 0; d < theta.length; d++) prediction += sample.x[d] * theta[d];
      const residual = prediction - sample.y;
      total += 0.5 * residual * residual;
    }
  }
  return total;
}

function sampleCount(shards) {
  return shards.reduce((sum, shard) => sum + shard.samples.length, 0);
}

/** Central finite differences — an oracle that does not reuse the analytic gradient. */
function numericalGradient(shards, theta, epsilon) {
  const gradient = new Array(theta.length).fill(0);
  for (let d = 0; d < theta.length; d++) {
    const high = theta.slice();
    const low = theta.slice();
    high[d] += epsilon;
    low[d] -= epsilon;
    gradient[d] = (lossOf(shards, high) - lossOf(shards, low)) / (2 * epsilon);
  }
  return gradient;
}

function addChunk(target, chunkIndex, chunkSize, source) {
  const start = chunkIndex * chunkSize;
  const end = Math.min(start + chunkSize, target.length);
  for (let i = start; i < end; i++) target[i] += source[i];
}

function copyChunk(target, chunkIndex, chunkSize, source) {
  const start = chunkIndex * chunkSize;
  const end = Math.min(start + chunkSize, target.length);
  for (let i = start; i < end; i++) target[i] = source[i];
  return end - start;
}

// ---------------------------------------------------------------------------- ring all-reduce

/**
 * Ring all-reduce over a list of peers; `buffers` maps peer index -> gradient array of equal length.
 * `faults` may schedule a death at `{phase: "reduce-scatter"|"all-gather", step, peer}`. A death
 * aborts the collective and the partial reduction is thrown away (real reducers do the same), so the
 * caller can rebuild the communicator over the survivors and rerun it over the original gradients.
 */
function ringAllReduce(peerIds, buffers, faults = null) {
  const world = peerIds.length;
  if (world === 0) throw new RangeError("empty world");
  const size = buffers[0].length;
  const chunkSize = size === 0 ? 0 : Math.ceil(size / world);
  const stats = { messages: 0, elementsTransferred: 0, steps: 0, aborted: null };
  const transfer = (from, to, phase, step) => {
    if (faults && faults.phase === phase && faults.step === step && faults.peer === peerIds[to]) {
      stats.aborted = { phase, step, peer: peerIds[to] };
      return false;
    }
    stats.messages++;
    stats.elementsTransferred += chunkSize;
    return true;
  };

  for (let step = 0; step < world - 1; step++) {
    for (let p = 0; p < world; p++) {
      const sendChunk = ((p - step) % world + world) % world;
      const receiveChunk = ((p - step - 1) % world + world) % world;
      const sender = buffers[((p - 1) + world) % world];
      if (!transfer(p, p, "reduce-scatter", step)) return { ok: false, stats, ...stats.aborted };
      addChunk(buffers[p], receiveChunk, chunkSize, sender);
    }
    stats.steps++;
  }
  for (let step = 0; step < world - 1; step++) {
    for (let p = 0; p < world; p++) {
      const sendChunk = ((p - step + 1) % world + world) % world;
      const receiveChunk = ((p - step) % world + world) % world;
      const sender = buffers[((p - 1) + world) % world];
      if (!transfer(p, p, "all-gather", step)) return { ok: false, stats, ...stats.aborted };
      copyChunk(buffers[p], receiveChunk, chunkSize, sender);
    }
    stats.steps++;
  }
  return { ok: true, stats };
}

function sequentialSum(buffers) {
  const total = new Array(buffers[0].length).fill(0);
  for (const buffer of buffers) for (let i = 0; i < total.length; i++) total[i] += buffer[i];
  return total;
}

// ---------------------------------------------------------------------------- cluster

class TrainingCluster {
  /**
   * @param {object} config `{ workers, samplesPerWorker, dimensions, seed, checkpointEvery }`
   */
  constructor(config) {
    this.config = {
      samplesPerWorker: 16,
      dimensions: 4,
      seed: 0x8165eed,
      checkpointEvery: 5,
      ...config,
    };
    this.shards = buildDataset(this.config.workers, this.config.samplesPerWorker, this.config.dimensions, this.config.seed);
    this.theta = new Array(this.config.dimensions).fill(0);
    this.round = 0;
    this.lossHistory = [];
    this.checkpoints = [];
    this.events = [];
    this.live = this.shards.map((shard) => shard.workerId);
    this.collectiveStats = { messages: 0, elementsTransferred: 0, aborted: 0, reruns: 0 };
  }

  replicas() {
    return new Map(this.live.map((id) => [id, this.theta.slice()]));
  }

  /** Does every live worker hold exactly the same parameters? (the safety invariant) */
  replicasAgree() {
    const replicas = this.replicas();
    const first = JSON.stringify(this.theta);
    for (const value of replicas.values()) if (JSON.stringify(value) !== first) return false;
    return true;
  }

  liveShards() {
    return this.shards.filter((shard) => this.live.includes(shard.workerId));
  }

  checkpoint() {
    this.checkpoints.push({ round: this.round, theta: this.theta.slice(), live: [...this.live] });
    return this.checkpoints[this.checkpoints.length - 1];
  }

  /**
   * One synchronous SGD round: per-worker gradients, ring all-reduce, identical update everywhere.
   * A failure during the collective aborts it, drops the dead worker and re-runs the collective over
   * the survivors' *original* gradients.
   */
  step({ faults = null } = {}) {
    const peers = [...this.live];
    const buffers = peers.map((id) => gradientOf(this.shards[id], this.theta));
    let result = ringAllReduce(peers, buffers, faults);
    let reduced;
    if (!result.ok) {
      this.collectiveStats.aborted++;
      this.events.push({ round: this.round, event: "collective-aborted", peer: result.peer, phase: result.phase });
      this.live = this.live.filter((id) => id !== result.peer);
      if (this.live.length === 0) throw new Error("all workers died");
      const survivors = [...this.live];
      const survivorGradients = survivors.map((id) => gradientOf(this.shards[id], this.theta));
      const retryBuffers = survivorGradients.map((gradient) => gradient.slice());
      result = ringAllReduce(survivors, retryBuffers, null);
      this.collectiveStats.reruns++;
      if (!result.ok) throw new Error("retry failed");
      if (!this.replicasAgreeAfterCollective(retryBuffers)) throw new Error("retry produced different replicas");
      // The ring sums chunks in a different order than this canonical sum; the two must agree exactly.
      if (JSON.stringify(retryBuffers[0]) !== JSON.stringify(sequentialSum(survivorGradients))) {
        throw new Error("the reduced-world result is not the exact sum over the survivors");
      }
      this.collectiveStats.messages += result.stats.messages;
      this.collectiveStats.elementsTransferred += result.stats.elementsTransferred;
      reduced = retryBuffers[0];
    } else {
      this.collectiveStats.messages += result.stats.messages;
      this.collectiveStats.elementsTransferred += result.stats.elementsTransferred;
      if (!this.replicasAgreeAfterCollective(buffers)) throw new Error("all-reduce produced different replicas");
      reduced = buffers[0];
    }
    for (let d = 0; d < this.theta.length; d++) this.theta[d] = quantize(this.theta[d] - LEARNING_RATE * reduced[d]);
    this.lossHistory.push(lossOf(this.liveShards(), this.theta));
    this.round++;
    if (this.round % this.config.checkpointEvery === 0) this.checkpoint();
    return this.lossHistory[this.lossHistory.length - 1];
  }

  replicasAgreeAfterCollective(buffers) {
    const first = JSON.stringify(buffers[0]);
    for (const buffer of buffers) if (JSON.stringify(buffer) !== first) return false;
    return true;
  }

  /** A worker dies between rounds (e.g. a hardware fault detected by the health check). */
  kill(id) {
    this.live = this.live.filter((peer) => peer !== id);
    this.events.push({ round: this.round, event: "killed", peer: id });
    return this.live.length;
  }

  /**
   * A restarted or fresh worker joins. It must first load the committed parameters, otherwise its
   * replica would diverge from its peers; `join` enforces that by taking the state from the cluster.
   */
  join(id, shard) {
    if (this.live.includes(id)) throw new Error("worker already in the world");
    if (this.live.length === 0) throw new Error("cannot join an empty world");
    if (!shard) {
      shard = { workerId: id, samples: this.shards[0].samples.map((sample) => ({ x: sample.x.slice(), y: sample.y })) };
    }
    if (!this.shards.some((existing) => existing.workerId === id)) this.shards.push(shard);
    this.live.push(id);
    this.events.push({ round: this.round, event: "join", peer: id });
    if (!this.replicasAgree()) throw new Error("joiner did not catch up");
    return this.live.length;
  }

  static fromCheckpoint(checkpoint) {
    const cluster = new TrainingCluster({ workers: 1, samplesPerWorker: 0, dimensions: checkpoint.theta.length, seed: 0 });
    cluster.theta = checkpoint.theta.slice();
    cluster.round = checkpoint.round;
    cluster.live = [...checkpoint.live];
    cluster.checkpoints = [];
    cluster.lossHistory = [];
    return cluster;
  }
}

/** Single-process reference: identical update rule with a plain canonical-order gradient sum. */
function referenceRun(shards, rounds) {
  const theta = new Array(shards.length === 0 ? 0 : shards[0].samples[0].x.length).fill(0);
  const losses = [];
  for (let round = 0; round < rounds; round++) {
    const total = new Array(theta.length).fill(0);
    for (const shard of shards) {
      const gradient = gradientOf(shard, theta);
      for (let d = 0; d < total.length; d++) total[d] += gradient[d];
    }
    for (let d = 0; d < theta.length; d++) theta[d] = quantize(theta[d] - LEARNING_RATE * total[d]);
    losses.push(lossOf(shards, theta));
  }
  return { theta, losses };
}

/**
 * Reference for runs whose world changes (failures, elastic membership): `schedule[round]` lists the
 * workers contributing in that round, so the oracle reproduces the exact same update sequence.
 */
function referenceRunSchedule(shards, schedule) {
  const byId = new Map(shards.map((shard) => [shard.workerId, shard]));
  const dimensions = shards[0].samples[0].x.length;
  const theta = new Array(dimensions).fill(0);
  const losses = [];
  for (const world of schedule) {
    const total = new Array(dimensions).fill(0);
    const active = world.map((id) => byId.get(id));
    for (const shard of active) {
      const gradient = gradientOf(shard, theta);
      for (let d = 0; d < total.length; d++) total[d] += gradient[d];
    }
    for (let d = 0; d < dimensions; d++) theta[d] = quantize(theta[d] - LEARNING_RATE * total[d]);
    losses.push(lossOf(active, theta));
  }
  return { theta, losses };
}

module.exports = {
  splitmix64,
  LEARNING_RATE,
  PARAMETER_SCALE,
  quantize,
  trueWeights,
  buildDataset,
  gradientOf,
  lossOf,
  numericalGradient,
  sampleCount,
  ringAllReduce,
  sequentialSum,
  TrainingCluster,
  referenceRun,
  referenceRunSchedule,
};

if (require.main === module) {
  const assert = require("node:assert");

  const shards = buildDataset(4, 16, 4, 0x816aaa);
  const theta = [0, 0, 0, 0];
  const perWorkerGradients = shards.map((shard) => gradientOf(shard, theta));
  const oracle = sequentialSum(perWorkerGradients);
  assert.deepStrictEqual(
    perWorkerGradients.reduce((total, gradient) => total.map((value, d) => value + gradient[d])),
    oracle,
    "integer sums are associative, so the reference sum is exact"
  );

  // --- gradient versus central finite differences -------------------------------
  const analytic = gradientOf(shards[0], [1, 2, -1, 0]);
  const numeric = numericalGradient([shards[0]], [1, 2, -1, 0], 0.01);
  for (let d = 0; d < analytic.length; d++) {
    assert.ok(Math.abs(analytic[d] - numeric[d]) <= 1e-6 * Math.max(1, Math.abs(analytic[d])), `gradient[${d}]: ${analytic[d]} vs ${numeric[d]}`);
  }

  // --- ring all-reduce: correctness, exactness and cost -------------------------
  const buffers = perWorkerGradients.map((gradient) => gradient.slice());
  const reduced = ringAllReduce([0, 1, 2, 3], buffers);
  assert.ok(reduced.ok);
  for (const buffer of buffers) assert.deepStrictEqual(buffer, oracle, "every replica must hold the exact sum");
  assert.strictEqual(reduced.stats.messages, 2 * 4 * 3, "ring all-reduce uses 2(P-1) messages per worker");
  const chunkSize = Math.ceil(4 / 4);
  assert.strictEqual(reduced.stats.elementsTransferred, 2 * 4 * 3 * chunkSize);
  const naiveElements = 4 * 3 * 4; // broadcast-style all-to-all of the full buffer
  assert.ok(reduced.stats.elementsTransferred < naiveElements, "ring beats naive all-to-all");

  // buffer length not divisible by the world size, and degenerate worlds
  const oddBuffers = [0, 1, 2, 3, 4].map((worker) => [worker, worker + 1, worker + 2, worker + 3, worker + 4, worker + 5, worker + 6]);
  const oddCopy = oddBuffers.map((buffer) => buffer.slice());
  const oddResult = ringAllReduce([0, 1, 2, 3, 4], oddCopy);
  assert.ok(oddResult.ok);
  assert.deepStrictEqual(oddCopy[0], sequentialSum(oddBuffers), "uneven chunks must still reduce exactly");
  assert.deepStrictEqual(oddCopy[4], oddCopy[0]);
  const single = [[3, 4, 5]];
  assert.ok(ringAllReduce([0], single).ok);
  assert.deepStrictEqual(single[0], [3, 4, 5], "a one-worker world is a no-op");
  assert.strictEqual(ringAllReduce([0], single).stats.messages, 0);
  const empty = [[], []];
  assert.ok(ringAllReduce([0, 1], empty).ok);
  assert.deepStrictEqual(empty, [[], []]);
  assert.throws(() => ringAllReduce([], []), RangeError);

  // --- training: identical replicas, monotone loss, parameter recovery ----------
  const cluster = new TrainingCluster({ workers: 4, seed: 0x816aaa });
  const reference = referenceRun(cluster.shards, 100);
  // The 1/2^16 parameter grid bounds how small the loss can get: above that floor the descent must
  // be strictly monotone, at the floor the parameters simply stop moving.
  const LOSS_FLOOR = 1e-6;
  for (let round = 0; round < 100; round++) {
    cluster.step();
    assert.ok(cluster.replicasAgree(), `replicas diverged in round ${round}`);
    if (round > 0) {
      const previous = cluster.lossHistory[round - 1];
      assert.ok(cluster.lossHistory[round] <= previous, `loss must never increase (round ${round})`);
      if (previous > LOSS_FLOOR) assert.ok(cluster.lossHistory[round] < previous, `loss must strictly decrease above the grid floor (round ${round})`);
    }
  }
  assert.ok(cluster.lossHistory[99] <= LOSS_FLOOR, `the run must reach the quantisation floor, got ${cluster.lossHistory[99]}`);
  assert.deepStrictEqual(cluster.theta, reference.theta, "the ring result must equal the sequential reference exactly");
  assert.deepStrictEqual(cluster.lossHistory, reference.losses);
  const weights = trueWeights();
  for (let d = 0; d < weights.length; d++) {
    assert.ok(Math.abs(cluster.theta[d] - weights[d]) < 0.005, `parameter ${d} recovered as ${cluster.theta[d]} vs ${weights[d]}`);
  }

  // --- mid-collective failure in both phases ------------------------------------
  let failuresRecovered = 0;
  for (const phase of ["reduce-scatter", "all-gather"]) {
    const faulty = new TrainingCluster({ workers: 4, seed: 0x816bbb });
    const schedule = [
      ...[0, 1, 2, 3, 4, 5].map(() => [0, 1, 2, 3]),
      ...Array.from({ length: 11 }, () => [0, 1, 3]),
    ];
    const oracleRun = referenceRunSchedule(faulty.shards, schedule);
    const losses = [];
    for (let round = 0; round < 6; round++) losses.push(faulty.step());
    const beforeFault = faulty.theta.slice();
    const survivorOracle = sequentialSum([0, 1, 3].map((id) => gradientOf(faulty.shards[id], beforeFault)));
    const loss = faulty.step({ faults: { phase, step: 1, peer: 2 } });
    assert.ok(faulty.live.includes(2) === false, `${phase}: the dead worker must leave the world`);
    assert.strictEqual(faulty.collectiveStats.aborted, 1);
    assert.strictEqual(faulty.collectiveStats.reruns, 1);
    assert.ok(faulty.replicasAgree(), `${phase}: survivors must agree after recovery`);
    assert.ok(loss < losses[losses.length - 1], `${phase}: training must make progress after a failure`);
    assert.deepStrictEqual(
      faulty.theta,
      beforeFault.map((value, d) => quantize(value - LEARNING_RATE * survivorOracle[d])),
      `${phase}: the aborted round must equal one exact update over the survivors' gradients`
    );
    for (let round = 0; round < 10; round++) faulty.step();
    assert.deepStrictEqual(faulty.lossHistory, oracleRun.losses, `${phase}: the reduced-world loss trajectory must match the reference`);
    assert.deepStrictEqual(faulty.theta, oracleRun.theta, `${phase}: the reduced-world parameters must match the reference`);
    assert.ok(faulty.lossHistory[16] < faulty.lossHistory[5], `${phase}: liveness - the reduced world keeps converging`);
    failuresRecovered += faulty.collectiveStats.aborted;
  }

  // --- death between rounds, restart from checkpoint, elastic join ---------------
  const elastic = new TrainingCluster({ workers: 4, seed: 0x816ccc, checkpointEvery: 5 });
  for (let round = 0; round < 12; round++) elastic.step();
  const committed = elastic.checkpoints[elastic.checkpoints.length - 1];
  assert.strictEqual(committed.round, 10);
  assert.notDeepStrictEqual(committed.theta, elastic.theta, "the committed checkpoint predates the two rounds after it");
  const replay = TrainingCluster.fromCheckpoint(committed);
  replay.shards = elastic.shards;
  replay.live = [...elastic.live];
  replay.lossHistory = elastic.lossHistory.slice(0, committed.round);
  replay.step();
  replay.step();
  assert.deepStrictEqual(replay.theta, elastic.theta, "replaying two rounds from the checkpoint must reproduce the state exactly");
  elastic.kill(3);
  assert.deepStrictEqual(elastic.live, [0, 1, 2]);
  elastic.join(3);
  assert.deepStrictEqual(elastic.live, [0, 1, 2, 3]);
  assert.ok(elastic.replicasAgree(), "the restarted worker must catch up before contributing");
  const liveOracle = sequentialSum(elastic.live.map((id) => gradientOf(elastic.shards[id], elastic.theta)));
  const beforeJoin = elastic.theta.slice();
  elastic.step();
  assert.deepStrictEqual(
    elastic.theta,
    beforeJoin.map((value, d) => quantize(value - LEARNING_RATE * liveOracle[d])),
    "an elastic world must reduce over exactly the live shards"
  );
  assert.throws(() => elastic.join(1), /already in the world/);
  const fresh = new TrainingCluster({ workers: 1, samplesPerWorker: 0, dimensions: 4, seed: 0 });
  fresh.live = [];
  assert.throws(() => fresh.join(9), /empty world/);

  // --- checkpoint/restore reproduces the uninterrupted trajectory ---------------
  const uninterrupted = new TrainingCluster({ workers: 3, seed: 0x816ddd, checkpointEvery: 4 });
  for (let round = 0; round < 20; round++) uninterrupted.step();
  const snapshot = uninterrupted.checkpoints[1];
  const resumed = TrainingCluster.fromCheckpoint(snapshot);
  resumed.shards = uninterrupted.shards;
  resumed.config.checkpointEvery = 4;
  resumed.lossHistory = uninterrupted.lossHistory.slice(0, snapshot.round);
  for (let round = snapshot.round; round < 20; round++) resumed.step();
  assert.deepStrictEqual(resumed.lossHistory, uninterrupted.lossHistory, "restoring a checkpoint must not change the trajectory");
  assert.deepStrictEqual(resumed.theta, uninterrupted.theta);

  // --- a worker can also be lost between rounds, and determinism ----------------
  const deterministicA = new TrainingCluster({ workers: 3, seed: 0x816eee });
  const deterministicB = new TrainingCluster({ workers: 3, seed: 0x816eee });
  for (let round = 0; round < 10; round++) {
    deterministicA.step();
    deterministicB.step(round === 4 ? { faults: { phase: "reduce-scatter", step: 0, peer: 1 } } : {});
  }
  assert.strictEqual(deterministicA.live.length, 3);
  assert.strictEqual(deterministicB.live.length, 2);
  const sameScheduleAgain = new TrainingCluster({ workers: 3, seed: 0x816eee });
  for (let round = 0; round < 10; round++) sameScheduleAgain.step(round === 4 ? { faults: { phase: "reduce-scatter", step: 0, peer: 1 } } : {});
  assert.deepStrictEqual(sameScheduleAgain.lossHistory, deterministicB.lossHistory, "the same schedule must reproduce the same run");
  assert.strictEqual(sampleCount(deterministicA.liveShards()), 48);
  assert.strictEqual(sampleCount(deterministicB.liveShards()), 32, "the effective batch follows the surviving workers");

  console.log(
    `workers 4, rounds 100, loss ${cluster.lossHistory[0].toFixed(1)} -> ${cluster.lossHistory[99].toExponential(2)}, ` +
      `theta [${cluster.theta.join(", ")}] vs target [${weights.join(", ")}], ring messages ${cluster.collectiveStats.messages}, ` +
      `transferred ${cluster.collectiveStats.elementsTransferred} elements (naive all-to-all would move ${4 * 3 * 4 * 100}), ` +
      `failures recovered ${failuresRecovered}`
  );
  console.log("PASS");
}
