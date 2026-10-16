/** Task 719 - Implement parameter server simulation for distributed ML training. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A parameter server (Li et al.) keeps the model on server shards and lets workers push
 * gradients / pull parameters over the network. This file simulates the single-shard design:
 *
 *   * `ParameterServer` owns `params`, a monotonically increasing `version` and the log of
 *     applied updates. `pull()` hands out an *immutable snapshot* plus the version it came
 *     from; `push()` applies lr * gradient and bumps the version. Pushes may arrive out of
 *     order, so every applied update records its staleness = (updates applied before it) -
 *     (version the worker read). Bounded-staleness SGD is exactly this model.
 *   * `scheduleSync` is the BSP/barrier mode: one round pulls one version, every worker
 *     computes on its shard, the barrier waits for the *slowest* worker, then the server
 *     applies the mean gradient -- one full-batch gradient step on the whole data set.
 *   * `scheduleAsync` is a discrete-event simulation of pipelined workers: pull events take
 *     snapshots (and the gradient is computed from that snapshot), pushes queue on a
 *     single-threaded server. No barrier, so a straggler only slows its own throughput.
 *
 * Simplified: abstract "ticks" instead of real sockets, one server shard, one gradient per
 * push (no inner mini-batches), fixed deterministic per-worker delays instead of a latency
 * distribution, and a small dense least-squares objective. What is still verified: that a
 * barrier round is one full-batch gradient step against an independent reference, that every
 * worker of a round reads the same version, the staleness recorded by the server against a
 * re-derivation from the pull log, the straggler's larger staleness, snapshot immutability,
 * convergence to the analytic optimum with the loss settling on the irreducible noise floor,
 * deterministic replay, degenerate inputs (single worker, lr = 0, empty cluster) and the
 * wall-clock claim that motivates asynchronous servers: with a straggler on the cluster,
 * async reaches a lower loss in the same budget.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

/** splitmix64 -> [0, 1); deterministic, so the whole simulation is reproducible. */
function makeRng(seed) {
  let state = (BigInt(seed >>> 0) + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
  return function next() {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z ^= z >> 31n;
    return Number(z >> 11n) / 9007199254740992;
  };
}

/** Mean-squared-error gradient of a shard: (2 / n) * X^T (X w - y). */
function shardGradient(shard, params) {
  const { features, targets } = shard;
  const n = features.length;
  const d = params.length;
  const gradient = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    const row = features[i];
    let residual = -targets[i];
    for (let k = 0; k < d; k++) residual += row[k] * params[k];
    const scale = (2 * residual) / n;
    for (let k = 0; k < d; k++) gradient[k] += scale * row[k];
  }
  return gradient;
}

/** Mean-squared error of a shard (the per-worker objective the gradient above descends). */
function shardLoss(shard, params) {
  const { features, targets } = shard;
  let loss = 0;
  for (let i = 0; i < features.length; i++) {
    let residual = -targets[i];
    for (let k = 0; k < params.length; k++) residual += features[i][k] * params[k];
    loss += residual * residual;
  }
  return loss / features.length;
}

/** Cluster objective: the mean of the shard losses (every shard is weighted equally). */
function combinedLoss(workers, params) {
  let loss = 0;
  for (const worker of workers) loss += shardLoss(worker.shard, params) / workers.length;
  return loss;
}

/** A worker's shard as one plain (features, targets) problem. */
const allSamples = (data) => ({ features: data.map((s) => s.x), targets: data.map((s) => s.y) });

/** Exact minimiser of the concatenated least squares through the normal equations. */
function leastSquaresSolution(data) {
  const d = data[0].x.length;
  const gram = Array.from({ length: d }, () => new Array(d).fill(0));
  const rhs = new Array(d).fill(0);
  for (const { x, y } of data) {
    for (let a = 0; a < d; a++) {
      rhs[a] += x[a] * y;
      for (let b = 0; b < d; b++) gram[a][b] += x[a] * x[b];
    }
  }
  const m = gram.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < d; col++) {
    let pivot = col;
    for (let r = col + 1; r < d; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    [m[col], m[pivot]] = [m[pivot], m[col]];
    for (let r = 0; r < d; r++) {
      if (r === col || m[col][col] === 0) continue;
      const factor = m[r][col] / m[col][col];
      for (let c = col; c <= d; c++) m[r][c] -= factor * m[col][c];
    }
  }
  return m.map((row, i) => row[d] / m[i][i]);
}

/** Power iteration on the MSE curvature: the largest eigenvalue caps the learning rate. */
function maxCurvature(data) {
  const d = data[0].x.length;
  let v = new Array(d).fill(0);
  v[0] = 1;
  for (let step = 0; step < 200; step++) {
    const next = new Array(d).fill(0);
    for (const { x } of data) {
      let dot = 0;
      for (let k = 0; k < d; k++) dot += x[k] * v[k];
      for (let k = 0; k < d; k++) next[k] += dot * x[k];
    }
    const norm = Math.hypot(...next);
    v = next.map((value) => value / norm);
  }
  let value = 0;
  for (const { x } of data) {
    let dot = 0;
    for (let k = 0; k < d; k++) dot += x[k] * v[k];
    value += dot * dot;
  }
  return (2 * value) / data.length;            // curvature of the MSE objective is 2 * X^T X / N
}

class ParameterServer {
  constructor(dimension, learningRate) {
    this.params = new Array(dimension).fill(0);
    this.version = 0;
    this.lr = learningRate;
    this.log = [];                             // { version, workerId, staleness, time, pullVersion }
  }

  /** Immutable read: later pushes must not be visible through an earlier snapshot. */
  pull() {
    return { params: this.params.slice(), version: this.version };
  }

  push(gradient, workerId, pullVersion, time) {
    const staleness = this.version - pullVersion;
    for (let i = 0; i < this.params.length; i++) this.params[i] -= this.lr * gradient[i];
    this.version++;
    this.log.push({ version: this.version, workerId, staleness, time, pullVersion });
    return staleness;
  }
}

class Worker {
  constructor(id, samples, { pullCost, computePerSample, pushCost }) {
    this.id = id;
    this.shard = allSamples(samples);
    this.pullCost = pullCost;
    this.computeCost = computePerSample * samples.length;
    this.pushCost = pushCost;
  }

  get cycleCost() {
    return this.pullCost + this.computeCost + this.pushCost;
  }
}

/** Barrier schedule: a round costs the slowest worker plus the server. */
function scheduleSync(workers, { server, rounds, serverCost = 0.02 }) {
  if (workers.length === 0) throw new Error("a parameter server needs at least one worker");
  const roundCost = Math.max(...workers.map((w) => w.cycleCost)) + serverCost;
  let time = 0;
  const pullVersions = [];
  for (let round = 0; round < rounds; round++) {
    const snapshot = server.pull();
    pullVersions.push(snapshot.version);
    const gradients = workers.map((w) => shardGradient(w.shard, snapshot.params));
    const mean = new Array(snapshot.params.length).fill(0);
    for (const gradient of gradients) {
      for (let i = 0; i < mean.length; i++) mean[i] += gradient[i] / workers.length;
    }
    time += roundCost;
    server.push(mean, -1, snapshot.version, time);
  }
  return { time, roundCost, pullVersions, updates: server.log.length };
}

/**
 * Discrete-event asynchronous schedule. Each worker cycles pull -> compute -> push without
 * waiting for anyone; the server applies payloads in arrival order, so a worker's gradient
 * lands on a newer parameter vector than the one it read (that gap is the staleness).
 */
function scheduleAsync(workers, { server, deadline, serverCost = 0.02 }) {
  if (workers.length === 0) throw new Error("a parameter server needs at least one worker");
  const pullEvents = [];
  for (const worker of workers) {
    for (let start = 0; start + worker.cycleCost <= deadline; start += worker.cycleCost) {
      pullEvents.push({ time: start, worker });
    }
  }
  pullEvents.sort((a, b) => (a.time - b.time) || (a.worker.id - b.worker.id));
  // A payload lands one worker cycle after its pull, and the single-threaded server applies
  // payloads in landing order, so an update's visibility time is fixed in advance -- that is
  // what lets the two event streams below be merged without a priority queue.
  const pushEvents = pullEvents.map((pull) => ({ land: pull.time + pull.worker.cycleCost, worker: pull.worker }));
  pushEvents.sort((a, b) => (a.land - b.land) || (a.worker.id - b.worker.id));
  let serverFree = 0;
  for (const push of pushEvents) {
    serverFree = Math.max(serverFree, push.land) + serverCost;
    push.applyTime = serverFree;
  }
  const pullVersions = new Map(workers.map((w) => [w.id, []]));
  const payloads = new Map(workers.map((w) => [w.id, []]));
  const losses = [];
  let pi = 0;
  let si = 0;
  while (pi < pullEvents.length || si < pushEvents.length) {
    // Ties go to the push: an update applied exactly at time t is visible to a pull at t.
    const skipPull = si < pushEvents.length
      && (pi >= pullEvents.length || pushEvents[si].applyTime <= pullEvents[pi].time);
    if (skipPull) {
      const push = pushEvents[si++];
      const payload = payloads.get(push.worker.id).shift();
      server.push(payload.gradient, push.worker.id, payload.version, push.applyTime);
      losses.push({ time: push.applyTime, loss: combinedLoss(workers, server.params) });
    } else {
      const pull = pullEvents[pi++];
      const snapshot = server.pull();
      pullVersions.get(pull.worker.id).push(snapshot.version);
      payloads.get(pull.worker.id).push({
        version: snapshot.version,
        gradient: shardGradient(pull.worker.shard, snapshot.params),
      });
    }
  }
  return { losses, time: serverFree, pullVersions, pulls: pullEvents.length, updates: server.log.length };
}

/** Round-robin split of the data over a deterministic cluster; the straggler is slower. */
function buildCluster(data, { workerCount, straggler, stragglerFactor }) {
  const shards = Array.from({ length: workerCount }, () => []);
  data.forEach((sample, i) => shards[i % workerCount].push(sample));
  return shards.map((shard, id) => new Worker(id, shard, {
    pullCost: 0.05,
    computePerSample: 0.005 * (id === straggler ? stragglerFactor : 1),
    pushCost: 0.05,
  }));
}

/** y = x . truth + small zero-mean noise, so the optimum has an irreducible loss floor. */
function makeProblem(rng, { samples, dimension }) {
  const truth = Array.from({ length: dimension }, () => rng() * 2 - 1);
  const data = Array.from({ length: samples }, () => {
    const x = Array.from({ length: dimension }, () => rng() * 2 - 1);
    const y = x.reduce((acc, v, k) => acc + v * truth[k], (rng() - 0.5) * 0.02);
    return { x, y };
  });
  return { truth, data };
}

module.exports = {
  makeRng, shardGradient, shardLoss, combinedLoss, allSamples, leastSquaresSolution,
  maxCurvature, ParameterServer, Worker, scheduleSync, scheduleAsync, buildCluster, makeProblem,
};

if (require.main === module) {
  const rng = makeRng(20260719);
  const { data } = makeProblem(rng, { samples: 40, dimension: 4 });
  const optimum = leastSquaresSolution(data);
  const curvature = maxCurvature(data);
  const stableLr = 1 / curvature;             // lr * curvature <= 1: stable, monotone descent
  const cluster = buildCluster(data, { workerCount: 4, straggler: 3, stragglerFactor: 5 });

  // -- a barrier round is exactly one full-batch gradient step --------------------------
  const syncServer = new ParameterServer(4, stableLr);
  const first = scheduleSync(cluster, { server: syncServer, rounds: 1 });
  const referenceGradient = shardGradient(allSamples(data), [0, 0, 0, 0]);
  for (let i = 0; i < 4; i++) {
    ok(Math.abs(syncServer.params[i] + stableLr * referenceGradient[i]) < 1e-12,
      `a barrier round must equal the full-batch step at coordinate ${i}`);
  }
  strictEqual(syncServer.log[0].staleness, 0, "a barrier round reads and writes one version");
  strictEqual(first.roundCost, Math.max(...cluster.map((w) => w.cycleCost)) + 0.02,
    "a barrier round costs the slowest worker plus the server");
  deepStrictEqual(first.pullVersions, [0]);

  // -- barrier consistency and convergence to the analytic optimum ----------------------
  const longServer = new ParameterServer(4, stableLr);
  const barrier = scheduleSync(cluster, { server: longServer, rounds: 500 });
  deepStrictEqual(barrier.pullVersions, Array.from({ length: 500 }, (_, i) => i),
    "round r must read version r for every worker -- that is the barrier");
  const syncDistance = Math.hypot(...longServer.params.map((v, i) => v - optimum[i]));
  const floor = combinedLoss(cluster, optimum);
  const trainedLoss = combinedLoss(cluster, longServer.params);
  ok(syncDistance < 1e-9, `sync training must converge to the optimum, distance=${syncDistance}`);
  ok(trainedLoss < floor * 1.000001 && trainedLoss > floor * 0.999999,
    `the loss must settle on the noise floor, got ${trainedLoss} vs floor ${floor}`);
  console.log(`sync: 500 rounds, distance=${syncDistance.toExponential(2)} loss=${trainedLoss.toExponential(3)}`);

  // -- asynchronous run, staleness re-derived from the pull log -------------------------
  const asyncServer = new ParameterServer(4, stableLr);
  const asyncRun = scheduleAsync(cluster, { server: asyncServer, deadline: 60 });
  const log = asyncServer.log;
  ok(log.length > 4, "the async run must apply several updates");
  const consumed = new Map();
  const stalenessByWorker = new Map();
  for (let k = 0; k < log.length; k++) {
    const entry = log[k];
    strictEqual(entry.version, k + 1, "versions must be consecutive");
    const seen = consumed.get(entry.workerId) || 0;
    const workerPulls = asyncRun.pullVersions.get(entry.workerId);
    // A worker's n-th push follows its n-th pull, so the version it read is workerPulls[n].
    strictEqual(entry.staleness, k - workerPulls[seen],
      `update ${k}: staleness must be (apply index) - (version read at that worker's pull)`);
    consumed.set(entry.workerId, seen + 1);
    const stats = stalenessByWorker.get(entry.workerId) || { count: 0, max: 0 };
    stats.count++;
    stats.max = Math.max(stats.max, entry.staleness);
    stalenessByWorker.set(entry.workerId, stats);
  }
  ok(log.every((e) => e.staleness >= 0), "staleness can never be negative");
  const straggler = stalenessByWorker.get(3);
  const fast = stalenessByWorker.get(0);
  ok(straggler.max > fast.max,
    `the straggler must accumulate more staleness (${straggler.max}) than a fast worker (${fast.max})`);
  ok(straggler.count < fast.count, "and it must complete fewer cycles in the same budget");
  console.log(`async: pulls=${asyncRun.pulls} updates=${log.length} maxStaleness=${straggler.max}`);

  // -- snapshot immutability, empty cluster, lr = 0 -------------------------------------
  const aliasServer = new ParameterServer(2, 1);
  const snapshot = aliasServer.pull();
  aliasServer.push([1, 1], 0, 0, 1);
  deepStrictEqual(snapshot.params, [0, 0], "a pulled snapshot must not alias the live parameters");
  throws(() => scheduleSync([], { server: new ParameterServer(2, 1), rounds: 1 }), /at least one worker/);
  throws(() => scheduleAsync([], { server: new ParameterServer(2, 1), deadline: 1 }), /at least one worker/);

  const frozen = new ParameterServer(4, 0);
  scheduleSync(cluster, { server: frozen, rounds: 10 });
  deepStrictEqual(frozen.params, [0, 0, 0, 0], "lr = 0 must leave the parameters alone");
  strictEqual(frozen.log.length, 10, "but the rounds still happened");

  // -- deterministic replay -------------------------------------------------------------
  const replay = () => {
    const server = new ParameterServer(4, stableLr);
    const run = scheduleAsync(cluster, { server, deadline: 30 });
    return run.losses.map((l) => l.loss).concat(server.log.map((e) => e.staleness));
  };
  deepStrictEqual(replay(), replay(), "identical inputs must replay identically");

  // -- one worker degenerates to plain SGD ---------------------------------------------
  const solo = buildCluster(data, { workerCount: 1, straggler: -1, stragglerFactor: 1 });
  const soloServer = new ParameterServer(4, stableLr);
  scheduleSync(solo, { server: soloServer, rounds: 500 });
  ok(Math.hypot(...soloServer.params.map((v, i) => v - optimum[i])) < 1e-9,
    "a single-worker server is plain batch GD and must converge too");

  // -- the payoff: a straggler stalls the barrier but not the pipeline -------------------
  const budget = 60;
  const slowLr = stableLr / 200;             // small enough that neither run can converge
  const budgetSyncServer = new ParameterServer(4, slowLr);      // inside the budget
  const budgetSync = scheduleSync(cluster, { server: budgetSyncServer, rounds: Math.floor(budget / barrier.roundCost) });
  const budgetAsyncServer = new ParameterServer(4, slowLr);
  const budgetAsync = scheduleAsync(cluster, { server: budgetAsyncServer, deadline: budget });
  const syncLoss = combinedLoss(cluster, budgetSyncServer.params);
  const asyncLoss = combinedLoss(cluster, budgetAsyncServer.params);
  ok(budgetSync.time <= budget && budgetAsync.time <= budget + 1, "both runs must respect the budget");
  ok(budgetAsyncServer.log.length > 3 * budgetSyncServer.log.length,
    `async must fit far more updates in the budget (${budgetAsyncServer.log.length} vs ${budgetSyncServer.log.length})`);
  ok(syncLoss > floor * 100, `the barrier must still be far from the floor, got ${syncLoss} vs ${floor}`);
  ok(asyncLoss < floor * 1.1, `async must have reached the floor region, got ${asyncLoss} vs ${floor}`);
  ok(asyncLoss * 100 < syncLoss,
    `async loss ${asyncLoss} must beat barrier loss ${syncLoss} by two orders in the same budget`);
  console.log(`budget ${budget}: barrier ${budgetSyncServer.log.length} updates loss=${syncLoss.toExponential(3)}` +
    ` | async ${budgetAsyncServer.log.length} updates loss=${asyncLoss.toExponential(3)}`);
  console.log(`curvature=${curvature.toFixed(3)} stable lr=${stableLr.toFixed(3)} noise floor=${floor.toExponential(3)}`);
  console.log("PASS");
}
