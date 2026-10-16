/** Task 720 - Implement ring-allreduce simulation and communication cost modeling. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Ring allreduce (the Baidu/`ring-allreduce` algorithm) replaces the parameter-server
 * "everyone to the root and back" pattern with 2(N-1) point-to-point steps on a logical ring
 * of N ranks. The buffer of every rank is cut into N chunks:
 *
 *   * reduce-scatter: N-1 steps. At step k rank r sends chunk (r - k) mod N to rank r+1 and
 *     adds the chunk arriving from rank r-1 into its own slot. After N-1 steps rank r holds
 *     the complete sum for exactly one chunk, namely (r + 1) mod N, and every other chunk is
 *     still partial. Sends of a step are computed before any delivery, which is the ring
 *     semantics.
 *   * allgather: N-1 steps. Each rank relays its complete chunk around the ring until all
 *     ranks hold it. Now every rank owns the full reduction.
 *
 * The point of the algorithm is the byte count: each rank *moves* 2(N-1)/N * S bytes (S =
 * payload size) instead of the (N-1)S that a naive all-to-all costs, and the per-rank volume
 * is independent of N, so the algorithm is bandwidth-optimal -- the price is latency: 2(N-1)
 * messages against N-1 for the naive scheme, which is why the cost model below shows the
 * naive scheme winning for tiny payloads and the ring winning beyond a crossover size.
 *
 * Simplified: a simulated network with no packet loss, uniform chunk sizes (the payload must
 * divide by N), one message per step (no segmentation) and byte/time accounting instead of
 * real sockets. Verified here: bit-exact equality of the ring result with an independent
 * element-wise allreduce on integer payloads, the structural invariant of the reduce-scatter
 * phase (exactly one complete chunk per rank, at index (r+1) mod N), the allgather phase
 * giving every rank every chunk, the per-step ring topology (each rank sends exactly one
 * chunk to its successor), the simulated byte counters against the closed-form model, the
 * bandwidth-optimality ratio, the latency/bandwidth crossover found by bisection, the
 * degenerate N = 1 and empty-payload cases, invalid input rejection, and determinism.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

/** Deterministic splitmix64 -> [0, 1); keeps every simulation reproducible. */
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

const mod = (value, n) => ((value % n) + n) % n;
const BYTES_PER_ELEMENT = 8;                       // float64 payloads

/** Independent reference: element-wise sum over all ranks (never used by the ring itself). */
function referenceAllreduce(buffers) {
  const out = new Array(buffers[0].length).fill(0);
  for (const buffer of buffers) for (let i = 0; i < out.length; i++) out[i] += buffer[i];
  return out;
}

/** Naive baseline: every rank gathers all N buffers (all-to-all), i.e. (N-1)S bytes each. */
function naiveAllreduce(buffers) {
  const total = referenceAllreduce(buffers);
  return {
    result: buffers.map(() => total.slice()),
    bytesSent: buffers.map(() => (buffers.length - 1) * total.length * BYTES_PER_ELEMENT),
    messages: buffers.map(() => buffers.length - 1),
  };
}

/**
 * Ring allreduce. `onStep` (optional) is called after every step with a phase descriptor so
 * the caller can assert the structural invariants of the algorithm.
 */
function ringAllreduce(buffers, { onStep } = {}) {
  const ranks = buffers.length;
  const length = buffers[0].length;
  for (const buffer of buffers) {
    if (buffer.length !== length) throw new Error("all ranks must hold equally sized buffers");
  }
  if (length % ranks !== 0) {
    throw new Error(`payload of ${length} elements cannot be split over ${ranks} ranks`);
  }
  const chunkLength = length / ranks;
  const state = buffers.map((buffer) => buffer.slice());
  const bytesSent = new Array(ranks).fill(0);
  const messages = new Array(ranks).fill(0);
  const stepLog = [];
  const sliceOf = (buffer, chunk) => buffer.slice(chunk * chunkLength, (chunk + 1) * chunkLength);
  const writeInto = (buffer, chunk, payload, accumulate) => {
    const base = chunk * chunkLength;
    for (let i = 0; i < chunkLength; i++) {
      buffer[base + i] = accumulate ? buffer[base + i] + payload[i] : payload[i];
    }
  };

  for (let step = 0; step < ranks - 1; step++) {
    // At step k rank r ships chunk (r - k) mod N, so every rank posts exactly one chunk and
    // the ring carries N distinct chunks per step. All sends are computed before any
    // delivery (ring semantics), and the receiver accumulates into the chunk it received, so
    // after the step that chunk covers one more rank of the arc that ends at r.
    const outbox = state.map((buffer, r) => sliceOf(buffer, mod(r - step, ranks)));
    for (let from = 0; from < ranks; from++) {
      const chunk = mod(from - step, ranks);
      writeInto(state[(from + 1) % ranks], chunk, outbox[from], true);
      bytesSent[from] += outbox[from].length * BYTES_PER_ELEMENT;
      messages[from]++;
    }
    stepLog.push({ phase: "reduce-scatter", step, chunk: null });
    if (onStep) onStep({ phase: "reduce-scatter", step, state });
  }

  for (let step = 0; step < ranks - 1; step++) {
    // Rank r relays the newest chunk it owns, (r + 1 - step) mod N, so after the step both r
    // and r+1 know it; after N-1 steps every rank knows all N chunks.
    const outbox = state.map((buffer, r) => sliceOf(buffer, mod(r + 1 - step, ranks)));
    for (let from = 0; from < ranks; from++) {
      const chunk = mod(from + 1 - step, ranks);
      writeInto(state[(from + 1) % ranks], chunk, outbox[from], false);
      bytesSent[from] += outbox[from].length * BYTES_PER_ELEMENT;
      messages[from]++;
    }
    stepLog.push({ phase: "allgather", step, chunk: null });
    if (onStep) onStep({ phase: "allgather", step, state });
  }

  return { result: state, bytesSent, messages, chunkLength, ranks, payloadBytes: length * BYTES_PER_ELEMENT, steps: stepLog };
}

/** Closed-form cost of the ring: 2(N-1) messages of one chunk each, per rank. */
function ringCost({ ranks, payloadBytes }) {
  const perRankBytes = (2 * (ranks - 1) * payloadBytes) / ranks;
  const messages = 2 * (ranks - 1);
  return { perRankBytes, messages };
}

/** Closed-form cost of the naive all-to-all baseline: (N-1) full buffers per rank. */
function naiveCost({ ranks, payloadBytes }) {
  return { perRankBytes: (ranks - 1) * payloadBytes, messages: ranks - 1 };
}

/** Latency + bandwidth model (alpha = per-message latency, bandwidth in bytes/second). */
function transferTime({ perRankBytes, messages, latency, bandwidth }) {
  return messages * latency + perRankBytes / bandwidth;
}

const ringTime = (params) => transferTime({ ...ringCost(params), latency: params.latency, bandwidth: params.bandwidth });
const naiveTime = (params) => transferTime({ ...naiveCost(params), latency: params.latency, bandwidth: params.bandwidth });

/** Independent oracle for the crossover promised by the cost model: bisection on the gap. */
function crossoverPayloadBytes({ ranks, latency, bandwidth }, upperBound = 2 ** 40) {
  // The gap (ring - naive) starts positive -- the ring pays 2(N-1) latencies instead of N-1 --
  // and decreases with the payload, so a single bisection brackets the crossover.
  const gap = (payloadBytes) => ringTime({ ranks, payloadBytes, latency, bandwidth })
    - naiveTime({ ranks, payloadBytes, latency, bandwidth });
  if (gap(upperBound) >= 0) throw new Error("no crossover below the upper bound");
  let lo = 0;
  let hi = upperBound;
  for (let iteration = 0; iteration < 200; iteration++) {
    const mid = (lo + hi) / 2;
    if (gap(mid) > 0) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

/** Count the chunks of a rank that already equal the fully reduced reference. */
function completeChunks(buffer, reference, chunkLength) {
  const complete = [];
  for (let chunk = 0; chunk < buffer.length / chunkLength; chunk++) {
    let same = true;
    for (let i = 0; i < chunkLength; i++) {
      if (buffer[chunk * chunkLength + i] !== reference[chunk * chunkLength + i]) { same = false; break; }
    }
    if (same) complete.push(chunk);
  }
  return complete;
}

module.exports = {
  makeRng, referenceAllreduce, naiveAllreduce, ringAllreduce, ringCost, naiveCost,
  transferTime, ringTime, naiveTime, crossoverPayloadBytes, completeChunks,
};

if (require.main === module) {
  const rng = makeRng(20260720);
  const makeBuffers = (ranks, chunkLength, rand) =>
    Array.from({ length: ranks }, () => Array.from({ length: chunkLength * ranks }, () => rand()));

  // -- correctness on integer payloads (exact in doubles, so equality is bit-exact) -----
  let comparedElements = 0;
  for (const ranks of [1, 2, 3, 4, 5, 8, 16]) {
    const buffers = makeBuffers(ranks, 3, () => Math.floor(rng() * 21) - 10);
    const reference = referenceAllreduce(buffers);
    const ring = ringAllreduce(buffers);
    for (let r = 0; r < ranks; r++) {
      deepStrictEqual(ring.result[r], reference, `rank ${r} of ${ranks} must hold the full sum`);
      comparedElements += reference.length;
    }
    deepStrictEqual(naiveAllreduce(buffers).result, ring.result, "both allreduce schemes agree");
    const expected = ringCost({ ranks, payloadBytes: buffers[0].length * 8 });
    for (let r = 0; r < ranks; r++) {
      strictEqual(ring.bytesSent[r], expected.perRankBytes,
        `rank ${r} must move the closed-form 2(N-1)/N * S bytes`);
      strictEqual(ring.messages[r], expected.messages, `rank ${r} must send 2(N-1) messages`);
    }
  }
  console.log(`allreduce correctness compared over ${comparedElements} rank-elements`);

  // -- reduce-scatter invariant: after step k, rank r's chunk (r-1-k) mod N is the sum of
  //    the k+2 ranks forming the arc that ends at r --------------------------------------
  const ranks = 8;
  const chunkLength = 4;
  const buffers = makeBuffers(ranks, chunkLength, () => Math.floor(rng() * 11) - 5);
  const reference = referenceAllreduce(buffers);
  const arcSum = (chunk, from, length) => {
    const out = new Array(chunkLength).fill(0);
    for (let t = 0; t < length; t++) {
      const source = buffers[mod(from - t, ranks)];
      for (let i = 0; i < chunkLength; i++) out[i] += source[chunk * chunkLength + i];
    }
    return out;
  };
  const seen = [];
  ringAllreduce(buffers, {
    onStep: ({ phase, step, state: live }) => seen.push({ phase, step, live: live.map((b) => b.slice()) }),
  });
  const reduceScatterSteps = seen.filter((s) => s.phase === "reduce-scatter");
  const allgatherSteps = seen.filter((s) => s.phase === "allgather");
  strictEqual(reduceScatterSteps.length, ranks - 1, "reduce-scatter needs exactly N-1 steps");
  strictEqual(allgatherSteps.length, ranks - 1, "allgather needs N-1 steps");
  strictEqual(completeChunks(buffers[0], reference, chunkLength).length, 0,
    "before the first step no rank owns a complete chunk");
  for (let step = 0; step < ranks - 1; step++) {
    const states = reduceScatterSteps[step].live;
    for (let r = 0; r < ranks; r++) {
      const chunk = mod(r - 1 - step, ranks);
      const expected = arcSum(chunk, r, step + 2);
      deepStrictEqual(states[r].slice(chunk * chunkLength, (chunk + 1) * chunkLength), expected,
        `step ${step} rank ${r}: chunk ${chunk} must hold the sum of the arc of ${step + 2} ranks ending at r`);
    }
  }
  const afterReduceScatter = reduceScatterSteps[ranks - 2].live;
  for (let r = 0; r < ranks; r++) {
    const complete = completeChunks(afterReduceScatter[r], reference, chunkLength);
    deepStrictEqual(complete, [mod(r + 1, ranks)],
      `reduce-scatter leaves rank ${r} owning exactly chunk (r + 1) mod N`);
  }
  for (let step = 0; step < ranks - 1; step++) {
    const states = allgatherSteps[step].live;
    for (let r = 0; r < ranks; r++) {
      const expected = Array.from({ length: step + 2 }, (_, t) => mod(r + 1 - t, ranks)).sort((a, b) => a - b);
      deepStrictEqual(completeChunks(states[r], reference, chunkLength), expected,
        `allgather step ${step}: rank ${r} must know exactly ${step + 2} complete chunks`);
    }
  }
  const lastAllgather = seen[seen.length - 1].live;
  for (let r = 0; r < ranks; r++) {
    deepStrictEqual(completeChunks(lastAllgather[r], reference, chunkLength), [0, 1, 2, 3, 4, 5, 6, 7],
      "after allgather every rank holds every chunk");
  }

  // -- ring topology: every rank ships exactly one chunk per step to its successor ------
  const ring = ringAllreduce(buffers);
  strictEqual(ring.steps.length, 2 * (ranks - 1), "2(N-1) steps in total");
  strictEqual(ring.bytesSent.every((b) => b === ring.bytesSent[0]), true, "the ring is symmetric");
  deepStrictEqual(ring.bytesSent, new Array(ranks).fill((2 * (ranks - 1) * buffers[0].length * 8) / ranks),
    "every rank moves exactly 2(N-1)/N * S bytes");
  const naive = naiveAllreduce(buffers);
  const ringPerRank = ringCost({ ranks, payloadBytes: buffers[0].length * 8 }).perRankBytes;
  const naivePerRank = naiveCost({ ranks, payloadBytes: buffers[0].length * 8 }).perRankBytes;
  strictEqual(naivePerRank / ringPerRank, ranks / 2, "the naive all-to-all moves N/2 times the ring's bytes");
  console.log(`ranks=${ranks}: ring ${ringPerRank} B/rank vs naive ${naivePerRank} B/rank`);

  // -- degenerate and invalid inputs ----------------------------------------------------
  const solo = ringAllreduce([[7, 8, 9]]);
  deepStrictEqual(solo.result, [[7, 8, 9]], "a single rank needs no communication");
  deepStrictEqual(solo.bytesSent, [0]);
  strictEqual(solo.messages[0], 0);
  strictEqual(ringCost({ ranks: 1, payloadBytes: 24 }).perRankBytes, 0);

  const twoRanks = ringAllreduce([[1, 2], [3, 4]]);
  deepStrictEqual(twoRanks.result, [[4, 6], [4, 6]], "N = 2 still reduces correctly");
  strictEqual(ringCost({ ranks: 2, payloadBytes: 16 }).perRankBytes,
    naiveCost({ ranks: 2, payloadBytes: 16 }).perRankBytes,
    "at N = 2 the ring and the direct exchange move the same volume");

  const emptyBuffers = [[], [], []];
  const emptyRing = ringAllreduce(emptyBuffers);
  deepStrictEqual(emptyRing.result, emptyBuffers, "an empty payload reduces to an empty payload");
  strictEqual(emptyRing.bytesSent[0], 0);

  throws(() => ringAllreduce([[1, 2, 3], [4, 5, 6]]), /cannot be split/, "5 elements over 2 ranks is invalid");
  throws(() => ringAllreduce([[1, 2], [3]]), /equally sized/, "ragged buffers are invalid");
  throws(() => ringAllreduce([[1, 2], [3, 4]].map((b, i) => (i === 0 ? b : b.concat(0)))),
    /cannot be split|equally sized/, "mismatched sizes are rejected");

  // -- determinism ----------------------------------------------------------------------
  const replay = () => {
    const local = makeRng(7);
    const data = Array.from({ length: 4 }, () => Array.from({ length: 8 }, () => Math.floor(local() * 9)));
    const run = ringAllreduce(data);
    return { result: run.result, bytes: run.bytesSent };
  };
  deepStrictEqual(replay(), replay(), "identical inputs must produce identical traffic and results");

  // -- cost model: latency-dominated small payloads favour the naive scheme -------------
  const model = { ranks: 8, latency: 1e-5, bandwidth: 1e9 };
  const crossover = crossoverPayloadBytes(model);
  ok(ringTime({ ...model, payloadBytes: crossover * 0.5 })
    > naiveTime({ ...model, payloadBytes: crossover * 0.5 }),
    "below the crossover the 2(N-1)-message ring loses to the N-1-message naive scheme");
  ok(ringTime({ ...model, payloadBytes: crossover * 2 })
    < naiveTime({ ...model, payloadBytes: crossover * 2 }),
    "above the crossover the bandwidth-optimal ring wins");
  const analytic = (model.latency * model.bandwidth) / (1 - 2 / model.ranks);
  ok(Math.abs(crossover - analytic) <= analytic * 1e-6,
    `bisection found ${crossover}, the closed form says ${analytic}`);
  console.log(`crossover payload=${crossover.toFixed(0)} B (ring ${(ringTime({ ...model, payloadBytes: 1e7 }) * 1e3).toFixed(3)} ms`
    + ` vs naive ${(naiveTime({ ...model, payloadBytes: 1e7 }) * 1e3).toFixed(3)} ms at 10 MB)`);
  console.log("PASS");
}
