/** Task 718 - Implement gradient compression techniques (top-k) and error feedback. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Distributed SGD rarely affords an all-reduce of the dense gradient d: each worker keeps
 * only the k entries of largest magnitude ("top-k sparsification") so the wire payload is
 * 2k numbers instead of d. Plain top-k, however, is biased: the coordinates it drops are
 * dropped *forever*, and with a skewed gradient the same handful of coordinates wins the
 * race every round while the rest never move.
 *
 * Error feedback (Seide et al. "1-bit SGD"; Stich et al. "Sparsified SGD with memory") keeps
 * a residual/memory buffer r and transmits the top-k of g_eff = g + r. The transmitted part
 * is subtracted from the buffer, the unsent part stays, hence the identity maintained by
 * every round, componentwise and exactly,
 *
 *     g + r_old = decoded + r_new          (nothing created, nothing lost)
 *
 * Because the buffer stores *sums* of the ignored gradients it eventually makes an ignored
 * coordinate win the next top-k race, and summing the identity over t gives
 * sum_t decoded_t = sum_t g_t - r_final, so the accumulated update of every coordinate
 * tracks its accumulated true gradient instead of a biased subset of it.
 *
 * This file implements the sparsifier (a bounded min-heap single pass, not a full sort) and
 * the training round, and verifies: top-k against a brute-force sort oracle including
 * magnitude ties and k clamped to [0, d]; the exactness of the accumulator identity and of
 * the cumulative-update identity derived from it; the lossless boundary k = d and the
 * degenerate boundaries k = 0 / empty gradient; and the two behavioural claims that motivate
 * error feedback -- (a) on a skewed *constant* gradient with k = 1 the baseline serves only
 * one coordinate ever, while error feedback serves all of them, and (b) on an
 * ill-conditioned quadratic error feedback reaches a far smaller loss in the same budget.
 * Deterministic inputs only (splitmix64 PRNG, no Math.random, no clock).
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

/** splitmix64: deterministic, seeded 64-bit stream mapped to [0, 1). */
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

/**
 * Top-k selection in one pass with a bounded min-heap of size k whose root is the *least*
 * significant kept entry. Significances are ordered by decreasing |value|, ties by
 * increasing index, which makes the selected set unique for any input (so the brute-force
 * oracle below must agree index-for-index, not merely up to reordering).
 * Returns { indices, values } with indices ascending (canonical, coalesced payload order).
 */
function topKSelect(values, k) {
  const d = values.length;
  const keep = Math.max(0, Math.min(d, Math.trunc(k)));
  if (keep === 0) return { indices: [], values: [] };
  const heap = [];
  const worse = (i, j) => {
    const mi = Math.abs(values[i]);
    const mj = Math.abs(values[j]);
    return mi !== mj ? mi < mj : i > j;
  };
  const swap = (a, b) => { const t = heap[a]; heap[a] = heap[b]; heap[b] = t; };
  const siftUp = (start) => {
    let i = start;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!worse(heap[i], heap[parent])) return;
      swap(i, parent);
      i = parent;
    }
  };
  const siftDown = (start) => {
    let i = start;
    for (;;) {
      const left = 2 * i + 1;
      const right = left + 1;
      let worst = i;
      if (left < heap.length && worse(heap[left], heap[worst])) worst = left;
      if (right < heap.length && worse(heap[right], heap[worst])) worst = right;
      if (worst === i) return;
      swap(i, worst);
      i = worst;
    }
  };
  for (let i = 0; i < d; i++) {
    if (heap.length < keep) {
      heap.push(i);
      siftUp(heap.length - 1);
    } else if (worse(heap[0], i)) {
      heap[0] = i;
      siftDown(0);
    }
  }
  const indices = heap.slice().sort((a, b) => a - b);
  return { indices, values: indices.map((i) => values[i]) };
}

/** Brute-force oracle: full sort by (|value| desc, index asc), then canonical index order. */
function topKBruteForce(values, k) {
  const keep = Math.max(0, Math.min(values.length, Math.trunc(k)));
  const order = [];
  for (let i = 0; i < values.length; i++) order.push(i);
  order.sort((a, b) => {
    const ma = Math.abs(values[a]);
    const mb = Math.abs(values[b]);
    return ma !== mb ? mb - ma : a - b;
  });
  const indices = order.slice(0, keep).sort((a, b) => a - b);
  return { indices, values: indices.map((i) => values[i]) };
}

/** Dense vector holding only the transmitted entries; everything else is exactly zero. */
function decompress(indices, values, dimension) {
  const dense = new Array(dimension).fill(0);
  for (let i = 0; i < indices.length; i++) dense[indices[i]] = values[i];
  return dense;
}

/** Wire cost of a round: float32 value + int32 index per transmitted entry. */
const payloadBytes = (k) => k * 8;
const denseBytes = (d) => d * 4;

class ErrorFeedbackTrainer {
  /**
   * @param {number} dimension
   * @param {number} k          entries transmitted per round
   * @param {number} lr         learning rate
   * @param {boolean} feedback  false reproduces the "compress the raw gradient" baseline
   */
  constructor(dimension, k, lr, feedback = true) {
    this.dimension = dimension;
    this.k = k;
    this.lr = lr;
    this.feedback = feedback;
    this.params = new Array(dimension).fill(0);
    this.residual = new Array(dimension).fill(0);
    this.updateCounts = new Array(dimension).fill(0);
    this.rounds = 0;
    this.transmitted = 0;
    this.transmittedMass = 0;
  }

  /** One communication round; returns the payload and the dense vector actually applied. */
  step(gradient) {
    const d = this.dimension;
    strictEqual(gradient.length, d, "gradient dimension mismatch");
    const effective = new Array(d);
    for (let i = 0; i < d; i++) effective[i] = gradient[i] + (this.feedback ? this.residual[i] : 0);
    const { indices, values } = topKSelect(effective, this.k);
    const dense = decompress(indices, values, d);
    for (const value of values) this.transmittedMass += Math.abs(value);
    for (let i = 0; i < d; i++) {
      if (dense[i] !== 0) this.updateCounts[i]++;
      this.params[i] -= this.lr * dense[i];
    }
    if (this.feedback) for (let i = 0; i < d; i++) this.residual[i] = effective[i] - dense[i];
    this.rounds++;
    this.transmitted += indices.length;
    return { indices, values, dense, effective };
  }

  /** Diagnostics only: fraction of coordinates that have ever been transmitted. */
  servedCoordinates() {
    return this.updateCounts.filter((c) => c > 0).length;
  }
}

const quadraticLoss = (params, weights, targets) => {
  let loss = 0;
  for (let i = 0; i < params.length; i++) {
    const error = params[i] - targets[i];
    loss += 0.5 * weights[i] * error * error;
  }
  return loss;
};

/** Deterministic orthonormal rows (Gram-Schmidt over rng draws) for a rotated quadratic. */
function orthonormalBasis(rank, rng) {
  const rows = [];
  for (let i = 0; i < rank; i++) {
    const row = Array.from({ length: rank }, () => rng() * 2 - 1);
    for (const previous of rows) {
      let projection = 0;
      for (let k = 0; k < rank; k++) projection += previous[k] * row[k];
      for (let k = 0; k < rank; k++) row[k] -= projection * previous[k];
    }
    const norm = Math.hypot(...row);
    rows.push(row.map((v) => v / norm));
  }
  return rows;
}

module.exports = {
  makeRng, topKSelect, topKBruteForce, decompress, ErrorFeedbackTrainer,
  payloadBytes, denseBytes, quadraticLoss, orthonormalBasis,
};

if (require.main === module) {
  const rng = makeRng(20260718);
  let bruteForceCompared = 0;

  // -- top-k against the brute-force oracle, including ties and all k boundaries --------
  for (let trial = 0; trial < 200; trial++) {
    const d = 1 + Math.floor(rng() * 12);
    const quantised = trial % 3 === 0;   // integer values force plenty of |value| ties
    const gradient = Array.from({ length: d }, () =>
      quantised ? Math.floor(rng() * 7) - 3 : rng() * 4 - 2);
    for (let k = -1; k <= d + 2; k++) {
      const got = topKSelect(gradient, k);
      const want = topKBruteForce(gradient, k);
      deepStrictEqual(got, want, `trial ${trial} k=${k} disagrees with the sort oracle`);
      bruteForceCompared++;
      strictEqual(got.indices.length, Math.max(0, Math.min(d, Math.trunc(k))),
        "payload must hold exactly clamp(k, 0, d) entries");
    }
  }

  // -- lossless boundary: k = d transmits the gradient exactly, k = 0 transmits nothing --
  const kd = new ErrorFeedbackTrainer(5, 5, 0.25);
  const g1 = [1, -2, 3, 0.5, -0.25];
  const r1 = kd.step(g1);
  deepStrictEqual(r1.dense, g1, "k = d must reconstruct the gradient losslessly");
  deepStrictEqual(kd.residual, [0, 0, 0, 0, 0], "a lossless round empties the memory buffer");
  deepStrictEqual(kd.params, g1.map((v) => -0.25 * v));
  strictEqual(payloadBytes(r1.indices.length), denseBytes(5) * (5 / 5) * 2,
    "lossless round costs 2x dense: value + index per entry");

  const k0 = new ErrorFeedbackTrainer(3, 0, 0.5);
  const r0 = k0.step([1, 2, 3]);
  deepStrictEqual(r0.dense, [0, 0, 0]);
  deepStrictEqual(k0.params, [0, 0, 0], "k = 0 must not move the model");
  deepStrictEqual(k0.residual, [1, 2, 3], "k = 0 must bank the whole gradient");
  strictEqual(k0.transmitted, 0);

  const empty = new ErrorFeedbackTrainer(0, 3, 0.5);
  const re = empty.step([]);
  deepStrictEqual(re.indices, []);
  strictEqual(empty.servedCoordinates(), 0);
  const negative = topKSelect([1, 2, 3], -2);
  deepStrictEqual(negative, { indices: [], values: [] }, "negative k clamps to zero");

  // -- the accumulator identity, cross-checked with an independent accumulator ----------
  const d = 6;
  const trainer = new ErrorFeedbackTrainer(d, 2, 0.05);
  const independentCarry = new Array(d).fill(0);   // sums the gradients of *ignored* coords
  const independentApplied = new Array(d).fill(0);
  const gradientTotal = new Array(d).fill(0);
  for (let round = 0; round < 60; round++) {
    const gradient = Array.from({ length: d }, () => rng() * 2 - 1);
    const transmitted = new Set();
    const payload = trainer.step(gradient);
    for (const i of payload.indices) transmitted.add(i);
    for (let i = 0; i < d; i++) {
      gradientTotal[i] += gradient[i];
      independentApplied[i] += payload.dense[i];
      // r_new = g_eff - decoded: zeroed by a successful transmission, otherwise the banked
      // sum of every gradient this coordinate has been denied so far.
      independentCarry[i] = transmitted.has(i) ? 0 : independentCarry[i] + gradient[i];
    }
    for (let i = 0; i < d; i++) {
      strictEqual(trainer.residual[i], independentCarry[i],
        `round ${round}: residual of coordinate ${i} must equal the independently banked sum`);
    }
  }
  for (let i = 0; i < d; i++) {
    ok(Math.abs(trainer.params[i] + 0.05 * independentApplied[i]) < 1e-12,
      `coordinate ${i}: params must equal -lr * sum(decoded) as accumulated by the test`);
  }
  // A final lossless round must deliver *all* banked gradient mass: sum_t decoded = sum_t g.
  trainer.k = d;
  const flush = trainer.step(new Array(d).fill(0));
  for (let i = 0; i < d; i++) independentApplied[i] += flush.dense[i];
  for (let i = 0; i < d; i++) {
    strictEqual(trainer.residual[i], 0, "the flush must empty the memory buffer");
    const scale = Math.max(1, Math.abs(gradientTotal[i]));
    ok(Math.abs(independentApplied[i] - gradientTotal[i]) <= 1e-12 * scale,
      `coordinate ${i}: error feedback must deliver the total gradient after a flush ` +
      `(got ${independentApplied[i]}, want ${gradientTotal[i]})`);
  }
  console.log(`top-k/sort-oracle agreements=${bruteForceCompared} accumulator-coords=${d}`);

  // -- (a) skewed *constant* gradient: the baseline starves, error feedback does not -----
  const skew = [4, 1, 0.5, 0.25];
  const rounds = 40;
  const withFeedback = new ErrorFeedbackTrainer(skew.length, 1, 0.1, true);
  const withoutFeedback = new ErrorFeedbackTrainer(skew.length, 1, 0.1, false);
  for (let round = 0; round < rounds; round++) {
    withFeedback.step(skew);
    withoutFeedback.step(skew);
  }
  strictEqual(withoutFeedback.servedCoordinates(), 1,
    "plain top-k on a constant skewed gradient can only ever serve the largest coordinate");
  strictEqual(withFeedback.servedCoordinates(), skew.length,
    "error feedback must eventually serve every coordinate");
  const counts = withFeedback.updateCounts;
  ok(counts.every((c) => c > 0), "no coordinate may be starved");
  ok(counts.every((c, i) => i === 0 || counts[i - 1] >= c),
    "services must be ordered by gradient magnitude (bigger gradient wins more often)");
  strictEqual(counts.reduce((a, b) => a + b, 0), rounds, "one service per round with k = 1");
  // Final flush: the feedback trainer has banked everything and can deliver the whole mass.
  withFeedback.k = skew.length;
  withFeedback.step(new Array(skew.length).fill(0));
  const expectedMass = rounds * skew.reduce((a, c) => a + Math.abs(c), 0);
  strictEqual(withFeedback.transmittedMass, expectedMass,
    "with error feedback a flush delivers the complete accumulated gradient mass");
  strictEqual(withoutFeedback.transmittedMass, rounds * Math.abs(skew[0]),
    "the baseline permanently drops every gradient it never selected");
  const improvement = (t) => -skew.reduce((acc, c, i) => acc + c * t.params[i], 0);
  ok(improvement(withFeedback) > improvement(withoutFeedback),
    `objective gain: feedback=${improvement(withFeedback)} baseline=${improvement(withoutFeedback)}`);
  console.log(`served coordinates: feedback=${withFeedback.servedCoordinates()} baseline=${withoutFeedback.servedCoordinates()}` +
    ` mass: feedback=${withFeedback.transmittedMass} baseline=${withoutFeedback.transmittedMass} of ${expectedMass}`);
  deepStrictEqual(withoutFeedback.residual, [0, 0, 0, 0], "the baseline keeps no memory at all");
  for (let i = 1; i < skew.length; i++) {
    strictEqual(withoutFeedback.params[i], 0, `baseline never touched coordinate ${i}`);
  }
  const delivered = (t) => t.transmittedMass;
  ok(delivered(withFeedback) > delivered(withoutFeedback),
    "feedback transmits more gradient mass than the starved baseline");

  // -- (b) rotated quadratic: same budget, markedly smaller loss with feedback -----------
  const rngB = makeRng(718718);          // independent stream: this test does not depend on
  const dim = 8;                          // how many draws the earlier checks consumed
  const targets = Array.from({ length: dim }, () => rngB() * 2 - 1);
  const weights = [16, 8, 4, 2, 1, 0.5, 0.25, 0.125];
  const basis = orthonormalBasis(dim, rngB);
  const applyA = (vector) => {
    const out = new Array(dim).fill(0);
    for (let j = 0; j < dim; j++) {
      let projection = 0;
      for (let k = 0; k < dim; k++) projection += basis[j][k] * vector[k];
      projection *= weights[j];
      for (let k = 0; k < dim; k++) out[k] += projection * basis[j][k];
    }
    return out;
  };
  const gradientOf = (params) => applyA(params.map((p, i) => p - targets[i]));
  /** f(x) = 0.5 * e^T A e with e = x - target, evaluated through the explicit A-apply. */
  const rotatedLoss = (params) => {
    const error = params.map((p, i) => p - targets[i]);
    const ae = applyA(error);
    let loss = 0;
    for (let i = 0; i < dim; i++) loss += 0.5 * error[i] * ae[i];
    return loss;
  };
  const efTrainer = new ErrorFeedbackTrainer(dim, 2, 0.05, true);
  const plainTrainer = new ErrorFeedbackTrainer(dim, 2, 0.05, false);
  let efLossAtQuarter = 0;
  for (let round = 0; round < 400; round++) {
    efTrainer.step(gradientOf(efTrainer.params));
    plainTrainer.step(gradientOf(plainTrainer.params));
    if (round === 99) efLossAtQuarter = rotatedLoss(efTrainer.params);
  }
  const efLoss = rotatedLoss(efTrainer.params);
  const plainLoss = rotatedLoss(plainTrainer.params);
  ok(efLoss < 1e-5, `feedback should reach the minimiser, got loss ${efLoss}`);
  ok(plainLoss > 1e-5, `the starved baseline should not reach the target loss, got ${plainLoss}`);
  ok(efLoss * 100 < plainLoss, `feedback loss ${efLoss} must be two orders below ${plainLoss}`);
  ok(efLossAtQuarter > efLoss, "the feedback run must still be making progress at the end");
  ok(efTrainer.transmittedMass > plainTrainer.transmittedMass,
    "error feedback shifts strictly more gradient mass across the wire");
  ok(Number.isFinite(plainLoss) && Number.isFinite(efLoss), "both runs must stay finite");
  console.log(`rotated quadratic: feedback=${efLoss.toExponential(3)} baseline=${plainLoss.toExponential(3)}`);
  console.log(`wire bytes/round: top-2=${payloadBytes(2)} dense=${denseBytes(dim)}`);
  console.log("PASS");
}
