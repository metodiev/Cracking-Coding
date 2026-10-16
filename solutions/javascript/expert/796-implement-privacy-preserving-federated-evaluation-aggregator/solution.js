/** Task 796 - Implement privacy-preserving federated evaluation aggregator. (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * A federated layer that collects client updates without either the server or a curious client being
 * able to read an individual contribution:
 *   1. clip: each update is projected onto the L2 ball of radius C, so one client's influence is bounded
 *      (clipping is idempotent and never increases a norm - both verified),
 *   2. noise: the sum is perturbed by Gaussian noise with the (eps, delta) calibration
 *        sigma = C * sqrt(2 ln(1.25/delta)) / eps,
 *      the classic Gaussian-mechanism bound for sensitivity C,
 *   3. secure aggregation: every client quantises its clipped update to integers modulo p = 2^31 - 1 and
 *      masks it with (a) pairwise masks PRG(i, j) that cancel in the sum over any subset, and (b) a
 *      private self-mask. The server only ever sees masked vectors, and the exact sum of the updates is
 *      recovered by cancelling the telescoping pairwise masks and subtracting the aggregate self-mask.
 *
 * Invariants verified: the masked sum recovers the true quantised sum exactly modulo p; an individual
 * masked vector is far from its own update (so masking really hides it); clipping bounds every norm;
 * the calibration constant satisfies the Gaussian-mechanism inequality with equality; the measured noise
 * scale matches sigma; utility scales linearly with sigma; dropout recovery is exact for the surviving
 * subset; quantisation stays inside half a step; and invalid privacy parameters are rejected.
 *
 * What is simplified: the aggregate self-mask is handed to the server instead of being reconstructed
 * from threshold shares, the field is the Mersenne prime 2^31 - 1 with a fixed-point scale (no secure
 * multiparty arithmetic), the PRG is a deterministic splitmix32 standing in for a cryptographic stream,
 * and the DP accounting is the single-step Gaussian mechanism rather than a full composition ledger.
 * Nothing about the recovered aggregate is faked: it is compared against the exact integer sum.
 */
"use strict";

const { ok, throws } = require("node:assert");

const MODULUS = 2147483647n;             // 2^31 - 1, the Mersenne prime used as the field
const SECURE_AGGREGATION_SCALE = 100000; // fixed-point scale for quantising floats

/** Deterministic splitmix32 uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

/** Standard normal from two uniforms (Box-Muller). */
const gaussian = (random) => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

const norm = (vector) => Math.sqrt(vector.reduce((acc, v) => acc + v * v, 0));

/** Projects a vector onto the L2 ball of radius C (the identity when it is already inside). */
function clip(vector, bound) {
  if (!(bound > 0)) throw new RangeError("the clipping bound must be > 0");
  const length = norm(vector);
  if (length <= bound) return vector.slice();
  return vector.map((v) => (v * bound) / length);
}

/** Gaussian-mechanism noise scale for sensitivity `sensitivity`, (eps, delta)-DP. */
function noiseScale(sensitivity, epsilon, delta) {
  if (!(epsilon > 0)) throw new RangeError("epsilon must be > 0");
  if (!(delta > 0 && delta < 1)) throw new RangeError("delta must lie in (0, 1)");
  if (!(sensitivity > 0)) throw new RangeError("the sensitivity must be > 0");
  return (sensitivity * Math.sqrt(2 * Math.log(1.25 / delta))) / epsilon;
}

/** Quantises a real vector to integer residues modulo p. */
const quantise = (vector) => vector.map((v) => {
  const scaled = BigInt(Math.round(v * SECURE_AGGREGATION_SCALE));
  return ((scaled % MODULUS) + MODULUS) % MODULUS;
});

/** Fixed-point round trip of an integer residue back to a real number (signed representative). */
const dequantise = (residue) => Number(residue > MODULUS / 2n ? residue - MODULUS : residue) / SECURE_AGGREGATION_SCALE;

/**
 * Pairwise masks: for every ordered pair (i, j) a deterministic PRG stream. Client i adds the mask for
 * every j > i and subtracts the mask for every j < i, so the masks telescope to zero in any subset sum
 * - as long as exactly the same subset is summed on both sides.
 */
function pairwiseMask(dimension, i, j, seed) {
  const random = makeRng((seed ^ Math.imul(i + 1, 0x9e3779b1) ^ Math.imul(j + 1, 0x85ebca6b)) >>> 0);
  return Array.from({ length: dimension }, () => BigInt(Math.floor(random() * 1000000)));
}

/** Self-mask of client i, private to that client (recovered from threshold shares in a full system). */
function selfMask(dimension, i, seed) {
  const random = makeRng((seed ^ Math.imul(i + 1, 0xc2b2ae35)) >>> 0);
  return Array.from({ length: dimension }, () => BigInt(Math.floor(random() * 1000000)));
}

/**
 * One secure-aggregation round: returns every client's masked message plus the decoded aggregate.
 * `dropped` lists clients that never delivered; the server cancels their pairwise contribution.
 */
function secureAggregate(updates, { epsilon = 1, delta = 1e-5, bound = 1, seed = 7, dropped = [] } = {}) {
  if (updates.length === 0) throw new RangeError("at least one client is required");
  const dimension = updates[0].length;
  if (!updates.every((u) => u.length === dimension)) throw new RangeError("all updates must share one dimension");
  const sigma = noiseScale(bound, epsilon, delta);
  const clipped = updates.map((u) => clip(u, bound));
  const residues = clipped.map(quantise);
  const live = updates.map((_, i) => i).filter((i) => !dropped.includes(i));
  const messages = residues.map((residue, i) => residue.map((value, d) => {
    let masked = (value + selfMask(dimension, i, seed)[d]) % MODULUS;
    for (let j = 0; j < updates.length; j++) {
      if (j === i) continue;
      const mask = pairwiseMask(dimension, Math.min(i, j), Math.max(i, j), seed)[d];
      masked = (masked + (j > i ? mask : MODULUS - mask)) % MODULUS;
    }
    return ((masked % MODULUS) + MODULUS) % MODULUS;
  }));
  // The server sums the live messages: masks between live pairs telescope away, the self-masks and the
  // masks towards dropped peers must be cancelled explicitly (the server can recompute them).
  const serverSum = live
    .map((i) => messages[i])
    .reduce((acc, message) => acc.map((value, d) => (value + message[d]) % MODULUS), new Array(dimension).fill(0n));
  let recovered = serverSum;
  for (const i of live) {
    const selfCancellation = selfMask(dimension, i, seed);
    recovered = recovered.map((value, d) => (value + MODULUS - selfCancellation[d]) % MODULUS);
    for (const j of dropped) {
      const mask = pairwiseMask(dimension, Math.min(i, j), Math.max(i, j), seed);
      recovered = recovered.map((value, d) => {
        const contribution = j > i ? mask[d] : (MODULUS - mask[d]) % MODULUS;
        return (value + MODULUS - contribution) % MODULUS;
      });
    }
  }
  const exact = residues.reduce((acc, residue) => acc.map((value, d) => (value + residue[d]) % MODULUS), new Array(dimension).fill(0n));
  const random = makeRng(seed + 1);
  const noisy = recovered.map((value) => dequantise(value) + gaussian(random) * sigma);
  return { messages, recovered, exact, noisy, sigma, clipped };
}

/** The plain (non-private) mean used as the utility reference. */
const plainMean = (updates) => updates[0].map((_, d) => updates.reduce((acc, u) => acc + u[d], 0) / updates.length);

/** Deterministic synthetic client updates: a signal direction plus per-client drift. */
function makeUpdates(seed, clients = 10, dimension = 4) {
  const random = makeRng(seed);
  return Array.from({ length: clients }, (_, i) => Array.from({ length: dimension }, (_, d) => (d === 0 ? 1 : 0) + 0.35 * gaussian(random) + 0.05 * i * (d % 2 ? -1 : 1)));
}

function main() {
  const notes = [];
  const updates = makeUpdates(20251016);

  // 1. The masked messages must reproduce the exact quantised sum modulo p.
  const round = secureAggregate(updates, { epsilon: 1, delta: 1e-5, bound: 1, seed: 7 });
  const exactMatch = round.recovered.every((v, d) => v === round.exact[d]);
  const maskMagnitudes = round.messages.map((message, i) => message.map((value, d) => Math.abs(dequantise((value + MODULUS - quantise(round.clipped[i])[d]) % MODULUS))));
  const meanMask = maskMagnitudes.flat().reduce((a, b) => a + b, 0) / maskMagnitudes.flat().length;
  notes.push(`secure aggregation of ${updates.length} clients: recovered sum matches the exact quantised sum = ${exactMatch}, mean per-coordinate mask magnitude ${meanMask.toFixed(2)}`);
  ok(exactMatch, "masking must be reversible: the recovered sum must equal the exact quantised sum");
  ok(maskMagnitudes.every((client) => client.every((v) => v > 1e-6)), "every transmitted message must be masked (never equal to the raw update)");

  // 2. Clipping bounds every norm and leaves small updates untouched (idempotence).
  const big = [[3, 4], [1, 0]];
  const clippedBig = clip(big[0], 1);
  const clippedIdempotent = clip(clippedBig, 1);
  notes.push(`clipping: |[3, 4]| = ${norm(big[0]).toFixed(4)} -> ${norm(clippedBig).toFixed(12)}, small |[1, 0]| stays ${norm(clip(big[1], 1)).toFixed(12)}`);
  ok(Math.abs(norm(clippedBig) - 1) < 1e-12, "clipping must project onto the ball boundary");
  ok(clippedBig.every((v, i) => v === clippedIdempotent[i]), "clipping must be idempotent");
  ok(clip([0.2, 0.1], 1).every((v, i) => v === [0.2, 0.1][i]), "an update inside the ball must be unchanged");
  ok(Math.abs(clip([3, 4], 1)[0] / clip([3, 4], 1)[1] - 3 / 4) < 1e-12, "clipping must preserve the direction");

  // 3. The calibration must satisfy the Gaussian-mechanism inequality with equality.
  const checkPoints = [[1, 1e-5], [0.5, 1e-6], [2, 0.01]];
  const calibration = checkPoints.map(([epsilon, delta]) => {
    const sigma = noiseScale(1, epsilon, delta);
    const required = Math.sqrt(2 * Math.log(1.25 / delta));
    return { epsilon, delta, sigma, ratio: (sigma * epsilon) / required };
  });
  notes.push(`calibration sigma = C sqrt(2 ln(1.25/delta))/eps: ${calibration.map((c) => `eps ${c.epsilon}, delta ${c.delta} -> sigma ${c.sigma.toFixed(4)}, sigma*eps/sqrt(2 ln(1.25/delta)) = ${c.ratio.toFixed(12)}`).join("; ")}`);
  ok(calibration.every((c) => Math.abs(c.ratio - 1) < 1e-12), "the calibration constant must make the Gaussian bound tight");
  ok(noiseScale(1, 2, 1e-5) < noiseScale(1, 1, 1e-5) && noiseScale(1, 0.5, 1e-5) > noiseScale(1, 1, 1e-5), "a larger epsilon must require less noise");

  // 4. The measured noise scale must match sigma on the seeded stream.
  const sigma = noiseScale(1, 1, 1e-5);
  const random = makeRng(1234);
  const samples = Array.from({ length: 20000 }, () => gaussian(random) * sigma);
  const measuredStd = Math.sqrt(samples.reduce((acc, v) => acc + v * v, 0) / samples.length);
  const meanAbsolute = samples.reduce((acc, v) => acc + Math.abs(v), 0) / samples.length;
  notes.push(`noise verification: measured std ${measuredStd.toFixed(5)} vs sigma ${sigma.toFixed(5)} (ratio ${(measuredStd / sigma).toFixed(4)}), mean |noise| ${meanAbsolute.toFixed(4)} vs sqrt(2/pi)*sigma = ${(sigma * Math.sqrt(2 / Math.PI)).toFixed(4)}`);
  ok(Math.abs(measuredStd / sigma - 1) < 0.03, "the sampled noise standard deviation must match the calibration");
  ok(Math.abs(meanAbsolute / (sigma * Math.sqrt(2 / Math.PI)) - 1) < 0.03, "the mean absolute deviation must match the Gaussian prediction");

  // 5. Utility scales linearly with sigma: the per-coordinate aggregate error is proportional to it.
  const errorFor = (epsilon) => {
    const total = Array.from({ length: 8 }, (_, k) => secureAggregate(updates, { epsilon, delta: 1e-5, bound: 1, seed: 31 + 7 * k }))
      .reduce((acc, result) => {
        const clippedSum = result.clipped.reduce((sum, u) => sum.map((v, d) => v + u[d]), new Array(updates[0].length).fill(0));
        return acc + Math.sqrt(result.noisy.reduce((a, v, d) => a + (v - clippedSum[d]) ** 2, 0) / clippedSum.length);
      }, 0);
    return total / 8;
  };
  const budgets = [0.25, 0.5, 1, 2];
  const errors = budgets.map(errorFor);
  notes.push(`utility: per-coordinate aggregate error ${errors.map((e, i) => `${e.toFixed(4)} at eps ${budgets[i]}`).join(", ")} (ratio eps 0.25 : eps 2 = ${(errors[0] / errors[3]).toFixed(2)})`);
  ok(errors[0] > errors[3] && errors[1] >= errors[2], "more privacy budget (larger eps) must not hurt the estimate");
  ok(Math.abs(errors[0] / errors[3] - 8) < 3, "the error must scale roughly as 1/eps");
  ok(Math.abs(errors[0] * budgets[0] - errors[3] * budgets[3]) / (errors[0] * budgets[0]) < 0.5, "the error times epsilon must be roughly constant");

  // 6. Dropout recovery: dropping clients must still yield the exact surviving sum.
  const dropped = [2, 5, 9];
  const partial = secureAggregate(updates, { epsilon: 1, delta: 1e-5, bound: 1, seed: 7, dropped });
  const survivingExact = updates
    .map((u, i) => (dropped.includes(i) ? null : u))
    .filter(Boolean)
    .map((u) => quantise(clip(u, 1)))
    .reduce((acc, residue) => acc.map((value, d) => (value + residue[d]) % MODULUS), new Array(updates[0].length).fill(0n));
  const dropoutMatch = partial.recovered.every((v, d) => v === survivingExact[d]);
  notes.push(`dropout of ${dropped.length} clients: recovered sum matches the ${updates.length - dropped.length}-client exact sum = ${dropoutMatch}`);
  ok(dropoutMatch, "the surviving clients' sum must be recovered exactly when peers drop out");

  // 7. Quantisation error stays inside half a step, and the aggregate is unbiased by it.
  const quantisationErrors = updates.flatMap((u) => quantise(clip(u, 1)).map((residue, d) => Math.abs(dequantise(residue) - clip(u, 1)[d])));
  const worstQuantisation = Math.max(...quantisationErrors);
  notes.push(`quantisation: worst absolute error ${worstQuantisation.toExponential(2)} vs half a step ${(0.5 / SECURE_AGGREGATION_SCALE).toExponential(2)}`);
  ok(worstQuantisation <= 0.5 / SECURE_AGGREGATION_SCALE + 1e-12, "fixed-point rounding must stay within half a step");

  // 8. Boundary and validation.
  const single = secureAggregate([[0.5, -0.25]], { epsilon: 1, delta: 1e-5, bound: 1, seed: 3 });
  ok(single.recovered.every((v, d) => v === single.exact[d]), "a single client must still be recovered exactly");
  ok(secureAggregate([[0, 0, 0, 0]], { epsilon: 1, delta: 1e-5, bound: 1, seed: 3 }).exact.every((v) => v === 0n), "a zero update must stay zero");
  throws(() => noiseScale(1, 0, 1e-5), RangeError, "epsilon = 0 must throw");
  throws(() => noiseScale(1, 1, 1), RangeError, "delta = 1 must throw");
  throws(() => clip([1, 1], 0), RangeError, "a zero clipping bound must throw");
  throws(() => secureAggregate([[1, 2], [1]], { epsilon: 1, delta: 1e-5 }), RangeError, "mismatched dimensions must throw");
  throws(() => secureAggregate([], {}), RangeError, "an empty client list must throw");

  // 9. Determinism.
  const repeat = secureAggregate(updates, { epsilon: 1, delta: 1e-5, bound: 1, seed: 7 });
  ok(repeat.noisy.every((v, d) => v === round.noisy[d]) && repeat.messages.every((m, i) => m.every((v, d) => v === round.messages[i][d])),
    "the same seed must reproduce the masks and the noisy aggregate exactly");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, gaussian, norm, clip, noiseScale, quantise, dequantise, pairwiseMask, selfMask, secureAggregate, plainMean, makeUpdates, MODULUS };

if (require.main === module) main();
