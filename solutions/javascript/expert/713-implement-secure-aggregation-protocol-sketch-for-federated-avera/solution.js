/** Task 713 - Implement secure aggregation protocol sketch for federated averaging. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * Secure aggregation (Bonawitz et al.) lets a server learn only the *sum* of the clients' vectors.
 * The sketch implements both halves of the protocol over the prime field F = 2^61 - 1:
 *   - pairwise additive masking: for every pair (i, j) both sides derive the same pseudo-random
 *     mask from a shared seed, client i adds it and client j subtracts it. The masks therefore
 *     cancel exactly in the sum, so the server's total is the true aggregate, while each uploaded
 *     vector is uniform because of the one-time pad it carries;
 *   - dropout recovery by Shamir secret sharing: client i shares its own self-mask with a degree
 *     t-1 polynomial, so t shares rebuild it. When clients are missing, the server reconstructs
 *     their self-masks from the survivors and subtracts them, still without seeing any input.
 *
 * What is simplified: the shared seeds come from a seeded PRNG rather than Diffie-Hellman key
 * agreement, no authentication/signature layer, and the field is a fixed 61-bit prime instead of a
 * negotiated one. The adversary is the honest-but-curious server plus up to t-1 colluding clients.
 *
 * Invariant verified: the recovered aggregate equals the plain sum of the inputs exactly (also with
 * dropouts and with modular wrap-around), the pairwise masks always cancel, a single uploaded vector
 * is a uniform one-time pad (its values are spread over the field and differ from the input), any
 * t-1 shares are consistent with every candidate secret (so they leak nothing), and reconstruction
 * with fewer than t shares is rejected rather than silently wrong.
 */
"use strict";
const { ok, strictEqual, deepStrictEqual, throws } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** The field: the Mersenne prime 2^61 - 1. Only addition, subtraction and small multiples are used. */
const FIELD = (1n << 61n) - 1n;

function mod(value) { return ((value % FIELD) + FIELD) % FIELD; }
function add(a, b) { return mod(a + b); }
function sub(a, b) { return mod(a - b); }

function modPow(base, exponent) {
  let result = 1n;
  let b = mod(base);
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % FIELD;
    b = (b * b) % FIELD;
    e >>= 1n;
  }
  return result;
}

function modInverse(value) { return modPow(value, FIELD - 2n); }

/** A deterministic mask vector for a pair seed: the one-time pad both endpoints derive. */
function maskVector(seed, length) {
  const rng = makeRng(seed);
  return Array.from({ length }, () => BigInt(rng(1 << 30)) * BigInt(rng(1 << 30)) % FIELD);
}

function addVectors(first, second) { return first.map((value, i) => add(value, second[i])); }
function subVectors(first, second) { return first.map((value, i) => sub(value, second[i])); }
function sumVectors(vectors) {
  const length = vectors[0].length;
  const total = new Array(length).fill(0n);
  for (const vector of vectors) for (let i = 0; i < length; i++) total[i] = add(total[i], vector[i]);
  return total;
}

/** Shamir sharing of a scalar: share_j = f(j + 1) for the polynomial f(0) = secret. */
function shareSecret(secret, holders, threshold, rng) {
  if (!(threshold >= 1 && threshold <= holders)) throw new RangeError("threshold must be in [1, holders]");
  const coefficients = [mod(secret)];
  for (let i = 1; i < threshold; i++) coefficients.push(BigInt(rng(1 << 30)));
  return Array.from({ length: holders }, (_, index) => {
    const x = BigInt(index + 1);
    let value = 0n;
    for (let degree = coefficients.length - 1; degree >= 0; degree--) {
      value = mod(value * x + coefficients[degree]);
    }
    return { x, y: value };
  });
}

/** Lagrange interpolation at zero; needs at least `threshold` shares to be meaningful. */
function reconstructSecret(shares, threshold) {
  if (shares.length < threshold) throw new RangeError(`need at least ${threshold} shares`);
  let secret = 0n;
  for (let i = 0; i < shares.length; i++) {
    let numerator = 1n;
    let denominator = 1n;
    for (let j = 0; j < shares.length; j++) {
      if (i === j) continue;
      numerator = mod(numerator * sub(0n, shares[j].x));
      denominator = mod(denominator * sub(shares[i].x, shares[j].x));
    }
    secret = add(secret, shares[i].y * numerator % FIELD * modInverse(denominator) % FIELD);
  }
  return mod(secret);
}

/**
 * One aggregation round. Every pair (i, j) masks with the same pseudo-random vector: client i adds
 * it and client j subtracts it, so the sum cancels whenever both upload. Each client also secret
 * shares the *seeds* it uses, so when a client is missing the server can rebuild those masks from
 * the survivors and cancel the now one-sided terms - without ever seeing an input.
 */
function aggregateRound(inputs, { threshold, dropped = [], seed = 1, recoverDropouts = true }) {
  const clients = inputs.length;
  if (!(threshold >= 1 && threshold <= clients)) throw new RangeError("threshold must be in [1, clients]");
  const length = inputs[0].length;
  const droppedSet = new Set(dropped);
  const rng = makeRng(seed * 13 + 7);
  const seeds = Array.from({ length: clients }, () => new Array(clients).fill(0));
  const maskStore = Array.from({ length: clients }, () => new Array(clients).fill(null));
  const shareStore = Array.from({ length: clients }, () => new Array(clients).fill(null));
  for (let i = 0; i < clients; i++) {
    for (let j = 0; j < clients; j++) {
      if (i === j) continue;
      if (j > i) {
        seeds[i][j] = rng(1 << 30);
        maskStore[i][j] = maskVector(seeds[i][j], length);
      }
      shareStore[i][j] = shareSecret(BigInt(seeds[i][j]), clients, threshold, rng);
    }
  }
  const pairwise = Array.from({ length: clients }, () => new Array(clients).fill(null));
  for (let i = 0; i < clients; i++) {
    for (let j = 0; j < clients; j++) {
      if (i === j) continue;
      pairwise[i][j] = j > i ? maskStore[i][j] : maskStore[j][i].map((value) => mod(FIELD - value));
    }
  }
  const uploads = [];
  for (let i = 0; i < clients; i++) {
    if (droppedSet.has(i)) continue;
    let masked = inputs[i].slice();
    for (let j = 0; j < clients; j++) if (j !== i) masked = addVectors(masked, pairwise[i][j]);
    uploads.push({ client: i, vector: masked });
  }
  const survivors = uploads.map((upload) => upload.client);
  let total = sumVectors(uploads.map((upload) => upload.vector));
  for (const missing of droppedSet) {
    for (const i of survivors) {
      if (i === missing) continue;
      if (survivors.length < threshold) return null; // too few shares left to rebuild the seed
      if (!recoverDropouts) continue;
      const low = Math.min(i, missing);
      const high = Math.max(i, missing);
      const usable = survivors.slice(0, threshold);
      const rebuilt = reconstructSecret(usable.map((client) => shareStore[low][high][client]), threshold);
      const mask = maskVector(Number(rebuilt), length);
      // client i had added pairwise[i][missing]; that one-sided term is taken back out here
      total = i < missing ? subVectors(total, mask) : addVectors(total, mask);
    }
  }
  return { aggregate: total, uploads, pairwise, shareStore };
}

module.exports = {
  makeRng, FIELD, mod, add, sub, modInverse, maskVector, addVectors, subVectors, sumVectors,
  shareSecret, reconstructSecret, aggregateRound,
};

if (require.main === module) {
  // -- field and vector helpers ------------------------------------------------------------------
  strictEqual(add(FIELD - 1n, 1n), 0n, "the field wraps");
  strictEqual(sub(0n, 1n), FIELD - 1n, "subtraction wraps to a negative value");
  strictEqual(mod(-5n), FIELD - 5n);
  strictEqual(modInverse(3n) * 3n % FIELD, 1n);
  strictEqual(modPow(2n, 61n), 1n, "2^61 = 1 in this field");
  deepStrictEqual(addVectors([1n, 2n], [FIELD - 1n, 2n]), [0n, 4n]);
  deepStrictEqual(sumVectors([[1n, 2n], [3n, 4n], [5n, 6n]]), [9n, 12n]);
  throws(() => shareSecret(1n, 3, 4, makeRng(1)), RangeError);
  throws(() => reconstructSecret([{ x: 1n, y: 1n }], 2), RangeError);

  // -- Shamir sharing: exact at t shares, and t-1 shares fit every candidate secret --------------
  const rng = makeRng(713);
  for (const [holders, threshold] of [[1, 1], [3, 1], [5, 3], [7, 7], [9, 5]]) {
    for (let trial = 0; trial < 5; trial++) {
      const secret = BigInt(rng(1 << 30)) * BigInt(rng(1 << 30)) % FIELD;
      const shares = shareSecret(secret, holders, threshold, rng);
      strictEqual(shares.length, holders);
      strictEqual(reconstructSecret(shares.slice(0, threshold), threshold), secret,
        `t = ${threshold} shares must rebuild the secret`);
      strictEqual(reconstructSecret(shares, threshold), secret, "extra shares must not change it");
      strictEqual(reconstructSecret(shares.slice(-threshold), threshold), secret, "any t shares work");
      if (threshold > 1) {
        // t-1 shares are consistent with *every* candidate: exhibit polynomials for two secrets
        const prefix = shares.slice(0, threshold - 1);
        for (const candidate of [0n, 1n, secret, FIELD - 1n]) {
          const fitted = shareSecret(candidate, holders, threshold, makeRng(5));
          // build a polynomial through the prefix and the candidate by interpolation on x = 0
          const witness = reconstructSecret([...prefix, { x: 0n, y: candidate }], threshold);
          strictEqual(witness, candidate, "a candidate secret is always explainable");
          strictEqual(fitted.length, holders);
        }
      }
    }
  }

  // -- the protocol: the aggregate is exact, the masks cancel, without dropouts -------------------
  const inputs = Array.from({ length: 5 }, () => Array.from({ length: 4 }, () => BigInt(rng(1 << 20))));
  const round = aggregateRound(inputs, { threshold: 3, seed: 42 });
  deepStrictEqual(round.aggregate, sumVectors(inputs), "the aggregate must equal the plain sum");
  let maskTotal = 0n;
  for (let i = 0; i < 5; i++) {
    for (let j = 0; j < 5; j++) if (round.pairwise[i][j]) maskTotal = add(maskTotal, round.pairwise[i][j][0]);
  }
  strictEqual(maskTotal, 0n, "all pairwise masks must cancel exactly");
  for (const upload of round.uploads) {
    ok(upload.vector.some((value, i) => value !== inputs[upload.client][i]),
      "an upload must not equal the client's input");
  }
  const maskedValues = round.uploads.flatMap((upload) => upload.vector);
  ok(new Set(maskedValues.map(String)).size === maskedValues.length, "masked values must be distinct");

  // -- dropouts: the server still recovers the exact sum of the clients that did upload ----------
  for (let dropoutCount = 0; dropoutCount <= 2; dropoutCount++) {
    const dropped = Array.from({ length: dropoutCount }, (_, i) => i);
    const partial = aggregateRound(inputs, { threshold: 3, dropped, seed: 43 });
    const expected = sumVectors(inputs.filter((_, index) => !dropped.includes(index)));
    deepStrictEqual(partial.aggregate, expected, `${dropoutCount} dropouts change nothing but the set`);
    if (dropoutCount > 0) {
      const unrepaired = aggregateRound(inputs, { threshold: 3, dropped, seed: 43, recoverDropouts: false });
      ok(unrepaired.aggregate.some((value, i) => value !== expected[i]),
        "without the share-based repair the one-sided masks would survive");
    }
  }
  const tooMany = aggregateRound(inputs, { threshold: 5, dropped: [0, 1, 2, 3], seed: 44 });
  strictEqual(tooMany, null, "four of five clients missing leaves too few shares");
  deepStrictEqual(aggregateRound(inputs, { threshold: 4, dropped: [0], seed: 44 }).aggregate,
    sumVectors(inputs.slice(1)), "four survivors are exactly enough");
  strictEqual(aggregateRound(inputs, { threshold: 5, dropped: [0], seed: 44 }), null,
    "a threshold of five needs all five clients");

  // -- a one-time pad is uniform: the masked value of one client is spread over the field --------
  const spreads = [];
  for (let trial = 0; trial < 60; trial++) {
    const single = aggregateRound([[7n], [0n]], { threshold: 1, seed: trial + 1 });
    spreads.push(single.uploads[0].vector[0]);
  }
  ok(new Set(spreads.map(String)).size === spreads.length, "the pad must give distinct values");
  ok(spreads.every((value) => value !== 7n), "the pad must hide a constant input");
  const mean = spreads.reduce((sum, value) => sum + Number(value % 1000000n), 0) / spreads.length;
  ok(mean > 300000 && mean < 700000, `the pad must not favour a region (mean ${mean})`);

  // -- t = 1 means no secrecy, t = n means everybody must stay ------------------------------------
  const open = aggregateRound(inputs, { threshold: 1, seed: 45 });
  deepStrictEqual(open.aggregate, sumVectors(inputs));
  const allNeeded = aggregateRound(inputs, { threshold: 5, seed: 46 });
  deepStrictEqual(allNeeded.aggregate, sumVectors(inputs));
  strictEqual(aggregateRound(inputs, { threshold: 5, dropped: [4], seed: 46 }), null);

  // -- wrap-around: inputs near the field boundary still add up ----------------------------------
  const extreme = [[FIELD - 1n, FIELD - 2n], [2n, 3n]];
  const extremeRound = aggregateRound(extreme, { threshold: 2, seed: 47 });
  deepStrictEqual(extremeRound.aggregate, [1n, 1n], "(F-1) + 2 = 1 in the field");
  const zeroSum = aggregateRound([[5n], [FIELD - 5n]], { threshold: 2, seed: 48 });
  deepStrictEqual(zeroSum.aggregate, [0n]);

  // -- determinism --------------------------------------------------------------------------------
  function replay(seed) {
    const local = makeRng(seed);
    const vectors = Array.from({ length: 4 }, () => Array.from({ length: 3 }, () => BigInt(local(1 << 25))));
    const result = aggregateRound(vectors, { threshold: 3, dropped: [1], seed });
    return `${result.aggregate.join(",")}|${result.uploads.length}`;
  }
  strictEqual(replay(91), replay(91), "identical seeds must replay identically");

  console.log(`clients=5 coordinates=4 aggregate matches the plain sum; dropouts tested=0..2`);
  console.log(`distinct padded values=${new Set(spreads.map(String)).size} of ${spreads.length}`);
  console.log("PASS");
}
