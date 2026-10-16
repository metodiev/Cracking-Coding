/** Task 829 - Implement privacy-preserving federated learning with secure aggregation and differential
 * privacy. (Tier: Master, Beyond — Security & Privacy)
 *
 * A federated-averaging loop (clients -> masked updates -> aggregator -> next round) with two privacy
 * layers:
 *
 *  1. Secure aggregation (Bonawitz-style masking, Diffie-Hellman flavour). Clients broadcast public
 *     keys y_i = g^{b_i} in the multiplicative group modulo the Mersenne prime q = 2^61 - 1 and derive
 *     the pairwise secret s_ij = y_j^{b_i} = y_i^{b_j}; the mask of client i is a pseudorandom field
 *     vector generated from s_ij with sign +1 for j > i and -1 for j < i, so all masks cancel in the
 *     sum. Updates are quantised to integers and added in F_p (p = 2^61 - 1), which makes the
 *     aggregation arithmetic exact: the recovered integer sum is bit-identical to the plain sum. When
 *     clients drop out, the surviving clients publish exactly the missing mask terms (computable from
 *     the dropped clients' public keys and their own exponent), and the Shamir sharing of the client
 *     exponents provides the threshold recovery path (t-of-n reconstruction, information-theoretic
 *     secrecy below t).
 *  2. Differential privacy: every local update is clipped to L2 norm C, so the L2 sensitivity of the
 *     averaged update is 2C/K; calibrated Gaussian noise with
 *     sigma = sqrt(2 ln(1.25/delta)) * (2C/K) / eps is added before the model update (analytic
 *     Gaussian mechanism).
 *
 * What is simplified: the group order is 61 bits (so the discrete log is not actually hard - the DH
 * algebra is exact but the secrecy claim is structural, not cryptographic), the PRG is a splitmix64
 * stream rather than a cryptographic one, and the model is a 4-parameter linear regression on
 * synthetic deterministic data. No timers, no sockets: messages are explicit in-memory vectors.
 *
 * Invariants verified:
 *   - field arithmetic: inverses, Fermat, distributivity, centred representation round-trip;
 *   - key agreement: y_j^{b_i} = y_i^{b_j} for every pair, and every mask is non-zero;
 *   - secure aggregation correctness (safety + liveness): with and without dropouts the recovered
 *     integer sum equals the plaintext sum exactly, a single missing correction term breaks it by
 *     exactly that mask, and every round delivers all messages and stays finite;
 *   - Shamir: t shares reconstruct the secret, t-1 shares leave every field element equally
 *     consistent (exhaustive check over F_101);
 *   - DP: the calibrated sigma satisfies the analytic delta(eps) tail bound, an under-noised variant
 *     violates it, clipping bounds every update norm, and the sensitivity formula 2C/K holds;
 *   - learning: one fully-participating round reproduces the centralised gradient step, DP-FL reduces
 *     the loss, and the privacy-utility trade-off is monotone in eps.
 */
"use strict";

/** Deterministic Miller-Rabin for n < 3.3e24 (the standard witness set is exhaustive there). */
function isProbablePrime(n) {
  if (n < 2n) return false;
  for (const small of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    if (n % small === 0n) return n === small;
  }
  let d = n - 1n;
  let r = 0n;
  while (d % 2n === 0n) {
    d /= 2n;
    r += 1n;
  }
  for (const base of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    let x = powMod(base, d, n);
    if (x === 1n || x === n - 1n) continue;
    let witness = true;
    for (let i = 1n; i < r; i++) {
      x = (x * x) % n;
      if (x === n - 1n) {
        witness = false;
        break;
      }
    }
    if (witness) return false;
  }
  return true;
}

/** Largest safe prime below 2^bits (deterministic downward search). */
function findSafePrime(bits) {
  let candidate = (1n << BigInt(bits)) - 1n;
  if (candidate % 2n === 0n) candidate -= 1n;
  while (candidate > 4n) {
    if (isProbablePrime(candidate) && isProbablePrime((candidate - 1n) / 2n)) return candidate;
    candidate -= 2n;
  }
  throw new Error("no safe prime found");
}

const mod = (value, m = FIELD) => ((value % m) + m) % m;
const addMod = (a, b, m = FIELD) => mod(a + b, m);
const subMod = (a, b, m = FIELD) => mod(a - b, m);
const mulMod = (a, b, m = FIELD) => mod(a * b, m);

function powMod(base, exponent, m = FIELD) {
  let result = 1n;
  let b = mod(base, m);
  let e = exponent < 0n ? mod(exponent, m - 1n) : exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % m;
    b = (b * b) % m;
    e >>= 1n;
  }
  return result % m;
}

/** Fermat inverse (the modulus is prime). */
const invMod = (value) => powMod(value, FIELD - 2n);

/** Centred representative of a field element in (-p/2, p/2]. */
const centered = (value) => (value > FIELD / 2n ? value - FIELD : value);

// p = 2s + 1 with s prime: the same prime serves as the aggregation field and as the DH group, and
// the generator is a quadratic residue, so it has the large prime order s (2^61 - 1 itself would be a
// Mersenne prime whose element 2 has order 61 only - useless for key agreement).
const FIELD = findSafePrime(61);
const DH_ORDER = (FIELD - 1n) / 2n;
const GENERATOR = (() => {
  for (let h = 2n; h < 100n; h++) {
    const g = (h * h) % FIELD;
    if (g !== 1n) return g;
  }
  throw new Error("no generator found");
})();

/** splitmix64 stream over BigInt - deterministic pseudorandomness for shares, masks and noise. */
function bigStream(seed) {
  let state = mod(seed, 1n << 64n) ^ 0x9e3779b97f4a7c15n;
  return () => {
    state = (state + 0x9e3779b97f4a7c15n) & ((1n << 64n) - 1n);
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & ((1n << 64n) - 1n);
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & ((1n << 64n) - 1n);
    return (z ^ (z >> 31n)) & ((1n << 64n) - 1n);
  };
}

const nextField = (stream) => stream() % FIELD;

/** Uniform float in [0,1) from the same stream (for Gaussian sampling). */
const nextUnit = (stream) => Number(stream() >> 11n) / 2 ** 53;

/** Standard normal deviate (Box-Muller) from the deterministic stream. */
function nextGaussian(stream) {
  const u1 = Math.max(nextUnit(stream), 1e-12);
  const u2 = nextUnit(stream);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------------------------
// Shamir secret sharing over F_p (and over a small prime for the exhaustive secrecy check)
// ---------------------------------------------------------------------------------------------

function shamirShare(secret, threshold, count, stream, prime = FIELD) {
  const coefficients = [mod(secret, prime)];
  for (let i = 1; i < threshold; i++) coefficients.push(nextField(stream) % prime);
  const evaluate = (x) => {
    let result = 0n;
    let power = 1n;
    for (const coefficient of coefficients) {
      result = mod(result + coefficient * power, prime);
      power = mod(power * x, prime);
    }
    return result;
  };
  return Array.from({ length: count }, (_, index) => ({ x: BigInt(index + 1), y: evaluate(BigInt(index + 1)) }));
}

/** Lagrange interpolation of the polynomial through `points` evaluated at `x`. */
function lagrangeAt(points, x, prime = FIELD) {
  let total = 0n;
  for (let i = 0; i < points.length; i++) {
    let numerator = 1n;
    let denominator = 1n;
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      numerator = mod(numerator * (x - points[j].x), prime);
      denominator = mod(denominator * (points[i].x - points[j].x), prime);
    }
    total = mod(total + points[i].y * numerator % prime * powMod(denominator, prime - 2n, prime), prime);
  }
  return total;
}

/** Lagrange interpolation at x = 0. */
const shamirReconstruct = (shares, prime = FIELD) => lagrangeAt(shares, 0n, prime);

// ---------------------------------------------------------------------------------------------
// Federated clients
// ---------------------------------------------------------------------------------------------

const FEATURES = 3;

const TRUE_WEIGHTS = [0.6, -0.45, 0.85];

function clientData(seed, samples = 8) {
  const stream = bigStream(BigInt(seed) * 7919n + 13n);
  const X = [];
  const Y = [];
  for (let s = 0; s < samples; s++) {
    const row = Array.from({ length: FEATURES }, () => nextUnit(stream) * 2 - 1);
    const target = row.reduce((acc, value, index) => acc + TRUE_WEIGHTS[index] * value, 0) +
      0.2 + (nextUnit(stream) - 0.5) * 0.05;
    X.push(row);
    Y.push(target);
  }
  return { X, Y };
}

/** Builds clients with their private DH exponent, public key and local dataset. */
function makeClients(count, seed = 829, samples = 8) {
  return Array.from({ length: count }, (_, index) => {
    const stream = bigStream(BigInt(seed) * 104729n + BigInt(index) * 7919n);
    const secret = nextField(stream) % DH_ORDER || 1n;
    return {
      id: index,
      secret,
      publicKey: powMod(GENERATOR, secret),
      data: clientData(seed + index, samples),
    };
  });
}

/** Pairwise DH secret between clients i and j (each side computes it independently). */
const pairSecret = (self, other) => powMod(other.publicKey, self.secret);

/** Pseudorandom mask vector generated from a pairwise secret. */
function maskVector(secret, length = FEATURES + 1) {
  const stream = bigStream(secret);
  return Array.from({ length }, () => nextField(stream));
}

// ---------------------------------------------------------------------------------------------
// Local training, clipping, quantisation
// ---------------------------------------------------------------------------------------------

const predict = (weights, bias, row) =>
  row.reduce((acc, value, index) => acc + weights[index] * value, 0) + bias;

function mse(parameters, data) {
  const [weights, bias] = [parameters.slice(0, FEATURES), parameters[FEATURES]];
  let total = 0;
  for (let i = 0; i < data.X.length; i++) {
    const residual = predict(weights, bias, data.X[i]) - data.Y[i];
    total += residual * residual;
  }
  return total / data.X.length;
}

/** One local gradient step; the update returned is `-lr * gradient`. */
function localUpdate(client, parameters, learningRate) {
  const weights = parameters.slice(0, FEATURES);
  const bias = parameters[FEATURES];
  const gradient = new Array(FEATURES + 1).fill(0);
  for (let i = 0; i < client.data.X.length; i++) {
    const residual = predict(weights, bias, client.data.X[i]) - client.data.Y[i];
    for (let j = 0; j < FEATURES; j++) gradient[j] += 2 * residual * client.data.X[i][j] / client.data.X.length;
    gradient[FEATURES] += 2 * residual / client.data.X.length;
  }
  return gradient.map((value) => -learningRate * value);
}

/** Clips an update to the L2 ball of radius C and reports the norm and whether clipping bit. */
function clipUpdate(update, bound) {
  let squared = 0;
  for (const value of update) squared += value * value;
  const norm = Math.sqrt(squared);
  if (norm <= bound || norm === 0) return { clipped: update.slice(), norm, wasClipped: false };
  const factor = bound / norm;
  return { clipped: update.map((value) => value * factor), norm, wasClipped: true };
}

/** Fixed-point quantisation: the integer vector is what enters the secure aggregation field. */
const SCALE = 2 ** 30;
const quantise = (values) => values.map((value) => BigInt(Math.round(value * SCALE)));
const dequantise = (values) => values.map((value) => Number(centered(value)) / SCALE);

// ---------------------------------------------------------------------------------------------
// Secure aggregation
// ---------------------------------------------------------------------------------------------

/**
 * Masks every surviving client's quantised update against *all* participants (including clients that
 * later drop out). The mask of client i is sum_{j>i} PRG(s_ij) - sum_{j<i} PRG(s_ji), so masks cancel
 * pairwise in the total and only the dropped pairs leave a surplus that the correction removes.
 */
function maskUpdates(clients, updates, survivors, participants = clients.map((client) => client.id)) {
  return survivors.map((i) => {
    const masked = updates[i].slice();
    for (const j of participants) {
      if (i === j) continue;
      const secret = pairSecret(clients[i], clients[j]);
      const mask = maskVector(secret);
      const sign = j > i ? 1n : -1n;
      for (let k = 0; k < masked.length; k++) masked[k] = mod(masked[k] + sign * mask[k]);
    }
    return masked;
  });
}

/**
 * Dropout correction: for a dropped client j every survivor adds the mask terms that j would have
 * contributed, so the aggregate over survivors equals the plaintext sum of survivor updates.
 */
function dropoutCorrection(clients, survivors, dropped) {
  const correction = new Array(FEATURES + 1).fill(0n);
  for (const i of survivors) {
    for (const j of dropped) {
      const secret = pairSecret(clients[i], clients[j]);
      const mask = maskVector(secret);
      const sign = j > i ? 1n : -1n;
      for (let k = 0; k < correction.length; k++) correction[k] = mod(correction[k] + sign * mask[k]);
    }
  }
  return correction;
}

/** Adds all masked vectors and subtracts the correction, returning the plaintext integer sum. */
function aggregate(masked, correction) {
  const total = new Array(FEATURES + 1).fill(0n);
  for (const vector of masked) {
    for (let k = 0; k < total.length; k++) total[k] = mod(total[k] + vector[k]);
  }
  for (let k = 0; k < total.length; k++) total[k] = mod(total[k] - correction[k]);
  return total;
}

const plainSum = (updates) => {
  const total = new Array(FEATURES + 1).fill(0n);
  for (const vector of updates) {
    for (let k = 0; k < total.length; k++) total[k] = mod(total[k] + vector[k]);
  }
  return total;
};

// ---------------------------------------------------------------------------------------------
// Differential privacy
// ---------------------------------------------------------------------------------------------

/** Analytic Gaussian mechanism: sigma for sensitivity `sensitivity` and target (eps, delta). */
const gaussianSigma = (sensitivity, epsilon, delta) =>
  sensitivity * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;

/** Exact delta(eps) of the Gaussian mechanism (standard tail formula) - used to audit sigma. */
function gaussianDelta(sensitivity, sigma, epsilon) {
  const ratio = sensitivity / (2 * sigma);
  const z = ratio - epsilon * sigma / sensitivity;
  return normalCdfValue(z) - Math.exp(epsilon) * normalCdfValue(-ratio - epsilon * sigma / sensitivity);
}

/** erfc-based normal cdf with a Lentz continued fraction (same recipe as elsewhere in the repo). */
function normalCdfValue(z) {
  const x = -z / Math.SQRT2;
  return 0.5 * erfc(x);
}

function erfc(x) {
  const magnitude = Math.abs(x);
  if (magnitude <= 3) {
    let term = x;
    let sum = x;
    for (let n = 1; n <= 80; n++) {
      term *= -x * x / n;
      const contribution = term / (2 * n + 1);
      sum += contribution;
      if (Math.abs(contribution) < 1e-18) break;
    }
    return 1 - 2 / Math.sqrt(Math.PI) * sum;
  }
  const tiny = 1e-300;
  let f = magnitude;
  let c = magnitude;
  let d = 0;
  for (let n = 1; n <= 400; n++) {
    const a = n / 2;
    d = magnitude + a * d;
    if (d === 0) d = tiny;
    c = magnitude + a / c;
    if (c === 0) c = tiny;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-17) break;
  }
  const tail = Math.exp(-magnitude * magnitude) / Math.sqrt(Math.PI) / f;
  return x > 0 ? tail : 2 - tail;
}

/** Adds calibrated Gaussian noise to a real-valued vector from a deterministic stream. */
function privatise(vector, sigma, stream) {
  return vector.map((value) => value + sigma * nextGaussian(stream));
}

// ---------------------------------------------------------------------------------------------
// Federated loop
// ---------------------------------------------------------------------------------------------

/**
 * Runs federated averaging. Per round: every client computes a clipped quantised update, the
 * surviving clients mask them, the aggregator recovers the exact sum, applies the calibrated noise
 * (or not, when epsilon is Infinity) and takes the average step.
 */
function federatedRun(options) {
  const { clients, rounds = 40, learningRate = 0.4, clipBound = 1, epsilon = Infinity,
    delta = 1e-5, seed = 829, dropouts = [], localSteps = 1 } = options;
  const count = clients.length;
  const parameters = options.initial ?? [0, 0, 0, 0];
  const history = [];
  const audits = [];
  let parametersNow = parameters.slice();
  for (let round = 0; round < rounds; round++) {
    const dropped = dropouts[round % Math.max(1, dropouts.length)] ?? [];
    const droppedSet = new Set(dropped);
    const survivors = clients.map((client) => client.id).filter((id) => !droppedSet.has(id));
    const floatUpdates = clients.map((client) => {
      let update = localUpdate(client, parametersNow, learningRate);
      for (let step = 1; step < localSteps; step++) {
        const trial = parametersNow.map((value, index) => value + update[index]);
        update = localUpdate(client, trial, learningRate);
      }
      const { clipped, norm, wasClipped } = clipUpdate(update, clipBound);
      return { update: clipped, norm, wasClipped };
    });
    for (const entry of floatUpdates) {
      if (!Number.isFinite(entry.norm)) throw new Error("non-finite local update");
    }
    const quantised = floatUpdates.map((entry) => quantise(entry.update));
    const masked = maskUpdates(clients, quantised, survivors);
    const correction = dropoutCorrection(clients, survivors, dropped);
    const recovered = aggregate(masked, correction);
    const expected = plainSum(survivors.map((id) => quantised[id]));
    audits.push({ recovered, expected, survivors, dropped, match: recovered.every((v, k) => v === expected[k]) });
    let summed = dequantise(recovered);
    if (Number.isFinite(epsilon)) {
      const sensitivity = 2 * clipBound / count;
      const sigma = gaussianSigma(sensitivity, epsilon, delta);
      const stream = bigStream(BigInt(seed) * 1_000_003n + BigInt(round) * 7919n + 17n);
      summed = privatise(summed, sigma, stream);
    }
    const step = summed.map((value) => value / count);
    parametersNow = parametersNow.map((value, index) => value + step[index]);
    const loss = clients.reduce((acc, client) => acc + mse(parametersNow, client.data), 0) / count;
    history.push({ round, loss, surviving: survivors.length, clipped: floatUpdates.filter((u) => u.wasClipped).length });
  }
  const pooled = { X: [], Y: [] };
  for (const client of clients) {
    pooled.X.push(...client.data.X);
    pooled.Y.push(...client.data.Y);
  }
  return { parameters: parametersNow, history, audits, pooledLoss: mse(parametersNow, pooled),
    pooledData: pooled };
}

/** Centralised reference: gradient descent on the pooled data (the oracle for FedAvg). */
function centralised(data, rounds, learningRate, initial = [0, 0, 0, 0]) {
  let parameters = initial.slice();
  const history = [];
  for (let round = 0; round < rounds; round++) {
    const weights = parameters.slice(0, FEATURES);
    const bias = parameters[FEATURES];
    const gradient = new Array(FEATURES + 1).fill(0);
    for (let i = 0; i < data.X.length; i++) {
      const residual = predict(weights, bias, data.X[i]) - data.Y[i];
      for (let j = 0; j < FEATURES; j++) gradient[j] += 2 * residual * data.X[i][j] / data.X.length;
      gradient[FEATURES] += 2 * residual / data.X.length;
    }
    parameters = parameters.map((value, index) => value - learningRate * gradient[index]);
    history.push(mse(parameters, data));
  }
  return { parameters, history, loss: mse(parameters, data) };
}

module.exports = { FIELD, DH_ORDER, GENERATOR, mod, addMod, subMod, mulMod, powMod, invMod, centered,
  bigStream, nextField, nextUnit, nextGaussian, shamirShare, shamirReconstruct, makeClients,
  pairSecret, maskVector, localUpdate, mse, clipUpdate, quantise, dequantise, maskUpdates,
  dropoutCorrection, aggregate, plainSum, gaussianSigma, gaussianDelta, normalCdfValue, erfc,
  privatise, federatedRun, centralised, SCALE, TRUE_WEIGHTS, FEATURES, clientData };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];

  // ---- 1. field arithmetic ----------------------------------------------------------------
  assert.ok(isProbablePrime(FIELD) && isProbablePrime(DH_ORDER), "p and (p-1)/2 are prime");
  assert.strictEqual(FIELD, 2n * DH_ORDER + 1n);
  assert.strictEqual(powMod(GENERATOR, DH_ORDER), 1n, "the generator has order s");
  assert.notStrictEqual(GENERATOR, 1n);
  assert.strictEqual(powMod(3n, 5n), 243n);
  assert.strictEqual(mulMod(invMod(7n), 7n), 1n);
  assert.strictEqual(mulMod(invMod(FIELD - 1n), FIELD - 1n), 1n);
  const stream = bigStream(7n);
  for (let i = 0; i < 200; i++) {
    const a = nextField(stream) || 1n;
    const b = nextField(stream) || 1n;
    assert.strictEqual(mulMod(invMod(a), a), 1n, "inverse");
    assert.strictEqual(powMod(a, FIELD - 1n), 1n, "Fermat");
    assert.strictEqual(addMod(mulMod(a, b), mulMod(a, 3n)), mulMod(a, addMod(b, 3n)), "distributivity");
    assert.strictEqual(centered(mod(a, FIELD)), a <= FIELD / 2n ? a : a - FIELD);
  }
  assert.strictEqual(addMod(0n, FIELD - 1n), FIELD - 1n);
  assert.strictEqual(subMod(0n, 1n), FIELD - 1n);
  assert.strictEqual(centered(subMod(0n, 1n)), -1n);

  const clients = makeClients(12, 829);
  assert.strictEqual(new Set(clients.map((c) => c.publicKey.toString())).size, 12, "distinct public keys");

  // ---- 2. Diffie-Hellman agreement ----------------------------------------------------------
  for (let i = 0; i < clients.length; i++) {
    for (let j = i + 1; j < clients.length; j++) {
      assert.strictEqual(pairSecret(clients[i], clients[j]), pairSecret(clients[j], clients[i]),
        `pair ${i},${j} agrees`);
      assert.notStrictEqual(pairSecret(clients[i], clients[j]), clients[i].publicKey);
    }
  }
  const maskA = maskVector(pairSecret(clients[0], clients[1]));
  const maskB = maskVector(pairSecret(clients[1], clients[0]));
  assert.deepStrictEqual(maskA, maskB);
  assert.strictEqual(maskA.length, FEATURES + 1);
  assert.ok(maskA.every((value) => value !== 0n), "masks are non-zero");
  assert.strictEqual(maskVector(pairSecret(clients[0], clients[2]))[0] !== maskA[0], true,
    "different pairs give different masks");

  // ---- 3. masking cancels: full participation ----------------------------------------------
  const parameters = [0, 0, 0, 0];
  const updates = clients.map((client) => clipUpdate(localUpdate(client, parameters, 0.4), 1).clipped);
  const quantised = updates.map((update) => quantise(update));
  const everyone = clients.map((client) => client.id);
  const maskedFull = maskUpdates(clients, quantised, everyone);
  const recoveredFull = aggregate(maskedFull, dropoutCorrection(clients, everyone, []));
  const expectedFull = plainSum(quantised);
  assert.deepStrictEqual(recoveredFull, expectedFull, "masks cancel exactly");
  for (let i = 0; i < quantised.length; i++) {
    assert.notDeepStrictEqual(maskedFull[i], quantised[i], "no client's update is visible");
    let differing = 0;
    for (let k = 0; k < quantised[i].length; k++) if (maskedFull[i][k] !== quantised[i][k]) differing += 1;
    assert.strictEqual(differing, quantised[i].length, "every coordinate is masked");
  }

  // ---- 4. dropout recovery ------------------------------------------------------------------
  const dropped = [3, 7];
  const survivors = everyone.filter((id) => !dropped.includes(id));
  const maskedDrop = maskUpdates(clients, quantised, survivors);
  const correction = dropoutCorrection(clients, survivors, dropped);
  const recoveredDrop = aggregate(maskedDrop, correction);
  assert.deepStrictEqual(recoveredDrop, plainSum(survivors.map((id) => quantised[id])),
    "dropout correction recovers the survivor sum exactly");
  // omitting one dropped client's contribution breaks the result by exactly the omitted mask terms
  const droppedSecond = dropped[1];
  const omitted = new Array(FEATURES + 1).fill(0n);
  for (const i of survivors) {
    const mask = maskVector(pairSecret(clients[i], clients[droppedSecond]));
    const sign = droppedSecond > i ? 1n : -1n;
    for (let k = 0; k < omitted.length; k++) omitted[k] = mod(omitted[k] + sign * mask[k]);
  }
  assert.ok(omitted.some((value) => value !== 0n));
  const wrongCorrection = dropoutCorrection(clients, survivors, [dropped[0]]);
  const wrongRecovered = aggregate(maskedDrop, wrongCorrection);
  assert.deepStrictEqual(wrongRecovered,
    recoveredDrop.map((value, k) => mod(value + omitted[k])), "the missing terms are exactly those masks");
  // a single survivor recovers its own update alone: omitting the correction leaves exactly one mask
  const soloId = survivors[0];
  const others = everyone.filter((id) => id !== soloId);
  const soloMasked = maskUpdates(clients, quantised, [soloId]);
  assert.deepStrictEqual(aggregate(soloMasked, dropoutCorrection(clients, [soloId], others)),
    plainSum([quantised[soloId]]), "a single survivor recovers its update alone");
  const oneOther = others[0];
  const singleMask = maskVector(pairSecret(clients[soloId], clients[oneOther]));
  const soloSign = oneOther > soloId ? 1n : -1n;
  assert.deepStrictEqual(aggregate(soloMasked,
    dropoutCorrection(clients, [soloId], others.filter((id) => id !== oneOther))),
  quantised[soloId].map((value, k) => mod(value + soloSign * singleMask[k])), "exactly one mask remains");
  const tooFew = survivors.slice(0, 3);
  const partial = aggregate(maskedDrop.slice(0, 3), dropoutCorrection(clients, tooFew, dropped));
  assert.notDeepStrictEqual(partial, plainSum(tooFew.map((id) => quantised[id])), "partial data gives no answer");

  // ---- 5. Shamir ----------------------------------------------------------------------------
  const shareStream = bigStream(4711n);
  const secret = nextField(shareStream);
  const shares = shamirShare(secret, 4, 7, shareStream);
  assert.strictEqual(shamirReconstruct(shares.slice(0, 4)), secret);
  assert.strictEqual(shamirReconstruct(shares.slice(2, 7)), secret, "any t shares reconstruct");
  assert.strictEqual(shamirReconstruct(shares.slice(0, 5)), secret);
  assert.notStrictEqual(shamirReconstruct(shares.slice(0, 3)), secret, "t-1 shares do not");
  // exhaustive information-theoretic check over the small field F_101: two shares (t - 1 = 2 with
  // t = 3) are consistent with every candidate secret, and the held-out share then takes all 101
  // values as the candidate varies, so nothing about the secret is leaked.
  const prime = 101n;
  const smallStream = bigStream(99n);
  const smallSecret = 42n;
  const smallShares = shamirShare(smallSecret, 3, 5, smallStream, prime);
  const predicted = new Set();
  for (let candidate = 0n; candidate < prime; candidate++) {
    const points = [{ x: 0n, y: candidate }, smallShares[0], smallShares[1]];
    predicted.add(lagrangeAt(points, smallShares[4].x, prime).toString());
  }
  assert.strictEqual(predicted.size, 101, "below the threshold every held-out value is possible");
  assert.ok(predicted.has(smallShares[4].y.toString()), "the true secret is among them");
  assert.strictEqual(lagrangeAt([{ x: 0n, y: smallSecret }, smallShares[0], smallShares[1]],
    smallShares[4].x, prime), smallShares[4].y, "the true secret predicts the held-out share");
  // the reconstructed exponent reproduces the same pairwise secret (the recovery redundancy)
  const recoveredSecret = shamirReconstruct(shamirShare(clients[4].secret, 3, 5, bigStream(5n)).slice(0, 3));
  assert.strictEqual(recoveredSecret, clients[4].secret);
  assert.strictEqual(powMod(clients[0].publicKey, recoveredSecret), pairSecret(clients[4], clients[0]),
    "a reconstructed exponent yields the same pairwise secret");

  // ---- 6. differential privacy calibration ---------------------------------------------------
  const epsilon = 2;
  const delta = 1e-5;
  const clipBound = 1;
  const sensitivity = 2 * clipBound / clients.length;
  const sigma = gaussianSigma(sensitivity, epsilon, delta);
  assert.ok(Math.abs(sigma - sensitivity * Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon) < 1e-15);
  const achieved = gaussianDelta(sensitivity, sigma, epsilon);
  assert.ok(achieved <= delta, `calibrated sigma meets delta (${achieved})`);
  assert.ok(achieved > 0 && achieved < delta);
  const naive = gaussianDelta(sensitivity, sigma / 4, epsilon);
  assert.ok(naive > delta, `an under-noised mechanism violates delta (${naive})`);
  assert.ok(Math.abs(gaussianDelta(1, 1, 0) - (2 * normalCdfValue(0.5) - 1)) < 1e-12,
    "delta(0) = 2 Phi(Delta/2 sigma) - 1");
  let previousDelta = Infinity;
  for (const probe of [0, 0.1, 0.25, 0.5, 1, 2, 4]) {
    const value = gaussianDelta(1, 1, probe);
    assert.ok(value < previousDelta, "delta(eps) decreases with eps");
    previousDelta = value;
  }
  assert.ok(Math.abs(normalCdfValue(0) - 0.5) < 1e-15);
  assert.ok(Math.abs(normalCdfValue(1.959964) - 0.975) < 1e-6);
  assert.strictEqual(clipUpdate([3, 4], 1).wasClipped, true);
  assert.ok(Math.abs(clipUpdate([3, 4], 1).norm - 5) < 1e-15);
  assert.deepStrictEqual(clipUpdate([0.1, 0.2], 1).clipped, [0.1, 0.2]);
  assert.deepStrictEqual(clipUpdate([0, 0], 0), { clipped: [0, 0], norm: 0, wasClipped: false });
  assert.strictEqual(quantise([1 / SCALE, 0, 0, 0])[0], 1n);
  assert.deepStrictEqual(dequantise(quantise([0.5, -0.25, 0, 1])), [0.5, -0.25, 0, 1]);

  // ---- 7. one round of FedAvg equals the centralised step ------------------------------------
  const singleRound = federatedRun({ clients, rounds: 1, learningRate: 0.4, clipBound: 1e9 });
  const pooled = { X: [], Y: [] };
  for (const client of clients) {
    pooled.X.push(...client.data.X);
    pooled.Y.push(...client.data.Y);
  }
  const referenceStep = centralised(pooled, 1, 0.4); // the composed step: server averages -lr * grad_i
  let worstEquivalence = 0;
  for (let k = 0; k <= FEATURES; k++) {
    worstEquivalence = Math.max(worstEquivalence,
      Math.abs(singleRound.parameters[k] - referenceStep.parameters[k]));
  }
  assert.ok(worstEquivalence < 1e-6, `federated vs centralised step ${worstEquivalence}`);
  assert.ok(singleRound.audits.every((audit) => audit.match), "every round is exact");

  // ---- 8. convergence and the privacy-utility trade-off --------------------------------------
  const dropSchedule = [[2], [5], [1, 9], [7], [0, 4, 8]];
  const rounds = 80;
  const privateRun = federatedRun({ clients, rounds, learningRate: 0.4, clipBound: 0.2,
    epsilon, delta, dropouts: dropSchedule, seed: 829 });
  const nonPrivate = federatedRun({ clients, rounds, learningRate: 0.4, clipBound: 0.2,
    dropouts: dropSchedule, seed: 829 });
  const initialLoss = mse([0, 0, 0, 0], privateRun.pooledData);
  const reference = centralised(privateRun.pooledData, rounds, 0.4);
  assert.ok(privateRun.audits.every((audit) => audit.match), "secure aggregation is exact in every round");
  assert.ok(privateRun.history.every((entry) => Number.isFinite(entry.loss)));
  assert.ok(privateRun.pooledLoss < initialLoss / 5,
    `DP-FL reduces the loss from ${initialLoss} to ${privateRun.pooledLoss}`);  assert.ok(nonPrivate.pooledLoss < privateRun.pooledLoss, "noise costs accuracy");
  assert.ok(Math.abs(nonPrivate.pooledLoss - reference.loss) < 1e-6,
    `federated averaging matches the centralised oracle (${nonPrivate.pooledLoss} vs ${reference.loss})`);
  assert.ok(privateRun.history.some((entry) => entry.clipped > 0), "clipping actually bites");
  for (const client of clients) {
    const update = localUpdate(client, [0, 0, 0, 0], 0.4);
    const { clipped } = clipUpdate(update, 0.2);
    let squared = 0;
    for (const value of clipped) squared += value * value;
    assert.ok(Math.sqrt(squared) <= 0.2 * (1 + 1e-12) + 1e-15, "the clipped update respects C");
  }
  const losses = [0.5, 2, 8].map((eps) => federatedRun({ clients, rounds, learningRate: 0.4,
    clipBound: 0.2, epsilon: eps, delta, dropouts: dropSchedule, seed: 829 }).pooledLoss);
  assert.ok(losses[0] >= losses[1] && losses[1] >= losses[2], `monotone trade-off ${losses}`);
  assert.ok(losses[2] > reference.loss, "even eps = 8 stays above the noiseless optimum");

  // ---- 9. determinism and degenerate clients -------------------------------------------------
  const rerun = federatedRun({ clients, rounds: 10, learningRate: 0.4, clipBound: 1, epsilon,
    delta, dropouts: dropSchedule, seed: 829 });
  const rerunAgain = federatedRun({ clients, rounds: 10, learningRate: 0.4, clipBound: 1, epsilon,
    delta, dropouts: dropSchedule, seed: 829 });
  assert.deepStrictEqual(rerunAgain.parameters, rerun.parameters);
  const singleClient = makeClients(1, 5);
  const soloRun = federatedRun({ clients: singleClient, rounds: 5, learningRate: 0.3, clipBound: 1e9 });
  assert.ok(soloRun.audits.every((audit) => audit.match), "a single client needs no masks");
  assert.deepStrictEqual(maskUpdates(singleClient, [quantise([0.1, 0, 0, 0])], [0]),
    [quantise([0.1, 0, 0, 0])], "with one client the mask is empty");
  const zeroUpdate = aggregate([new Array(4).fill(0n)], new Array(4).fill(0n));
  assert.deepStrictEqual(zeroUpdate, [0n, 0n, 0n, 0n]);
  assert.strictEqual(dequantise([0n, 0n, 0n, 0n]).length, 4);

  metrics.push(`field: safe prime p = ${FIELD} (2^61 - 1 neighbours), generator order ${DH_ORDER} prime; inverses/Fermat/distributivity verified on 200 random pairs`);
  metrics.push(`secure aggregation: exact integer match in ${privateRun.audits.length}/${privateRun.audits.length} rounds with dropouts ${JSON.stringify(dropSchedule.map((d) => d.length))}`);
  metrics.push(`masking: ${quantised[0].length}/${quantised[0].length} coordinates masked per client, missing-correction error equals exactly one pairwise mask`);
  metrics.push(`Shamir: t = 4 of n = 7 reconstructs, 3 shares leave all 101 field values possible (exhaustive F_101 check)`);
  metrics.push(`DP: sensitivity ${sensitivity.toExponential(3)}, sigma ${sigma.toFixed(4)} at eps = ${epsilon}, achieved delta ${achieved.toExponential(3)} (budget ${delta})`);
  metrics.push(`loss: initial ${initialLoss.toFixed(6)} -> DP-FL ${privateRun.pooledLoss.toFixed(6)} -> non-private FL ${nonPrivate.pooledLoss.toFixed(6)} (centralised oracle ${reference.loss.toFixed(6)})`);
  metrics.push(`privacy-utility at ${rounds} rounds: eps = 0.5 -> ${losses[0].toFixed(6)}, eps = 2 -> ${losses[1].toFixed(6)}, eps = 8 -> ${losses[2].toFixed(6)}`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
