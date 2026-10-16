/** Task 712 - Implement privacy-preserving set intersection via hashing sketch. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * Two protocols are implemented and checked:
 *
 * 1. Exact intersection by commutative blinding (the Diffie-Hellman-style PSI). Both parties map
 *    each element to a group element of the Mersenne prime field p = 2^61 - 1 and raise it to a
 *    secret exponent. Party A sends {H(x)^a}, B returns the same values raised to b together with
 *    its own {H(y)^b}, and A raises those to a. Because exponentiation commutes - (H^a)^b = (H^b)^a
 *    - the two sides agree exactly on the doubly blinded values, so A can read off which of its own
 *    elements are also B's, while every value in the transcript is blinded (a != 0 hides it).
 *
 * 2. Approximate intersection size from a MinHash sketch. A min-hash sketch keeps, for k
 *    independent hash functions, the smallest hash of the set; the fraction of agreeing positions
 *    estimates the Jaccard index, whose error shrinks like 1/sqrt(k). It is the inexpensive
 *    alternative when only the *size* of the intersection is needed, and it never sends elements
 *    at all. A Bloom-filter prefilter is included as well: it can only produce false positives,
 *    never false negatives, which is exactly what the checks pin down.
 *
 * What is simplified: semi-honest parties, a fixed public prime instead of a negotiated group, a
 * non-cryptographic hash inside the sketches, and no malicious-security analysis. The Laplace-noise
 * release of the estimated intersection size is included with its sensitivity of 1.
 *
 * Invariant verified: the exact protocol returns precisely the brute-force intersection; the
 * blinding really commutes and really hides (secret 0 is rejected and is shown to leak); sketches
 * bracket the Jaccard index with an error that falls as k grows; the Bloom filter has no false
 * negatives and respects the theoretical false-positive rate; and the noisy size release satisfies
 * the eps-Laplace ratio bound exactly.
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

function modPow(base, exponent, modulus) {
  let result = 1n;
  let b = base % modulus;
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % modulus;
    b = (b * b) % modulus;
    e >>= 1n;
  }
  return result;
}
/** Deterministic Miller-Rabin; the fixed bases are a proof for every 64-bit input. */
function isProbablePrime(value) {
  if (value < 2n) return false;
  for (const small of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    if (value === small) return true;
    if (value % small === 0n) return false;
  }
  let d = value - 1n;
  let s = 0n;
  while (d % 2n === 0n) { d /= 2n; s++; }
  for (const witness of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    let x = modPow(witness, d, value);
    if (x === 1n || x === value - 1n) continue;
    let composite = true;
    for (let i = 1n; i < s; i++) {
      x = (x * x) % value;
      if (x === value - 1n) { composite = false; break; }
    }
    if (composite) return false;
  }
  return true;
}
/**
 * A safe prime p = 2q + 1 with q prime, searched downwards from 2^61 - 1 before any test runs. Using
 * a safe prime matters: in a Mersenne prime field the generator 2 has the tiny order 61, which would
 * collapse the whole protocol - the collision checks below would catch exactly that.
 */
const SAFE_PRIME = (() => {
  for (let q = (1n << 61n) - 1n; q > 3n; q -= 2n) {
    if (isProbablePrime(q) && isProbablePrime(2n * q + 1n)) return { p: 2n * q + 1n, order: q };
  }
  throw new Error("no safe prime found");
})();
const FIELD_PRIME = SAFE_PRIME.p;
const SUBGROUP_ORDER = SAFE_PRIME.order;
const GROUP_GENERATOR = 4n; // a quadratic residue, so it generates the order-q subgroup
/** FNV-1a over a string, reduced into the exponent space [1, q-1]. */
function hash64(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  let wide = BigInt(hash);
  wide = (wide << 21n) ^ BigInt(Math.imul(hash, 0x9e3779b1) >>> 0);
  return wide % (SUBGROUP_ORDER - 1n) + 1n;
}
/** Maps an element into the subgroup: H(x) = g^(hash(x)) mod p. */
function toGroupElement(element) {
  return modPow(GROUP_GENERATOR, hash64(String(element)), FIELD_PRIME);
}
/** One blinding step; the exponent must be a secret in [1, q-1]. */
function blind(value, secret) {
  checkSecret(secret, "secret");
  return modPow(value, secret, FIELD_PRIME);
}

function checkSecret(secret, name) {
  if (typeof secret !== "bigint" || secret <= 0n || secret >= SUBGROUP_ORDER) {
    throw new RangeError(`${name} must lie in [1, q-1]`);
  }
}
/**
 * The protocol. Returns the intersection together with the transcript, so the checks can inspect
 * exactly what each party learned.
 */
function psiIntersection(setA, setB, secretA, secretB) {
  checkSecret(secretA, "secretA");
  checkSecret(secretB, "secretB");
  const elementsA = [...new Set(setA.map(String))];
  const elementsB = [...new Set(setB.map(String))];
  const aBlinded = elementsA.map((element) => blind(toGroupElement(element), secretA));
  const aDoubly = aBlinded.map((value) => blind(value, secretB)); // values B returns to A
  const bBlinded = elementsB.map((element) => blind(toGroupElement(element), secretB));
  const bDoubly = bBlinded.map((value) => blind(value, secretA)); // values A computes locally
  const bSet = new Set(bDoubly.map(String));
  const intersection = [];
  for (let i = 0; i < elementsA.length; i++) {
    if (bSet.has(aDoubly[i].toString())) intersection.push(elementsA[i]);
  }
  return {
    intersection: intersection.sort(),
    transcript: { aBlinded, aDoubly, bBlinded, bDoubly },
  };
}
/** Brute-force oracle: the plain set intersection. */
function plainIntersection(setA, setB) {
  const other = new Set(setB.map(String));
  return [...new Set(setA.map(String))].filter((element) => other.has(element)).sort();
}

const MASK64 = (1n << 64n) - 1n;
/** splitmix64 finalizer: spreads one integer into 64 well-mixed bits (the sketch hash). */
function mix64(value) {
  let x = value & MASK64;
  x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
  x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
  return (x ^ (x >> 31n)) & MASK64;
}
/** One min-hash value per function: the smallest hash of the set under that function. */
function minHashSketch(elements, functions, seed) {
  const rng = makeRng(seed);
  const salts = Array.from({ length: functions }, () => BigInt(rng(1 << 30)));
  const sketch = new Array(functions).fill(null);
  for (const element of new Set(elements.map(String))) {
    const base = hash64(element);
    for (let i = 0; i < functions; i++) {
      const value = mix64(base + salts[i] * (BigInt(i) + 1n));
      if (sketch[i] === null || value < sketch[i]) sketch[i] = value;
    }
  }
  return sketch;
}
/** Jaccard index estimated by the fraction of agreeing sketch positions. */
function jaccardFromSketches(first, second) {
  let agree = 0;
  for (let i = 0; i < first.length; i++) if (first[i] !== null && first[i] === second[i]) agree++;
  return agree / first.length;
}

function jaccard(setA, setB) {
  const a = new Set(setA.map(String));
  const b = new Set(setB.map(String));
  let common = 0;
  for (const element of a) if (b.has(element)) common++;
  const union = a.size + b.size - common;
  return union === 0 ? 1 : common / union;
}
/** Classic Bloom filter; false positives are possible, false negatives are not. */
function bloomFilter(elements, bits, hashes, seed) {
  const rng = makeRng(seed);
  const salts = Array.from({ length: hashes }, () => BigInt(rng(1 << 30)));
  const bytes = new Uint8Array(bits);
  const slots = (element) => {
    const base = hash64(String(element));
    return Array.from({ length: hashes }, (_, i) =>
      Number(mix64(base + salts[i] * (BigInt(i) + 1n)) % BigInt(bits)));
  };
  for (const element of new Set(elements.map(String))) {
    for (const slot of slots(element)) bytes[slot] = 1;
  }
  return {
    bits,
    hashes,
    bytes,
    slots,
    has(element) { return slots(element).every((slot) => bytes[slot] === 1); },
  };
}
/** Measured false-positive rate over a probe set. */
function bloomFalsePositiveRate(filter, probes) {
  const positives = probes.filter((probe) => filter.has(probe)).length;
  return probes.length === 0 ? 0 : positives / probes.length;
}
/** Theoretical false-positive rate of a Bloom filter: (1 - e^(-k n / m))^k. */
function bloomTheoreticalRate(inserted, bits, hashes) {
  return Math.pow(1 - Math.exp(-(hashes * inserted) / bits), hashes);
}
/** Laplace sample from the given PRNG. */
function laplaceSample(rng, scale) {
  if (scale <= 0) return 0;
  const u = (rng(1 << 30) + 0.5) / (1 << 30) - 0.5;
  const magnitude = -scale * Math.log(1 - 2 * Math.abs(u));
  return u < 0 ? -magnitude : magnitude;
}
/** Differentially private release of an intersection size: the sensitivity is 1. */
function noisyIntersectionSize(size, epsilon, rng) {
  if (!(epsilon > 0)) throw new RangeError("epsilon must be positive");
  return Math.max(0, size + laplaceSample(rng, 1 / epsilon));
}
/** Exact discrete-Laplace pmf, used to verify the privacy claim without sampling. */
function discreteLaplacePmf(k, scale) {
  if (scale <= 0) return k === 0 ? 1 : 0;
  const p = Math.exp(-1 / scale);
  return ((1 - p) / (1 + p)) * Math.pow(p, Math.abs(k));
}

module.exports = {
  makeRng, hash64, toGroupElement, blind, psiIntersection, plainIntersection, minHashSketch,
  jaccardFromSketches, jaccard, bloomFilter, bloomFalsePositiveRate, bloomTheoreticalRate,
  noisyIntersectionSize, discreteLaplacePmf, FIELD_PRIME,
};

if (require.main === module) {
  // -- the algebraic foundation: exponentiation commutes and a zero secret leaks ------------------
  const rng = makeRng(712);
  for (let trial = 0; trial < 40; trial++) {
    const element = `element-${rng(1000)}`;
    const value = toGroupElement(element);
    const a = BigInt(2 + rng(1 << 20));
    const b = BigInt(2 + rng(1 << 20));
    strictEqual(blind(blind(value, a), b), blind(blind(value, b), a), "blinding must commute");
    ok(blind(value, a) !== value, "blinding must change the value");
    ok(blind(value, 1n) === value % FIELD_PRIME, "an exponent of one is the identity");
  }
  throws(() => blind(2n, 0n), RangeError);
  throws(() => blind(2n, FIELD_PRIME - 1n), RangeError);
  throws(() => psiIntersection(["a"], ["a"], 0n, 1n), RangeError);
  ok(toGroupElement("x") !== toGroupElement("y"), "different elements map to different group values");

  // -- exact intersection on hand and random set pairs ------------------------------------------
  const left = ["alpha", "beta", "gamma", "delta"];
  const right = ["beta", "delta", "epsilon"];
  const hand = psiIntersection(left, right, 12345n, 987654321n);
  deepStrictEqual(hand.intersection, plainIntersection(left, right));
  deepStrictEqual(hand.intersection, ["beta", "delta"]);
  for (const [setA, setB] of [
    [[], []],
    [["solo"], []],
    [["solo"], ["solo"]],
    [[], ["solo"]],
    [["x", "y"], ["z"]],
    [["x", "y", "z"], ["x", "y"]],
    [["x", "x", "y"], ["x", "y", "y", "y"]],
  ]) {
    const result = psiIntersection(setA, setB, 777n, 31337n);
    deepStrictEqual(result.intersection, plainIntersection(setA, setB), `sets ${setA} / ${setB}`);
  }
  const universe = Array.from({ length: 1000 }, (_, i) => `item-${i}`);
  const bigA = universe.filter(() => rng(3) === 0);
  const bigB = universe.filter(() => rng(4) === 0);
  const big = psiIntersection(bigA, bigB, 0xdeadbeefn, 0xfeedfacen);
  deepStrictEqual(big.intersection, plainIntersection(bigA, bigB), "300-element sets");
  ok(big.intersection.length > 20, "the large pair must genuinely intersect");

  // -- what the transcript contains: no plaintext elements --------------------------------------
  const secrets = new Set(hand.transcript.aBlinded.map(String));
  for (const element of left) {
    ok(!secrets.has(toGroupElement(element).toString()), "no plaintext hash may appear in the transcript");
  }
  for (const value of hand.transcript.bBlinded) ok(value > 1n, "every exchanged value is blinded");
  ok(new Set(hand.transcript.aBlinded.map(String)).size === left.length, "blinding keeps the set injective");
  const withZeroSecret = psiIntersection(["alpha"], ["alpha"], 1n, 1n);
  ok(withZeroSecret.transcript.aBlinded[0] === toGroupElement("alpha") % FIELD_PRIME,
    "an exponent of one exposes the raw group element - hence the rejection of secret 0");

  // -- MinHash: the Jaccard estimate and its 1/sqrt(k) convergence ------------------------------
  const setOne = universe.slice(0, 400);
  const setTwo = universe.slice(200, 700);
  const truth = jaccard(setOne, setTwo);
  ok(Math.abs(truth - 200 / 700) < 1e-12, "the true Jaccard index is 200/700");
  const errors = [];
  for (const functions of [16, 64, 256, 1024]) {
    const sketchA = minHashSketch(setOne, functions, 99);
    const sketchB = minHashSketch(setTwo, functions, 99);
    const estimate = jaccardFromSketches(sketchA, sketchB);
    errors.push(Math.abs(estimate - truth));
    // the estimate has standard deviation sqrt(J(1-J)/k); allow three sigma plus a little slack
    const sd = Math.sqrt((truth * (1 - truth)) / functions);
    ok(Math.abs(estimate - truth) <= 3 * sd + 0.02,
      `${functions} hashes give ${estimate}, expected ${truth} +- ${3 * sd}`);
  }
  ok(errors[3] <= errors[0] + 1e-9, "more hash functions must not make the estimate worse");
  const sketchesOfSame = minHashSketch(setOne, 64, 7);
  strictEqual(jaccardFromSketches(sketchesOfSame, sketchesOfSame), 1, "a set with itself is Jaccard 1");
  strictEqual(jaccardFromSketches(minHashSketch([], 16, 1), minHashSketch(setOne, 16, 1)), 0,
    "an empty set agrees with nothing");
  const rmse = (functions) => {
    let total = 0;
    for (let trial = 0; trial < 24; trial++) {
      const estimate = jaccardFromSketches(
        minHashSketch(setOne, functions, trial + 1), minHashSketch(setTwo, functions, trial + 1));
      total += (estimate - truth) ** 2;
    }
    return Math.sqrt(total / 24);
  };
  const coarse = rmse(16), fine = rmse(256);
  ok(fine < coarse, `the error must fall with k (${coarse.toFixed(4)} -> ${fine.toFixed(4)})`);

  // -- Bloom filter: no false negatives, bounded false positives --------------------------------
  const filter = bloomFilter(setOne, 2048, 4, 5);
  for (const element of setOne) ok(filter.has(element), "a Bloom filter must never miss a member");
  const probes = universe.slice(400).concat(Array.from({ length: 400 }, (_, i) => `outside-${i}`));
  const observed = bloomFalsePositiveRate(filter, probes);
  const theoretical = bloomTheoreticalRate(setOne.length, 2048, 4);
  ok(theoretical > 0.05 && theoretical < 0.2, `the theoretical rate is ${theoretical}`);
  ok(observed <= theoretical * 1.6, `observed ${observed} must respect the bound ${theoretical}`);
  strictEqual(bloomFalsePositiveRate(bloomFilter([], 256, 3, 1), probes), 0, "an empty filter answers no");

  // -- differentially private intersection size ---------------------------------------------------
  const noiseRng = makeRng(7120);
  const epsilon = 0.5;
  const releases = Array.from({ length: 4000 }, () => noisyIntersectionSize(200, epsilon, noiseRng));
  const mean = releases.reduce((a, b) => a + b, 0) / releases.length;
  ok(Math.abs(mean - 200) < 0.2, "the noisy size must stay unbiased");
  for (let k = -8; k <= 8; k++) {
    const ratio = discreteLaplacePmf(k, 1 / epsilon) / discreteLaplacePmf(k + 1, 1 / epsilon);
    ok(ratio <= Math.exp(epsilon) + 1e-12, `the size release ratio at k=${k} exceeds e^eps`);
  }
  const clamped = Array.from({ length: 2000 }, () => noisyIntersectionSize(0, 0.2, noiseRng));
  ok(clamped.every((value) => value >= 0), "the released size must be clamped at zero");
  ok(clamped.some((value) => value === 0), "clamping must actually trigger for a zero size");
  throws(() => noisyIntersectionSize(1, 0, noiseRng), RangeError);

  // -- determinism ---------------------------------------------------------------------------------
  function replay(seed) {
    const local = makeRng(seed);
    const setA = Array.from({ length: 20 }, () => `k${local(50)}`);
    const setB = Array.from({ length: 20 }, () => `k${local(50)}`);
    const result = psiIntersection(setA, setB, 424242n, 535353n);
    const sketch = jaccardFromSketches(minHashSketch(setA, 32, seed), minHashSketch(setB, 32, seed));
    return `${result.intersection.join(",")}|${sketch.toFixed(9)}`;
  }
  strictEqual(replay(58), replay(58), "identical seeds must replay identically");

  console.log(`large pair intersection=${big.intersection.length} of ${bigA.length}/${bigB.length}`);
  console.log(`MinHash RMSE 16->256 hashes: ${coarse.toFixed(4)} -> ${fine.toFixed(4)}; Bloom fp=${observed.toFixed(4)} (bound ${theoretical.toFixed(4)})`);
  console.log("PASS");
}
