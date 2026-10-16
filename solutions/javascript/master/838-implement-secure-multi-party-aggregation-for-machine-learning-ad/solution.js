/** Task 838 - Implement secure multi-party aggregation for machine learning (additive secret sharing).
 *  (Tier: Master, CTCI Beyond — Security & Privacy)
 *
 * Secure aggregation for federated learning: n parties hold private gradient vectors and one server
 * must learn only their *sum* (the model update). The implementation follows the Bonawitz-style
 * protocol structure:
 *
 *  1. Pairwise key agreement (toy Diffie-Hellman over Z_p*, p = 2^127 - 1) gives every pair an
 *     independent seed; pair (i, j) derives mask u_ij.
 *  2. Party i hides its input as  y_i = x_i + b_i + SUM_{j>i} u_ij - SUM_{j<i} u_ji.
 *     The pairwise terms cancel in the plain sum; the self mask b_i keeps a lone surviving party's
 *     input hidden from the server.
 *  3. Every b_i and every pairwise term u_ij is Shamir-shared (t-of-n). When parties drop, the server
 *     adds the shares it still holds and interpolates the *sum* polynomial at 0, so it re-derives
 *     SUM_{i in S} b_i and each orphaned u_ij — never an individual b_i and never an input.
 *  4. Multiplication of two secrets (secure inner product, i.e. gradient dot products) uses one
 *     pre-computed Beaver triple per product: the opened differences d = x - a and e = y - b are
 *     public, the product shares stay additive, and reconstruction needs t shares.
 *
 * Field: F_p with p = 2^127 - 1 (Mersenne prime); BigInt arithmetic with square-and-multiply
 * exponentiation. All randomness comes from explicitly seeded splitmix64, so runs are reproducible.
 *
 * Toy / simplified, stated honestly: the DH group uses a 127-bit modulus but only 32-bit private
 * exponents, so discrete log is not hard — the protocol structure, the mask-cancellation algebra,
 * Shamir recoverability and dropout robustness are genuine, the confidentiality against a real
 * adversary is not. Transport, rounds and serialisation are collapsed into direct calls.
 *
 * Invariants verified below (safety + liveness, and confidentiality as a measurement):
 *  - exactness: aggregateSurvivors equals the plaintext sum of the *surviving* inputs for every
 *    dropout set of size <= n - t, including the full-attendance case;
 *  - liveness boundary: below t survivors the protocol returns null instead of a wrong value;
 *  - confidentiality: the server-visible masked values never equal the input, spread over ~87% of
 *    all residues mod 97 across 200 independent sessions, and stay spread even when the aggregate is
 *    fixed;
 *  - homomorphic tooling: Beaver-triple products and secure inner products match an independent
 *    plaintext oracle, the secure mean is exact, and t - 1 shares never reveal a product.
 */
"use strict";

const FIELD_PRIME = (1n << 127n) - 1n; // Mersenne prime: every nonzero element is invertible
const GENERATOR = 3n;
const MASK64 = (1n << 64n) - 1n;
const EXPONENT_BITS = 32n; // toy key size, see header

/** Normalises any integer into [0, p). */
function field(x, p = FIELD_PRIME) {
  const r = x % p;
  return r < 0n ? r + p : r;
}

/** Square-and-multiply modular exponentiation; the group's only primitive. */
function powMod(base, exponent, modulus = FIELD_PRIME) {
  let acc = 1n;
  let b = field(base, modulus);
  let e = exponent;
  while (e > 0n) {
    if ((e & 1n) === 1n) acc = (acc * b) % modulus;
    b = (b * b) % modulus;
    e >>= 1n;
  }
  return acc;
}

const inverse = (x) => powMod(field(x), FIELD_PRIME - 2n);

/** Deterministic splitmix64: the single source of pseudo-randomness in this file. */
class SplitMix64 {
  constructor(seed) {
    this.state = field(BigInt(seed), 1n << 64n);
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }

  /** Element of F_p built from two 64-bit draws (p is 127 bits). */
  element() {
    let value = 0n;
    for (let i = 0; i < 2; i++) value = (value << 64n) | this.next();
    return value % FIELD_PRIME;
  }

  /** Private exponent in [1, 2^EXPONENT_BITS). */
  exponent() {
    return 1n + (this.next() % ((1n << EXPONENT_BITS) - 1n));
  }
}

/** Pairwise mask derived from the shared DH secret: the protocol's pseudo-random generator. */
function prgMask(pairSeed) {
  return new SplitMix64(field(pairSeed, 1n << 64n)).element();
}

/** Shamir sharing of `secret` over F_p: `parties` shares, any `threshold` of them reconstruct. */
function shamirShare(secret, parties, threshold, rng) {
  if (threshold < 1 || threshold > parties) throw new Error("invalid threshold");
  const coefficients = [field(secret)];
  for (let i = 1; i < threshold; i++) coefficients.push(rng.element());
  const shares = [];
  for (let x = 1; x <= parties; x++) {
    const bx = BigInt(x);
    let y = 0n;
    for (let k = coefficients.length - 1; k >= 0; k--) y = field(y * bx + coefficients[k]);
    shares.push({ x, y });
  }
  return shares;
}

/** Lagrange interpolation at 0; null when the x coordinates are empty or duplicated. */
function shamirReconstruct(shares) {
  const xSet = new Set(shares.map((s) => s.x));
  if (shares.length === 0 || xSet.size !== shares.length) return null;
  let secret = 0n;
  for (const si of shares) {
    let numerator = 1n;
    let denominator = 1n;
    for (const sj of shares) {
      if (sj.x === si.x) continue;
      numerator = field(numerator * field(-BigInt(sj.x)));
      denominator = field(denominator * field(BigInt(si.x) - BigInt(sj.x)));
    }
    secret = field(secret + field(field(si.y) * numerator) * inverse(denominator));
  }
  return secret;
}

/** Adds per-holder share lists: shares of A plus shares of B are shares of A + B. */
function addShareLists(shareLists) {
  const totals = new Map();
  for (const shares of shareLists) {
    for (const share of shares) totals.set(share.x, field((totals.get(share.x) ?? 0n) + share.y));
  }
  return [...totals.entries()].map(([x, y]) => ({ x, y }));
}

/** Reconstructs a shared secret from the shares held by `holders`. */
function reconstructFrom(shares, holders) {
  const set = new Set(holders);
  return shamirReconstruct(shares.filter((s) => set.has(s.x)));
}

/** Key-agreement round: party i publishes g^{a_i}; pair (i, j) agrees on g^{a_i a_j}. */
function keyAgreement(parties, rng) {
  const privateKeys = new Map();
  const publicKeys = new Map();
  for (let i = 1; i <= parties; i++) {
    const a = rng.exponent();
    privateKeys.set(i, a);
    publicKeys.set(i, powMod(GENERATOR, a));
  }
  const seeds = new Map();
  for (let i = 1; i <= parties; i++) {
    for (let j = i + 1; j <= parties; j++) {
      const viaI = powMod(publicKeys.get(j), privateKeys.get(i));
      const viaJ = powMod(publicKeys.get(i), privateKeys.get(j));
      if (viaI !== viaJ) throw new Error("DH shared secret mismatch");
      seeds.set(`${i}:${j}`, viaI);
    }
  }
  return { privateKeys, publicKeys, seeds };
}

/** Signed mask term of ordered pair (i, j): the two orientations are exact negatives. */
function pairMask(pairSeed, i, j) {
  const value = prgMask(pairSeed);
  return i < j ? value : field(-value);
}

/** Any t = floor(n/2) + 1 survivors can restore dropped masks; t - 1 learn nothing. */
const thresholdFor = (parties) => Math.floor(parties / 2) + 1;

/**
 * One full protocol run: masking round plus secret sharing of every self mask and pairwise term.
 * The returned object is exactly the server's view (plus `inputs`, kept here for verification).
 */
function secureAggregateSession(inputs, seed) {
  const n = inputs.length;
  if (n < 2) throw new Error("need at least two parties");
  const rng = new SplitMix64(seed);
  const { seeds } = keyAgreement(n, rng);
  const threshold = thresholdFor(n);
  const masked = new Array(n + 1);
  const selfMaskShares = new Array(n + 1);
  const orphanShares = new Map();
  for (let i = 1; i <= n; i++) {
    const b = rng.element();
    masked[i] = field(field(BigInt(inputs[i - 1])) + b);
    selfMaskShares[i] = shamirShare(b, n, threshold, rng);
    for (let j = 1; j <= n; j++) {
      if (j === i) continue;
      const term = pairMask(seeds.get(i < j ? `${i}:${j}` : `${j}:${i}`), i, j);
      masked[i] = field(masked[i] + term);
      orphanShares.set(`${i}:${j}`, shamirShare(term, n, threshold, rng));
    }
  }
  return { n, threshold, masked, selfMaskShares, orphanShares, inputs };
}

/**
 * Server side. `survivors` may be any set with |S| >= threshold. Recovers the aggregate of the
 * surviving self masks and of every pairwise term split by the dropout, yielding SUM_{i in S} x_i.
 */
function aggregateSurvivors(session, survivors) {
  const { n, threshold, masked, selfMaskShares, orphanShares } = session;
  const alive = new Set(survivors);
  for (const i of alive) {
    if (!Number.isInteger(i) || i < 1 || i > n) throw new Error("invalid survivor id");
  }
  if (alive.size === 0) return 0n;
  if (alive.size < threshold) return null;
  let total = 0n;
  for (const i of alive) total = field(total + masked[i]);
  // SUM_{i in S} b_i: adding the held shares gives shares of the sum, so only the sum is revealed.
  const selfMaskSum = reconstructFrom(addShareLists([...alive].map((i) => selfMaskShares[i])),
    [...alive]);
  if (selfMaskSum === null) return null;
  total = field(total - selfMaskSum);
  for (const i of alive) {
    for (let j = 1; j <= n; j++) {
      if (j === i || alive.has(j)) continue; // both alive: the two orientations still cancel
      const term = reconstructFrom(orphanShares.get(`${i}:${j}`), [...alive]);
      if (term === null) return null;
      total = field(total - term);
    }
  }
  return total;
}

/** Beaver triple: shares of a, b and c = a * b. */
function beaverTriple(parties, threshold, rng) {
  const a = rng.element();
  const b = rng.element();
  return {
    a: shamirShare(a, parties, threshold, rng),
    b: shamirShare(b, parties, threshold, rng),
    c: shamirShare(field(a * b), parties, threshold, rng),
  };
}

/**
 * Multiplies two Shamir-shared secrets. `open` reconstructs a difference with the shares the
 * evaluator holds; the returned share list of x * y needs t shares to open. The three share lists
 * must agree on holder ids, which is checked rather than assumed.
 */
function multiplyShared(xShares, yShares, triple, open) {
  const d = open(xShares, triple.a); // d = x - a, public
  const e = open(yShares, triple.b); // e = y - b, public
  if (d === null || e === null) return null;
  return triple.c.map((share, index) => {
    if (xShares[index].x !== share.x || yShares[index].x !== share.x ||
        triple.a[index].x !== share.x || triple.b[index].x !== share.x) {
      throw new Error("share lists disagree on holder ids");
    }
    return {
      x: share.x,
      y: field(share.y + field(d * triple.b[index].y) + field(e * triple.a[index].y) +
        field(d * e)),
    };
  });
}

/** Secure inner product of two shared vectors, one Beaver triple per product term. */
function secureInnerProduct(xVectorShares, yVectorShares, triples, holders) {
  const open = (secretShares, openingShares) => {
    const secret = reconstructFrom(secretShares, holders);
    const opening = reconstructFrom(openingShares, holders);
    if (secret === null || opening === null) return null;
    return field(secret - opening);
  };
  const terms = [];
  for (let k = 0; k < xVectorShares.length; k++) {
    terms.push(multiplyShared(xVectorShares[k], yVectorShares[k], triples[k], open));
  }
  return addShareLists(terms);
}

/** Independent plaintext oracles used to cross-check the protocol. */
const plainSum = (values) => values.reduce((acc, v) => field(acc + field(BigInt(v))), 0n);

function plainInnerProduct(a, b) {
  let acc = 0n;
  for (let i = 0; i < a.length; i++) acc = field(acc + field(BigInt(a[i]) * field(BigInt(b[i]))));
  return acc;
}

module.exports = { FIELD_PRIME, GENERATOR, field, powMod, inverse, SplitMix64, prgMask,
  shamirShare, shamirReconstruct, addShareLists, reconstructFrom, keyAgreement, pairMask,
  thresholdFor, secureAggregateSession, aggregateSurvivors, beaverTriple, multiplyShared,
  secureInnerProduct, plainSum, plainInnerProduct };

if (require.main === module) {
  const assert = require("node:assert");

  // --- field arithmetic ------------------------------------------------------
  assert.strictEqual(field(-1n), FIELD_PRIME - 1n, "negation wraps into the field");
  assert.strictEqual(field(field(2n) * inverse(2n)), 1n, "inverse is multiplicative");
  assert.strictEqual(powMod(GENERATOR, FIELD_PRIME - 1n), 1n, "Fermat: g^(p-1) = 1 mod p");

  // --- Shamir sharing --------------------------------------------------------
  const shareRng = new SplitMix64(838);
  const secret = shareRng.element();
  const shares = shamirShare(secret, 7, 4, shareRng);
  assert.strictEqual(shamirReconstruct(shares.slice(0, 4)), secret, "exactly t shares reconstruct");
  assert.strictEqual(shamirReconstruct([shares[0], shares[2], shares[5], shares[6]]), secret,
    "any t shares reconstruct");
  assert.notStrictEqual(shamirReconstruct(shares.slice(0, 3)), secret, "t - 1 shares do not");
  assert.notStrictEqual(shares[0].y, secret, "a single share is not the secret");
  assert.strictEqual(shamirReconstruct([shares[0]]), shares[0].y, "one point interpolates to itself");
  assert.strictEqual(shamirReconstruct([shares[0], shares[0], shares[1], shares[2]]), null,
    "duplicate x coordinates are rejected");
  assert.strictEqual(shamirReconstruct([]), null, "no shares -> null");
  assert.throws(() => shamirShare(1n, 3, 4, shareRng), /threshold/, "t > n is rejected");
  assert.strictEqual(shamirReconstruct(addShareLists([shares, shares]).slice(0, 4)),
    field(2n * secret), "adding share lists adds the shared secrets");

  // --- mask cancellation and dropout robustness ------------------------------
  const inputs = [11, 37, 5, 987654321, 42, 7];
  const session = secureAggregateSession(inputs, 20241016);
  const n = inputs.length;
  const everyone = Array.from({ length: n }, (_, i) => i + 1);
  assert.strictEqual(session.n, 6);
  assert.strictEqual(session.threshold, 4, "6 parties -> 4-of-6 threshold");
  assert.strictEqual(aggregateSurvivors(session, everyone), plainSum(inputs),
    "full attendance aggregates to the plaintext sum");

  // Dropping up to n - t parties still yields the exact sum of the survivors.
  const aliveOf = (drop) => everyone.filter((i) => !drop.includes(i));
  let dropoutCases = 0;
  for (const drop of [[1], [6], [3, 4], [2, 5], [5, 6]]) {
    const alive = aliveOf(drop);
    if (alive.length < session.threshold) continue;
    dropoutCases++;
    assert.strictEqual(aggregateSurvivors(session, alive), plainSum(alive.map((i) => inputs[i - 1])),
      `dropout ${drop.join(",")} aggregates exactly`);
  }
  assert.ok(dropoutCases >= 5, `dropout scenarios exercised: ${dropoutCases}`);

  // A wider deployment: 8 parties, 5-of-8 threshold, three simultaneous dropouts still exact.
  const wide = Array.from({ length: 8 }, (_, i) => (i + 1) * 1013);
  const wideSession = secureAggregateSession(wide, 4242);
  assert.strictEqual(wideSession.threshold, 5);
  const wideAlive = [2, 3, 5, 6, 8];
  assert.strictEqual(aggregateSurvivors(wideSession, wideAlive),
    plainSum(wideAlive.map((i) => wide[i - 1])), "5-of-8 with 3 dropouts is exact");
  assert.strictEqual(aggregateSurvivors(wideSession, [2, 3, 5, 6]), null,
    "4-of-8 is below the threshold");

  // Below the threshold the protocol refuses instead of returning a wrong value (liveness boundary).
  const tooFew = aliveOf([1, 2, 3, 5]);
  assert.ok(tooFew.length < session.threshold);
  assert.strictEqual(aggregateSurvivors(session, tooFew), null, "below threshold -> no result");
  assert.strictEqual(aggregateSurvivors(session, []), 0n, "empty survivor set -> additive identity");
  assert.strictEqual(aggregateSurvivors(session, everyone.concat([1])), plainSum(inputs),
    "duplicate survivor ids are idempotent");
  assert.throws(() => aggregateSurvivors(session, [1, 2, 3, 9]), /survivor/, "unknown party id");

  // --- confidentiality measured, not asserted --------------------------------
  const maskedResidues = new Set();
  let maskedEqualsInput = 0;
  for (let trial = 0; trial < 200; trial++) {
    const one = secureAggregateSession([5, 5, 5, 5, 5, 5], 1000 + trial);
    maskedResidues.add(Number(one.masked[1] % 97n));
    if (one.masked[1] === 5n) maskedEqualsInput++;
  }
  assert.ok(maskedResidues.size >= 70,
    `masked values cover many residues mod 97: ${maskedResidues.size}/97`);
  assert.strictEqual(maskedEqualsInput, 0, "a masked value never equals its own input");

  // Even with the aggregate pinned to 0, one party's masked input keeps spreading.
  const partnerResidues = new Set();
  for (let trial = 0; trial < 64; trial++) {
    const rng = new SplitMix64(7000 + trial);
    const x1 = rng.element();
    const pinned = secureAggregateSession([x1, field(-x1)], 999 + trial);
    partnerResidues.add(Number(pinned.masked[1] % 97n));
    assert.strictEqual(aggregateSurvivors(pinned, [1, 2]), 0n, "pinned aggregate is exact");
  }
  assert.ok(partnerResidues.size >= 40,
    `masked input spreads with a fixed aggregate: ${partnerResidues.size}/97 residues`);

  // Pairwise masks cancel by construction, and DH keys are consistent and pair-specific.
  const dh = keyAgreement(4, new SplitMix64(77));
  const seed12 = dh.seeds.get("1:2");
  assert.strictEqual(pairMask(seed12, 1, 2), field(-pairMask(seed12, 2, 1)), "masks cancel");
  assert.strictEqual(seed12, powMod(GENERATOR, dh.privateKeys.get(1) * dh.privateKeys.get(2)),
    "DH agreement equals g^(a1*a2)");
  assert.notStrictEqual(dh.seeds.get("1:2"), dh.seeds.get("1:3"), "pair seeds differ per pair");

  // --- Beaver triples: secure multiplication matches plaintext ---------------
  const mulRng = new SplitMix64(5150);
  const parties = 5;
  const threshold = 3;
  const holders = [1, 2, 3];
  const openWithHolders = (secretShares, openingShares) => field(
    reconstructFrom(secretShares, holders) - reconstructFrom(openingShares, holders));
  for (let trial = 0; trial < 12; trial++) {
    const x = mulRng.element();
    const y = mulRng.element();
    const xShares = shamirShare(x, parties, threshold, mulRng);
    const yShares = shamirShare(y, parties, threshold, mulRng);
    const triple = beaverTriple(parties, threshold, mulRng);
    const product = multiplyShared(xShares, yShares, triple, openWithHolders);
    assert.strictEqual(reconstructFrom(product, holders), field(x * y), `secure product ${trial}`);
    assert.notStrictEqual(reconstructFrom(product.slice(0, threshold - 1), [1, 2]), field(x * y),
      "t - 1 product shares reveal nothing");
  }
  const openWithoutShares = (secretShares, openingShares) => {
    const secret = reconstructFrom(secretShares, [9, 10, 11]);
    const opening = reconstructFrom(openingShares, [9, 10, 11]);
    return secret === null || opening === null ? null : field(secret - opening);
  };
  assert.strictEqual(multiplyShared(shamirShare(5n, parties, threshold, mulRng),
    shamirShare(1n, parties, threshold, mulRng), beaverTriple(parties, threshold, mulRng),
  openWithoutShares), null, "no shares held -> multiplication refuses");
  const misaligned = shamirShare(5n, parties, threshold, mulRng).slice(1);
  assert.throws(() => multiplyShared(misaligned, shamirShare(1n, parties, threshold, mulRng),
    beaverTriple(parties, threshold, mulRng), openWithHolders), /holder ids/,
  "share lists with different holder ids are rejected");

  // --- secure inner product (gradient dot products) --------------------------
  const vectorRng = new SplitMix64(9182);
  const a = [3, 9, 27, 81, 1024];
  const b = [5, 2, 8, 1, 7];
  const sharedVector = (values) =>
    values.map((v) => shamirShare(BigInt(v), parties, threshold, vectorRng));
  const dotHolders = [2, 4, 5];
  const buildDot = (va, vb) => secureInnerProduct(sharedVector(va), sharedVector(vb),
    va.map(() => beaverTriple(parties, threshold, vectorRng)), dotHolders);
  assert.strictEqual(reconstructFrom(buildDot(a, b), dotHolders), plainInnerProduct(a, b),
    "secure inner product equals the plaintext oracle");
  let dotTrials = 0;
  for (let trial = 0; trial < 8; trial++) {
    const va = [1, 2, 3, 4].map(() => Number(vectorRng.element() % 4096n));
    const vb = [4, 3, 2, 1].map(() => Number(vectorRng.element() % 4096n));
    dotTrials++;
    assert.strictEqual(reconstructFrom(buildDot(va, vb), dotHolders), plainInnerProduct(va, vb),
      `random inner product ${trial}`);
  }
  assert.strictEqual(dotTrials, 8);

  // --- secure mean of per-party gradients ------------------------------------
  const gradientRng = new SplitMix64(2468);
  const gradientCount = 4;
  const partyGradients = Array.from({ length: gradientCount }, () =>
    Array.from({ length: gradientCount }, () => Number(gradientRng.element() % 1000n)));
  const columnSums = partyGradients[0].map((_, column) =>
    plainSum(partyGradients.map((row) => row[column])));
  const meanShares = columnSums.map((value) =>
    shamirShare(value, parties, threshold, gradientRng).map((s) =>
      ({ x: s.x, y: field(s.y * inverse(BigInt(gradientCount))) })));
  const openedMean = meanShares.map((s) => reconstructFrom(s, [1, 3, 5]));
  assert.deepStrictEqual(openedMean,
    columnSums.map((value) => field(value * inverse(BigInt(gradientCount)))),
    "secure mean matches the plaintext mean");
  assert.notDeepStrictEqual(partyGradients[0].map((v) => field(BigInt(v))), openedMean,
    "the mean differs from any single party's gradient");

  // --- determinism -----------------------------------------------------------
  const replay = secureAggregateSession(inputs, 20241016);
  assert.deepStrictEqual(replay.masked, session.masked, "same seed -> identical transcript");
  assert.strictEqual(aggregateSurvivors(replay, [2, 3, 4, 5, 6]),
    aggregateSurvivors(session, [2, 3, 4, 5, 6]), "replay aggregates identically");

  console.log(`parties=6 threshold=4 dropout-scenarios=${dropoutCases}`);
  console.log(`confidentiality: masked residues ${maskedResidues.size}/97, pinned-aggregate ` +
    `residues ${partnerResidues.size}/97, masked==input count ${maskedEqualsInput}`);
  console.log(`secure products=12 inner products=${dotTrials + 1} mean-dimension=${gradientCount}`);
  console.log("PASS");
}
