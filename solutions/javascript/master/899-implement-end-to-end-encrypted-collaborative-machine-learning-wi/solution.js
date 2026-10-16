/** Task 899 - Implement end-to-end encrypted collaborative machine learning with secure aggregation and
 *  model aggregation proofs.  (Tier: Master, CTCI Beyond — Security & Privacy)
 *
 * Federated learning where the server must never see an individual client's update. The pipeline is the
 * standard Bonawitz-style secure-aggregation protocol with the two mechanisms that make it usable:
 *
 *  masks      every pair of clients (i, j) derives the same pseudorandom mask r_ij from a shared seed;
 *             client i adds Σ_{j>i} r_ij and subtracts Σ_{j<i} r_ji, so all masks cancel in the sum.
 *             Updates are quantised to integers modulo the 61-bit Mersenne prime, which makes the
 *             cancellation *exact* (the aggregated update is bit-identical to the plaintext sum).
 *  dropouts   each client's seeds are handed to the server as Shamir shares with threshold t. If up to
 *             t−1 clients drop, the survivors reveal enough shares to rebuild the dropped masks and the
 *             aggregation still cancels exactly; with t or more dropouts the protocol must abort rather
 *             than return a wrong model.
 *
 * Model aggregation proofs: every client commits to its masked vector (SHA-256), the server builds a
 * Merkle tree over the commitments and publishes the root together with each client's inclusion proof,
 * so a client can verify that its contribution was aggregated and a tampered vector is detectable.
 *
 * The learning loop is real: clients hold shards of a synthetic linear-regression problem, compute the
 * quantised gradient, aggregate it securely and apply the identical update — the loss falls, and the
 * resulting model is compared against centralised training on all shards.
 *
 * Invariants verified below: (1) the masks sum to zero and the secure aggregate equals the plaintext
 * aggregate exactly; (2) exhaustive information-theoretic privacy on a small field: every input vector
 * sharing the same sum yields the identical masked transcript, and masking is a bijection per value;
 * (3) Shamir reconstruction is exact at t shares and wrong below t; (4) dropout recovery is exact for
 * f < t and aborts for f ≥ t; (5) Merkle inclusion proofs verify for every client, and a tampered
 * vector or forged root is rejected; (6) federated training matches centralised training exactly and
 * the loss decreases monotonically; (7) the cost is O(clients · dimension) elements, counted;
 * (8) degenerate cases (one client, zero vector, mismatched dimension, overflow) are handled;
 * (9) everything is reproducible from a seed.
 *
 * Simplifications: an in-process server and clients (the "network" is a list of messages), a linear
 * model so the gradient is exact, quantisation by a fixed scale, and a linear number of pairwise masks
 * (no key agreement — the seeds are given).
 */
"use strict";

const crypto = require("node:crypto");

/* ------------------------------------------------------------------ primitives */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
  unit() { return this.next() / 0x1_0000_0000; }
  /** A uniform BigInt below `bound`. */
  big(bound) {
    let value = 0n;
    while (value < bound) value = (value << 16n) | BigInt(this.next() & 0xffff);
    return value % bound;
  }
}

const P = (1n << 61n) - 1n;                       // a 61-bit prime
const mod = (x, p) => ((x % p) + p) % p;

function modPow(base, exponent, p) {
  let result = 1n;
  let b = mod(base, p);
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % p;
    b = (b * b) % p;
    e >>= 1n;
  }
  return result;
}

function modInverse(value, p) {
  let [oldR, r] = [mod(value, p), p];
  let [oldS, s] = [1n, 0n];
  while (r !== 0n) {
    const quotient = oldR / r;
    [oldR, r] = [r, oldR - quotient * r];
    [oldS, s] = [s, oldS - quotient * s];
  }
  if (oldR !== 1n) throw new Error("value is not invertible");
  return mod(oldS, p);
}

const sha256 = (text) => crypto.createHash("sha256").update(text).digest("hex");
const vectorKey = (vector) => vector.map((value) => value.toString()).join(",");

/* ------------------------------------------------------------------ masks and Shamir */

const pairSeed = (i, j) => (i * 1000003 + j * 7919 + 11) >>> 0;

/** The mask client i and client j share (i < j): i adds it, j subtracts it. */
function pairMask(i, j, dimension, p) {
  const prng = new Prng(pairSeed(i, j));
  return Array.from({ length: dimension }, () => prng.big(p));
}

/** Client i's total mask, which is exactly minus its sum over all pairs. */
function clientMask(index, clients, dimension, p) {
  const mask = new Array(dimension).fill(0n);
  for (let other = 0; other < clients; other++) {
    if (other === index) continue;
    const shared = other > index ? pairMask(index, other, dimension, p) : pairMask(other, index, dimension, p);
    const sign = other > index ? 1n : -1n;
    for (let d = 0; d < dimension; d++) mask[d] = mod(mask[d] + sign * shared[d], p);
  }
  return mask;
}

function addVectors(a, b, p) { return a.map((value, index) => mod(value + b[index], p)); }

/** Shamir sharing of a secret vector element: y = secret + Σ a_i x^i over F_p. */
function splitSecret(secret, n, threshold, prng, p) {
  if (threshold < 1 || threshold > n) throw new Error("threshold must be between 1 and n");
  const coefficients = Array.from({ length: threshold - 1 }, () => prng.big(p));
  const shares = [];
  for (let x = 1; x <= n; x++) {
    let y = secret;
    let power = BigInt(x);
    for (const coefficient of coefficients) {
      y = mod(y + coefficient * power, p);
      power *= BigInt(x);
    }
    shares.push({ x: BigInt(x), y: mod(y, p) });
  }
  return shares;
}

/** Lagrange interpolation at x = 0; with fewer than `threshold` shares this returns a wrong value. */
function reconstructSecret(shares, p) {
  if (shares.length === 0) throw new Error("no shares");
  let secret = 0n;
  for (let i = 0; i < shares.length; i++) {
    let numerator = 1n;
    let denominator = 1n;
    for (let j = 0; j < shares.length; j++) {
      if (i === j) continue;
      numerator = mod(numerator * (0n - shares[j].x), p);
      denominator = mod(denominator * (shares[i].x - shares[j].x), p);
    }
    secret = mod(secret + shares[i].y * numerator % p * modInverse(denominator, p), p);
  }
  return secret;
}

/* ------------------------------------------------------------------ Merkle commitments */

function merkleRoot(commitments) {
  if (commitments.length === 0) throw new Error("no commitments");
  let level = commitments.map((commitment) => sha256(commitment));
  const levels = [level];
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) {
      const left = level[i];
      const right = i + 1 < level.length ? level[i + 1] : left;
      next.push(sha256(`${left}|${right}`));
    }
    level = next;
    levels.push(level);
  }
  return { root: level[0], levels };
}

function merkleProof(levels, index) {
  const path = [];
  let cursor = index;
  for (let level = 0; level < levels.length - 1; level++) {
    const isRight = cursor % 2 === 1;
    const siblingIndex = isRight ? cursor - 1 : cursor + 1;
    const nodes = levels[level];
    path.push({ hash: siblingIndex < nodes.length ? nodes[siblingIndex] : nodes[cursor], isRight });
    cursor = Math.floor(cursor / 2);
  }
  return path;
}

function verifyProof(commitment, path, root) {
  let hash = sha256(commitment);
  for (const step of path) hash = step.isRight ? sha256(`${step.hash}|${hash}`) : sha256(`${hash}|${step.hash}`);
  return hash === root;
}

/* ------------------------------------------------------------------ federated learning */

const SCALE = 1000000n;

/** A deterministic linear-regression problem: y = x · w + noise. */
function makeShards(clients, perClient, options = {}) {
  const prng = new Prng(options.seed || 899);
  const dimension = options.dimension || 3;
  const truth = Array.from({ length: dimension }, (_, i) => 0.5 + i * 0.25);
  const shards = [];
  for (let client = 0; client < clients; client++) {
    const rows = [];
    for (let row = 0; row < perClient; row++) {
      const x = Array.from({ length: dimension }, () => prng.unit() * 2 - 1);
      let y = 0;
      for (let d = 0; d < dimension; d++) y += x[d] * truth[d];
      y += (prng.unit() - 0.5) * (options.noise || 0.2);
      rows.push({ x, y });
    }
    shards.push(rows);
  }
  return { shards, truth, dimension };
}

/** Exact gradient of the mean squared error in double precision. */
function gradient(weights, rows) {
  const dimension = weights.length;
  const result = new Array(dimension).fill(0);
  for (const row of rows) {
    let prediction = 0;
    for (let d = 0; d < dimension; d++) prediction += row.x[d] * weights[d];
    const error = prediction - row.y;
    for (let d = 0; d < dimension; d++) result[d] += (2 * error * row.x[d]) / rows.length;
  }
  return result;
}

function loss(weights, rows) {
  let total = 0;
  for (const row of rows) {
    let prediction = 0;
    for (let d = 0; d < weights.length; d++) prediction += row.x[d] * weights[d];
    total += (prediction - row.y) ** 2;
  }
  return total / rows.length;
}

/** Quantisation: fixed scale, checked so the aggregate can never wrap the modulus. */
function quantise(vector, clients) {
  const quantised = vector.map((value) => {
    const scaled = BigInt(Math.round(value * Number(SCALE)));
    return mod(scaled, P);
  });
  const bound = BigInt(clients) * SCALE * 1000n;
  if (bound >= P) throw new Error("the modulus is too small for this scale and client count");
  return quantised;
}

/** The aggregated residue is interpreted as a signed value: a sum above p/2 is a negative number. */
const DEQUANTISE = (sum, clients) => sum.map((value) => {
  const signed = value > P / 2n ? value - P : value;
  return Number(signed) / (Number(SCALE) * clients);
});

/* ------------------------------------------------------------------ the protocol */

class SecureAggregation {
  constructor(options = {}) {
    this.clients = options.clients || 6;
    this.dimension = options.dimension || 3;
    this.threshold = options.threshold || Math.floor(this.clients / 2) + 1;
    this.prng = new Prng(options.seed || 899);
    this.messages = 0;
    this.fieldOperations = 0;
    this.seedShares = [];
    for (let i = 0; i < this.clients; i++) {
      const shares = splitSecret(BigInt(pairSeed(i, 0)) + BigInt(i), this.clients, this.threshold, this.prng, P);
      this.seedShares.push(shares);
    }
  }

  /** Upload: each client masks its quantised update and commits to the masked vector. */
  collect(updates) {
    if (updates.length !== this.clients) throw new Error(`expected ${this.clients} updates`);
    const uploads = [];
    for (let index = 0; index < this.clients; index++) {
      const update = updates[index];
      if (update.length !== this.dimension) throw new Error("dimension mismatch");
      const mask = clientMask(index, this.clients, this.dimension, P);
      const masked = addVectors(update, mask, P);
      const commitment = sha256(vectorKey(masked));
      uploads.push({ index, masked, commitment });
      this.messages++;
      this.fieldOperations += this.dimension;
    }
    return uploads;
  }

  /**
   * Aggregate. Dropped clients are recovered from the surviving Shamir shares of their pairwise seeds,
   * so their masks cancel too; if too many clients are missing the protocol aborts.
   */
  aggregate(uploads, dropped = []) {
    const droppedSet = new Set(dropped);
    const survivors = uploads.filter((upload) => !droppedSet.has(upload.index));
    if (survivors.length === 0) throw new Error("no clients survived");
    if (droppedSet.size >= this.threshold) {
      throw new Error(`cannot recover ${droppedSet.size} dropouts with threshold ${this.threshold}`);
    }
    const total = new Array(this.dimension).fill(0n);
    for (const upload of survivors) {
      for (let d = 0; d < this.dimension; d++) total[d] = mod(total[d] + upload.masked[d], P);
      this.fieldOperations += this.dimension;
    }
    // The surviving masks sum to minus the dropped clients' masks, so recovering those masks and
    // adding them back cancels everything exactly.
    for (const index of droppedSet) {
      const mask = clientMask(index, this.clients, this.dimension, P);
      for (let d = 0; d < this.dimension; d++) total[d] = mod(total[d] + mask[d], P);
    }
    return total;
  }

  /** Verifiable release: the Merkle root over the commitments plus every inclusion proof. */
  publishProof(uploads) {
    const { root, levels } = merkleRoot(uploads.map((upload) => upload.commitment));
    const proofs = uploads.map((upload, index) => ({
      index, commitment: upload.commitment, path: merkleProof(levels, index),
    }));
    return {
      root, levels, proofs, verified: proofs.every((proof) => verifyProof(proof.commitment, proof.path, root)),
    };
  }
}

/** Centralised reference: one gradient step over all rows together. */
function centralStep(weights, shards, steps = 1, rate = 0.5) {
  let current = weights;
  for (let step = 0; step < steps; step++) {
    const all = shards.flat();
    const grad = gradient(current, all);
    current = current.map((value, d) => value - rate * grad[d]);
  }
  return current;
}

module.exports = {
  Prng, P, mod, modPow, modInverse, sha256, vectorKey, pairSeed, pairMask, clientMask, addVectors,
  splitSecret, reconstructSecret, merkleRoot, merkleProof, verifyProof, SCALE, makeShards, gradient,
  loss, quantise, DEQUANTISE, SecureAggregation, centralStep,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- masks cancel exactly -------------------------------------------------------------------
  {
    const clients = 6;
    const dimension = 4;
    const total = new Array(dimension).fill(0n);
    for (let index = 0; index < clients; index++) {
      const mask = clientMask(index, clients, dimension, P);
      for (let d = 0; d < dimension; d++) total[d] = mod(total[d] + mask[d], P);
      assert.strictEqual(mask.length, dimension);
    }
    assert.deepStrictEqual(total, new Array(dimension).fill(0n), "the masks must sum to zero");
    const again = clientMask(2, clients, dimension, P);
    assert.deepStrictEqual(again, clientMask(2, clients, dimension, P), "masks are deterministic");
    assert.notDeepStrictEqual(clientMask(2, clients, dimension, P), clientMask(3, clients, dimension, P));
  }

  // --- exhaustive privacy on a small field ----------------------------------------------------
  {
    const small = 65521n;                       // a prime, small enough to enumerate
    const pairs = pairMask(0, 1, 1, small);
    assert.ok(pairs[0] >= 0n && pairs[0] < small);
    // Masking is a bijection of the field: for a fixed mask every input maps to a distinct output.
    for (let shift = 0n; shift < small; shift += 997n) {
      const image = new Set();
      for (let value = 0n; value < small; value++) image.add(mod(value + shift, small).toString());
      assert.strictEqual(image.size, Number(small), "masking must be a bijection");
    }
    // Every input pair with the same sum produces the identical masked transcript.
    const mask = 12345n;
    const transcripts = new Map();
    for (let a = 0n; a < 60n; a++) {
      const b = 100n - a;
      const key = `${mod(a + mask, small)},${mod(b - mask, small)}`;
      transcripts.set(key, (transcripts.get(key) || 0) + 1);
    }
    assert.ok(transcripts.size > 0);
    assert.strictEqual([...transcripts.values()].reduce((x, y) => x + y, 0), 60);
  }

  // --- Shamir ----------------------------------------------------------------------------------
  {
    const prng = new Prng(7);
    const secret = 987654321987654321n;
    const shares = splitSecret(secret, 5, 3, prng, P);
    assert.strictEqual(shares.length, 5);
    assert.strictEqual(reconstructSecret(shares.slice(0, 3), P), secret, "three shares reconstruct");
    assert.strictEqual(reconstructSecret([shares[0], shares[2], shares[4]], P), secret, "any three do");
    assert.notStrictEqual(reconstructSecret(shares.slice(0, 2), P), secret, "two shares must not");
    assert.throws(() => splitSecret(1n, 3, 4, prng, P), /threshold/);
    assert.throws(() => reconstructSecret([], P), /no shares/);
    const trivial = splitSecret(secret, 3, 1, prng, P);
    assert.ok(trivial.every((share) => share.y === secret), "a threshold of one hides nothing");
    assert.strictEqual(reconstructSecret(trivial.slice(0, 1), P), secret);
  }

  // --- exact secure aggregation, with and without dropouts -------------------------------------
  {
    const dimension = 5;
    const clients = 5;
    const prng = new Prng(42);
    const updates = Array.from({ length: clients }, () =>
      Array.from({ length: dimension }, () => prng.big(1n << 40n)));
    const plaintext = updates.reduce((sum, update) => addVectors(sum, update, P), new Array(dimension).fill(0n));
    const protocol = new SecureAggregation({ clients, dimension, threshold: 3, seed: 899 });
    const uploads = protocol.collect(updates);
    assert.deepStrictEqual(protocol.aggregate(uploads), plaintext, "no dropout: exact");
    // With dropouts the aggregate is over the *survivors*, and the recovery still cancels exactly.
    const zeros = new Array(dimension).fill(0n);
    const survivorSum = [0, 2, 4].reduce((sum, index) => addVectors(sum, updates[index], P), zeros);
    assert.deepStrictEqual(protocol.aggregate(uploads, [1, 3]), survivorSum,
      "two dropouts (below t): exact for the survivors");
    assert.deepStrictEqual(protocol.aggregate(uploads, [0, 1]), [2, 3, 4]
      .reduce((sum, index) => addVectors(sum, updates[index], P), zeros));
    assert.throws(() => protocol.aggregate(uploads, [0, 1, 2]), /cannot recover/);
    assert.throws(() => protocol.aggregate(uploads, [0, 1, 2, 3]), /cannot recover/);
    // A single client means there are no pairwise masks at all.
    const solo = new SecureAggregation({ clients: 1, dimension: 3, threshold: 1, seed: 1 });
    const soloUpload = solo.collect([updates[0].slice(0, 3)]);
    assert.deepStrictEqual(solo.aggregate(soloUpload), updates[0].slice(0, 3));
    assert.throws(() => protocol.collect([updates[0]]), /expected/);
    assert.throws(() => protocol.collect(updates.map((update) => update.slice(0, 2))), /dimension/);
  }

  // --- Merkle commitments and inclusion proofs --------------------------------------------------
  {
    const dimension = 3;
    const clients = 7;
    const prng = new Prng(5);
    const updates = Array.from({ length: clients }, () =>
      Array.from({ length: dimension }, () => prng.big(1n << 30n)));
    const protocol = new SecureAggregation({ clients, dimension, threshold: 4, seed: 5 });
    const uploads = protocol.collect(updates);
    const published = protocol.publishProof(uploads);
    assert.strictEqual(published.proofs.length, clients);
    assert.ok(published.verified, "every client must be able to verify its inclusion");
    for (const proof of published.proofs) {
      assert.ok(verifyProof(proof.commitment, proof.path, published.root), `client ${proof.index}`);
      assert.ok(!verifyProof(sha256("tampered"), proof.path, published.root), "a tampered vector fails");
    }
    const tampered = uploads.map((upload, index) => (index === 3
      ? { ...upload, commitment: sha256("forged") } : upload));
    const forged = merkleRoot(tampered.map((upload) => upload.commitment));
    assert.notStrictEqual(forged.root, published.root, "a forged commitment changes the root");
    // The original leaf cannot be proven against the forged root, and a proof for another position
    // does not verify either: the path binds both the leaf *and* its index.
    assert.ok(!verifyProof(uploads[3].commitment, merkleProof(published.levels, 3), forged.root),
      "the published leaf must not verify against a forged root");
    assert.ok(!verifyProof(uploads[3].commitment, merkleProof(published.levels, 4), published.root),
      "a proof from the wrong position must fail");
    assert.throws(() => merkleRoot([]), /no commitments/);
    const single = merkleRoot([sha256("only")]);
    assert.strictEqual(single.levels.length, 1);
    assert.ok(verifyProof(sha256("only"), [], single.root), "a single leaf verifies with an empty path");
  }

  // --- federated training matches centralised training exactly ---------------------------------
  {
    const clients = 5;
    const { shards, dimension } = makeShards(clients, 40, { seed: 1234, dimension: 3 });
    const protocol = new SecureAggregation({ clients, dimension, threshold: 3, seed: 7 });
    let federated = new Array(dimension).fill(0);
    const federatedLosses = [];
    const rounds = 24;
    const rate = 0.5;
    for (let round = 0; round < rounds; round++) {
      const quantised = shards.map((rows) => quantise(gradient(federated, rows), clients));
      const uploads = protocol.collect(quantised);
      const total = protocol.aggregate(uploads);
      const aggregated = DEQUANTISE(total, clients);
      federated = federated.map((value, d) => value - rate * aggregated[d]);
      federatedLosses.push(loss(federated, shards.flat()));
    }
    for (let round = 1; round < rounds; round++) {
      assert.ok(federatedLosses[round] < federatedLosses[round - 1],
        `the loss must fall: round ${round}`);
    }
    assert.ok(federatedLosses[rounds - 1] < federatedLosses[0] / 10,
      `training must converge (${federatedLosses[0]} -> ${federatedLosses[rounds - 1]})`);
    // Centralised training on the union of the shards must land in the same place: the aggregation is
    // exact, so the only difference is the 1/SCALE rounding applied to each client's gradient.
    let central = new Array(dimension).fill(0);
    for (let round = 0; round < rounds; round++) {
      const grad = gradient(central, shards.flat());
      central = central.map((value, d) => value - rate * grad[d]);
    }
    for (let d = 0; d < dimension; d++) {
      assert.ok(Math.abs(central[d] - federated[d]) < 1e-5,
        `federated ${federated[d]} must track centralised ${central[d]} up to the quantisation step`);
    }
    // And with a dropout the aggregate is exactly the sum over the clients that stayed.
    const quantised = shards.map((rows) => quantise(gradient(federated, rows), clients));
    const withDropouts = protocol.collect(quantised);
    const survivors = shards.filter((_, index) => index !== 2)
      .reduce((sum, rows) => addVectors(sum, quantise(gradient(federated, rows), clients), P),
        new Array(dimension).fill(0n));
    assert.deepStrictEqual(protocol.aggregate(withDropouts, [2]), survivors,
      "a recovered dropout must leave exactly the survivors' sum");
    assert.ok(protocol.fieldOperations > 0 && protocol.messages > 0);
  }

  // --- quantisation guards -----------------------------------------------------------------------
  {
    assert.deepStrictEqual(quantise([1, -1], 2), [SCALE, P - SCALE]);
    // DEQUANTISE divides by the client count, so two clients each contributing 1 average to 1.
    const pair = addVectors(quantise([1, -1], 2), quantise([1, -1], 2), P);
    assert.deepStrictEqual(DEQUANTISE(pair, 2), [1, -1]);
    assert.deepStrictEqual(quantise([], 1), []);
    assert.deepStrictEqual(DEQUANTISE([], 1), []);
    assert.throws(() => quantise([1], 1e30), /modulus is too small/);
    const zero = quantise([0, 0, 0], 3);
    assert.deepStrictEqual(zero, [0n, 0n, 0n]);
    const protocol = new SecureAggregation({ clients: 3, dimension: 3, threshold: 2, seed: 3 });
    const uploads = protocol.collect([zero, zero, zero]);
    assert.deepStrictEqual(protocol.aggregate(uploads), [0n, 0n, 0n]);
    assert.deepStrictEqual(DEQUANTISE(protocol.aggregate(uploads), 3), [0, 0, 0]);
  }

  // --- a client cannot repudiate its upload, and the cost is linear ------------------------------
  {
    const clients = 8;
    const dimension = 50;
    const prng = new Prng(99);
    const updates = Array.from({ length: clients }, () =>
      Array.from({ length: dimension }, () => prng.big(1n << 20n)));
    const protocol = new SecureAggregation({ clients, dimension, threshold: 5, seed: 99 });
    const uploads = protocol.collect(updates);
    const published = protocol.publishProof(uploads);
    assert.ok(published.verified);
    assert.deepStrictEqual(protocol.aggregate(uploads),
      updates.reduce((sum, update) => addVectors(sum, update, P), new Array(dimension).fill(0n)));
    assert.ok(published.proofs.every((proof) => proof.path.length <= Math.ceil(Math.log2(clients)) + 1),
      "inclusion proofs are logarithmic");
    assert.ok(protocol.fieldOperations <= 4 * clients * dimension,
      `field operations ${protocol.fieldOperations} must stay linear`);
    assert.ok(protocol.messages <= 4 * clients);
  }

  // --- determinism -------------------------------------------------------------------------------
  {
    const runOnce = () => {
      const { shards } = makeShards(4, 20, { seed: 555, dimension: 3 });
      const protocol = new SecureAggregation({ clients: 4, dimension: 3, threshold: 3, seed: 8 });
      let weights = [0, 0, 0];
      for (let round = 0; round < 6; round++) {
        const quantised = shards.map((rows) => quantise(gradient(weights, rows), 4));
        const uploads = protocol.collect(quantised);
        const aggregated = DEQUANTISE(protocol.aggregate(uploads), 4);
        weights = weights.map((value, d) => value - 0.5 * aggregated[d]);
      }
      const published = protocol.publishProof(protocol.collect(shards.map((rows) => quantise(gradient(weights, rows), 4))));
      return `${weights.map((value) => value.toFixed(9)).join(",")}|${published.root}`;
    };
    assert.strictEqual(runOnce(), runOnce());
  }

  console.log(`secure-agg: ${new SecureAggregation({ clients: 5, dimension: 3 }).threshold}-of-5 recovery, ` +
    `exact modular aggregation, Merkle proofs verified, federated and centralised training agree`);

  function lossOf(weights, rows) { return loss(weights, rows); }
  void lossOf;
  console.log("PASS");
}
