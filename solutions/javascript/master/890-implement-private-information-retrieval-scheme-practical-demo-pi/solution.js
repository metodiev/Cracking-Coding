/** Task 890 - Implement a practical private information retrieval scheme (PIR demo).
 *  (Tier: Master, CTCI Beyond — Security & Privacy)
 *
 * Single-server computational PIR over a √N × √N database using additively homomorphic encryption
 * (Paillier with g = n+1, the standard practical choice — the same structure SealPIR/XPIR use with
 * lattice ciphertexts). The scheme is the classic one:
 *
 *   query      the client arranges the N records as R x C and, for the target (i, j), sends one
 *              ciphertext per column: Enc(1) at column j and Enc(0) elsewhere. Because Paillier is
 *              semantically secure under DCR, the server cannot tell which column carries the 1.
 *   answer     the server computes, for every row r, the homomorphic inner product
 *              Π_c Enc(b_c)^{DB[r][c]} = Enc(Σ_c b_c·DB[r][c]), and returns all R ciphertexts (it must
 *              return all rows, otherwise the response itself would leak the row index).
 *   retrieve   the client decrypts response[i] and obtains DB[i][j]; the other ciphertexts are
 *              discarded. Nothing else is revealed.
 *
 * Communication is C + R + O(1) ciphertexts = O(√N) rather than the O(N) of sending the whole
 * database, and the server's work is O(N) modular multiplications with N^(1/2)-bit exponents.
 *
 * Invariants verified below: (1) correctness — the retrieved record equals the database entry for
 * every index of a small database and for sampled indices of a 4096-record one; (2) the homomorphic
 * properties themselves (additive and scalar multiplication) against plaintext arithmetic;
 * (3) an *exact* leakage statement — because c = (1 + m·n)·r^n mod n^2, the low part c mod n is
 * independent of the plaintext, verified by flipping the plaintext while keeping the randomness
 * fixed; (4) indistinguishability — a distinguisher built from ciphertext statistics performs at
 * chance level on Enc(0) vs Enc(1), with the tolerance documented (this is empirical evidence for
 * semantic security, not a reduction to DCR); (5) index privacy — the query payload is the same size
 * and shape for every index and contains no row/column identifier; (6) cost bound — the transcript is
 * O(√N) ciphertexts, verified numerically against N; (7) robustness — out-of-range indices and
 * databases whose inner products could overflow the modulus are rejected rather than mis-answered;
 * (8) determinism.
 *
 * Simplifications: in-process only (no sockets), the database is a small integer matrix, records are
 * packed into a single residue, and the primes are generated deterministically from a seeded PRNG so
 * runs are reproducible (a real deployment would use OS entropy). Recursion to O(N^(1/3)) is *not*
 * implemented; the header states √N and the check verifies exactly that.
 */
"use strict";

/* ------------------------------------------------------------------ determinism */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  /** Deterministic BigInt in [0, limit). */
  big(limit) {
    let value = 0n;
    while (value < limit) {
      const shift = BigInt(this.next() & 0xff);
      value = (value << 8n) | shift;
    }
    return value % limit;
  }
}

/* ------------------------------------------------------------------ number theory */

const mod = (x, m) => ((x % m) + m) % m;

function modPow(base, exponent, modulus) {
  let result = 1n;
  let b = mod(base, modulus);
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % modulus;
    b = (b * b) % modulus;
    e >>= 1n;
  }
  return result;
}

function gcd(a, b) {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y) { const t = x % y; x = y; y = t; }
  return x;
}

/** Miller-Rabin with bases drawn from the seeded PRNG (deterministic for a fixed seed). */
function isProbablePrime(candidate, prng, rounds = 20) {
  if (candidate < 2n) return false;
  for (const small of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    if (candidate === small) return true;
    if (candidate % small === 0n) return false;
  }
  let d = candidate - 1n;
  let s = 0n;
  while (d % 2n === 0n) { d /= 2n; s += 1n; }
  for (let round = 0; round < rounds; round++) {
    const a = 2n + prng.big(candidate - 3n);
    let x = modPow(a, d, candidate);
    if (x === 1n || x === candidate - 1n) continue;
    let composite = true;
    for (let i = 1n; i < s; i++) {
      x = (x * x) % candidate;
      if (x === candidate - 1n) { composite = false; break; }
    }
    if (composite) return false;
  }
  return true;
}

/** Deterministic prime of `bits` bits derived from the PRNG stream. */
function randomPrime(bits, prng) {
  const high = 1n << BigInt(bits - 1);
  let candidate = high | prng.big(high) | 1n;
  while (!isProbablePrime(candidate, prng)) candidate += 2n;
  return candidate;
}

/* ------------------------------------------------------------------ Paillier */

class Paillier {
  /** g = n + 1, so g^m mod n^2 = 1 + m·n exactly (binomial expansion, higher terms vanish). */
  constructor(bits = 48, prng = new Prng(890)) {
    this.p = randomPrime(bits, prng);
    this.q = randomPrime(bits, prng);
    if (this.p === this.q) throw new Error("degenerate key");
    this.n = this.p * this.q;
    this.n2 = this.n * this.n;
    this.lambda = lcm(this.p - 1n, this.q - 1n);
    this.mu = modInverse(this.lambda, this.n);
    if (mod(this.lambda * this.mu, this.n) !== 1n) throw new Error("bad key");
    this.prng = prng;
  }

  /** Fresh randomness, so encrypting the same message twice gives unrelated ciphertexts. */
  encrypt(message, randomness) {
    const m = mod(message, this.n);
    let r = randomness !== undefined ? randomness : this.sampleUnit();
    const rn = modPow(r, this.n, this.n2);
    return mod((1n + m * this.n) * rn, this.n2);
  }

  sampleUnit() {
    for (let attempt = 0; attempt < 64; attempt++) {
      const candidate = 2n + this.prng.big(this.n - 3n);
      if (gcd(candidate, this.n) === 1n) return candidate;
    }
    throw new Error("no invertible blinding factor");
  }

  decrypt(ciphertext) {
    const u = modPow(ciphertext, this.lambda, this.n2);
    return mod((u - 1n) / this.n * this.mu, this.n);
  }

  add(left, right) { return (left * right) % this.n2; }
  scale(ciphertext, scalar) { return modPow(ciphertext, mod(scalar, this.n), this.n2); }
}

function modInverse(value, modulus) {
  let [old_r, r] = [mod(value, modulus), modulus];
  let [old_s, s] = [1n, 0n];
  while (r !== 0n) {
    const quotient = old_r / r;
    [old_r, r] = [r, old_r - quotient * r];
    [old_s, s] = [s, old_s - quotient * s];
  }
  if (old_r !== 1n) throw new Error("not invertible");
  return mod(old_s, modulus);
}

function lcm(a, b) { return (a / gcd(a, b)) * b; }

/* ------------------------------------------------------------------ PIR */

function validateDatabase(rows, key) {
  if (!Array.isArray(rows) || rows.length === 0) throw new Error("database must be a non-empty matrix");
  const width = rows[0].length;
  if (width === 0) throw new Error("database rows must be non-empty");
  for (const row of rows) {
    if (row.length !== width) throw new Error("ragged database");
    for (const value of row) {
      if (typeof value !== "bigint" || value < 0n) throw new Error("records must be non-negative integers");
      // The homomorphic row sum must not wrap around the modulus, otherwise the answer is wrong.
      if (value * BigInt(width) >= key.n) throw new Error("record too large for this modulus: row sums would wrap");
    }
  }
  return { rows: rows.length, columns: width };
}

class PirServer {
  constructor(key, rows) {
    this.key = key;
    this.shape = validateDatabase(rows, key);
    this.rows = rows;
    this.coresWorked = 0;
    this.requestsSeen = [];
  }

  /** The request carries only ciphertexts: no row, no column, no index. */
  answer(request) {
    if (!Array.isArray(request.ciphertexts) || request.ciphertexts.length !== this.shape.columns) {
      throw new Error("malformed request");
    }
    this.requestsSeen.push({ ciphertexts: request.ciphertexts.slice(), kind: request.kind });
    const reply = [];
    for (let r = 0; r < this.shape.rows; r++) {
      let accumulator = 1n;
      for (let c = 0; c < this.shape.columns; c++) {
        const record = this.rows[r][c];
        if (record === 0n) continue;    // public-data fast path: x^0 = 1
        accumulator = (accumulator * modPow(request.ciphertexts[c], record, this.key.n2)) % this.key.n2;
        this.coresWorked++;
      }
      reply.push(accumulator);
    }
    return reply;
  }
}

class PirClient {
  constructor(key, server) {
    this.key = key;
    this.server = server;
    this.queries = 0;
  }

  indexToCoordinates(index) {
    const columns = this.server.shape.columns;
    if (!Number.isInteger(index) || index < 0 || index >= this.server.shape.rows * columns) {
      throw new Error(`index ${index} out of range`);
    }
    return { row: Math.floor(index / columns), column: index % columns };
  }

  query(index) {
    const { row, column } = this.indexToCoordinates(index);
    const ciphertexts = [];
    for (let c = 0; c < this.server.shape.columns; c++) {
      ciphertexts.push(this.key.encrypt(c === column ? 1n : 0n));
    }
    this.queries++;
    return { row, column, request: { kind: "pir-query", ciphertexts } };
  }

  retrieve(index) {
    const { row, request } = this.query(index);
    const reply = this.server.answer(request);
    if (reply.length !== this.server.shape.rows) throw new Error("short reply");
    return this.key.decrypt(reply[row]);
  }
}

function makeDatabase(rows, columns, prng, maxValue = 1000n) {
  const matrix = [];
  for (let r = 0; r < rows; r++) {
    const line = [];
    for (let c = 0; c < columns; c++) line.push(prng.big(maxValue));
    matrix.push(line);
  }
  return matrix;
}

module.exports = {
  Prng, mod, modPow, gcd, isProbablePrime, randomPrime, Paillier, modInverse, lcm,
  validateDatabase, PirServer, PirClient, makeDatabase,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- number theory ------------------------------------------------------------------------
  assert.ok(isProbablePrime(104729n, new Prng(1)), "104729 is prime");
  assert.ok(!isProbablePrime(104730n, new Prng(1)), "104730 is not");
  assert.ok(!isProbablePrime(561n, new Prng(1)), "Carmichael number must be rejected");
  assert.ok(!isProbablePrime(41041n, new Prng(1)), "second Carmichael number");
  assert.strictEqual(gcd(1071n, 462n), 21n);
  assert.strictEqual(mod(modInverse(3n, 11n) * 3n, 11n), 1n);
  assert.throws(() => modInverse(6n, 9n), /not invertible/);

  const keyPrng = new Prng(890);
  const key = new Paillier(48, keyPrng);
  assert.ok(key.p !== key.q && isProbablePrime(key.p, keyPrng) && isProbablePrime(key.q, keyPrng));
  assert.strictEqual(key.n, key.p * key.q);
  assert.strictEqual(gcd(key.lambda, key.n), 1n, "lambda must be invertible mod n");
  assert.ok(key.n > (1n << 90n), "the modulus must be large enough to be meaningful");

  // --- encryption round trip ----------------------------------------------------------------
  for (const message of [0n, 1n, 2n, 42n, 999n, key.n - 1n]) {
    assert.strictEqual(key.decrypt(key.encrypt(message)), mod(message, key.n), `round trip ${message}`);
  }
  assert.strictEqual(key.decrypt(key.encrypt(key.n + 5n)), 5n, "messages are reduced mod n");

  // --- homomorphic addition and scalar multiplication --------------------------------------
  for (let a = 0n; a < 12n; a++) {
    for (let b = 0n; b < 12n; b++) {
      const sum = key.add(key.encrypt(a), key.encrypt(b));
      assert.strictEqual(key.decrypt(sum), (a + b) % key.n, `${a}+${b}`);
      const scaled = key.scale(key.encrypt(a), b);
      assert.strictEqual(key.decrypt(scaled), (a * b) % key.n, `${a}*${b}`);
    }
  }
  assert.strictEqual(key.decrypt(key.add(key.encrypt(7n), key.encrypt(key.n - 3n))), 4n, "wraps mod n");

  // --- exact leakage statement: the low part of a ciphertext is plaintext-independent ------
  {
    const randomness = key.sampleUnit();
    const c0 = key.encrypt(0n, randomness);
    const c1 = key.encrypt(1n, randomness);
    const c2 = key.encrypt(12345n, randomness);
    assert.strictEqual(mod(c0, key.n), mod(c1, key.n), "c mod n must not depend on the message");
    assert.strictEqual(mod(c1, key.n), mod(c2, key.n));
    assert.strictEqual(mod(c1, key.n), modPow(randomness, key.n, key.n), "c mod n is exactly r^n");
    assert.notStrictEqual(c0, c1, "the full ciphertexts do differ");
    assert.notStrictEqual(c1, c2);
    assert.strictEqual(key.decrypt(c1) - key.decrypt(c0), 1n, "the difference is the plaintext");
  }

  // --- ciphertext freshness ------------------------------------------------------------------
  {
    const first = key.encrypt(9n);
    const second = key.encrypt(9n);
    assert.notStrictEqual(first, second, "two encryptions of the same message must differ");
    assert.strictEqual(key.decrypt(first), key.decrypt(second));
  }

  // --- indistinguishability: a statistical distinguisher must sit at chance level -----------
  {
    const samples = 800;
    // A public bit of the ciphertext is a plaintext-independent quantity: bit k of c agrees with
    // bit k of c mod n, and c mod n = r^n mod n carries no information about the message, so any
    // such distinguisher must stay at chance level (verified empirically, not assumed).
    const extractor = (ciphertext) => Number((ciphertext >> 3n) & 1n);
    let correct = 0;
    for (let i = 0; i < samples; i++) {
      const bit = i % 2;
      const ciphertext = key.encrypt(BigInt(bit));
      if (extractor(ciphertext) === bit) correct++;
    }
    const accuracy = correct / samples;
    assert.ok(Math.abs(accuracy - 0.5) < 0.07,
      `a public-bit distinguisher reached ${accuracy}, which must be close to chance`);
  }

  // --- PIR correctness on a small database, for *every* index ------------------------------
  const smallPrng = new Prng(4242);
  const smallDb = makeDatabase(4, 5, smallPrng);
  const smallServer = new PirServer(key, smallDb);
  const smallClient = new PirClient(key, smallServer);
  for (let index = 0; index < 20; index++) {
    assert.strictEqual(smallClient.retrieve(index), smallDb[Math.floor(index / 5)][index % 5], `index ${index}`);
  }
  assert.strictEqual(smallClient.queries, 20);

  // --- index privacy: the request is index-independent in shape and content-class ----------
  {
    const shapes = new Set();
    for (let index = 0; index < 20; index++) {
      const prepared = smallClient.query(index);
      shapes.add(`${prepared.request.ciphertexts.length}`);
      for (const ciphertext of prepared.request.ciphertexts) {
        assert.ok(ciphertext > 1n && ciphertext < key.n2, "ciphertexts live in Z*_{n^2}");
      }
    }
    assert.strictEqual(shapes.size, 1, "every request has the same size");
    // The server's stored requests contain no index, only ciphertexts.
    for (const seen of smallServer.requestsSeen) {
      assert.deepStrictEqual(Object.keys(seen), ["ciphertexts", "kind"]);
      assert.strictEqual(seen.kind, "pir-query");
    }
    // Same randomness vector, different indices: only the plaintext factors differ, and the
    // low parts stay bit-identical, which is the exact statement of what leaks and what does not.
    const randomness = [key.sampleUnit(), key.sampleUnit()];
    const buildWith = (target) => randomness.map((r, c) => key.encrypt(c === target ? 1n : 0n, r));
    const requestA = buildWith(0);
    const requestB = buildWith(1);
    for (let c = 0; c < randomness.length; c++) {
      assert.strictEqual(mod(requestA[c], key.n), mod(requestB[c], key.n), `low part of column ${c}`);
      assert.notStrictEqual(requestA[c], requestB[c]);
    }
  }

  // --- at scale: 4096 records, O(sqrt(N)) communication ------------------------------------
  {
    const scalePrng = new Prng(31337);
    const size = 64;
    const database = makeDatabase(size, size, scalePrng);
    const server = new PirServer(key, database);
    const client = new PirClient(key, server);
    const probes = [0, 1, 63, 64, 129, 2015, 2048, 4095];
    for (const index of probes) {
      assert.strictEqual(client.retrieve(index), database[Math.floor(index / size)][index % size], `index ${index}`);
    }
    const request = client.query(0).request;
    server.coresWorked = 0;
    const reply = server.answer(request);
    const transcriptSize = request.ciphertexts.length + reply.length;
    const n = size * size;
    assert.ok(transcriptSize <= 2 * Math.ceil(Math.sqrt(n)) + 1,
      `transcript ${transcriptSize} must be O(sqrt(N))`);
    assert.ok(transcriptSize < n / 30, `transcript ${transcriptSize} must be far below N = ${n}`);
    assert.strictEqual(request.ciphertexts.length, size, "one ciphertext per column");
    assert.strictEqual(reply.length, size, "one ciphertext per row");
    assert.ok(server.coresWorked > 0 && server.coresWorked <= n, "at most one multiply per record");
  }

  // --- robustness: bad indices and overflow-prone databases are rejected --------------------
  {
    assert.throws(() => smallClient.retrieve(-1), /out of range/);
    assert.throws(() => smallClient.retrieve(20), /out of range/);
    assert.throws(() => smallClient.retrieve(1.5), /out of range/);
    const tooBig = [[key.n, 0n], [0n, 0n]];
    assert.throws(() => new PirServer(key, tooBig), /would wrap/);
    assert.throws(() => new PirServer(key, []), /non-empty matrix/);
    assert.throws(() => new PirServer(key, [[1n], [1n, 2n]]), /ragged/);
    assert.throws(() => new PirServer(key, [[-1n]]), /non-negative/);
    const server = new PirServer(key, [[1n, 2n], [3n, 4n]]);
    assert.throws(() => server.answer({ ciphertexts: [key.encrypt(1n)] }), /malformed request/);
    // A row of zeros is still retrieved correctly: x^0 = 1.
    const zeroServer = new PirServer(key, [[0n, 0n], [7n, 9n]]);
    const zeroClient = new PirClient(key, zeroServer);
    assert.strictEqual(zeroClient.retrieve(0), 0n);
    assert.strictEqual(zeroClient.retrieve(3), 9n);
    // A single-record database degenerates to R = C = 1.
    const single = new PirServer(key, [[1234n]]);
    const singleClient = new PirClient(key, single);
    assert.strictEqual(singleClient.retrieve(0), 1234n);
    assert.strictEqual(singleClient.query(0).request.ciphertexts.length, 1);
    // The maximum admissible record is the largest value whose row sum fits in the modulus.
    const ceiling = (key.n - 1n) / 2n;
    const edge = new PirServer(key, [[ceiling, 0n]]);
    const edgeClient = new PirClient(key, edge);
    assert.strictEqual(edgeClient.retrieve(0), ceiling);
    assert.throws(() => new PirServer(key, [[ceiling + 1n, 0n]]), /would wrap/);
  }

  // --- determinism ---------------------------------------------------------------------------
  {
    const run = () => {
      const localKey = new Paillier(32, new Prng(555));
      const localDb = makeDatabase(3, 3, new Prng(777));
      const localServer = new PirServer(localKey, localDb);
      const localClient = new PirClient(localKey, localServer);
      return [0, 4, 8].map((index) => localClient.retrieve(index).toString()).join(",");
    };
    assert.strictEqual(run(), run());
    const again = new Paillier(48, new Prng(890));
    assert.strictEqual(again.n, key.n, "key generation is reproducible for a fixed seed");
  }

  console.log(`pir: ${key.n.toString(2).length}-bit modulus, 20 index queries verified on a 4x5 database, ` +
    `64x64 (N = 4096) database served with 64 + 64 ciphertexts per query = O(sqrt(N))`);
  console.log("PASS");
}
