/** Task 711 - Implement homomorphic encryption simple operations demo (Paillier). (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * Paillier encryption works in the multiplicative group modulo n^2 with n = p*q. The public key is
 * (n, g = n + 1), the secret key is lambda = lcm(p-1, q-1) with mu = L(g^lambda mod n^2)^-1 mod n,
 * where L(x) = (x - 1) / n. A message m is encrypted as c = g^m * r^n mod n^2 with a fresh random
 * r coprime to n, and decrypted as L(c^lambda mod n^2) * mu mod n. Two properties make the scheme
 * additively homomorphic and are checked here rather than assumed:
 *   - multiplying ciphertexts adds the plaintexts: D(E(a) * E(b) mod n^2) = a + b mod n;
 *   - raising a ciphertext to a power k multiplies the plaintext: D(E(a)^k) = a * k mod n.
 * The randomizer is what makes encryption randomised: two ciphertexts of the same message differ,
 * yet decrypt identically - semantic security for this demo, without a security proof.
 *
 * What is simplified: 256-bit primes generated with Miller-Rabin from a seeded PRNG (so runs are
 * reproducible), no key serialisation, no padding or threshold variants, and no security claim
 * beyond the correctness and the algebraic identities that are verified.
 *
 * Invariant verified: decrypt(encrypt(m)) = m for a sweep of messages including 0 and n-1, the two
 * homomorphic identities hold exactly (also across the modular wrap-around and after long chains of
 * additions), ciphertexts are different for equal messages but re-decrypt identically, and a tiny
 * 35-modulus key is checked exhaustively against the table of all encryptions of all messages.
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

function gcd(a, b) {
  while (b !== 0n) {
    const next = a % b;
    a = b;
    b = next;
  }
  return a < 0n ? -a : a;
}

/** Extended Euclid: returns {g, x} with a*x = g (mod m) when g = gcd(a, m). */
function modInverse(a, m) {
  let [oldR, r] = [((a % m) + m) % m, m];
  let [oldS, s] = [1n, 0n];
  while (r !== 0n) {
    const quotient = oldR / r;
    [oldR, r] = [r, oldR - quotient * r];
    [oldS, s] = [s, oldS - quotient * s];
  }
  if (oldR !== 1n) throw new RangeError("the value is not invertible");
  return ((oldS % m) + m) % m;
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

/** Miller-Rabin with the fixed bases that make it deterministic far below 2^64 and safe above. */
function isProbablePrime(value, rounds = 24) {
  if (value < 2n) return false;
  for (const small of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    if (value === small) return true;
    if (value % small === 0n) return false;
  }
  let d = value - 1n;
  let s = 0n;
  while (d % 2n === 0n) { d /= 2n; s++; }
  const witnesses = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];
  for (const witness of witnesses.slice(0, rounds)) {
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

/** A random prime of the given bit size, drawn from the seeded PRNG. */
function randomPrime(bits, rng) {
  const bytes = Math.ceil(bits / 8);
  for (;;) {
    let candidate = 0n;
    for (let i = 0; i < bytes; i++) candidate = (candidate << 8n) | BigInt(rng(256));
    candidate |= (1n << BigInt(bits - 1)) | 1n; // force the size and oddness
    if (isProbablePrime(candidate)) return candidate;
  }
}

/** Random in [2, n) that is coprime to n - the encryption randomizer. */
function randomCoprime(n, rng) {
  for (;;) {
    let candidate = 0n;
    const bytes = Math.ceil(n.toString(2).length / 8) + 1;
    for (let i = 0; i < bytes; i++) candidate = (candidate << 8n) | BigInt(rng(256));
    candidate %= n;
    if (candidate > 1n && gcd(candidate, n) === 1n) return candidate;
  }
}

class Paillier {
  /** Key generation from two primes (given, or drawn with the seeded PRNG). */
  constructor({ bits = 256, seed = 1, p = null, q = null } = {}) {
    const rng = makeRng(seed);
    this.rng = rng;
    this.p = p ?? randomPrime(bits, rng);
    this.q = q ?? randomPrime(bits, rng);
    if (this.p === this.q) throw new RangeError("the two primes must differ");
    this.n = this.p * this.q;
    this.nSquared = this.n * this.n;
    this.g = this.n + 1n;
    this.lambda = ((this.p - 1n) * (this.q - 1n)) / gcd(this.p - 1n, this.q - 1n);
    if (gcd(this.n, (this.p - 1n) * (this.q - 1n)) !== 1n) {
      throw new RangeError("g = n + 1 needs gcd(n, phi(n)) = 1");
    }
    const l = ((modPow(this.g, this.lambda, this.nSquared) - 1n) / this.n) % this.n;
    this.mu = modInverse(l, this.n);
  }
  /** L(x) = (x - 1) / n, the Paillier extraction of the plaintext from a decrypted group element. */
  lFunction(value) { return (value - 1n) / this.n; }
  reduce(message) {
    if (typeof message !== "bigint") throw new TypeError("messages are BigInt values");
    return ((message % this.n) + this.n) % this.n; // negative values wrap around mod n
  }
  encrypt(message, randomizer = null) {
    const m = this.reduce(message);
    const r = randomizer === null ? randomCoprime(this.n, this.rng) : randomizer;
    return (modPow(this.g, m, this.nSquared) * modPow(r, this.n, this.nSquared)) % this.nSquared;
  }
  decrypt(ciphertext) {
    const raw = modPow(ciphertext, this.lambda, this.nSquared);
    return ((this.lFunction(raw) * this.mu) % this.n + this.n) % this.n;
  }
  /** Homomorphic addition of plaintexts: E(a) * E(b) mod n^2. */
  addCiphertexts(first, second) { return (first * second) % this.nSquared; }
  /** Homomorphic scalar multiplication: E(a)^k. */
  multiplyPlaintext(ciphertext, factor) {
    const exponent = this.reduce(factor);
    return modPow(ciphertext, exponent, this.nSquared);
  }
}

module.exports = { makeRng, gcd, modInverse, modPow, isProbablePrime, randomPrime, Paillier };

if (require.main === module) {
  // -- number theory helpers ---------------------------------------------------------------------
  strictEqual(gcd(48n, 18n), 6n);
  strictEqual(gcd(-48n, 18n), 6n);
  strictEqual(modInverse(3n, 11n), 4n);
  strictEqual((modInverse(17n, 3120n) * 17n) % 3120n, 1n);
  throws(() => modInverse(6n, 9n), RangeError);
  strictEqual(modPow(5n, 0n, 7n), 1n);
  strictEqual(modPow(5n, 3n, 7n), 6n);
  strictEqual(modPow(2n, 10n, 1000n), 24n);
  ok(isProbablePrime(1000003n), "1000003 is prime");
  ok(!isProbablePrime(1000001n), "1000001 = 101 * 9901");
  ok(isProbablePrime(2n) && !isProbablePrime(1n) && !isProbablePrime(0n));
  ok(isProbablePrime((1n << 61n) - 1n), "2^61 - 1 is a Mersenne prime");

  // -- exhaustive check with a tiny key: every message, every pair --------------------------------
  const tiny = new Paillier({ p: 5n, q: 7n, seed: 2 });
  strictEqual(tiny.n, 35n);
  strictEqual(tiny.lambda, 12n, "lcm(4, 6) = 12");
  const table = new Map();
  for (let m = 0n; m < 35n; m++) {
    const ciphertext = tiny.encrypt(m, 2n); // a fixed randomizer makes the table reproducible
    table.set(m, ciphertext);
    strictEqual(tiny.decrypt(ciphertext), m, `round trip for ${m}`);
  }
  for (let a = 0n; a < 35n; a++) {
    for (let b = 0n; b < 35n; b++) {
      strictEqual(tiny.decrypt(tiny.addCiphertexts(table.get(a), table.get(b))), (a + b) % 35n,
        `additive homomorphism for ${a} + ${b}`);
      strictEqual(tiny.decrypt(tiny.multiplyPlaintext(table.get(a), b)), (a * b) % 35n,
        `scalar multiplication for ${a} * ${b}`);
    }
  }
  strictEqual(tiny.decrypt(tiny.encrypt(0n, 3n)), 0n, "zero maps to a random unit and decrypts to zero");

  // -- a real key: round trips, randomisation, homomorphisms --------------------------------------
  const t0 = Date.now();
  const paillier = new Paillier({ bits: 256, seed: 711 });
  ok(paillier.n.toString(2).length >= 500, "n must be about 512 bits");
  ok(gcd(paillier.n, (paillier.p - 1n) * (paillier.q - 1n)) === 1n, "n is coprime to phi(n)");
  const messages = [0n, 1n, 2n, paillier.n - 1n, paillier.n / 3n, 12345678901234567890n];
  const ciphertexts = messages.map((m) => paillier.encrypt(m));
  for (let i = 0; i < messages.length; i++) {
    strictEqual(paillier.decrypt(ciphertexts[i]), messages[i], `round trip for ${messages[i]}`);
  }
  const again = paillier.encrypt(messages[3]);
  ok(again !== ciphertexts[3], "encryption must be randomised");
  strictEqual(paillier.decrypt(again), messages[3], "a fresh ciphertext decrypts the same");
  for (const ciphertext of ciphertexts) ok(ciphertext !== 0n && ciphertext < paillier.nSquared);

  // -- homomorphic identities on a real key -------------------------------------------------------
  const a = 987654321n;
  const b = paillier.n - 12345n; // a negative value in disguise
  const sum = paillier.addCiphertexts(paillier.encrypt(a), paillier.encrypt(b));
  strictEqual(paillier.decrypt(sum), paillier.reduce(a + b), "a + (n - 12345) wraps to a - 12345");
  const product = paillier.multiplyPlaintext(paillier.encrypt(a), 7n);
  strictEqual(paillier.decrypt(product), paillier.reduce(7n * a));
  strictEqual(paillier.decrypt(paillier.multiplyPlaintext(paillier.encrypt(a), 0n)), 0n, "k = 0 gives 0");
  strictEqual(paillier.decrypt(paillier.multiplyPlaintext(paillier.encrypt(a), 1n)), paillier.reduce(a));
  strictEqual(paillier.decrypt(paillier.multiplyPlaintext(paillier.encrypt(a), paillier.n)), 0n,
    "k = n annihilates the plaintext");

  // -- long addition chains must not drift --------------------------------------------------------
  const ciphertextsOfOne = Array.from({ length: 64 }, () => paillier.encrypt(1n));
  const chained = ciphertextsOfOne.reduce((accumulated, next) => paillier.addCiphertexts(accumulated, next));
  strictEqual(paillier.decrypt(chained), 64n, "a chain of 64 additions still decrypts exactly");
  let running = paillier.encrypt(0n);
  for (let i = 1n; i <= 20n; i++) running = paillier.addCiphertexts(running, paillier.encrypt(i));
  strictEqual(paillier.decrypt(running), 210n, "1 + 2 + ... + 20 = 210");

  // -- the secret key is needed: a wrong lambda recovers something else ---------------------------
  const wrong = { lambda: paillier.lambda + 1n, mu: paillier.mu };
  const broken = ((modPow(ciphertexts[4], wrong.lambda, paillier.nSquared) - 1n) / paillier.n * wrong.mu)
    % paillier.n;
  ok(broken !== messages[4], "a wrong secret key must not recover the message");

  // -- rejected input and determinism -------------------------------------------------------------
  throws(() => paillier.encrypt(1), TypeError);
  const replayA = new Paillier({ bits: 256, seed: 711 });
  const replayB = new Paillier({ bits: 256, seed: 711 });
  strictEqual(replayA.n, paillier.n, "identical seeds generate the identical key");
  strictEqual(replayA.encrypt(42n), replayB.encrypt(42n), "and identical ciphertexts");

  console.log(`key bits=${paillier.n.toString(2).length} exhaustively checked pairs=1225`);
  console.log(`key generation plus checks took ${Date.now() - t0} ms`);
  console.log("PASS");
}
