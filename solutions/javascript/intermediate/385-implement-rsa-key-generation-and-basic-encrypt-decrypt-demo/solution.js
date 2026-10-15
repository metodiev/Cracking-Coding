/** Task 385 - Implement RSA key generation and basic encrypt/decrypt (demo). (Tier: Intermediate, CTCI Beyond — Security & Privacy) */
"use strict";

function modPower(base, exponent, modulus) {
  let result = 1n % modulus;
  let b = base % modulus;
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % modulus;
    b = (b * b) % modulus;
    e >>= 1n;
  }
  return result;
}

function extendedGcd(a, b) {
  let oldR = a, r = b;
  let oldS = 1n, s = 0n;
  while (r !== 0n) {
    const q = oldR / r;
    [oldR, r] = [r, oldR - q * r];
    [oldS, s] = [s, oldS - q * s];
  }
  return [oldR, oldS];
}

function modInverse(a, modulus) {
  const [g, x] = extendedGcd(((a % modulus) + modulus) % modulus, modulus);
  if (g !== 1n) throw new Error("modular inverse does not exist");
  return ((x % modulus) + modulus) % modulus;
}

function gcd(a, b) {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y !== 0n) [x, y] = [y, x % y];
  return x;
}

/** Trial division, used only to reject obviously bad demo primes. */
function isSmallPrime(n) {
  if (n < 2n) return false;
  for (let i = 2n; i * i <= n; i++) {
    if (n % i === 0n) return false;
  }
  return true;
}

/** Returns [n, e, d, phi]; throws on bad demo parameters. */
function keygen(p, q, e) {
  if (!isSmallPrime(p) || !isSmallPrime(q) || p === q)
    throw new Error("p and q must be distinct primes");
  const n = p * q;
  const phi = (p - 1n) * (q - 1n);
  if (gcd(e, phi) !== 1n)
    throw new Error("public exponent must be coprime with phi(n)");
  return [n, e, modInverse(e, phi), phi];
}

function encrypt(message, e, n) {
  if (message < 0n || message >= n)
    throw new Error("message must satisfy 0 <= m < n");
  return modPower(message, e, n);
}

function decrypt(ciphertext, d, n) {
  if (ciphertext < 0n || ciphertext >= n)
    throw new Error("ciphertext must satisfy 0 <= c < n");
  return modPower(ciphertext, d, n);
}

module.exports = { modPower, modInverse, keygen, encrypt, decrypt };

if (require.main === module) {
  const assert = require("node:assert");

  // classic small keypair p=61, q=53 (n = 3233, phi = 3120, e = 17, d = 2753)
  const [n, e, d, phi] = keygen(61n, 53n, 17n);
  assert.deepStrictEqual([n, e, d, phi], [3233n, 17n, 2753n, 3120n]);
  assert.strictEqual(n, 61n * 53n);
  assert.strictEqual(phi, 60n * 52n);
  assert.strictEqual((e * d) % phi, 1n);
  assert.notStrictEqual(d, e);

  // round trip for several messages, including the boundaries 0 and n-1
  for (const message of [0n, 1n, 2n, 42n, 65n, 1000n, 3232n]) {
    assert.strictEqual(decrypt(encrypt(message, e, n), d, n), message, `round trip ${message}`);
  }
  assert.strictEqual(encrypt(0n, e, n), 0n);
  assert.strictEqual(encrypt(1n, e, n), 1n);           // documented fixed points
  assert.strictEqual(encrypt(n - 1n, e, n), n - 1n);   // (-1)^e = -1 for odd e
  for (const message of [2n, 42n, 65n, 1000n]) {
    assert.notStrictEqual(encrypt(message, e, n), message);
  }

  // wrong key: applying the public exponent to the ciphertext does not recover the message
  const ciphertext42 = encrypt(42n, e, n);
  assert.strictEqual(ciphertext42, 2557n);
  assert.strictEqual(decrypt(ciphertext42, e, n), 1001n);
  assert.notStrictEqual(decrypt(ciphertext42, e, n), 42n);

  // second keypair with e = 3: p=101, q=113 -> n=11413, phi=11200, d=7467
  const [n2, e2, d2, phi2] = keygen(101n, 113n, 3n);
  assert.deepStrictEqual([n2, e2, d2, phi2], [11413n, 3n, 7467n, 11200n]);
  assert.strictEqual((e2 * d2) % phi2, 1n);
  for (const message of [0n, 1n, 2n, 5n, 42n, 999n, 11412n]) {
    assert.strictEqual(decrypt(encrypt(message, e2, n2), d2, n2), message, `round trip 2 ${message}`);
  }
  const ciphertext7 = encrypt(7n, e2, n2);
  assert.strictEqual(decrypt(ciphertext7, e2, n2), 8652n);
  assert.notStrictEqual(decrypt(ciphertext7, e2, n2), 7n);

  // keygen rejects bad demo parameters (documented errors)
  assert.throws(() => keygen(7n, 13n, 3n));
  assert.throws(() => keygen(9n, 53n, 17n));
  assert.throws(() => keygen(61n, 61n, 17n));
  assert.throws(() => keygen(61n, 53n, 4n));

  // encryption domain is checked
  assert.throws(() => encrypt(3233n, e, n));
  assert.throws(() => encrypt(-1n, e, n));

  console.log("PASS");
}
