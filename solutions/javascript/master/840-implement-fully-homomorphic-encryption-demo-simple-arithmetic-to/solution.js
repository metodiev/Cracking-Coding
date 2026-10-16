/** Task 840 - Implement fully homomorphic encryption demo simple arithmetic (toy).
 *  (Tier: Master, CTCI Beyond — Security & Privacy)
 *
 * DGHV-style *fully* homomorphic encryption over the integers (van Dijk–Gentry–Halevi–Vaikuntanathan):
 * the smallest scheme that really supports unbounded circuits once the noise is refreshed.
 *
 *  - secret key: an odd prime p with `etaBits` bits;
 *  - ciphertext of a bit m: c = q * p + 2r + m, with q of about `gammaBits` bits and |r| < 2^rhoBits;
 *  - decrypt: (c mod p) mod 2, correct exactly while the signed residue noise = 2r + m stays inside
 *    (-p/2, p/2) — the *noise budget* this demo revolves around;
 *  - public key: a sorted list of zero-encryptions x_i = q_i p + 2 r_i with x_0 the largest (the
 *    scheme's requirement). The evaluator encrypts by adding a random subset of them, so it never
 *    needs p;
 *  - addition adds noises, multiplication multiplies them: the noise of c1 * c2 is exactly
 *    (2 r1 + m1)(2 r2 + m2). XOR is additive, AND is multiplicative, NOT adds the trivial ciphertext 1;
 *  - refresh (the client-aided round): the key holder decrypts and re-encrypts with fresh noise, which
 *    makes the reachable depth unbounded. Real bootstrapping removes the key holder from that loop;
 *    that trust assumption is the documented simplification of this demo.
 *
 * Everything is deterministic: keys, noise and samples all come from seeded splitmix64, and primality
 * is decided by Miller–Rabin with the fixed base set that is proven for every n < 3.3e24.
 *
 * Simplified / toy aspects (honest): eta = 64 and gamma = 192, so the approximate-GCD problem behind
 * the scheme is trivially breakable and this is *not* cryptographically secure. What is real: the
 * algebra, the exact noise identities, the correctness region |noise| < p/2, the public-key path, the
 * depth/refresh behaviour, and every quantity measured below.
 *
 * Invariants verified below:
 *  - correctness (safety): decrypt(encrypt(m)) = m for 200 fresh ciphertexts, the full XOR/AND/NOT
 *    truth table, all 64 three-bit additions of a ripple-carry adder, and public-key ciphertexts;
 *  - budget/liveness: decryption is the intended bit exactly iff |noise| < p/2 — scanned across the
 *    boundary, including the two wrapping values p/2 and p/2 + 1;
 *  - noise identities: noise(c1 + c2) = noise(c1) + noise(c2) and noise(c1 * c2) = (2r1+m1)(2r2+m2)
 *    hold exactly, and refresh restores noise <= 2 * 2^rho;
 *  - measured depth: the longest AND chain that still decrypts is measured for three parameter sets,
 *    the next level is shown to fail, and with refresh a depth-12 chain stays exact;
 *  - confidentiality measurements: ciphertext parity agrees with the plaintext bit only ~half the
 *    time, and gcd(c1, c2) of two ciphertexts stays tiny — the two cheapest naive attacks fail;
 *  - cost: ciphertext expansion in bits per plaintext bit, and the homomorphic operation count.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;
// Bases proven to decide primality for every n < 3.3e24, far beyond the 64-bit secret primes used here.
const MILLER_RABIN_BASES = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];

/** Deterministic splitmix64: the only source of pseudo-randomness. */
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

  /** Uniform integer in [0, 2^width). */
  bits(width) {
    const w = BigInt(width);
    let value = 0n;
    for (let filled = 0n; filled < w; filled += 64n) value = (value << 64n) | this.next();
    return value & ((1n << w) - 1n);
  }

  /** Signed integer with |value| < 2^width. */
  signed(width) {
    const magnitude = this.bits(width);
    return this.next() & 1n ? magnitude : -magnitude;
  }

  bit() {
    return Number(this.next() & 1n);
  }
}

function field(x, p) {
  const r = x % p;
  return r < 0n ? r + p : r;
}

function powMod(base, exponent, modulus) {
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

/** Miller–Rabin with a fixed base set: deterministic over the ranges used here. */
function isProbablePrime(candidate) {
  if (candidate < 2n) return false;
  for (const base of MILLER_RABIN_BASES) {
    if (candidate === base) return true;
    if (candidate % base === 0n) return false;
  }
  let d = candidate - 1n;
  let s = 0;
  while (d % 2n === 0n) {
    d /= 2n;
    s++;
  }
  for (const base of MILLER_RABIN_BASES) {
    let x = powMod(base, d, candidate);
    if (x === 1n || x === candidate - 1n) continue;
    let witness = true;
    for (let i = 1; i < s; i++) {
      x = (x * x) % candidate;
      if (x === candidate - 1n) {
        witness = false;
        break;
      }
    }
    if (witness) return false;
  }
  return true;
}

/** Random odd prime with exactly `bits` bits. */
function generatePrime(bits, rng) {
  const high = 1n << BigInt(bits - 1);
  for (let attempt = 0; attempt < 4000; attempt++) {
    const candidate = high | (rng.bits(bits - 1) | 1n);
    if (isProbablePrime(candidate)) return candidate;
  }
  throw new Error("no prime found");
}

function greatestCommonDivisor(a, b) {
  let x = a < 0n ? -a : a;
  let y = b < 0n ? -b : b;
  while (y > 0n) {
    const next = x % y;
    x = y;
    y = next;
  }
  return x;
}

const bitLength = (value) => (value === 0n ? 1 : value.toString(2).length);
const absolute = (value) => (value < 0n ? -value : value);

/** Signed residue in (-p/2, p/2]: the channel the scheme decodes from. */
function centeredResidue(c, p) {
  const r = field(c, p);
  return r > p >> 1n ? r - p : r;
}

const BASE_PARAMS = { etaBits: 64, gammaBits: 192, rhoBits: 12, samples: 12, subsetSize: 8 };

/** Key holder: owns p, produces ciphertexts, decrypts and refreshes. */
class KeyHolder {
  constructor(params, seed) {
    this.params = params;
    this.rng = new SplitMix64(seed);
    this.p = generatePrime(params.etaBits, this.rng);
    this.publicKey = this.createPublicKey();
  }

  /** x_i = q_i p + 2 r_i, sorted with the largest first (the scheme requires max = x_0). */
  createPublicKey() {
    const qBits = this.params.gammaBits - this.params.etaBits;
    const samples = [];
    for (let i = 0; i < this.params.samples; i++) {
      const q = this.rng.bits(qBits) | 1n;
      const r = this.rng.signed(this.params.rhoBits);
      samples.push(q * this.p + 2n * r);
    }
    samples.sort((a, b) => (b > a ? 1 : b < a ? -1 : 0));
    return { samples, params: this.params };
  }

  /** Secret-key encryption of one bit. */
  encrypt(bit) {
    const qBits = this.params.gammaBits - this.params.etaBits;
    const q = this.rng.bits(qBits) | (1n << BigInt(qBits - 1)); // top bit set keeps gamma bits
    const r = this.rng.signed(this.params.rhoBits);
    return q * this.p + 2n * r + BigInt(bit & 1);
  }

  /** Ciphertext whose noise is exactly the requested value (the boundary scan needs this). */
  encryptWithNoise(noise) {
    const value = BigInt(noise);
    const bit = value & 1n;
    return 2n * ((value - bit) / 2n) + bit;
  }

  /** (c mod p) mod 2: the parity of the signed residue. */
  decrypt(ciphertext) {
    return Number(field(centeredResidue(ciphertext, this.p), 2n));
  }

  /** The signed residue, i.e. the noise 2r + m as long as the budget was never exceeded. */
  noise(ciphertext) {
    return centeredResidue(ciphertext, this.p);
  }

  /** Remaining budget as seen from the ciphertext alone; wrapping makes this optimistic. */
  budgetUsed(ciphertext) {
    return (this.p >> 1n) - absolute(this.noise(ciphertext));
  }

  /** The client-aided refresh round: decrypt and re-encrypt with fresh, small noise. */
  refresh(ciphertext) {
    return this.encrypt(this.decrypt(ciphertext));
  }
}

/** Evaluator: holds only the public key, never p. */
class Evaluator {
  constructor(publicKey) {
    this.publicKey = publicKey;
    this.rng = new SplitMix64(0xa11ce);
    this.operations = 0;
  }

  /** Public-key encryption: the bit plus a random subset of the zero-encryptions. */
  encrypt(bit) {
    const { samples, params } = this.publicKey;
    let c = BigInt(bit & 1);
    const used = new Set();
    while (used.size < params.subsetSize) {
      used.add(Number(this.rng.bits(16) % BigInt(samples.length)));
    }
    for (const index of used) c += samples[index];
    this.operations++;
    return c;
  }

  add(left, right) {
    this.operations++;
    return left + right;
  }

  xor(left, right) {
    return this.add(left, right);
  }

  not(ciphertext) {
    this.operations++;
    return ciphertext + 1n;
  }

  multiply(left, right) {
    this.operations++;
    return left * right;
  }

  /**
   * Evaluates a gate list. Wire ids: inputs 0..k-1, then one wire per gate in program order.
   * `refresh` is applied to every multiplication result when supplied.
   */
  evaluate(circuit, inputs, refresh) {
    const wires = inputs.slice();
    for (const gate of circuit) {
      let value;
      if (gate.op === "xor") value = this.xor(wires[gate.a], wires[gate.b]);
      else if (gate.op === "and") value = this.multiply(wires[gate.a], wires[gate.b]);
      else if (gate.op === "not") value = this.not(wires[gate.a]);
      else throw new Error(`unknown gate ${gate.op}`);
      if (gate.op === "and" && refresh !== undefined) value = refresh(value);
      wires.push(value);
    }
    return { wires, outputs: (circuit.outputs ?? []).map((wire) => wires[wire]) };
  }
}

/** Ripple-carry adder over `bits` bits; inputs a_0..a_{b-1}, b_0..b_{b-1}, outputs s_0..s_{b-1}, carry. */
function adderCircuit(bits) {
  const circuit = [];
  const inputCount = 2 * bits;
  const emit = (op, a, b) => {
    circuit.push({ op, a, b });
    return inputCount + circuit.length - 1;
  };
  const sums = [];
  let carry = null;
  for (let i = 0; i < bits; i++) {
    const a = i;
    const b = bits + i;
    const partial = emit("xor", a, b); // a XOR b
    if (carry === null) {
      sums.push(partial);
      carry = emit("and", a, b); // carry into the next position is a AND b
    } else {
      sums.push(emit("xor", partial, carry));
      const carryProduct = emit("and", partial, carry); // (a XOR b) AND carry_in
      const lower = emit("and", a, b); // a AND b
      const either = emit("xor", carryProduct, lower); // OR as XOR of the terms
      const both = emit("and", carryProduct, lower);
      carry = emit("xor", either, both);
    }
  }
  circuit.outputs = sums.concat([carry]);
  return circuit;
}

/** Chain of `depth` AND gates multiplying input 0 into the running product. */
function andChainCircuit(depth) {
  const inputCount = 2;
  const circuit = [{ op: "and", a: 0, b: 1 }];
  for (let i = 1; i < depth; i++) circuit.push({ op: "and", a: 0, b: inputCount + circuit.length - 1 });
  circuit.outputs = [inputCount + circuit.length - 1];
  return circuit;
}

module.exports = { MASK64, MILLER_RABIN_BASES, SplitMix64, field, powMod, isProbablePrime,
  generatePrime, greatestCommonDivisor, bitLength, absolute, centeredResidue, BASE_PARAMS,
  KeyHolder, Evaluator, adderCircuit, andChainCircuit };

if (require.main === module) {
  const assert = require("node:assert");
  const params = BASE_PARAMS;
  const holder = new KeyHolder(params, 840);
  const evaluator = new Evaluator(holder.publicKey);
  const smallRng = new SplitMix64(4242);

  // --- key material ----------------------------------------------------------
  assert.strictEqual(holder.p % 2n, 1n, "the secret modulus is odd");
  assert.ok(isProbablePrime(holder.p), "the secret modulus passes Miller-Rabin");
  assert.strictEqual(bitLength(holder.p), params.etaBits, "eta bits of secret modulus");
  assert.strictEqual(isProbablePrime(holder.p * holder.p), false, "Miller-Rabin rejects composites");
  assert.strictEqual(isProbablePrime(4n), false, "small composites are rejected");
  assert.strictEqual(isProbablePrime(2n), true, "the smallest prime is accepted");
  assert.strictEqual(new KeyHolder(params, 840).p, holder.p, "key generation is deterministic");
  assert.notStrictEqual(new KeyHolder(params, 841).p, holder.p, "a different seed yields a new key");

  // --- correctness of secret-key encryption ----------------------------------
  let freshCiphertexts = 0;
  for (let i = 0; i < 200; i++) {
    const m = smallRng.bit();
    const c = holder.encrypt(m);
    assert.strictEqual(holder.decrypt(c), m, `decrypt(encrypt(${m})) rounds back`);
    assert.ok(bitLength(c) >= params.gammaBits - 1 && bitLength(c) <= params.gammaBits,
      `ciphertexts carry ~gamma bits (got ${bitLength(c)})`);
    assert.ok(absolute(holder.noise(c)) <= 2n * ((1n << BigInt(params.rhoBits)) - 1n) + 1n,
      "fresh ciphertext noise sits at the fresh level");
    freshCiphertexts++;
  }
  assert.strictEqual(freshCiphertexts, 200);
  assert.strictEqual(holder.decrypt(0n), 0, "the trivial ciphertext of 0 decrypts to 0");
  assert.strictEqual(holder.decrypt(1n), 1, "the trivial ciphertext of 1 decrypts to 1");

  // --- public key: zero-encryptions, and the evaluator never sees p ----------
  assert.strictEqual("p" in evaluator, false, "the evaluator holds no secret modulus");
  assert.strictEqual(holder.publicKey.samples.length, params.samples);
  assert.ok(holder.publicKey.samples[0] >= holder.publicKey.samples[params.samples - 1],
    "x_0 is the largest zero-encryption, as the scheme requires");
  for (const sample of holder.publicKey.samples) {
    assert.strictEqual(holder.decrypt(sample), 0, "public-key samples encrypt zero");
    assert.ok(absolute(holder.noise(sample)) <= 2n * (1n << BigInt(params.rhoBits)),
      "sample noise stays below 2^(rho+1)");
  }
  const publicNoiseLimit = 2n * BigInt(params.subsetSize) * (1n << BigInt(params.rhoBits));
  let publicCiphertexts = 0;
  for (let i = 0; i < 40; i++) {
    const m = smallRng.bit();
    const c = evaluator.encrypt(m);
    assert.strictEqual(holder.decrypt(c), m, "public-key ciphertexts decrypt correctly");
    assert.ok(2n * absolute(holder.noise(c)) <= 2n * publicNoiseLimit,
      "public-key noise respects the subset bound");
    publicCiphertexts++;
  }
  assert.strictEqual(publicCiphertexts, 40);

  // --- confidentiality measurements -----------------------------------------
  // Naive attack 1: read the parity of the ciphertext; it must agree with m only by chance.
  let parityMatches = 0;
  const distinctCiphertexts = new Set();
  for (let i = 0; i < 200; i++) {
    const c = holder.encrypt(1);
    if (c % 2n === 1n) parityMatches++;
    distinctCiphertexts.add(c);
  }
  assert.ok(parityMatches > 60 && parityMatches < 140,
    `ciphertext parity leaks nothing (parity was 1 in ${parityMatches}/200 encryptions of 1)`);
  assert.strictEqual(distinctCiphertexts.size, 200, "every encryption is a fresh ciphertext");
  // Naive attack 2: gcd of two ciphertexts must stay tiny, so p is not exposed.
  let maxGcdBits = 0;
  for (let i = 0; i < 50; i++) {
    maxGcdBits = Math.max(maxGcdBits,
      bitLength(greatestCommonDivisor(holder.encrypt(1), holder.encrypt(0))));
  }
  assert.ok(maxGcdBits <= 8, `gcd(c1, c2) stays tiny (max ${maxGcdBits} bits)`);

  // --- the noise budget is exactly |noise| < p/2 -----------------------------
  const halfP = holder.p >> 1n;
  let insideCases = 0;
  let outsideCases = 0;
  for (const delta of [-6n, -4n, -2n, -1n, 0n, 1n, 2n, 3n, 5n]) {
    const noise = halfP + delta;
    const intendedBit = Number(noise & 1n);
    const c = holder.encryptWithNoise(noise);
    const inside = 2n * absolute(noise) < holder.p;
    if (inside) {
      insideCases++;
      assert.strictEqual(holder.noise(c), noise, `noise ${delta}: inside the budget, exact round trip`);
      assert.strictEqual(holder.decrypt(c), intendedBit, `noise ${delta}: decodes to the intended bit`);
    } else {
      outsideCases++;
      assert.strictEqual(holder.noise(c), noise - holder.p,
        `noise ${delta}: wraps around p, the loss the budget forbids`);
      assert.strictEqual(holder.decrypt(c), 1 - intendedBit,
        `noise ${delta}: the decoded bit flips, which is the failure mode`);
      assert.ok(absolute(holder.noise(c)) < absolute(noise),
        `noise ${delta}: the wrapped residue looks harmless from the ciphertext alone`);
    }
  }
  assert.ok(insideCases >= 4 && outsideCases >= 4, "both sides of the boundary were exercised");
  assert.strictEqual(centeredResidue(halfP, holder.p), halfP, "p/2 itself stays positive");
  assert.strictEqual(centeredResidue(halfP + 1n, holder.p), -halfP, "p/2 + 1 wraps to -p/2");

  // --- truth tables for XOR / AND / NOT -------------------------------------
  for (const m1 of [0, 1]) {
    for (const m2 of [0, 1]) {
      const c1 = holder.encrypt(m1);
      const c2 = holder.encrypt(m2);
      assert.strictEqual(holder.decrypt(evaluator.xor(c1, c2)), m1 ^ m2,
        `XOR(${m1},${m2}) is homomorphic`);
      assert.strictEqual(holder.decrypt(evaluator.multiply(c1, c2)), m1 & m2,
        `AND(${m1},${m2}) is homomorphic`);
      assert.strictEqual(holder.decrypt(evaluator.not(c1)), 1 - m1, `NOT(${m1}) is homomorphic`);
      assert.strictEqual(holder.decrypt(evaluator.not(evaluator.xor(c1, c2))), 1 - (m1 ^ m2),
        `XNOR(${m1},${m2}) is homomorphic`);
    }
  }
  assert.strictEqual(holder.decrypt(evaluator.add(evaluator.not(1n), evaluator.not(1n))), 0,
    "1 + 1 in the encrypted domain is 0 mod 2");

  // --- exact noise identities ------------------------------------------------
  let identityTrials = 0;
  for (let trial = 0; trial < 20; trial++) {
    const m1 = smallRng.bit();
    const m2 = smallRng.bit();
    const c1 = holder.encrypt(m1);
    const c2 = holder.encrypt(m2);
    const r1 = (holder.noise(c1) - BigInt(m1)) / 2n;
    const r2 = (holder.noise(c2) - BigInt(m2)) / 2n;
    assert.strictEqual(holder.noise(evaluator.add(c1, c2)),
      holder.noise(c1) + holder.noise(c2), "addition adds noises exactly");
    const product = evaluator.multiply(c1, c2);
    assert.strictEqual(holder.noise(product), (2n * r1 + BigInt(m1)) * (2n * r2 + BigInt(m2)),
      "multiplication noise is the product of the scaled noises");
    assert.strictEqual(holder.decrypt(product), m1 & m2, "and the product still decrypts");
    identityTrials++;
  }
  assert.strictEqual(identityTrials, 20);

  // --- refresh restores the noise budget -------------------------------------
  const one = holder.encrypt(1);
  let chain = holder.encrypt(1);
  let maxRefreshedNoise = 0n;
  for (let level = 0; level < 12; level++) {
    chain = holder.refresh(evaluator.multiply(chain, one));
    maxRefreshedNoise = max(maxRefreshedNoise, absolute(holder.noise(chain)));
  }
  assert.ok(maxRefreshedNoise <= 2n * (1n << BigInt(params.rhoBits)),
    `refresh keeps noise at the fresh level (max ${maxRefreshedNoise})`);
  assert.strictEqual(holder.decrypt(chain), 1, "a refreshed chain of 12 ANDs stays exact");

  // --- measured reachable depth without refresh ------------------------------
  // The guarantee: while the *true* noise |n0^k * n1| stays below p/2 the AND chain must decrypt
  // exactly; past that point correctness may survive by luck (an even wrap count keeps the parity),
  // so the probe also records the first depth at which a wrong bit is actually observed.
  const depthProbe = (etaBits, rhoBits, maxDepth) => {
    const probeParams = { ...params, etaBits, rhoBits, gammaBits: 3 * etaBits };
    const probeHolder = new KeyHolder(probeParams, 99);

    let budgetDepth = 0;
    let failureDepth = 0;
    let guaranteedChecks = 0;
    for (let trial = 0; trial < 8; trial++) {
      const input = probeHolder.encrypt(1);
      const base = probeHolder.encrypt(1);
      const inputNoise = probeHolder.noise(input);
      let accumulator = base;
      let trueNoise = probeHolder.noise(base);
      for (let depth = 1; depth <= maxDepth; depth++) {
        accumulator = input * accumulator;
        trueNoise = inputNoise * trueNoise;
        const insideBudget = 2n * absolute(trueNoise) < probeHolder.p;
        const bit = probeHolder.decrypt(accumulator);
        if (insideBudget) {
          assert.strictEqual(bit, 1,
            `eta${etaBits}/rho${rhoBits} depth ${depth}: inside the budget, therefore exact`);
          if (trial === 0) {
            budgetDepth = depth;
            guaranteedChecks++;
          }
        } else if (bit !== 1 && failureDepth === 0) {
          failureDepth = depth;
        }
      }
    }
    assert.ok(guaranteedChecks >= 1, "the guaranteed region was actually exercised");
    assert.ok(failureDepth === 0 || failureDepth >= budgetDepth,
      `failures only begin past the budget (budget ${budgetDepth}, first failure ${failureDepth})`);
    return { budgetDepth, failureDepth };
  };
  const shallow = depthProbe(32, 4, 24);
  const deeper = depthProbe(64, 4, 24);
  const noisiest = depthProbe(64, 12, 24);
  assert.ok(shallow.budgetDepth >= 2, `a narrow modulus budgets depth ${shallow.budgetDepth}`);
  assert.ok(deeper.budgetDepth > shallow.budgetDepth,
    `more modulus bits budget more depth (${deeper.budgetDepth} > ${shallow.budgetDepth})`);
  assert.ok(noisiest.budgetDepth < deeper.budgetDepth,
    `more initial noise budgets less depth (${noisiest.budgetDepth} < ${deeper.budgetDepth})`);
  assert.ok(deeper.failureDepth > 0, `unrefreshed depth really breaks (first failure at ${deeper.failureDepth})`);
  assert.ok(noisiest.failureDepth > 0, "and it breaks sooner with noisier randomness");
  assert.ok(deeper.failureDepth <= 24, "the probe stayed inside its depth window");

  // --- ripple-carry adder: circuit logic first, then the encrypted evaluation -
  const evaluatePlain = (circuit, inputs) => {
    const wires = inputs.slice();
    for (const gate of circuit) {
      const left = wires[gate.a];
      if (gate.op === "xor") wires.push(left ^ wires[gate.b]);
      else if (gate.op === "and") wires.push(left & wires[gate.b]);
      else wires.push(1 - left);
    }
    return (circuit.outputs ?? []).map((wire) => wires[wire]);
  };
  const bitsFromNumber = (value, width) =>
    Array.from({ length: width }, (_, i) => (value >> i) & 1);
  const numberFromBits = (bits) => bits.reduce((acc, bit, i) => acc | (bit << i), 0);

  const adder = adderCircuit(3);
  let plainAdditions = 0;
  for (let a = 0; a < 8; a++) {
    for (let b = 0; b < 8; b++) {
      const plain = evaluatePlain(adder, bitsFromNumber(a, 3).concat(bitsFromNumber(b, 3)));
      assert.strictEqual(numberFromBits(plain.slice(0, 3)), (a + b) & 7,
        `plaintext adder low bits for ${a}+${b}`);
      assert.strictEqual(plain[3], a + b > 7 ? 1 : 0, `plaintext adder carry for ${a}+${b}`);
      plainAdditions++;
    }
  }
  assert.strictEqual(plainAdditions, 64, "the adder logic was checked on all 64 pairs");

  let exactAdditions = 0;
  for (let a = 0; a < 8; a++) {
    for (let b = 0; b < 8; b++) {
      const inputs = [];
      for (let i = 0; i < 3; i++) inputs.push(holder.encrypt((a >> i) & 1));
      for (let i = 0; i < 3; i++) inputs.push(holder.encrypt((b >> i) & 1));
      const { outputs } = evaluator.evaluate(adder, inputs, (wire) => holder.refresh(wire));
      const bitsOut = outputs.map((wire) => holder.decrypt(wire));
      assert.strictEqual(numberFromBits(bitsOut.slice(0, 3)), (a + b) & 7,
        `encrypted adder low bits for ${a}+${b}`);
      assert.strictEqual(bitsOut[3], a + b > 7 ? 1 : 0, `encrypted adder carry for ${a}+${b}`);
      exactAdditions++;
    }
  }
  assert.strictEqual(exactAdditions, 64, "all 64 encrypted additions were checked with refresh");

  // Without refresh the same adder overruns its budget: measure how many pairs still come out exact.
  const smallAdder = adderCircuit(2);
  let exactWithoutRefresh = 0;
  for (let a = 0; a < 4; a++) {
    for (let b = 0; b < 4; b++) {
      const inputs = bitsFromNumber(a, 2).concat(bitsFromNumber(b, 2)).map((bit) => holder.encrypt(bit));
      const { outputs } = evaluator.evaluate(smallAdder, inputs);
      const bitsOut = outputs.map((wire) => holder.decrypt(wire));
      const sum = numberFromBits(bitsOut.slice(0, 2));
      const carry = bitsOut[2];
      if (sum === ((a + b) & 3) && carry === (a + b > 3 ? 1 : 0)) exactWithoutRefresh++;

    }
  }
  assert.ok(exactWithoutRefresh < 16,
    `without refresh only ${exactWithoutRefresh}/16 additions stay exact, which is why refresh exists`);

  // --- cost: expansion and operation counts ----------------------------------
  const expansion = bitLength(holder.encrypt(1));
  assert.ok(expansion >= params.gammaBits - 1 && expansion <= params.gammaBits,
    `ciphertext expansion is ~gamma bits per plaintext bit (${expansion})`);
  assert.ok(evaluator.operations > 100, `homomorphic operations executed: ${evaluator.operations}`);
  const multiplications = adder.filter((gate) => gate.op === "and").length;
  assert.ok(multiplications > 0 && multiplications < adder.length,
    `the adder uses ${multiplications} multiplications of ${adder.length} gates`);

  console.log(`key: eta=${params.etaBits} gamma=${params.gammaBits} rho=${params.rhoBits} ` +
    `secret-prime-bits=${bitLength(holder.p)}`);
  console.log(`unrefreshed AND depth (budget/failure): eta32/rho4=${shallow.budgetDepth}/${shallow.failureDepth}, ` +
    `eta64/rho4=${deeper.budgetDepth}/${deeper.failureDepth}, ` +
    `eta64/rho12=${noisiest.budgetDepth}/${noisiest.failureDepth}; refreshed depth=12 exact`);
  console.log(`confidentiality: parity matched ${parityMatches}/200, max gcd bits=${maxGcdBits}; ` +
    `expansion=${expansion} bits/bit, ops=${evaluator.operations}`);
  console.log("PASS");
}

/** BigInt max, used by the refresh noise tracker. */
function max(a, b) {
  return a > b ? a : b;
}
