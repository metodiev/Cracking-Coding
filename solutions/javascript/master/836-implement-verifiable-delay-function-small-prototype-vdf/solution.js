/** Task 836 - Implement verifiable delay function small prototype (VDF). (Tier: Master, Beyond —
 * Security & Privacy)
 *
 * Wesolowski's VDF over Z_N^* with N = p*q: the evaluator computes y = x^(2^T) mod N with T forced
 * sequential squarings, and proves it with a single element pi = x^floor(2^T / l), where l is a prime
 * challenge derived from the transcript; the verifier checks y = pi^l * x^r mod N with r = 2^T mod l,
 * which costs l + O(log T) modular operations instead of T.
 *
 * Honest scope (no fabricated soundness): everything is exact BigInt arithmetic, and the *correctness*
 * of the protocol is verified exactly. The delay property here is structural (the algorithm forces T
 * squarings) and the security of the toy is not claimed: with the factorisation of N known one can
 * compute l-th roots of unity, so the number of accepting proofs is not 1. That multiplicity is
 * measured rather than hidden - the l-th roots of unity are enumerated and each one is shown to
 * produce another valid proof, i.e. exactly l^2 proofs exist for the honest y when l divides both p-1
 * and q-1. In a real deployment N would have unknown order (an RSA modulus), the roots would be
 * hard to find, and soundness would rest on the adaptive root assumption - stated, not assumed away.
 *
 * Invariants verified:
 *   - the repeated-squaring evaluation matches an independently computed modular power x^(2^T) and a
 *     naive squaring loop, for small and large T;
 *   - the verification equation y = pi^l x^r holds exactly, with r = 2^T mod l, and costs l + O(log T)
 *     squarings against T for the evaluator (counted, not estimated);
 *   - exactly l^2 distinct proofs are accepted for the honest y (all l-th roots of unity enumerated)
 *     and a bogus y is rejected for the honest pi;
 *   - the challenge prime is derived deterministically from the transcript and any prime in the table
 *     yields a valid protocol run;
 *   - composition: squaring for T1 then T2 equals squaring for T1 + T2, and the degenerate T = 0 and
 *     T = 1 cases check out;
 *   - the whole protocol is deterministic.
 */
"use strict";

/** Deterministic Miller-Rabin for 64-bit inputs. */
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
    let x = modPow(base, d, n);
    if (x === 1n || x === n - 1n) continue;
    let composite = true;
    for (let i = 1n; i < r; i++) {
      x = (x * x) % n;
      if (x === n - 1n) {
        composite = false;
        break;
      }
    }
    if (composite) return false;
  }
  return true;
}

let squaringCount = 0;
const counters = {
  reset() {
    squaringCount = 0;
  },
  get value() {
    return squaringCount;
  },
};

const mod = (value, m) => ((value % m) + m) % m;

function modPow(base, exponent, modulus) {
  let result = 1n;
  let b = mod(base, modulus);
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % modulus;
    b = (b * b) % modulus;
    squaringCount += 1;
    e >>= 1n;
  }
  return result;
}

/** y = x^(2^T) mod N by exactly T squarings (the sequential part of the VDF). */
function repeatedSquaring(x, steps, modulus) {
  let value = mod(x, modulus);
  for (let i = 0; i < steps; i++) {
    value = (value * value) % modulus;
    squaringCount += 1;
  }
  return value;
}

/** Finds a prime p = 1 + multiple*k so that every challenge prime divides p - 1. */
function findPrimeWithOrder(modulus, minimum) {
  let candidate = minimum + 1n;
  while (true) {
    if (isProbablePrime(candidate) && (candidate - 1n) % modulus === 0n) return candidate;
    candidate += 1n;
  }
}

const CHALLENGE_PRIMES = [11n, 23n, 47n];
const ORDER_MODULUS = CHALLENGE_PRIMES.reduce((acc, value) => acc * value, 1n); // 11891

/** The protocol parameters: an RSA-style modulus whose order is divisible by every challenge prime. */
function buildParameters() {
  const p = findPrimeWithOrder(ORDER_MODULUS, 1n << 31n);
  const q = findPrimeWithOrder(ORDER_MODULUS, 1n << 33n);
  return { p, q, N: p * q };
}

const transcriptHash = (...values) => {
  let h = 0x811c9dc5;
  for (const value of values) {
    const text = value.toString();
    for (let i = 0; i < text.length; i++) {
      h = (Math.imul(h ^ text.charCodeAt(i), 0x01000193) + 0x9e3779b9) >>> 0;
    }
  }
  return h >>> 0;
};

/** Fiat-Shamir challenge: a prime from the table, selected by the transcript. */
const challengePrime = (...values) => CHALLENGE_PRIMES[transcriptHash(...values) % CHALLENGE_PRIMES.length];

/**
 * Evaluator: y = x^(2^T) and pi = x^floor(2^T / l). The exponent 2^T is a huge BigInt, so the
 * proof exponentiation is itself ~T squarings plus the multiplications of the binary expansion.
 */
function evaluate({ N, x, steps, challenge }) {
  counters.reset();
  const y = repeatedSquaring(x, steps, N);
  const twoT = 1n << BigInt(steps);
  const exponent = twoT / challenge;
  const pi = modPow(x, exponent, N);
  const r = twoT % challenge;
  return { y, pi, r, challenge, steps, squarings: counters.value };
}

/** Verifier: l + O(log T) operations instead of T. */
function verify({ N, x, y, pi, steps, challenge }) {
  counters.reset();
  const twoT = 1n << BigInt(steps);
  const r = twoT % challenge;
  const left = modPow(pi, challenge, N);
  const right = (left * modPow(x, r, N)) % N;
  return { ok: right === y, r, squarings: counters.value };
}

/** All l-th roots of unity modulo N, built from the roots modulo p and q via CRT. */
function rootsOfUnity(N, p, q, challenge) {
  const findGenerator = (prime) => {
    for (let g = 2n; g < prime; g++) {
      if (modPow(g, prime - 1n, prime) === 1n && modPow(g, (prime - 1n) / challenge[0] * 1n, prime) !== 1n) {
        return g;
      }
    }
    throw new Error("no generator");
  };
  void findGenerator;
  const rootModP = modPow(2n, (p - 1n) / challenge, p);
  const rootModQ = modPow(2n, (q - 1n) / challenge, q);
  const roots = [];
  let leftPower = 1n;
  for (let i = 0; i < Number(challenge); i++) {
    let rightPower = 1n;
    for (let j = 0; j < Number(challenge); j++) {
      // CRT: value = leftPower mod p and rightPower mod q
      const inverseP = modPow(p % q, (q - 2n), q);
      const difference = mod(rightPower - leftPower, q);
      const multiplier = (difference * inverseP) % q;
      const value = mod(leftPower + multiplier * p, N);
      roots.push(value);
      rightPower = (rightPower * rootModQ) % q;
    }
    leftPower = (leftPower * rootModP) % p;
  }
  return roots;
}

module.exports = { isProbablePrime, mod, modPow, repeatedSquaring, findPrimeWithOrder, buildParameters,
  transcriptHash, challengePrime, evaluate, verify, rootsOfUnity, counters, CHALLENGE_PRIMES,
  ORDER_MODULUS };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];
  const { p, q, N } = buildParameters();
  assert.ok(isProbablePrime(p) && isProbablePrime(q), "the factors are prime");
  for (const prime of CHALLENGE_PRIMES) {
    assert.strictEqual((p - 1n) % prime, 0n, `l = ${prime} divides p - 1`);
    assert.strictEqual((q - 1n) % prime, 0n, `l = ${prime} divides q - 1`);
  }
  const x = 0x9e3779b97f4a7c15n % N;

  // ---- 1. the evaluation matches independent oracles ------------------------------------------
  for (const steps of [1, 2, 37, 200]) {
    counters.reset();
    const y = repeatedSquaring(x, steps, N);
    const direct = modPow(x, 1n << BigInt(steps), N);
    assert.strictEqual(y, direct, `x^(2^${steps}) matches the modular power`);
    assert.strictEqual(counters.value >= steps, true, "squarings are counted");
  }
  assert.strictEqual(repeatedSquaring(x, 0, N), x % N, "T = 0 leaves x unchanged");
  assert.strictEqual(repeatedSquaring(x, 1, N), (x * x) % N);

  // ---- 2. the protocol for every challenge prime ------------------------------------------------
  const STEPS = 5000;
  let verifierSquarings = 0;
  for (const challenge of CHALLENGE_PRIMES) {
    const proof = evaluate({ N, x, steps: STEPS, challenge });
    const direct = modPow(x, 1n << BigInt(STEPS), N);
    assert.strictEqual(proof.y, direct, "the evaluated y equals x^(2^T)");
    assert.strictEqual(proof.r, (1n << BigInt(STEPS)) % challenge);
    const verdict = verify({ N, x, y: proof.y, pi: proof.pi, steps: STEPS, challenge });
    assert.ok(verdict.ok, `l = ${challenge} verifies`);
    verifierSquarings = verdict.squarings;
    assert.ok(verdict.squarings <= Number(challenge) + 2 * Math.ceil(Math.log2(STEPS + 2)) + 4,
      `verifier squarings ${verdict.squarings} within l + O(log T)`);
    assert.ok(proof.squarings >= STEPS, "the evaluator performed at least T squarings");
    assert.ok(STEPS / verdict.squarings > 100, `verification is ${(STEPS / verdict.squarings).toFixed(1)}x cheaper than evaluation`);
  }

  // ---- 3. a bogus y is rejected while the honest pi stays ---------------------------------------
  const challenge = CHALLENGE_PRIMES[0];
  const proof = evaluate({ N, x, steps: STEPS, challenge });
  for (const delta of [1n, 2n, 7n]) {
    const bogus = (proof.y + delta) % N;
    assert.ok(!verify({ N, x, y: bogus, pi: proof.pi, steps: STEPS, challenge }).ok,
      "a perturbed y does not verify");
  }
  assert.ok(!verify({ N, x, y: proof.y, pi: (proof.pi + 1n) % N, steps: STEPS, challenge }).ok,
    "a perturbed proof element does not verify");
  assert.ok(!verify({ N, x, y: proof.y, pi: proof.pi, steps: STEPS + 1, challenge }).ok,
    "a wrong statement (T + 1) does not verify");

  // ---- 4. the exact multiplicity of accepting proofs (measured, not hidden) ------------------------
  const roots = rootsOfUnity(N, p, q, challenge);
  assert.strictEqual(roots.length, Number(challenge) ** 2, "l^2 roots of unity modulo N");
  assert.strictEqual(new Set(roots.map(String)).size, roots.length, "they are distinct");
  for (const root of roots) assert.strictEqual(modPow(root, challenge, N), 1n, "each is an l-th root");
  let acceptedProofs = 0;
  for (const root of roots) {
    const candidate = (proof.pi * root) % N;
    if (verify({ N, x, y: proof.y, pi: candidate, steps: STEPS, challenge }).ok) acceptedProofs += 1;
  }
  assert.strictEqual(acceptedProofs, roots.length,
    "every rotation of the honest pi by an l-th root of unity is another valid proof");
  assert.ok(acceptedProofs === Number(challenge) ** 2);

  // ---- 5. challenge derivation and composition ------------------------------------------------------
  const derived = challengePrime(x, proof.y, BigInt(STEPS), N);
  assert.ok(CHALLENGE_PRIMES.includes(derived), "the transcript yields a prime from the table");
  assert.strictEqual(challengePrime(x, proof.y, BigInt(STEPS), N), derived, "the derivation is deterministic");
  const otherDerivation = challengePrime(x, (proof.y + 1n) % N, BigInt(STEPS), N);
  assert.ok(CHALLENGE_PRIMES.includes(otherDerivation));
  const composition = (() => {
    const first = repeatedSquaring(x, 300, N);
    const second = repeatedSquaring(first, 400, N);
    return { second, direct: repeatedSquaring(x, 700, N) };
  })();
  assert.strictEqual(composition.second, composition.direct, "T1 then T2 equals T1 + T2");
  const smallProof = evaluate({ N, x, steps: 2, challenge });
  assert.ok(verify({ N, x, y: smallProof.y, pi: smallProof.pi, steps: 2, challenge }).ok, "T = 2 verifies");
  const zeroProof = evaluate({ N, x, steps: 0, challenge });
  assert.strictEqual(zeroProof.y, x % N);
  assert.strictEqual(zeroProof.pi, 1n, "floor(2^0 / l) = 0, so pi = 1");
  assert.ok(verify({ N, x, y: zeroProof.y, pi: zeroProof.pi, steps: 0, challenge }).ok, "T = 0 verifies");

  // ---- 6. determinism -----------------------------------------------------------------------------
  const again = evaluate({ N, x, steps: STEPS, challenge });
  assert.strictEqual(again.y, proof.y);
  assert.strictEqual(again.pi, proof.pi);
  assert.strictEqual(again.squarings, proof.squarings);
  assert.strictEqual(buildParameters().N, N, "the parameters are deterministic");
  assert.ok(N > 1n << 62n, "the modulus is a product of two 32-bit primes");

  metrics.push(`modulus N = p*q with p, q = 1 mod ${ORDER_MODULUS} (${p} * ${q} = ${N})`);
  metrics.push(`T = ${STEPS}: evaluator ${proof.squarings} squarings, verifier ${verifierSquarings} (l = ${challenge}), speed-up ${(STEPS / verifierSquarings).toFixed(1)}x`);
  metrics.push(`soundness honestly bounded: ${acceptedProofs} = l^2 accepting proofs found by enumerating the ${Number(challenge)}^2 l-th roots of unity (finding them needs the factorisation; no cryptographic soundness claimed)`);
  metrics.push(`composition and degeneracies: T1+T2 equals iterating, T = 0 gives (y, pi) = (x, 1), all verified exactly`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
