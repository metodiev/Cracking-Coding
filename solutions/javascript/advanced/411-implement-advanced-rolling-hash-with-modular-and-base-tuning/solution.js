/** Task 411 - Implement advanced rolling hash with modular and base tuning. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Polynomial hashing: hash(s) = sum(s[i] * base^(n-1-i)) mod modulus. Tuning notes:
 *   * double hashing uses bases 131/137 with the large primes 1_000_000_007 / 1_000_000_009.
 *   * JS numbers cannot hold the products of ~1e9-sized values exactly (2^53 limit),
 *     so every hash computation here runs on BigInt and never mixes in Number values.
 *   * the demo pair modulus 65537 with base 2 is deliberately bad: 2^16 = -1 and 2^32 = 1
 *     (mod 65537), so same-length strings that only shift a character by 32 positions
 *     collide — that is the constructed single-hash collision asserted below.
 *   * base 1 degenerates the hash to the plain character sum, so anagrams collide.
 *   * prefix hashes make every substring fingerprint O(1); MultiHash combines several
 *     (base, modulus) pairs so an accidental equality must beat all of them at once.
 */
"use strict";

const MOD1 = 1000000007n;
const MOD2 = 1000000009n;
const BASE1 = 131n;
const BASE2 = 137n;
const DEMO_MOD = 65537n;
const DEMO_BASE = 2n;

function polyHash(s, base, mod) {
  let h = 0n;
  for (let i = 0; i < s.length; i++) {
    h = (h * base + BigInt(s.charCodeAt(i))) % mod;
  }
  return h;
}

class RollingHash {
  constructor(s, base, mod) {
    this.base = base;
    this.mod = mod;
    this.pref = new Array(s.length + 1).fill(0n);
    for (let i = 0; i < s.length; i++) {
      this.pref[i + 1] = (this.pref[i] * base + BigInt(s.charCodeAt(i))) % mod;
    }
    this.pow = new Array(s.length + 1).fill(1n);
    for (let i = 1; i <= s.length; i++) {
      this.pow[i] = (this.pow[i - 1] * base) % mod;
    }
  }

  get(l, r) {
    return ((this.pref[r] - this.pref[l] * this.pow[r - l]) % this.mod + this.mod) % this.mod;
  }
}

class MultiHash {
  constructor(s, specs) {
    this.hashes = specs.map(([base, mod]) => new RollingHash(s, base, mod));
  }

  fingerprint(l, r) {
    return this.hashes.map((h) => h.get(l, r));
  }

  equal(l1, r1, l2, r2) {
    if (r1 - l1 !== r2 - l2) return false;
    const a = this.fingerprint(l1, r1);
    const b = this.fingerprint(l2, r2);
    return a.every((v, i) => v === b[i]);
  }
}

let seed = 2024;

function lcgString(length, alphabet) {
  let out = "";
  for (let i = 0; i < length; i++) {
    seed = (seed * 75 + 74) % 65537;
    out += alphabet[seed % alphabet.length];
  }
  return out;
}

module.exports = { polyHash, RollingHash, MultiHash, lcgString, MOD1, MOD2, BASE1, BASE2, DEMO_MOD, DEMO_BASE };

if (require.main === module) {
  const assert = require("node:assert");

  // hand values: "ab" = 97*131 + 98, "abc" = 97*131^2 + 98*131 + 99
  assert.strictEqual(polyHash("ab", BASE1, MOD1), 12805n);
  assert.strictEqual(polyHash("abc", BASE1, MOD1), 1677554n);

  const h = new RollingHash("abc", BASE1, MOD1);
  assert.strictEqual(h.get(0, 3), 1677554n);
  assert.strictEqual(h.get(0, 1), 97n);
  assert.strictEqual(h.get(1, 2), 98n);
  assert.strictEqual(h.get(2, 3), 99n);
  assert.strictEqual(h.get(1, 3), polyHash("bc", BASE1, MOD1));
  assert.strictEqual(h.get(0, 0), 0n);
  assert.strictEqual(h.get(3, 3), 0n);

  // constructed single-hash collision: base 2, modulus 65537, length 33,
  // characters 32 positions apart swap places — weights repeat because 2^32 = 1.
  const s = "a" + "c".repeat(31) + "b";
  const t = "b" + "c".repeat(31) + "a";
  assert.notStrictEqual(s, t);
  assert.strictEqual(s.length, 33);
  assert.strictEqual(polyHash(s, DEMO_BASE, DEMO_MOD), polyHash(t, DEMO_BASE, DEMO_MOD));
  assert.strictEqual(
    new RollingHash(s, DEMO_BASE, DEMO_MOD).get(0, 33),
    new RollingHash(t, DEMO_BASE, DEMO_MOD).get(0, 33)
  );

  // double hashing with tuned large primes separates them again
  const specs = [[BASE1, MOD1], [BASE2, MOD2]];
  const dblS = new MultiHash(s, specs);
  const dblT = new MultiHash(t, specs);
  assert.notDeepStrictEqual(dblS.fingerprint(0, 33), dblT.fingerprint(0, 33));

  // base 1 degenerates to the character sum
  assert.strictEqual(polyHash("ab", 1n, MOD1), polyHash("ba", 1n, MOD1));
  assert.notStrictEqual(polyHash("ab", BASE1, MOD1), polyHash("ba", BASE1, MOD1));
  assert.strictEqual(polyHash("A", 131n, 256n), 65n);

  for (let round = 0; round < 6; round++) {
    const text = lcgString(61, "abcde");
    const multi = new MultiHash(text, specs);
    for (let i = 0; i < text.length; i++) {
      for (let j = i; j <= text.length; j++) {
        assert.ok(multi.equal(i, j, i, j), `reflexive ${i} ${j}`);
      }
    }
    for (let q = 0; q < 60; q++) {
      seed = (seed * 75 + 74) % 65537;
      const l1 = seed % text.length;
      seed = (seed * 75 + 74) % 65537;
      const l2 = seed % text.length;
      seed = (seed * 75 + 74) % 65537;
      const maxSpan = Math.min(text.length - l1, text.length - l2);
      const span = seed % (maxSpan + 1);
      const oracle = text.slice(l1, l1 + span) === text.slice(l2, l2 + span);
      assert.strictEqual(multi.equal(l1, l1 + span, l2, l2 + span), oracle, `oracle ${q}`);
    }
  }

  // every substring fingerprint equals the direct polynomial hash
  const miss = "mississippi";
  for (const [base, mod] of specs) {
    const rh = new RollingHash(miss, base, mod);
    for (let l = 0; l <= miss.length; l++) {
      for (let r = l; r <= miss.length; r++) {
        assert.strictEqual(rh.get(l, r), polyHash(miss.slice(l, r), base, mod), `direct ${l} ${r}`);
      }
    }
  }

  console.log("PASS");
}
