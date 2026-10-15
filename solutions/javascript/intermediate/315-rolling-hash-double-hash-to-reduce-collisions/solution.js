/** Task 315 - Rolling hash with two moduli to reduce collisions. (Tier: Intermediate, CTCI Ch 1) */
"use strict";

const MOD1 = 1_000_000_007;
const MOD2 = 998_244_353;
const BASE = 131;

const MOD_WEAK = 65537;
const BASE_WEAK = 65536;               // == -1 (mod 65537), a deliberately bad base

class DoubleRollingHash {
  constructor(s) {
    const n = s.length;
    this.h1 = new Array(n + 1).fill(0);
    this.h2 = new Array(n + 1).fill(0);
    this.p1 = new Array(n + 1).fill(1);
    this.p2 = new Array(n + 1).fill(1);
    for (let i = 0; i < n; i++) {
      const code = s.charCodeAt(i);
      this.h1[i + 1] = (this.h1[i] * BASE + code) % MOD1;
      this.h2[i + 1] = (this.h2[i] * BASE + code) % MOD2;
      this.p1[i + 1] = (this.p1[i] * BASE) % MOD1;
      this.p2[i + 1] = (this.p2[i] * BASE) % MOD2;
    }
  }

  /** Hash of s[lo..hi) as the pair (mod MOD1, mod MOD2). */
  hash(lo, hi) {
    const length = hi - lo;
    const a = ((this.h1[hi] - (this.h1[lo] * this.p1[length]) % MOD1 + MOD1) % MOD1 + MOD1) % MOD1;
    const b = ((this.h2[hi] - (this.h2[lo] * this.p2[length]) % MOD2 + MOD2) % MOD2 + MOD2) % MOD2;
    return [a, b];
  }

  equal(lo1, hi1, lo2, hi2) {
    const left = this.hash(lo1, hi1);
    const right = this.hash(lo2, hi2);
    return left[0] === right[0] && left[1] === right[1];
  }
}

/** Rolling hash built from scratch, no prefix arrays; used as a cross-check. */
function directHash(s) {
  let h1 = 0;
  let h2 = 0;
  for (let i = 0; i < s.length; i++) {
    const code = s.charCodeAt(i);
    h1 = (h1 * BASE + code) % MOD1;
    h2 = (h2 * BASE + code) % MOD2;
  }
  return [h1, h2];
}

/** Single-modulus hash with a base that is -1 modulo the modulus (alternating sum). */
function weakHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * BASE_WEAK + s.charCodeAt(i)) % MOD_WEAK;
  return h;
}

function lcgStrings(length, count, alphabet = "abca", seed = 3150) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += alphabet[seed % alphabet.length];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { MOD1, MOD2, BASE, MOD_WEAK, BASE_WEAK, DoubleRollingHash, directHash, weakHash, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual((MOD1 - 1) % 2, 0);
  assert.strictEqual((MOD2 - 1) % 2, 0);
  assert.notStrictEqual(MOD1, MOD2);

  const s = "banana";
  const rolling = new DoubleRollingHash(s);
  assert.deepStrictEqual(rolling.hash(1, 3), directHash("an"));
  assert.deepStrictEqual(rolling.hash(0, 2), directHash("ba"));
  assert.ok(rolling.equal(1, 3, 3, 5));                        // "an" == "an"
  assert.ok(!rolling.equal(0, 2, 3, 5));                       // "ba" != "an"
  assert.ok(rolling.equal(1, 6, 1, 6));

  // Deliberate collision of a weak single-modulus hash, separated by the double hash
  assert.notStrictEqual("ab", "bc");
  assert.strictEqual(weakHash("ab"), weakHash("bc"));          // 97*(-1)+98 == 98*(-1)+99
  assert.strictEqual(weakHash("ab"), 1);
  assert.notStrictEqual(weakHash("ab"), weakHash("ba"));
  assert.notDeepStrictEqual(new DoubleRollingHash("ab").hash(0, 2), new DoubleRollingHash("bc").hash(0, 2));
  assert.notDeepStrictEqual(directHash("ab"), directHash("bc"));
  assert.notStrictEqual(directHash("ab")[0], directHash("bc")[0]);

  // O(1) equality queries must agree with naive slicing, systematically and randomly
  const texts = ["a", "banana", "mississippi", "aaaaaaaa", ...lcgStrings(12, 20, "abca"), ...lcgStrings(16, 10, "abcd")];
  let checked = 0;
  for (const text of texts) {
    const rollingHash = new DoubleRollingHash(text);
    const n = text.length;
    for (let length = 1; length <= n; length++) {
      for (let lo1 = 0; lo1 + length <= n; lo1++) {
        for (let lo2 = lo1; lo2 + length <= n; lo2++) {
          const expected = text.slice(lo1, lo1 + length) === text.slice(lo2, lo2 + length);
          assert.strictEqual(rollingHash.equal(lo1, lo1 + length, lo2, lo2 + length), expected,
            `${text} ${lo1} ${lo2} ${length}`);
          checked++;
        }
      }
    }
    for (let lo = 0; lo <= n; lo++) {
      for (let hi = lo; hi <= n; hi++) {
        assert.deepStrictEqual(rollingHash.hash(lo, hi), directHash(text.slice(lo, hi)), `${text} ${lo} ${hi}`);
      }
    }
  }
  assert.ok(checked > 1000);

  console.log("PASS");
}
