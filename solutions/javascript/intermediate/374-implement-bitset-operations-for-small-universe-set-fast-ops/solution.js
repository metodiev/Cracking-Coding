/**
 * Task 374 - Bitset (bit array) with fast set operations for a small universe. (Tier: Intermediate, CTCI Ch 5)
 *
 * A fixed universe of n_bits (here <= 256) is packed into 32-bit words.  Every
 * operation touches only whole words, so add/remove/contains are O(1), count() is
 * O(words) using a popcount, and union/intersection are O(words) instead of
 * O(universe).  Iteration extracts the lowest set bit of each word in turn, which
 * yields indices in strictly ascending order without scanning empty stretches.
 *
 * Everything is cross-checked against the plain set oracle (javascript Set) on
 * deterministic LCG universes: membership of every index, count() against the oracle
 * size, ascending iteration against the sorted oracle, and union/intersection against
 * the oracle set operations.  Boundary indices 0, 31, 32 and n_bits-1 are asserted
 * explicitly.
 *
 * The implementation only uses 32-bit words, so the sign quirks of each language are
 * handled identically: words are treated as unsigned bit patterns (>>> 0, Math.clz32
 * and a SWAR popcount that never leaves the safe integer range).
 */
"use strict";

const UNIVERSE = 256;

class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** SWAR popcount of a 32-bit word (value coerced to unsigned). */
function popcount32(value) {
  let v = value >>> 0;
  v -= (v >>> 1) & 0x55555555;
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  v = (v + (v >>> 4)) & 0x0f0f0f0f;
  return ((v * 0x01010101) >>> 24) >>> 0;
}

/** Bitset over indices 0 .. bits-1, backed by 32-bit words. */
class IntBitset {
  constructor(bits) {
    if (bits <= 0) throw new Error("universe must be positive");
    this.bits = bits;
    this.words = new Array(Math.ceil(bits / 32)).fill(0);
    this.universe = 0;
  }

  add(index) {
    this.validate(index);
    this.words[index >> 5] |= 1 << (index & 31);
  }

  remove(index) {
    this.validate(index);
    this.words[index >> 5] &= ~(1 << (index & 31));
  }

  contains(index) {
    this.validate(index);
    return ((this.words[index >> 5] >>> (index & 31)) & 1) === 1;
  }

  count() {
    let total = 0;
    for (const word of this.words) total += popcount32(word);
    return total;
  }

  union(other) {
    this.checkSameUniverse(other);
    const result = new IntBitset(this.bits);
    for (let i = 0; i < this.words.length; i++) result.words[i] = this.words[i] | other.words[i];
    return result;
  }

  intersection(other) {
    this.checkSameUniverse(other);
    const result = new IntBitset(this.bits);
    for (let i = 0; i < this.words.length; i++) result.words[i] = this.words[i] & other.words[i];
    return result;
  }

  /** All members as indices in ascending order. */
  iterate() {
    const members = [];
    for (let wi = 0; wi < this.words.length; wi++) {
      let word = this.words[wi] >>> 0;
      while (word !== 0) {
        const low = word & -word;
        members.push(wi * 32 + (31 - Math.clz32(low)));
        word = (word ^ low) >>> 0;
      }
    }
    return members;
  }

  validate(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.bits) {
      throw new Error(`index out of universe: ${index}`);
    }
  }

  checkSameUniverse(other) {
    if (this.bits !== other.bits) throw new Error("universe size mismatch");
  }
}

module.exports = { IntBitset };

if (require.main === module) {
  const assert = require("node:assert");

  // --- boundary behaviour ----------------------------------------------
  const empty = new IntBitset(UNIVERSE);
  assert.strictEqual(empty.count(), 0, "empty count");
  assert.deepStrictEqual(empty.iterate(), [], "empty iterate");

  const boundary = new IntBitset(UNIVERSE);
  for (const index of [0, 31, 32, 33, 63, 64, 255]) boundary.add(index);
  assert.deepStrictEqual(boundary.iterate(), [0, 31, 32, 33, 63, 64, 255], "boundary iterate");
  assert.strictEqual(boundary.count(), 7, "boundary count");
  assert.ok(boundary.contains(0) && boundary.contains(255), "boundary contains");
  assert.ok(!boundary.contains(1) && !boundary.contains(254), "boundary non-members");
  boundary.add(0);
  assert.strictEqual(boundary.count(), 7, "add is idempotent");
  boundary.remove(1);
  assert.strictEqual(boundary.count(), 7, "remove absent is a no-op");
  boundary.remove(0);
  assert.ok(!boundary.contains(0) && boundary.count() === 6, "remove works");

  // --- explicit small example -------------------------------------------
  const first = new IntBitset(64);
  const second = new IntBitset(64);
  for (const index of [1, 5, 40, 63]) first.add(index);
  for (const index of [5, 7, 40, 41]) second.add(index);
  assert.deepStrictEqual(first.iterate(), [1, 5, 40, 63], "first iterate");
  assert.deepStrictEqual(second.iterate(), [5, 7, 40, 41], "second iterate");
  assert.deepStrictEqual(first.union(second).iterate(), [1, 5, 7, 40, 41, 63], "union");
  assert.deepStrictEqual(first.intersection(second).iterate(), [5, 40], "intersection");
  assert.strictEqual(first.union(second).count(), 6, "union count");
  assert.strictEqual(first.intersection(second).count(), 2, "intersection count");
  assert.strictEqual(new IntBitset(64).union(first).count(), 4, "union with empty");
  assert.strictEqual(new IntBitset(64).intersection(first).count(), 0, "intersection with empty");

  const full = new IntBitset(UNIVERSE);
  for (let index = 0; index < UNIVERSE; index++) full.add(index);
  assert.strictEqual(full.count(), UNIVERSE, "full count");
  assert.strictEqual(full.iterate().length, UNIVERSE, "full iterate size");
  const spark = new IntBitset(UNIVERSE);
  for (const index of [1, 5, 40, 63, 255]) spark.add(index);
  assert.strictEqual(full.union(spark).count(), UNIVERSE, "full union stays full");
  assert.deepStrictEqual(full.union(spark).iterate(), [...Array(UNIVERSE).keys()], "full union iterate");
  assert.deepStrictEqual(full.intersection(spark).iterate(), spark.iterate(), "full intersection");
  assert.strictEqual(new IntBitset(UNIVERSE).intersection(full).count(), 0, "empty intersection full");

  // --- LCG cross-checks against the plain set oracle --------------------
  const lcg = new Lcg(37413);
  for (let trial = 0; trial < 40; trial++) {
    const oracleA = new Set();
    const oracleB = new Set();
    const bitsA = new IntBitset(UNIVERSE);
    const bitsB = new IntBitset(UNIVERSE);

    for (let step = 0; step < 60; step++) {
      let index = lcg.next(UNIVERSE);
      let action = lcg.next(UNIVERSE);
      if (action % 3 === 0) {
        bitsA.remove(index);
        oracleA.delete(index);
      } else {
        bitsA.add(index);
        oracleA.add(index);
      }
      index = lcg.next(UNIVERSE);
      action = lcg.next(UNIVERSE);
      if (action % 3 === 0) {
        bitsB.remove(index);
        oracleB.delete(index);
      } else {
        bitsB.add(index);
        oracleB.add(index);
      }
    }

    for (let index = 0; index < UNIVERSE; index++) {
      assert.strictEqual(bitsA.contains(index), oracleA.has(index), `membership A ${trial}`);
      assert.strictEqual(bitsB.contains(index), oracleB.has(index), `membership B ${trial}`);
    }
    assert.strictEqual(bitsA.count(), oracleA.size, `count A ${trial}`);
    assert.strictEqual(bitsB.count(), oracleB.size, `count B ${trial}`);
    const sortedA = [...oracleA].sort((x, y) => x - y);
    const sortedB = [...oracleB].sort((x, y) => x - y);
    assert.deepStrictEqual(bitsA.iterate(), sortedA, `iterate A ${trial}`);
    assert.deepStrictEqual(bitsB.iterate(), sortedB, `iterate B ${trial}`);

    const unionOracle = new Set([...oracleA, ...oracleB]);
    const intersectionOracle = new Set([...oracleA].filter((v) => oracleB.has(v)));
    const union = bitsA.union(bitsB);
    const intersection = bitsA.intersection(bitsB);
    assert.strictEqual(union.count(), unionOracle.size, `union count ${trial}`);
    assert.strictEqual(intersection.count(), intersectionOracle.size, `intersection count ${trial}`);
    assert.deepStrictEqual(union.iterate(), [...unionOracle].sort((x, y) => x - y), `union iterate ${trial}`);
    assert.deepStrictEqual(intersection.iterate(), [...intersectionOracle].sort((x, y) => x - y),
      `intersection iterate ${trial}`);
    for (let index = 0; index < UNIVERSE; index += 7) {
      assert.strictEqual(union.contains(index), unionOracle.has(index), `union membership ${trial}`);
      assert.strictEqual(intersection.contains(index), intersectionOracle.has(index),
        `intersection membership ${trial}`);
    }
  }
  console.log("PASS");
}
