/** Task 336 - Design a snapshot array (persisting versions basic). (Tier: Intermediate, CTCI Ch 7) */
"use strict";

/**
 * SnapshotArray keeps, per index, the growing list of (snapId, value) writes; get() binary
 * searches the last write with snapId <= requested, so set/snap are O(1) and get is
 * O(log V). Cross-checked against a naive implementation that stores a full copy of every
 * array version on deterministic LCG operation sequences.
 */

class SnapshotArray {
  constructor(length) {
    if (length < 1) throw new Error("length must be >= 1");
    this.length = length;
    this.snapCount = 0;
    // history[i] holds [snapId, value] pairs with strictly increasing snap ids; the entry
    // for snap 0 defaults to 0 so untouched indices read as 0 without extra storage.
    this.history = Array.from({ length }, () => [[0, 0]]);
  }

  set(index, value) {
    if (index < 0 || index >= this.length) throw new Error("index out of range");
    const entries = this.history[index];
    const last = entries[entries.length - 1];
    if (last[0] === this.snapCount) {
      last[1] = value;                       // overwrite the current (unsealed) write
    } else {
      entries.push([this.snapCount, value]);
    }
  }

  snap() {
    return this.snapCount++;
  }

  get(index, snapId) {
    if (index < 0 || index >= this.length) throw new Error("index out of range");
    if (snapId < 0 || snapId >= this.snapCount) throw new Error("unknown snapshot id");
    const entries = this.history[index];
    let lo = 0, hi = entries.length - 1;
    while (lo < hi) {                        // last entry with snapId <= requested
      const mid = (lo + hi + 1) >> 1;
      if (entries[mid][0] <= snapId) lo = mid;
      else hi = mid - 1;
    }
    return entries[lo][1];
  }
}

class NaiveSnapshotArray {
  constructor(length) {
    this.current = new Array(length).fill(0);
    this.versions = [];
  }

  set(index, value) {
    this.current[index] = value;
  }

  snap() {
    this.versions.push(this.current.slice());
    return this.versions.length - 1;
  }

  get(index, snapId) {
    return this.versions[snapId][index];
  }
}

module.exports = { SnapshotArray, NaiveSnapshotArray };

if (require.main === module) {
  const assert = require("node:assert");

  // Canonical LeetCode sequence.
  const arr = new SnapshotArray(3);
  arr.set(0, 5);
  assert.strictEqual(arr.snap(), 0);
  arr.set(0, 6);
  assert.strictEqual(arr.get(0, 0), 5);
  assert.strictEqual(arr.snap(), 1);
  assert.strictEqual(arr.get(0, 0), 5);
  assert.strictEqual(arr.get(0, 1), 6);
  assert.strictEqual(arr.get(1, 0), 0);
  assert.strictEqual(arr.get(2, 1), 0);

  // Writes before the first snap are visible in snapshot 0; later writes are not.
  const arr2 = new SnapshotArray(2);
  arr2.set(1, 7);
  arr2.set(1, 9);                              // same snapshot 0: overwrites 7
  assert.strictEqual(arr2.snap(), 0);
  assert.strictEqual(arr2.get(1, 0), 9);
  arr2.set(1, 11);
  assert.strictEqual(arr2.get(1, 0), 9);       // unsealed write invisible
  assert.strictEqual(arr2.snap(), 1);
  assert.strictEqual(arr2.get(1, 1), 11);

  // Snapshots with no writes in between repeat the previous value.
  const arr3 = new SnapshotArray(1);
  arr3.set(0, 1);
  const first = arr3.snap();
  const second = arr3.snap();
  arr3.set(0, 2);
  const third = arr3.snap();
  assert.deepStrictEqual([first, second, third], [0, 1, 2]);
  assert.deepStrictEqual([arr3.get(0, 0), arr3.get(0, 1), arr3.get(0, 2)], [1, 1, 2]);
  assert.throws(() => arr3.get(0, 3), /unknown snapshot/);

  // LCG operation sequences against the full-copy oracle.
  let seed = 3141;
  const length = 8;
  const fast = new SnapshotArray(length);
  const naive = new NaiveSnapshotArray(length);
  for (let step = 0; step < 3000; step++) {
    seed = (seed * 75 + 74) % 65537;
    const choice = seed % 10;
    seed = (seed * 75 + 74) % 65537;
    if (choice < 5) {
      const index = seed % length;
      seed = (seed * 75 + 74) % 65537;
      const value = seed % 1000;
      fast.set(index, value);
      naive.set(index, value);
    } else if (choice < 7) {
      assert.strictEqual(fast.snap(), naive.snap(), `random snap ${step}`);
    } else if (naive.versions.length > 0) {
      const index = seed % length;
      seed = (seed * 75 + 74) % 65537;
      const snapId = seed % naive.versions.length;
      assert.strictEqual(fast.get(index, snapId), naive.get(index, snapId), `random get ${step}`);
    }
    if (naive.versions.length > 0) {
      const latest = naive.versions.length - 1;
      for (let probe = 0; probe < length; probe++) {
        assert.strictEqual(fast.get(probe, latest), naive.get(probe, latest), `probe ${step}`);
      }
    }
  }
  console.log("PASS");
}
