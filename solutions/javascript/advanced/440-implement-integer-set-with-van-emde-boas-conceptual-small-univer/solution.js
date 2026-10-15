/** Task 440 - Implement integer set with van Emde Boas (conceptual small universe). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Documented simplification of a van Emde Boas set on the small universe U = 256 = 16 x 16: the top
 * level keeps min/max, a 16-entry summary of cluster occupancy and 16 lazily created clusters, while
 * the summary and every cluster is the base case - a 16-bit member bitmask (BitSet16) whose bit scans
 * replace the recursion below u = 16. The decisive vEB trick is kept: the minimum is stored outside
 * the clusters, so delete(min) promotes the successor from the cluster/summary structure. Operations
 * MEMBER / SUCCESSOR / PREDECESSOR / INSERT / DELETE therefore match a full recursive vEB
 * (O(1) bit work at this two-level scale). Assertions cover hand sequences that cross cluster
 * boundaries, empty the structure and exercise successor/predecessor at both universe ends, plus LCG
 * operation scripts cross-checked against a sorted-set oracle with full 0..255 sweeps.
 */
"use strict";

class BitSet16 {
  constructor() {
    this.mask = 0;
  }

  insert(value) {
    this.mask |= 1 << value;
  }

  remove(value) {
    this.mask &= ~(1 << value);
  }

  member(value) {
    return ((this.mask >> value) & 1) === 1;
  }

  empty() {
    return this.mask === 0;
  }

  static lowestBit(bits) {
    let index = 0;
    while ((bits & 1) === 0) {
      bits >>= 1;
      index++;
    }
    return index;
  }

  static highestBit(bits) {
    let index = -1;
    while (bits !== 0) {
      bits >>= 1;
      index++;
    }
    return index;
  }

  minimum() {
    return this.mask === 0 ? null : BitSet16.lowestBit(this.mask);
  }

  maximum() {
    return this.mask === 0 ? null : BitSet16.highestBit(this.mask);
  }

  successor(value) {
    const higher = this.mask >> (value + 1);
    return higher === 0 ? null : value + 1 + BitSet16.lowestBit(higher);
  }

  predecessor(value) {
    const lower = this.mask & ((1 << value) - 1);
    return lower === 0 ? null : BitSet16.highestBit(lower);
  }
}

class VanEmdeBoas {
  constructor() {
    this.min = null;
    this.max = null;
    this.count = 0;
    this.summary = new BitSet16();
    this.clusters = new Array(16).fill(null);
  }

  size() {
    return this.count;
  }

  member(value) {
    if (this.min !== null && value === this.min) return true;
    if (this.max !== null && value === this.max) return true;
    const cluster = this.clusters[value >> 4];
    return cluster !== null && cluster.member(value & 15);
  }

  insert(value) {
    if (this.member(value)) return;
    if (this.min === null) {
      this.min = value;
      this.max = value;
      this.count = 1;
      return;
    }
    if (value < this.min) {
      const previousMin = this.min;
      this.min = value;
      value = previousMin;
    }
    if (value > this.max) this.max = value;
    const high = value >> 4;
    const low = value & 15;
    if (this.clusters[high] === null) {
      this.clusters[high] = new BitSet16();
      this.summary.insert(high);
    }
    this.clusters[high].insert(low);
    this.count++;
  }

  delete(value) {
    if (!this.member(value)) return false;
    if (this.count === 1) {
      this.min = null;
      this.max = null;
      this.summary = new BitSet16();
      this.clusters = new Array(16).fill(null);
      this.count = 0;
      return true;
    }
    if (value === this.min) {
      const promoted = this._successorStored(value);
      this.min = promoted;
      this._deleteStored(promoted);
      return true;
    }
    this._deleteStored(value);
    return true;
  }

  _successorStored(value) {
    const high = value >> 4;
    const low = value & 15;
    const cluster = this.clusters[high];
    if (cluster !== null && low < 15) {
      const within = cluster.successor(low);
      if (within !== null) return (high << 4) | within;
    }
    const nextHigh = this.summary.successor(high);
    return (nextHigh << 4) | this.clusters[nextHigh].minimum();
  }

  _deleteStored(value) {
    const high = value >> 4;
    const low = value & 15;
    this.clusters[high].remove(low);
    if (this.clusters[high].empty()) {
      this.summary.remove(high);
      this.clusters[high] = null;
    }
    this.count--;
    if (value === this.max) {
      const top = this.summary.maximum();
      this.max = top === null ? this.min : (top << 4) | this.clusters[top].maximum();
    }
  }

  successor(value) {
    if (this.min === null || value >= this.max) return null;
    if (value < this.min) return this.min;
    const high = value >> 4;
    const low = value & 15;
    const cluster = this.clusters[high];
    if (cluster !== null && low < 15) {
      const within = cluster.successor(low);
      if (within !== null) return (high << 4) | within;
    }
    const nextHigh = this.summary.successor(high);
    if (nextHigh === null) return null;
    return (nextHigh << 4) | this.clusters[nextHigh].minimum();
  }

  predecessor(value) {
    if (this.min === null || value <= this.min) return null;
    if (value > this.max) return this.max;
    const high = value >> 4;
    const low = value & 15;
    const cluster = this.clusters[high];
    if (cluster !== null && low > 0) {
      const within = cluster.predecessor(low);
      if (within !== null) return (high << 4) | within;
    }
    const previous = this.summary.predecessor(high);
    if (previous === null) return this.min;
    return (previous << 4) | this.clusters[previous].maximum();
  }
}

function sortedInsert(oracle, value) {
  let lo = 0;
  let hi = oracle.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (oracle[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  if (lo < oracle.length && oracle[lo] === value) return;
  oracle.splice(lo, 0, value);
}

module.exports = { BitSet16, VanEmdeBoas, sortedInsert };

if (require.main === module) {
  const assert = require("node:assert");
  let veb = new VanEmdeBoas();
  assert.strictEqual(veb.size(), 0);
  assert.strictEqual(veb.member(0), false);
  assert.strictEqual(veb.successor(0), null);
  assert.strictEqual(veb.predecessor(255), null);
  assert.strictEqual(veb.delete(0), false);

  veb.insert(200);
  assert.strictEqual(veb.size(), 1);
  assert.ok(veb.member(200) && !veb.member(199));
  assert.strictEqual(veb.successor(200), null);
  assert.strictEqual(veb.successor(5), 200);
  assert.strictEqual(veb.predecessor(200), null);
  assert.strictEqual(veb.predecessor(255), 200);
  assert.strictEqual(veb.delete(200), true);
  assert.strictEqual(veb.delete(200), false);
  assert.strictEqual(veb.size(), 0);

  veb.insert(5);
  veb.insert(100);
  veb.insert(200);
  assert.strictEqual(veb.size(), 3);
  assert.strictEqual(veb.successor(5), 100);
  assert.strictEqual(veb.successor(4), 5);
  assert.strictEqual(veb.predecessor(100), 5);
  assert.strictEqual(veb.predecessor(101), 100);
  assert.strictEqual(veb.successor(199), 200);
  assert.strictEqual(veb.delete(100), true);
  assert.strictEqual(veb.member(100), false);
  assert.strictEqual(veb.successor(5), 200);
  assert.strictEqual(veb.predecessor(200), 5);
  assert.strictEqual(veb.delete(5), true);
  assert.strictEqual(veb.successor(199), 200);
  assert.strictEqual(veb.predecessor(199), null);
  assert.strictEqual(veb.delete(200), true);
  assert.strictEqual(veb.successor(0), null);

  veb.insert(0);
  veb.insert(15);
  veb.insert(16);
  veb.insert(17);
  veb.insert(255);
  assert.strictEqual(veb.successor(0), 15);
  assert.strictEqual(veb.successor(15), 16);
  assert.strictEqual(veb.successor(16), 17);
  assert.strictEqual(veb.successor(17), 255);
  assert.strictEqual(veb.predecessor(255), 17);
  assert.strictEqual(veb.predecessor(17), 16);
  assert.strictEqual(veb.predecessor(16), 15);
  assert.strictEqual(veb.predecessor(0), null);
  assert.strictEqual(veb.delete(16), true);
  assert.strictEqual(veb.successor(15), 17);
  assert.strictEqual(veb.delete(0), true);
  assert.strictEqual(veb.min, 15);
  assert.strictEqual(veb.delete(15), true);
  assert.strictEqual(veb.min, 17);
  assert.strictEqual(veb.delete(17), true);
  assert.strictEqual(veb.min, 255);
  assert.strictEqual(veb.max, 255);
  assert.strictEqual(veb.delete(255), true);
  assert.strictEqual(veb.size(), 0);
  assert.strictEqual(veb.min, null);

  veb = new VanEmdeBoas();
  const oracle = [];
  let seed = 4400;
  for (let step = 0; step < 600; step++) {
    seed = (seed * 75 + 74) % 65537;
    const op = seed % 5;
    seed = (seed * 75 + 74) % 65537;
    const value = seed % 256;
    if (op <= 1) {
      veb.insert(value);
      sortedInsert(oracle, value);
    } else if (op === 2) {
      const expected = oracle.includes(value);
      if (expected) oracle.splice(oracle.indexOf(value), 1);
      assert.strictEqual(veb.delete(value), expected, `random delete ${step}`);
    } else if (op === 3) {
      assert.strictEqual(veb.member(value), oracle.includes(value), `random member ${step}`);
    } else {
      let expectedSuccessor = null;
      let expectedPredecessor = null;
      for (const candidate of oracle) {
        if (candidate > value && expectedSuccessor === null) expectedSuccessor = candidate;
        if (candidate < value) expectedPredecessor = candidate;
      }
      assert.strictEqual(veb.successor(value), expectedSuccessor, `random successor ${step}`);
      assert.strictEqual(veb.predecessor(value), expectedPredecessor, `random predecessor ${step}`);
    }
    assert.strictEqual(veb.size(), oracle.length, `random size ${step}`);
    if (step % 50 === 0) {
      for (let probe = 0; probe < 256; probe++) {
        assert.strictEqual(veb.member(probe), oracle.includes(probe), `sweep member ${step} ${probe}`);
      }
    }
  }
  assert.ok(oracle.length > 0);
  for (let probe = 0; probe < 256; probe++) {
    assert.strictEqual(veb.member(probe), oracle.includes(probe), `final member ${probe}`);
    let expectedSuccessor = null;
    let expectedPredecessor = null;
    for (const candidate of oracle) {
      if (candidate > probe && expectedSuccessor === null) expectedSuccessor = candidate;
      if (candidate < probe) expectedPredecessor = candidate;
    }
    assert.strictEqual(veb.successor(probe), expectedSuccessor, `final successor ${probe}`);
    assert.strictEqual(veb.predecessor(probe), expectedPredecessor, `final predecessor ${probe}`);
  }
  console.log("PASS");
}
