/** Task 482 - Implement bloom filter per SSTable prototype and test. (Tier: Advanced, CTCI Ch 14)
 *
 * Prototype of an LSM-style lookup: each SSTable owns a bloom filter sized to 10 bits per key with
 * k = 7 double-hashed probes, and a sorted entry list for the exact confirmation step. Hashing is
 * FNV-1a 32-bit; probe j uses (h1 + j * h2) % m with h2 = fnv1a(key + "!") | 1 (odd, so the probe
 * sequence never collapses). Lookup consults every table's filter first (false positives only cost
 * an exact binary search, never a wrong answer).
 *
 * Documented false-positive estimate (bits m, keys n, probes k): p = (1 - e^(-k*n/m))^k ~ 0.0082
 * per table; for the combined union the estimate is 1 - prod(1 - p_t) ~ 0.0244. The test pins the
 * exact deterministic false-positive counts and asserts the measured rate stays within a
 * documented 3x-theory bound, plus zero false negatives for every inserted key.
 */
"use strict";

class BloomFilter {
  constructor(bits, hashes) {
    this.m = bits;
    this.k = hashes;
    this.words = new Uint8Array((bits + 7) >> 3);
    this.count = 0;
  }

  static fnv1a(text) {
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h;
  }

  probes(key) {
    const h1 = BloomFilter.fnv1a(key);
    const h2 = (BloomFilter.fnv1a(key + "!") | 1) >>> 0;
    const idx = [];
    for (let j = 0; j < this.k; j++) idx.push((h1 + j * h2) % this.m);
    return idx;
  }

  add(key) {
    for (const i of this.probes(key)) this.words[i >> 3] |= 1 << (i & 7);
    this.count += 1;
  }

  maybeContains(key) {
    for (const i of this.probes(key)) {
      if ((this.words[i >> 3] & (1 << (i & 7))) === 0) return false;
    }
    return true;
  }
}

class SSTable {
  constructor(entries, bitsPerKey, hashes) {
    this.entries = entries.slice().sort();
    this.bloom = new BloomFilter(bitsPerKey * this.entries.length, hashes);
  }

  build(keys) {
    for (const key of keys) this.bloom.add(key);
    return this;
  }

  containsExact(key) {
    let lo = 0;
    let hi = this.entries.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (this.entries[mid] === key) return true;
      if (this.entries[mid] < key) lo = mid + 1;
      else hi = mid - 1;
    }
    return false;
  }

  contains(key) {
    return this.bloom.maybeContains(key) && this.containsExact(key);
  }
}

class Store {
  constructor(tables) {
    this.tables = tables;
  }

  get(key) {
    for (const table of this.tables) if (table.contains(key)) return true;
    return false;
  }

  bloomAny(key) {
    for (const table of this.tables) if (table.bloom.maybeContains(key)) return true;
    return false;
  }
}

function lcgValues(count, seed, limit) {
  const values = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push(seed % limit);
  }
  return values;
}

function theoreticalFpRate(keys, bits, hashes) {
  return Math.pow(1.0 - Math.exp((-hashes * keys) / bits), hashes);
}

function buildStore(sizes, startSeed) {
  let seed = startSeed;
  const tables = [];
  for (let i = 0; i < sizes.length; i++) {
    const size = sizes[i];
    const one = [];
    for (let j = 0; j < size; j++) one.push(`t${i}:${j}`);
    seed = (seed * 75 + 74) % 65537;
    const rotation = seed % size;
    const keys = one.slice(rotation).concat(one.slice(0, rotation));
    tables.push(new SSTable(keys, 10, 7).build(keys));
  }
  return tables;
}

module.exports = { BloomFilter, SSTable, Store };

if (require.main === module) {
  const assert = require("node:assert");

  const tiny = new BloomFilter(64, 4);
  tiny.add("alpha");
  assert.ok(tiny.maybeContains("alpha"));
  assert.ok(!new BloomFilter(64, 4).maybeContains("alpha"));

  const sizes = [150, 180, 210];
  const tables = buildStore(sizes, 482);
  const store = new Store(tables);
  for (let i = 0; i < 3; i++) {
    assert.strictEqual(tables[i].entries.length, sizes[i]);
    assert.strictEqual(tables[i].bloom.m, sizes[i] * 10);
    assert.strictEqual(tables[i].bloom.k, 7);
  }

  let presentQueries = 0;
  for (const table of tables) {
    for (const key of table.entries) {
      assert.ok(table.bloom.maybeContains(key), `no false negative ${key}`);
      assert.ok(store.get(key), `store finds ${key}`);
      presentQueries += 1;
    }
  }
  assert.strictEqual(presentQueries, 540);

  const union = Array.from(new Set(tables.flatMap((t) => t.entries))).sort();
  for (const value of lcgValues(200, 9001, 540)) {
    const key = union[value];
    assert.ok(store.get(key) && store.bloomAny(key), `present probe ${key}`);
  }
  const absent = lcgValues(240, 9002, 65537).map((v) => `absent:${v}`);
  for (const key of absent) {
    assert.ok(!union.includes(key), `absent build ${key}`);
    assert.ok(!store.get(key), `absent lookup ${key}`);
  }

  const perTableFp = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    for (const key of absent) if (tables[i].bloom.maybeContains(key)) perTableFp[i] += 1;
  }
  let combinedFp = 0;
  for (const key of absent) if (store.bloomAny(key)) combinedFp += 1;
  // Pinned from the deterministic construction; identical in all four languages.
  assert.deepStrictEqual(perTableFp, [2, 1, 1]);
  assert.strictEqual(combinedFp, 4);

  const pTable = [];
  let pUnionComplement = 1.0;
  for (let i = 0; i < 3; i++) {
    const t = tables[i];
    pTable.push(theoreticalFpRate(sizes[i], t.bloom.m, t.bloom.k));
    pUnionComplement *= 1.0 - pTable[i];
    assert.ok(perTableFp[i] <= 3 * pTable[i] * absent.length + 3, `theory bound ${i}`);
    assert.ok(perTableFp[i] / absent.length <= 0.05, `rate bound ${i}`);
    assert.strictEqual(Math.round(pTable[i] * 10000), 82, `theoretical p ${i}`);
  }
  const pUnion = 1.0 - pUnionComplement;
  assert.ok(combinedFp <= 3 * pUnion * absent.length + 3, "union bound");
  assert.ok(combinedFp / absent.length <= 0.05, "union rate bound");
  assert.strictEqual(Math.round(pUnion * 10000), 244, `theoretical union ${pUnion}`);

  const bitsIn = (b) => {
    let n = b;
    let count = 0;
    while (n) {
      n &= n - 1;
      count += 1;
    }
    return count;
  };
  const setBits = tables.map((t) => t.bloom.words.reduce((acc, b) => acc + bitsIn(b), 0));
  assert.deepStrictEqual(setBits, [768, 905, 1052]);
  for (let i = 0; i < 3; i++) assert.ok(setBits[i] < tables[i].bloom.m, `not saturated ${i}`);
  console.log("PASS");
}
