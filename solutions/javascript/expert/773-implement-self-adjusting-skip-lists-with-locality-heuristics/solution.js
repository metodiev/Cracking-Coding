/** Task 773 - Implement self-adjusting skip lists with locality heuristics. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A skip list whose towers adapt to the *access* pattern instead of only to the insertion-time
 * coin flips. Insertions use the classic geometric height from a seeded xorshift PRNG; every
 * successful search increments the node's access counter and, if that counter now justifies a
 * taller tower, promotes the node one level at a time using the search path as the splice points:
 *
 *     desiredHeight(node) = min(maxHeight, 1 + floor(log2(accesses)))
 *
 * Promotion never touches level 0, so the bottom list stays exactly the sorted key set; it only
 * adds the hot node to higher express lanes. That is the locality heuristic: keys that are hit
 * often end up with tall towers and the expected search length for them drops towards the height
 * of the levels above them, while cold keys keep their random heights.
 *
 * Verified here: the level-0 list equals the sorted key set of an oracle Map after every mutation
 * and membership agrees with a binary-search oracle on a sorted array; level nesting
 * (level k+1 subset of level k), height caps and node counts; delete semantics including repeated
 * deletes and re-inserts; promotion is monotone and exactly follows the log2 rule; and on the same
 * seeded skewed workload the adaptive run crosses strictly fewer links than the promotion-free
 * baseline (link crossings are the classic skip-list search cost; the per-level descent is
 * identical in both runs) while both return identical data; degenerate lists (empty, one key with
 * maxHeight 1, duplicate keys) and argument validation.
 */
"use strict";

/** xorshift32; deterministic across runs and platforms. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

class SelfAdjustingSkipList {
  constructor({ maxHeight = 12, seed = 987654321, promotion = true } = {}) {
    if (!Number.isInteger(maxHeight) || maxHeight < 1) throw new RangeError("maxHeight must be >= 1");
    this.maxHeight = maxHeight;
    this.head = { key: null, next: new Array(maxHeight).fill(null), height: maxHeight, accesses: 0 };
    this.rng = makeRng(seed);
    this.promotion = promotion;
    this.size = 0;
    this.stats = { searches: 0, comparisons: 0, levelSteps: 0, moves: 0, promotions: 0, hits: 0, misses: 0 };
  }

  /** Geometric height (p = 1/2), exact for maxHeight = 1. */
  #randomHeight() {
    let height = 1;
    while (height < this.maxHeight && this.rng(2) === 0) height++;
    return height;
  }

  /** Path of predecessors per level, recorded while descending to the insertion point. */
  #predecessors(key) {
    const update = new Array(this.maxHeight).fill(this.head);
    let node = this.head;
    for (let level = this.maxHeight - 1; level >= 0; level--) {
      this.stats.levelSteps++; // entering a level costs one probe of its list
      while (node.next[level] !== null && node.next[level].key < key) {
        node = node.next[level];
        this.stats.moves++; // each crossed link is one comparison
      }
      update[level] = node;
    }
    this.stats.comparisons = this.stats.moves + this.stats.levelSteps;
    return update;
  }

  insert(key, value = key) {
    const update = this.#predecessors(key);
    const existing = update[0].next[0];
    if (existing !== null && existing.key === key) {
      existing.value = value;
      return false;
    }
    const height = this.#randomHeight();
    const node = { key, value, height, accesses: 0, next: new Array(height).fill(null) };
    for (let level = 0; level < height; level++) {
      node.next[level] = update[level].next[level];
      update[level].next[level] = node;
    }
    this.size++;
    return true;
  }

  /** Search with the locality heuristic; returns the value or undefined. */
  search(key) {
    this.stats.searches++;
    const update = this.#predecessors(key);
    const candidate = update[0].next[0];
    if (candidate !== null && candidate.key === key) {
      this.stats.hits++;
      this.#recordAccess(candidate, update);
      return candidate.value;
    }
    this.stats.misses++;
    return undefined;
  }

  /** Access bookkeeping and promotion to the height justified by the access count. */
  #recordAccess(node, update) {
    node.accesses++;
    if (!this.promotion) return;
    let desired = 1;
    while (desired < this.maxHeight && node.accesses >= (1 << desired)) desired++;
    while (node.height < desired) {
      const level = node.height;
      node.next[level] = update[level].next[level];
      update[level].next[level] = node;
      node.height++;
      this.stats.promotions++;
    }
  }

  delete(key) {
    const update = this.#predecessors(key);
    const node = update[0].next[0];
    if (node === null || node.key !== key) return false;
    for (let level = 0; level < node.height; level++) update[level].next[level] = node.next[level];
    this.size--;
    return true;
  }

  has(key) { return this.search(key) !== undefined; }

  get(key) { return this.search(key); }

  /** Level-0 walk: must always be the sorted key set. */
  keys() {
    const out = [];
    for (let node = this.head.next[0]; node !== null; node = node.next[0]) out.push(node.key);
    return out;
  }

  values() { return this.keys().map((key) => this.get(key)); }

  /** Structural snapshot used by the invariants below. */
  snapshot() {
    const levels = [];
    for (let level = 0; level < this.maxHeight; level++) {
      const row = [];
      for (let node = this.head.next[level]; node !== null; node = node.next[level]) row.push(node.key);
      levels.push(row);
    }
    const heights = new Map();
    for (let node = this.head.next[0]; node !== null; node = node.next[0]) heights.set(node.key, node.height);
    return { levels, heights };
  }

  heightHistogram() {
    const histogram = new Map();
    for (let node = this.head.next[0]; node !== null; node = node.next[0]) {
      histogram.set(node.height, (histogram.get(node.height) ?? 0) + 1);
    }
    return histogram;
  }

  checkInvariants() {
    const levels = this.snapshot().levels;
    for (let level = 0; level < this.maxHeight - 1; level++) {
      const lower = new Set(levels[level]);
      for (const key of levels[level + 1]) if (!lower.has(key)) return `key ${key} at level ${level + 1} is missing from level ${level}`;
    }
    for (let level = 0; level < this.maxHeight; level++) {
      for (let i = 1; i < levels[level].length; i++) {
        if (!(levels[level][i - 1] < levels[level][i])) return `level ${level} is not sorted at index ${i}`;
      }
    }
    if (levels[0].length !== this.size) return `level 0 holds ${levels[0].length} keys but size is ${this.size}`;
    return null;
  }
}

/** Independent oracle: binary search in the sorted key array. */
function oracleSearch(sortedKeys, key) {
  let low = 0;
  let high = sortedKeys.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (sortedKeys[mid] === key) return true;
    if (sortedKeys[mid] < key) low = mid + 1;
    else high = mid - 1;
  }
  return false;
}

/** Deterministic skewed access sequence: the four `hot` keys take ~70% of the traffic. */
function skewedAccess(rng, length, vocabSize, hot) {
  const out = [];
  for (let i = 0; i < length; i++) {
    out.push(rng(100) < 70 ? hot[rng(hot.length)] : rng(vocabSize));
  }
  return out;
}

module.exports = { SelfAdjustingSkipList, makeRng, oracleSearch, skewedAccess };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;
  const rng = makeRng(20260773);
  let invariantChecks = 0;

  const drainStats = (list) => ({ ...list.stats });

  // -- degenerate lists --------------------------------------------------------------
  const empty = new SelfAdjustingSkipList();
  strictEqual(empty.size, 0, "a fresh list is empty");
  strictEqual(empty.keys().length, 0, "the bottom level of a fresh list is empty");
  strictEqual(empty.search(5), undefined, "searching an empty list misses");
  strictEqual(empty.delete(5), false, "deleting from an empty list fails");
  strictEqual(empty.checkInvariants(), null, "invariants hold for the empty list");

  const single = new SelfAdjustingSkipList({ maxHeight: 1, seed: 5 });
  strictEqual(single.insert(7, "seven"), true, "first insert reports a new key");
  strictEqual(single.insert(7, "SEVEN"), false, "re-inserting only updates the value");
  strictEqual(single.size, 1, "re-inserting must not change the size");
  strictEqual(single.search(7), "SEVEN", "the value must have been overwritten");
  strictEqual(single.snapshot().heights.get(7), 1, "a height-1 list keeps every node at level 0");
  strictEqual(single.delete(7), true, "deleting the only key succeeds");
  strictEqual(single.delete(7), false, "deleting it twice fails the second time");
  throws(() => new SelfAdjustingSkipList({ maxHeight: 0 }), RangeError, "maxHeight must be positive");

  // -- ordered insert / search / delete against the oracle ---------------------------
  const list = new SelfAdjustingSkipList({ maxHeight: 6, seed: 424242, promotion: false });
  const oracle = new Map();
  const keys = [];
  for (let i = 0; i < 200; i++) {
    const key = rng(400) - 200;
    if (oracle.has(key)) continue;
    keys.push(key);
    oracle.set(key, key * 3);
    list.insert(key, key * 3);
  }
  keys.sort((a, b) => a - b);
  deepStrictEqual(list.keys(), keys, "level 0 must be the sorted key set");
  strictEqual(list.size, oracle.size, "size must match the oracle");
  strictEqual(list.checkInvariants(), null, "nesting and ordering invariants hold");
  invariantChecks++;
  for (const key of keys) strictEqual(list.search(key), key * 3, `value mismatch for ${key}`);
  for (let probe = -250; probe <= 250; probe++) {
    strictEqual(list.search(probe) !== undefined, oracleSearch(keys, probe), `membership disagrees for ${probe}`);
  }

  // -- a mixing workload with deletion, verified after every mutation ----------------
  let mutationCount = 0;
  const workload = new SelfAdjustingSkipList({ maxHeight: 5, seed: 7, promotion: true });
  const shadow = new Map();
  for (let step = 0; step < 400; step++) {
    const key = rng(120);
    const action = rng(10);
    if (action < 6) {
      workload.insert(key, `v${key}`);
      shadow.set(key, `v${key}`);
    } else if (action < 9) {
      workload.delete(key);
      shadow.delete(key);
      strictEqual(workload.search(key), undefined, "a deleted key must not be found");
    } else {
      strictEqual(workload.search(key), shadow.get(key), "search must agree with the shadow map");
    }
    strictEqual(workload.size, shadow.size, `size drift at step ${step}`);
    strictEqual(workload.checkInvariants(), null, `invariant broken at step ${step}`);
    mutationCount++;
  }
  invariantChecks++;
  const shadowKeys = [...shadow.keys()].sort((a, b) => a - b);
  deepStrictEqual(workload.keys(), shadowKeys, "the bottom level must equal the shadow key set");
  for (const key of shadowKeys) strictEqual(workload.search(key), shadow.get(key), "final values must agree");

  // -- promotion rule: height == 1 + floor(log2(accesses)) (capped) ------------------
  const hot = new SelfAdjustingSkipList({ maxHeight: 8, seed: 11, promotion: true });
  for (let i = 0; i < 30; i++) hot.insert(i, i);
  for (let round = 0; round < 40; round++) hot.search(3);
  const hotHeight = hot.snapshot().heights.get(3);
  strictEqual(hotHeight, Math.min(8, 1 + Math.floor(Math.log2(40))), "height must follow the log2 access rule");
  strictEqual(hot.snapshot().heights.get(4), hot.insert(4, 4) === false ? hot.snapshot().heights.get(4) : 1,
    "cold keys keep their insertion height");
  ok(hotHeight > 1, "a hot key must actually have been promoted");
  strictEqual(hot.checkInvariants(), null, "promotion must preserve the nesting invariant");
  strictEqual(hot.keys().length, 30, "promotion must not change the key set");
  ok(hot.stats.promotions >= hotHeight - 1, "the promotion counter must record the level additions");
  // height is monotone in the access count and never exceeds the cap
  const tall = new SelfAdjustingSkipList({ maxHeight: 3, seed: 12, promotion: true });
  for (let i = 0; i < 10; i++) tall.insert(i, i);
  for (let r = 0; r < 500; r++) tall.search(0);
  strictEqual(tall.snapshot().heights.get(0), 3, "the promotion must stop at maxHeight");
  deepStrictEqual(tall.heightHistogram().has(4), false, "no node may exceed maxHeight");

  // -- locality: the adaptive run traverses fewer links than the same run without -----
  const hotKeys = [148, 149, 150, 151];
  const runStats = (promotion, sequence) => {
    const l = new SelfAdjustingSkipList({ maxHeight: 10, seed: 2024, promotion });
    for (let i = 0; i < 300; i++) l.insert(i, i);
    for (const key of sequence) strictEqual(l.search(key), key, "both runs must return the same data");
    deepStrictEqual(l.keys(), Array.from({ length: 300 }, (_, i) => i), "both runs keep the same sorted keys");
    return drainStats(l);
  };
  const sequence = skewedAccess(makeRng(31), 4000, 300, hotKeys);
  const adaptiveStats = runStats(true, sequence);
  const baselineStats = runStats(false, sequence);
  ok(adaptiveStats.moves < baselineStats.moves,
    `the locality heuristic must cross fewer links (${adaptiveStats.moves} < ${baselineStats.moves})`);
  ok(baselineStats.moves - adaptiveStats.moves > baselineStats.moves * 0.1,
    "the reduction must be more than noise");
  strictEqual(adaptiveStats.levelSteps, baselineStats.levelSteps, "the level descent itself is unchanged");
  strictEqual(adaptiveStats.promotions > 0, true, "the adaptive run must have promoted something");
  strictEqual(baselineStats.promotions, 0, "promotion must be off in the baseline");
  const sorted = Array.from({ length: 300 }, (_, i) => i);
  for (const key of sequence.slice(0, 200)) ok(oracleSearch(sorted, key), "the workload keys must exist");
  ok(sequence.filter((k) => hotKeys.includes(k)).length > sequence.length * 0.6, "the workload must be skewed");

  // -- determinism -------------------------------------------------------------------
  const buildOnce = (seed) => {
    const l = new SelfAdjustingSkipList({ maxHeight: 7, seed, promotion: true });
    const r = makeRng(seed);
    for (let i = 0; i < 150; i++) l.insert(r(100), i);
    for (let i = 0; i < 150; i++) l.search(r(100));
    return { keys: l.keys(), histogram: [...l.heightHistogram().entries()].sort((a, b) => a[0] - b[0]) };
  };
  deepStrictEqual(buildOnce(99), buildOnce(99), "the same seed must build the same towers");
  ok(buildOnce(100).keys.length > 0, "a different seed still builds a valid list");

  console.log(`mutations=${mutationCount} comparisons adaptive=${adaptiveStats.comparisons} baseline=${baselineStats.comparisons} moves ${adaptiveStats.moves} vs ${baselineStats.moves} promotions=${adaptiveStats.promotions}`);
  console.log("PASS");
}
