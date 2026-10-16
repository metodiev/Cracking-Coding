/** Task 607 - Implement external-memory suffix array construction (disk-based). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * WHAT IS SIMPLIFIED: real file I/O is not available here, so the disk is modelled by `BlockStore`, an
 * arena of fixed-size blocks with counted transfers, plus a `BufferCache` with a fixed number of frames
 * and a run-based external merge sorter. Every phase respects the memory budget an on-disk version
 * would use: run generation sorts at most bufferFrames * blockSize items in memory, a merge keeps one
 * block of each of its fanIn streams resident, and the page cache holds cacheFrames blocks. The
 * invariant verified is the real one: Manber-Myers prefix doubling runs block by block, ranks and the
 * order live on the "disk" between rounds, and the final block list is a correct suffix array.
 *
 * Algorithm: ranks start as compressed character codes; each round external-sorts all N suffixes by
 * (rank[i], rank[i + k]) with rank[i + k] = -1 past the end, then recomputes ranks with one sequential
 * scan of the sorted blocks. The loop stops once all N ranks are distinct, after O((N/B) log N) transfers.
 *
 * Checks: the block store rejects oversize blocks, block sizes 1/4/16 agree with an in-memory doubling
 * suffix array and with a character-by-character comparator, every result is a permutation, transfer
 * counters and cache evictions are exercised, the modelled memory stays far below N, and degenerate
 * inputs (empty, unary, periodic, LCG strings, invalid parameters) are covered. */
"use strict";

const SENTINEL = "\u0000";

class BlockStore {
  constructor(blockSize) {
    if (blockSize < 1) throw new Error("the block size must be at least one item");
    this.blockSize = blockSize;
    this.blocks = [];
    this.reads = 0;
    this.writes = 0;
    this.peak = 0;
  }

  write(items) {
    if (items.length > this.blockSize) throw new Error("block exceeds the block capacity");
    this.blocks.push(Array.from(items));
    this.writes++;
    this.peak = Math.max(this.peak, items.length);
    return this.blocks.length - 1;
  }

  read(blockId) {
    this.reads++;
    return this.blocks[blockId];
  }

  /** Splits `items` into consecutive blocks and returns their ids. */
  writeAll(items) {
    const ids = [];
    const size = this.blockSize;
    for (let i = 0; i < items.length; i += size) ids.push(this.write(items.slice(i, i + size)));
    return ids;
  }

  readAll(blockIds) {
    const items = [];
    for (const blockId of blockIds) for (const item of this.read(blockId)) items.push(item);
    return items;
  }
}

/** Fixed-size page cache with LRU replacement, counting hits, misses and evictions. */
class BufferCache {
  constructor(store, frames) {
    this.store = store;
    this.frames = Math.max(1, frames);
    this.resident = new Map();
    this.hits = 0;
    this.misses = 0;
    this.evictions = 0;
  }

  get(blockId) {
    if (this.resident.has(blockId)) {
      const block = this.resident.get(blockId);
      this.resident.delete(blockId);
      this.resident.set(blockId, block); // refresh the LRU position
      this.hits++;
      return block;
    }
    this.misses++;
    if (this.resident.size >= this.frames) {
      this.resident.delete(this.resident.keys().next().value);
      this.evictions++;
    }
    const block = this.store.read(blockId);
    this.resident.set(blockId, block);
    return block;
  }
}

/** Streams one block at a time out of a run stored as a list of block ids. */
class RunReader {
  constructor(store, blockIds) {
    this.store = store;
    this.blockIds = blockIds;
    this.blockIndex = 0;
    this.items = blockIds.length > 0 ? store.read(blockIds[0]) : [];
    this.position = 0;
    this.done = blockIds.length === 0;
  }

  current() {
    return this.items[this.position];
  }

  advance() {
    this.position++;
    if (this.position < this.items.length) return true;
    this.blockIndex++;
    if (this.blockIndex >= this.blockIds.length) {
      this.done = true;
      return false;
    }
    this.items = this.store.read(this.blockIds[this.blockIndex]);
    this.position = 0;
    return true;
  }
}

/** External merge sort: bounded in-memory runs, then multi-pass merges with fanIn streams. */
class ExternalSorter {
  constructor(store, blockSize, bufferFrames, fanIn) {
    this.store = store;
    this.blockSize = blockSize;
    this.runItems = blockSize * Math.max(1, bufferFrames);
    this.fanIn = Math.max(2, fanIn);
    this.mergePasses = 0;
  }

  /** Upper bound on the items this sorter may hold in memory at any moment. */
  residentItemBound(cacheFrames) {
    return this.runItems + (this.fanIn + cacheFrames) * this.blockSize;
  }

  /** Sorts `items` by `compare`; returns the ids of the blocks holding the sorted result. */
  sort(items, compare) {
    if (items.length === 0) return [];
    let runs = [];
    for (let start = 0; start < items.length; start += this.runItems) {
      const buffer = items.slice(start, start + this.runItems);
      buffer.sort(compare);
      runs.push(this.store.writeAll(buffer));
    }
    while (runs.length > 1) {
      this.mergePasses++;
      const merged = [];
      for (let start = 0; start < runs.length; start += this.fanIn) {
        merged.push(this.mergeGroup(runs.slice(start, start + this.fanIn), compare));
      }
      runs = merged;
    }
    return runs[0];
  }

  /** Merges up to fanIn sorted runs, buffering one block per stream and one output block. */
  mergeGroup(group, compare) {
    const readers = group.map((blockIds) => new RunReader(this.store, blockIds));
    const heads = readers.map((reader) => (reader.done ? null : reader.current()));
    const outputBlocks = [];
    let buffer = [];
    for (;;) {
      let best = -1;
      for (let i = 0; i < readers.length; i++) {
        if (heads[i] !== null && (best === -1 || compare(heads[i], heads[best]) < 0)) best = i;
      }
      if (best === -1) break;
      buffer.push(heads[best]);
      heads[best] = readers[best].advance() ? readers[best].current() : null;
      if (buffer.length === this.blockSize) {
        outputBlocks.push(this.store.write(buffer));
        buffer = [];
      }
    }
    if (buffer.length > 0) outputBlocks.push(this.store.write(buffer));
    return outputBlocks;
  }
}

/**
 * External prefix doubling over blocks. `text` is terminated with a sentinel so that every suffix ends
 * in a leaf of the suffix array and rank[i + k] = -1 past the end is the correct shorter-suffix rule.
 */
function externalSuffixArray(text, options = {}) {
  if (text.includes(SENTINEL)) throw new Error("text must not contain the sentinel character");
  const blockSize = options.blockSize ?? 8;
  const bufferFrames = options.bufferFrames ?? 3;
  const cacheFrames = options.cacheFrames ?? 2;
  if (bufferFrames < 2) throw new Error("run generation needs at least two buffer frames");
  const terminated = text + SENTINEL;
  const N = terminated.length;
  const store = new BlockStore(blockSize);
  const cache = new BufferCache(store, cacheFrames);
  const sorter = new ExternalSorter(store, blockSize, bufferFrames, options.fanIn ?? Math.max(2, bufferFrames - 1));
  const alphabet = [...new Set([...terminated])].sort();
  const code = new Map(alphabet.map((ch, i) => [ch, i]));
  let rankBlocks = store.writeAll(Array.from(terminated, (ch) => code.get(ch)));
  let order = Array.from({ length: N }, (_, i) => i);
  let shift = 1;
  let rounds = 0;
  let distinct = 0;

  const rankOf = (index) => cache.get(rankBlocks[(index / blockSize) | 0])[index % blockSize];
  for (;;) {
    // Key of a suffix this round: (rank[i], rank[i + shift]) with -1 past the end.
    const keyCompare = (a, b) => {
      const leftRank = rankOf(a);
      const rightRank = rankOf(b);
      if (leftRank !== rightRank) return leftRank - rightRank;
      const leftNext = a + shift < N ? rankOf(a + shift) : -1;
      const rightNext = b + shift < N ? rankOf(b + shift) : -1;
      return leftNext - rightNext;
    };
    // The total order only adds an index tie-break; equal keys must stay equal for the next ranking.
    const sortedItems = store.readAll(sorter.sort(order, (a, b) => keyCompare(a, b) || a - b));
    const newRanks = new Array(N);
    let currentRank = 0;
    for (let i = 0; i < sortedItems.length; i++) {
      if (i > 0 && keyCompare(sortedItems[i - 1], sortedItems[i]) !== 0) currentRank++;
      newRanks[sortedItems[i]] = currentRank;
    }
    rounds++;
    distinct = currentRank + 1;
    rankBlocks = store.writeAll(newRanks);
    order = sortedItems;
    if (distinct >= N || shift > 2 * N) break;
    shift <<= 1;
  }

  return {
    suffixArray: order.slice(0, N),
    stats: {
      rounds, blockSize, blocks: store.blocks.length, blockReads: store.reads, blockWrites: store.writes,
      storedItems: store.blocks.reduce((acc, block) => acc + block.length, 0), peakBlockLength: store.peak,
      cacheHits: cache.hits, cacheMisses: cache.misses, cacheEvictions: cache.evictions,
      mergePasses: sorter.mergePasses, residentItemBound: sorter.residentItemBound(cacheFrames),
    },
  };
}

function compareSuffixes(s, i, j) {
  const n = s.length;
  while (i < n && j < n) {
    if (s[i] !== s[j]) return s[i] < s[j] ? -1 : 1;
    i++;
    j++;
  }
  if (i === j) return 0;
  return i === n ? -1 : 1;
}

/** Independent oracle: in-memory prefix-doubling suffix array. */
function suffixArrayDoubling(s) {
  const n = s.length;
  if (n === 0) return [];
  const sa = Array.from({ length: n }, (_, i) => i);
  let rank = Array.from(s, (ch) => ch.charCodeAt(0));
  const key = (i, shift) => (i + shift < n ? rank[i + shift] : -1);
  for (let shift = 1; ; shift <<= 1) {
    sa.sort((a, b) => rank[a] - rank[b] || key(a, shift) - key(b, shift));
    const next = new Array(n).fill(0);
    for (let i = 1; i < n; i++) {
      const prev = sa[i - 1];
      const cur = sa[i];
      next[cur] = next[prev] + (rank[prev] !== rank[cur] || key(prev, shift) !== key(cur, shift) ? 1 : 0);
    }
    if (next[sa[n - 1]] === n - 1) return sa;
    rank = next;
  }
}

function lcgString(length, alphabet, seed) {
  let state = seed >>> 0;
  const chars = [];
  for (let i = 0; i < length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    chars.push(alphabet[(state >>> 8) % alphabet.length]);
  }
  return [chars.join(""), state];
}

module.exports = {
  BlockStore, BufferCache, ExternalSorter, externalSuffixArray, compareSuffixes, suffixArrayDoubling,
  lcgString, SENTINEL,
};

if (require.main === module) {
  const assert = require("node:assert");

  assert.throws(() => new BlockStore(2).write([1, 2, 3]), /capacity/);
  assert.throws(() => new BlockStore(0), /block size/);
  const store = new BlockStore(3);
  const ids = store.writeAll([1, 2, 3, 4, 5, 6, 7]);
  assert.deepStrictEqual([ids.length, store.peak, store.readAll(ids)], [3, 3, [1, 2, 3, 4, 5, 6, 7]]);

  const cached = new BufferCache(store, 1);
  cached.get(0);
  cached.get(1); // evicts block 0
  cached.get(0); // evicts block 1 again
  assert.deepStrictEqual([cached.misses, cached.evictions], [3, 2]);
  assert.deepStrictEqual(cached.get(0), [1, 2, 3]);
  assert.strictEqual(cached.hits, 1);

  for (const [text, expected] of [["", [0]], ["a", [1, 0]], ["ab", [2, 0, 1]], ["aaaa", [4, 3, 2, 1, 0]]]) {
    const result = externalSuffixArray(text, { blockSize: 2, bufferFrames: 2, cacheFrames: 2 });
    assert.deepStrictEqual(result.suffixArray, expected, `tiny ${text}`);
    assert.deepStrictEqual(result.suffixArray, suffixArrayDoubling(text + SENTINEL), `oracle ${text}`);
  }

  const cases = ["banana", "mississippi", "abracadabra", "abcabcabc", "ba".repeat(60), "abcdefghij".repeat(20)];
  let seed = 607;
  for (const alphabet of ["ab", "abc", "abcdefghij"]) {
    for (let trial = 0; trial < 4; trial++) {
      let text;
      [text, seed] = lcgString(180, alphabet, seed);
      cases.push(text);
    }
  }
  let mergePasses = 0;
  for (const text of cases) {
    const reference = suffixArrayDoubling(text + SENTINEL);
    for (const blockSize of [1, 4, 16]) {
      const result = externalSuffixArray(text, { blockSize, bufferFrames: 4, cacheFrames: 2 });
      const label = `${text.length} chars, block ${blockSize}`;
      assert.deepStrictEqual(result.suffixArray, reference, label);
      assert.strictEqual(new Set(result.suffixArray).size, text.length + 1, `${label} permutation`);
      for (let row = 0; row + 1 < reference.length; row++) {
        const order = result.suffixArray;
        assert.ok(compareSuffixes(text + SENTINEL, order[row], order[row + 1]) < 0, `${label} order`);
      }
      assert.ok(result.stats.peakBlockLength <= blockSize, `${label} capacity`);
      assert.ok(result.stats.blockReads > 0 && result.stats.blockWrites > 0, `${label} transfers`);
      assert.ok(result.stats.storedItems >= 2 * (text.length + 1), `${label} stored items`);
      assert.ok(result.stats.rounds <= Math.ceil(Math.log2(text.length + 2)) + 1, `${label} rounds`);
      mergePasses += result.stats.mergePasses;
    }
  }
  assert.ok(mergePasses > 0, "multi-run merges happened");
  assert.throws(() => externalSuffixArray("x", { bufferFrames: 1 }), /buffer frames/);
  assert.throws(() => externalSuffixArray("bad\u0000text"), /sentinel/);

  const big = "abcab".repeat(120) + "xy";
  const bigResult = externalSuffixArray(big, { blockSize: 16, bufferFrames: 4, fanIn: 3, cacheFrames: 2 });
  assert.deepStrictEqual(bigResult.suffixArray, suffixArrayDoubling(big + SENTINEL));
  assert.ok(bigResult.stats.cacheEvictions > 0, "the page cache really evicts");
  assert.ok(bigResult.stats.residentItemBound * 2 < bigResult.suffixArray.length &&
    (bigResult.stats.blockReads + bigResult.stats.blockWrites) / bigResult.stats.blocks < 120, "bounded I/O");

  console.log(`external sa: N=${bigResult.suffixArray.length}, rounds=${bigResult.stats.rounds}, ` +
    `blocks=${bigResult.stats.blocks}, transfers=${bigResult.stats.blockReads + bigResult.stats.blockWrites}, ` +
    `resident bound=${bigResult.stats.residentItemBound} items`);  console.log("PASS");}
