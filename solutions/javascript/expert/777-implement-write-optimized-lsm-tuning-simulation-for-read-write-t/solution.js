/** Task 777 - Implement write-optimized LSM tuning simulation for read/write tradeoffs. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A size-tiered LSM simulator: writes land in a memtable, the memtable is flushed to L0 as one run
 * when it is full, and a level holding more than `ratio` runs merges all of them into a single run
 * that is pushed to the next level. Every merge rewrites bytes, which is what write amplification
 * measures; every lookup walks the memtable and then the levels newest-first, binary-searching one
 * run per level (a run counts as one probe, which is the level of detail Bloom filters would hide).
 * Tombstones survive until they reach the last level, so deletions stay correct and their space is
 * only reclaimed by the final compaction.
 *
 * The parameters being "tuned" are the memtable size, the level ratio and the level count. The
 * simulator reports write amplification, probes per lookup and live-vs-total space; the point of the
 * exercise is that these trade against each other (a larger ratio compacts less often but leaves
 * more runs to probe).
 *
 * What is verified: every point lookup equals a shadow hash map (so overwrites and tombstones are
 * genuinely right); the write amplification computed by the running counter equals the same figure
 * recomputed from the compaction log; every level holds at most `ratio` runs and all runs are
 * sorted; the tombstone space is reclaimed only at the bottom level; larger ratios write less and
 * probe more on the same deterministic workload; and degenerate configurations (empty tree, one
 * key, no compaction, pure-delete workload) behave.
 */
"use strict";

class Run {
  constructor(entries) {
    entries.sort((a, b) => a.key - b.key);
    this.entries = entries;
    this.bytes = entries.reduce((sum, e) => sum + e.bytes, 0);
  }

  /** Binary search for the key; returns the entry or undefined. */
  get(key) {
    let low = 0;
    let high = this.entries.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (this.entries[mid].key === key) return this.entries[mid];
      if (this.entries[mid].key < key) low = mid + 1;
      else high = mid - 1;
    }
    return undefined;
  }

  get liveBytes() { return this.entries.filter((e) => !e.tombstone).reduce((sum, e) => sum + e.bytes, 0); }
}

class LSMTree {
  constructor({ memtableBytes = 64, ratio = 4, levels = 4, valueBytes = 4 } = {}) {
    if (!(memtableBytes > 0) || ratio < 2 || levels < 1) throw new RangeError("invalid LSM tuning");
    this.memtableBytes = memtableBytes;
    this.ratio = ratio;
    this.maxLevels = levels;
    this.valueBytes = valueBytes;
    this.memtable = new Map(); // key -> value (insertion ordered, flushed as a whole)
    this.memtableSize = 0;
    this.levels = Array.from({ length: levels }, () => []);
    this.log = []; // every byte ever written, in order, for the independent reconciliation
    this.stats = { userBytes: 0, writtenBytes: 0, flushes: 0, compactions: 0, probes: 0, lookups: 0 };
  }

  put(key, value) {
    const previous = this.memtable.get(key);
    if (previous !== undefined) this.memtableSize -= this.#entryBytes(previous);
    this.memtable.set(key, value);
    this.memtableSize += this.#entryBytes(value);
    this.stats.userBytes += this.#entryBytes(value);
    this.stats.writtenBytes += this.#entryBytes(value);
    this.log.push(this.#entryBytes(value));
    if (this.memtableSize >= this.memtableBytes) this.#flush();
  }

  remove(key) {
    const previous = this.memtable.get(key);
    if (previous !== undefined) this.memtableSize -= this.#entryBytes(previous);
    this.memtable.set(key, null); // tombstone
    this.memtableSize += this.#tombstoneBytes();
    this.stats.userBytes += this.#tombstoneBytes();
    this.stats.writtenBytes += this.#tombstoneBytes();
    this.log.push(this.#tombstoneBytes());
    if (this.memtableSize >= this.memtableBytes) this.#flush();
  }

  #entryBytes(value) { return value === null ? this.#tombstoneBytes() : 8 + value.length * this.valueBytes; }
  #tombstoneBytes() { return 8; }

  #toEntry(key, value) {
    if (value === null) return { key, value: null, tombstone: true, bytes: this.#tombstoneBytes() };
    return { key, value, tombstone: false, bytes: this.#entryBytes(value) };
  }

  #flush() {
    if (this.memtable.size === 0) return;
    const entries = [...this.memtable.entries()].map(([key, value]) => this.#toEntry(key, value));
    this.memtable = new Map();
    this.memtableSize = 0;
    this.levels[0].push(new Run(entries));
    this.stats.flushes++;
    this.#normalise(0);
  }

  /** Merges a level while it holds too many runs, pushing the merged run down one level. */
  #normalise(level) {
    while (this.levels[level].length > this.ratio) {
      const runs = this.levels[level];
      this.levels[level] = [];
      const merged = new Map();
      for (const run of runs) for (const entry of run.entries) merged.set(entry.key, entry);
      const entries = [...merged.values()];
      const bytes = entries.reduce((sum, e) => sum + e.bytes, 0);
      this.stats.writtenBytes += bytes;
      this.stats.compactions++;
      this.log.push(bytes);
      const atBottom = level + 1 >= this.maxLevels;
      const kept = atBottom ? entries.filter((e) => !e.tombstone) : entries; // tombstones die at the end
      if (atBottom) {
        this.levels[this.maxLevels - 1].push(new Run(kept));
        if (kept.length !== entries.length) {
          const dead = entries.filter((e) => e.tombstone).reduce((sum, e) => sum + e.bytes, 0);
          this.stats.writtenBytes += 0; // dropping data is not a write
          this.stats.reclaimed = (this.stats.reclaimed ?? 0) + dead;
        }
        return;
      }
      this.levels[level + 1].push(new Run(kept));
      this.#normalise(level + 1);
    }
  }

  /** Newest-first lookup; counts one probe per run inspected. */
  get(key) {
    this.stats.lookups++;
    if (this.memtable.has(key)) {
      this.stats.probes++;
      return this.memtable.get(key) ?? undefined;
    }
    for (let level = 0; level < this.maxLevels; level++) {
      const runs = this.levels[level];
      for (let index = runs.length - 1; index >= 0; index--) { // newest run of the level first
        this.stats.probes++;
        const hit = runs[index].get(key);
        if (hit !== undefined) return hit.tombstone ? undefined : hit.value;
      }
    }
    return undefined;
  }

  get writeAmplification() { return this.stats.userBytes === 0 ? 0 : this.stats.writtenBytes / this.stats.userBytes; }
  get probesPerLookup() { return this.stats.lookups === 0 ? 0 : this.stats.probes / this.stats.lookups; }

  /** Live bytes still present anywhere versus total bytes stored (space amplification). */
  space() {
    let live = 0;
    let total = 0;
    for (const run of this.levels.flat()) {
      total += run.bytes;
      live += run.liveBytes;
    }
    return { live, total, amplification: live === 0 ? 1 : total / live };
  }

  runCounts() { return this.levels.map((level) => level.length); }

  /** Level ordering, sorting and occupancy invariants plus the analytic compaction reconciliation. */
  checkInvariants() {
    for (let level = 0; level < this.maxLevels; level++) {
      if (this.levels[level].length > this.ratio) return `level ${level} holds ${this.levels[level].length} runs`;
      for (const run of this.levels[level]) {
        for (let i = 1; i < run.entries.length; i++) {
          if (!(run.entries[i - 1].key < run.entries[i].key)) return "run is not sorted";
        }
      }
    }
    const fromLog = this.log.reduce((sum, bytes) => sum + bytes, 0);
    if (fromLog !== this.stats.writtenBytes) return `write counter ${this.stats.writtenBytes} != log total ${fromLog}`;
    return null;
  }
}

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Deterministic mixed workload: 70% puts, 15% deletes, 15% point lookups. */
function runWorkload(tree, shadow, seed, operations, keySpace) {
  const rng = makeRng(seed);
  for (let i = 0; i < operations; i++) {
    const draw = rng(100);
    const key = rng(keySpace);
    if (draw < 70) {
      const value = `v${rng(50)}`;
      tree.put(key, value);
      shadow.set(key, value);
    } else if (draw < 85) {
      tree.remove(key);
      shadow.delete(key);
    } else {
      const expected = shadow.get(key);
      const actual = tree.get(key);
      if (actual !== expected) return { error: `lookup ${key}: ${actual} != ${expected}` };
    }
  }
  return { error: null };
}

module.exports = { LSMTree, Run, makeRng, runWorkload };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, ok, throws } = assert;

  // -- degenerate configurations ----------------------------------------------------
  const empty = new LSMTree();
  strictEqual(empty.get(1), undefined, "an empty LSM has no keys");
  strictEqual(empty.writeAmplification, 0, "no writes means no amplification");
  strictEqual(empty.checkInvariants(), null, "invariants hold for an empty LSM");
  strictEqual(empty.writeAmplification, 0, "the counter and the empty log agree");
  throws(() => new LSMTree({ ratio: 1 }), RangeError, "a ratio below 2 is invalid");
  throws(() => new LSMTree({ levels: 0 }), RangeError, "at least one level is required");
  const single = new LSMTree({ memtableBytes: 1024 });
  single.put(5, "five");
  strictEqual(single.get(5), "five", "one key is found in the memtable");
  strictEqual(single.get(6), undefined, "a missing key misses");
  strictEqual(single.checkInvariants(), null, "invariants hold with a single memtable entry");
  single.remove(5);
  strictEqual(single.get(5), undefined, "a tombstone hides the value");
  strictEqual(single.stats.flushes, 0, "nothing was flushed for such small writes");

  // -- correctness against the shadow map, plus the counter/log reconciliation ------
  for (const seed of [1, 2, 3, 4]) {
    const tree = new LSMTree({ memtableBytes: 48, ratio: 3, levels: 4 });
    const shadow = new Map();
    const { error } = runWorkload(tree, shadow, seed, 1500, 60);
    strictEqual(error, null, `the LSM must agree with the shadow map (seed ${seed})`);
    strictEqual(tree.checkInvariants(), null, `invariants must hold after the workload (seed ${seed})`);
    for (const [key, value] of shadow) strictEqual(tree.get(key), value, "the final map must agree");
    ok(tree.stats.flushes > 0 && tree.stats.compactions > 0, "the workload must have flushed and compacted");
    ok(tree.writeAmplification >= 1, "write amplification is at least one");
  }

  // -- tombstones are only reclaimed at the bottom level ----------------------------
  const reclaiming = new LSMTree({ memtableBytes: 24, ratio: 2, levels: 2 });
  for (let i = 0; i < 40; i++) reclaiming.put(i, `v${i}`);
  for (let i = 0; i < 40; i++) reclaiming.remove(i);
  reclaiming.put(999, "alive");
  strictEqual(reclaiming.get(5), undefined, "deleted keys stay deleted");
  strictEqual(reclaiming.get(999), "alive", "the surviving key is readable");
  strictEqual(reclaiming.checkInvariants(), null, "invariants hold after the deletes");
  ok(reclaiming.stats.reclaimed > 0, "the bottom level must have reclaimed tombstone space");
  const bottom = reclaiming.levels[reclaiming.maxLevels - 1].flat();
  ok(bottom.every((run) => run.entries.every((e) => !e.tombstone)),
    "no tombstone may survive in the bottom level");

  // -- the tuning trade-off: bigger ratio, less writing, more probing --------------
  const measure = (ratio) => {
    const tree = new LSMTree({ memtableBytes: 40, ratio, levels: 5 });
    const shadow = new Map();
    const { error } = runWorkload(tree, shadow, 9, 2000, 200);
    strictEqual(error, null, "the measured configurations must all be correct");
    return { ratio, writeAmp: tree.writeAmplification, probes: tree.probesPerLookup, stats: tree.stats };
  };
  const small = measure(2);
  const large = measure(8);
  ok(large.writeAmp < small.writeAmp, `a larger ratio must write less (${large.writeAmp} < ${small.writeAmp})`);
  ok(large.stats.compactions < small.stats.compactions, "a larger ratio must compact less often");
  ok(large.probes > small.probes, `a larger ratio must probe more runs (${large.probes} > ${small.probes})`);
  const bigMemtable = new LSMTree({ memtableBytes: 400, ratio: 4, levels: 5 });
  runWorkload(bigMemtable, new Map(), 9, 2000, 200);
  ok(bigMemtable.writeAmplification < measure(4).writeAmp, "a bigger memtable must write less");

  // -- space amplification and pure-update workloads --------------------------------
  const updater = new LSMTree({ memtableBytes: 32, ratio: 3, levels: 3 });
  const shadow = new Map();
  for (let i = 0; i < 300; i++) {
    const value = `v${i % 7}`;
    updater.put(i % 20, value);
    shadow.set(i % 20, value);
  }
  for (const [key, value] of shadow) strictEqual(updater.get(key), value, "overwrites must resolve newest-first");
  strictEqual(updater.checkInvariants(), null, "invariants hold for the overwrite workload");
  const space = updater.space();
  ok(space.total >= space.live && space.amplification >= 1, "total space cannot be smaller than live space");
  ok(space.amplification < 20, "20 hot keys with this tuning must not blow space up");

  // -- determinism ------------------------------------------------------------------
  const replay = () => {
    const tree = new LSMTree({ memtableBytes: 48, ratio: 3, levels: 4 });
    runWorkload(tree, new Map(), 77, 800, 80);
    return { stats: tree.stats, counts: tree.runCounts(), amp: tree.writeAmplification };
  };
  const first = replay();
  const second = replay();
  strictEqual(JSON.stringify(first), JSON.stringify(second), "the same seed must replay identically");

  console.log(`ratio2 waf=${small.writeAmp.toFixed(2)} probes=${small.probes.toFixed(2)} | ratio8 waf=${large.writeAmp.toFixed(2)} probes=${large.probes.toFixed(2)} | memtable400 waf=${bigMemtable.writeAmplification.toFixed(2)}`);
  console.log("PASS");
}
