/** Task 916 - Implement large-scale streaming joins with approximate state and bounded memory.
 *  (Tier: Master, CTCI Beyond — Stream Processing & Approximate State)
 *
 * A streaming inner join whose build-side state is capped: a Space-Saving table keeps at most `capacity`
 * keys; on overflow the least frequent key is evicted and the newcomer inherits its count, so every
 * retained estimate is an over-approximation with error at most N/capacity and — the property the checks
 * lean on — every key whose true frequency exceeds N/capacity survives. Probe records are answered from
 * that state, so the join is *incomplete but never wrong*: a row is emitted only when a stored value
 * really has that key, which is exactly the soundness/some-recall trade-off a bounded-memory operator has
 * to make (the alternative, unbounded state, is what the memory check rules out).
 *
 * Checks: state never exceeds the cap, exact agreement with the full join when the cap is large enough,
 * soundness (no false positives) at every cap, the Space-Saving survival guarantee against exact counts,
 * recall that improves with the cap, the memory saved against an unbounded join, degenerate streams
 * (empty sides, cap one, a single repeated key) and reproducibility.
 */
"use strict";

const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { return this.next() % bound; }
  float() { return this.next() / 4294967296; }
}

/** Space-Saving summary: bounded keys, counts never below the truth, error at most N/capacity. */
class SpaceSaving {
  constructor(capacity) {
    assert.ok(Number.isInteger(capacity) && capacity > 0, `capacity must be a positive integer, got ${capacity}`);
    this.capacity = capacity;
    this.entries = new Map();
    this.seen = 0;
    this.evictions = 0;
    this.maxSize = 0;
  }
  insert(key, value) {
    this.seen += 1;
    const existing = this.entries.get(key);
    if (existing) {
      existing.count += 1;
      existing.values.push(value);
    } else if (this.entries.size < this.capacity) {
      this.entries.set(key, { count: 1, values: [value] });
    } else {
      let victim = null;
      for (const [candidate, entry] of this.entries) {
        if (victim === null || entry.count < this.entries.get(victim).count) victim = candidate;
      }
      const floor = this.entries.get(victim).count;
      this.entries.delete(victim);
      this.evictions += 1;
      this.entries.set(key, { count: floor + 1, values: [value] });
    }
    this.maxSize = Math.max(this.maxSize, this.entries.size);
    return this;
  }
  lookup(key) { return this.entries.get(key) ?? null; }
  estimate(key) { return this.entries.get(key)?.count ?? 0; }
}

/** Streamed inner join against bounded build state; returns the emitted rows and the state. */
function boundedJoin(buildStream, probeStream, capacity) {
  const state = new SpaceSaving(capacity);
  const emitted = [];
  for (const record of buildStream) state.insert(record.key, record.value);
  for (const probe of probeStream) {
    const entry = state.lookup(probe.key);
    if (!entry) continue;
    for (const value of entry.values) emitted.push({ key: probe.key, probe: probe.id, build: value });
  }
  return { emitted, state };
}

/** Exact join over the same streams (the oracle). */
function exactJoin(buildStream, probeStream) {
  const rows = [];
  for (const probe of probeStream) {
    for (const record of buildStream) {
      if (record.key === probe.key) rows.push({ key: probe.key, probe: probe.id, build: record.value });
    }
  }
  return rows;
}

const countsOf = (stream) => {
  const counts = new Map();
  for (const record of stream) counts.set(record.key, (counts.get(record.key) ?? 0) + 1);
  return counts;
};

const rowKey = (row) => `${row.key}|${row.probe}|${row.build}`;

/** Zipf-ish key popularity so a bounded state actually has to choose what to keep. */
function makeStreams(keySpace, buildCount, probeCount, skew, seed) {
  const prng = new Prng(seed);
  const drawKey = () => `key-${Math.min(keySpace - 1, Math.floor(Math.pow(prng.float(), skew) * keySpace))}`;
  const build = Array.from({ length: buildCount }, (_, i) => ({ key: drawKey(), value: i }));
  const probe = Array.from({ length: probeCount }, (_, i) => ({ key: drawKey(), id: i }));
  return { build, probe };
}

function main() {
  const keySpace = 400;
  const { build, probe } = makeStreams(keySpace, 6000, 3000, 2.0, 0x5eed0916);
  const counts = countsOf(build);
  const truth = exactJoin(build, probe);

  // 1. bounded state — the table never grows past its cap, whatever the stream does.
  {
    for (const capacity of [1, 8, 64, keySpace, keySpace * 2]) {
      const { state } = boundedJoin(build, probe, capacity);
      assert.ok(state.entries.size <= capacity, `state ${state.entries.size} exceeds cap ${capacity}`);
      assert.ok(state.maxSize <= capacity, "the peak size is inside the cap too");
      assert.strictEqual(state.seen, build.length, "every build record is accounted for");
    }
  }

  // 2. exactness — with a cap that covers every distinct key the join is the full join.
  {
    const { emitted, state } = boundedJoin(build, probe, keySpace);
    assert.strictEqual(state.evictions, 0, "no key was evicted");
    assert.strictEqual(state.entries.size, counts.size, "every distinct key is held");
    const expected = new Set(truth.map(rowKey));
    assert.strictEqual(emitted.length, truth.length, "row count matches the exact join");
    assert.ok(emitted.every((row) => expected.has(rowKey(row))), "every emitted row is a true join row");
    assert.deepStrictEqual([...new Set(emitted.map((row) => rowKey(row)))].length, new Set(truth.map(rowKey)).size,
      "and the distinct rows match as well");
  }

  // 3. soundness — at every cap the join is incomplete at worst, never wrong.
  {
    const expected = new Set(truth.map(rowKey));
    for (const capacity of [1, 2, 16, 100, 250]) {
      const { emitted } = boundedJoin(build, probe, capacity);
      assert.ok(emitted.every((row) => expected.has(rowKey(row))),
        `cap ${capacity} emitted a row that is not in the exact join`);
      assert.ok(emitted.length <= truth.length, "a bounded join cannot invent rows");
    }
  }

  // 4. the Space-Saving guarantee — survivors are exactly the frequent keys, with bounded error.
  {
    const capacity = 120;
    const { state } = boundedJoin(build, probe, capacity);
    const threshold = build.length / capacity;
    let survivors = 0;
    for (const [key, count] of counts) {
      const entry = state.lookup(key);
      if (count > threshold) {
        assert.ok(entry !== null, `key ${key} with count ${count} must survive (threshold ${threshold})`);
        survivors += 1;
      }
      if (entry !== null) {
        assert.ok(entry.count >= count, "the estimate never drops below the truth");
        assert.ok(entry.count <= count + threshold, `estimate error for ${key} exceeds N/capacity`);
      }
    }
    assert.ok(survivors >= 5, `only ${survivors} keys passed the survival threshold`);
    assert.ok(state.entries.size <= capacity);
    assert.ok(state.evictions > 0, "the cap really did bind");
  }

  // 5. the memory/recall trade-off — a bigger cap is never worse and quickly approaches exact.
  {
    const recall = (capacity) => {
      const { emitted } = boundedJoin(build, probe, capacity);
      return emitted.length / truth.length;
    };
    const small = recall(4);
    const medium = recall(50);
    const large = recall(keySpace);
    assert.ok(small < medium, `recall ${small.toFixed(3)} -> ${medium.toFixed(3)} must improve with the cap`);
    assert.ok(medium <= large + 1e-9, "recall is bounded above by the exact join");
    assert.strictEqual(large, 1, "a large enough cap recovers the whole join");
    const { state } = boundedJoin(build, probe, 50);
    assert.ok(state.entries.size <= 50 && counts.size > 300,
      `bounded state ${state.entries.size} vs ${counts.size} distinct keys in the stream`);
    const threshold = build.length / 50;
    const heavy = probe.filter((record) => counts.get(record.key) > threshold);
    const answered = new Set(boundedJoin(build, probe, 50).emitted.map((row) => row.probe));
    assert.ok(heavy.length > 0, "the stream has keys above the survival threshold");
    assert.ok(heavy.every((record) => answered.has(record.id)),
      "every probe on a key above N/capacity is answered");
  }

  // 6. boundary — empty inputs, one key, cap one, and a probe that never matches.
  {
    const { emitted, state } = boundedJoin([], [], 10);
    assert.deepStrictEqual(emitted, []);
    assert.strictEqual(state.entries.size, 0);
    assert.strictEqual(state.seen, 0);
    const solo = Array.from({ length: 30 }, (_, i) => ({ key: "same", value: i }));
    const soloProbe = [{ key: "same", id: 1 }, { key: "missing", id: 2 }];
    const single = boundedJoin(solo, soloProbe, 1);
    assert.strictEqual(single.state.entries.size, 1, "one key fits in a cap of one");
    assert.strictEqual(single.emitted.length, 30, "the repeated key is fully joined");
    assert.ok(single.emitted.every((row) => row.key === "same"));
    assert.deepStrictEqual(boundedJoin(solo, soloProbe, 1).emitted.length, exactJoin(solo, soloProbe).length);
    assert.strictEqual(boundedJoin([], [{ key: "k", id: 0 }], 5).emitted.length, 0, "no build side, no rows");
    assert.strictEqual(boundedJoin([{ key: "k", value: 1 }], [], 5).emitted.length, 0, "no probe side, no rows");
    assert.throws(() => new SpaceSaving(0), /capacity must be a positive integer/);
  }

  // 7. the counter is what makes this sound — evictions lose rows, they never invent them.
  {
    const capacity = 20;
    const { emitted, state } = boundedJoin(build, probe, capacity);
    assert.ok(state.evictions > 0, "keys were evicted");
    assert.ok(emitted.length < truth.length, "eviction costs recall");
    const evicted = [...counts.keys()].filter((key) => state.lookup(key) === null);
    assert.ok(evicted.length > 0, "some keys are gone");
    const evictedRows = new Set(truth.filter((row) => evicted.includes(row.key)).map(rowKey));
    assert.ok(emitted.every((row) => !evictedRows.has(rowKey(row))), "no row is emitted for an evicted key");
    assert.ok(emitted.every((row) => counts.has(row.key)), "every emitted key exists in the build stream");
  }

  // 8. determinism — streams, state and output all replay exactly.
  {
    const again = makeStreams(keySpace, 6000, 3000, 2.0, 0x5eed0916);
    assert.deepStrictEqual(again.build, build);
    assert.deepStrictEqual(again.probe, probe);
    assert.deepStrictEqual(boundedJoin(build, probe, 50).emitted, boundedJoin(build, probe, 50).emitted);
    assert.deepStrictEqual([...boundedJoin(build, probe, 50).state.entries.keys()],
      [...boundedJoin(build, probe, 50).state.entries.keys()]);
    assert.notDeepStrictEqual(makeStreams(keySpace, 6000, 3000, 2.0, 0x5eed0917).build, build);
  }

  const bounded = boundedJoin(build, probe, 120);
  console.log(`streaming-join: ${build.length} build / ${probe.length} probe records over ${counts.size} keys; ` +
    `cap 120 holds ${bounded.state.entries.size} keys with ${bounded.state.evictions} evictions, emitting ` +
    `${bounded.emitted.length}/${truth.length} rows that are all true matches (cap ${keySpace} is exact)`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = { Prng, SpaceSaving, boundedJoin, exactJoin, countsOf, makeStreams, rowKey };
