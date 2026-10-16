/** Task 657 - Implement distributed join shuffle simulation in map-reduce. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Simulates the two distributed join strategies a MapReduce/Spark planner chooses between:
 *
 *   Reduce-side join (shuffle join): the map phase tags every record of both inputs and emits
 *   it to exactly one of P reduce partitions determined by `hash(joinKey) % P`. Both sides use
 *   the same hash, so every key lands in the same partition on both sides — that *partition
 *   locality invariant* is what makes the per-partition join equivalent to a single-node join.
 *   Each reducer then builds a hash table from its slice of the small side and probes it with
 *   its slice of the big side. Every record is shuffled exactly once.
 *
 *   Map-side (broadcast) join: when one input is small, replicate it to all P partitions and
 *   shuffle only the big side. Correct only if the whole small side fits in a partition's
 *   memory — a simplified assumption of this sketch, together with "one round, no combiner, no
 *   failure/re-execution, no real network".
 *
 * Verified: the join output equals a brute-force nested-loop join (inner and left outer) for
 * several partition counts, the partition counts sum back to the input sizes (no record lost or
 * duplicated by the shuffle), the concatenation of the per-partition results equals the global
 * result, broadcast and shuffle paths agree, and the skew report really exposes a hot key.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Deterministic FNV-1a over the string form of a key. */
function hashKey(key) {
  const text = typeof key === "string" ? key : String(key);
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

const keyOf = (row, joinKey) => (row[joinKey] === undefined ? null : row[joinKey]);

function validatePartitions(partitions) {
  if (!Number.isInteger(partitions) || partitions < 1) {
    throw new RangeError(`partition count must be a positive integer, got ${partitions}`);
  }
}

/** Map phase: tag each record and bucket it by hash(joinKey) % P (the shuffle write). */
function mapAndShuffle(rows, tag, joinKey, partitions) {
  validatePartitions(partitions);
  const buckets = Array.from({ length: partitions }, () => []);
  for (const row of rows) buckets[hashKey(keyOf(row, joinKey)) % partitions].push({ tag, row });
  return buckets;
}

/** Reduce phase for one partition: hash the right side, probe with the left side. */
function joinPartition(leftRows, rightRows, joinKey, mode) {
  const rightByKey = new Map();
  for (const row of rightRows) {
    const key = keyOf(row, joinKey);
    if (key === null) continue;
    if (!rightByKey.has(key)) rightByKey.set(key, []);
    rightByKey.get(key).push(row);
  }
  const out = [];
  for (const left of leftRows) {
    const key = keyOf(left, joinKey);
    const matches = key === null ? undefined : rightByKey.get(key);
    if (matches === undefined) {
      if (mode === "left") out.push({ left, right: null });
      continue;
    }
    for (const right of matches) out.push({ left, right });
  }
  return out;
}

function emptyStats(partitions) {
  return {
    partitions,
    perPartition: Array.from({ length: partitions }, (_, index) =>
      ({ index, left: 0, right: 0, matches: 0 })),
    shuffledRecords: 0,
    broadcastRecords: 0,
    matches: 0,
  };
}

/** Reduce-side (shuffle) join. `mode` is "inner" or "left". */
function shuffleJoin(left, right, joinKey, options = {}) {
  const partitions = options.partitions ?? 4;
  const mode = options.mode ?? "inner";
  validatePartitions(partitions);
  const leftBuckets = mapAndShuffle(left, "left", joinKey, partitions);
  const rightBuckets = mapAndShuffle(right, "right", joinKey, partitions);
  const stats = emptyStats(partitions);
  stats.shuffledRecords = left.length + right.length;
  const rows = [];
  for (let p = 0; p < partitions; p++) {
    const leftRows = leftBuckets[p].map((entry) => entry.row);
    const rightRows = rightBuckets[p].map((entry) => entry.row);
    const produced = joinPartition(leftRows, rightRows, joinKey, mode);
    stats.perPartition[p].left = leftRows.length;
    stats.perPartition[p].right = rightRows.length;
    stats.perPartition[p].matches = produced.length;
    stats.matches += produced.length;
    rows.push(...produced);
  }
  return { rows, stats, leftBuckets, rightBuckets };
}

/**
 * Map-side (broadcast) join: the build side is replicated to all P partitions, only the probe
 * side is shuffled. Output always keeps the parameter roles: `left` comes from the first
 * argument, `right` from the second, whatever the build side is.
 */
function broadcastJoin(left, right, joinKey, options = {}) {
  const partitions = options.partitions ?? 4;
  const mode = options.mode ?? "inner";
  const buildSide = options.buildSide ?? "right";
  validatePartitions(partitions);
  if (buildSide !== "right" && buildSide !== "left") {
    throw new RangeError(`build side must be "left" or "right", got ${buildSide}`);
  }
  const buildIsRight = buildSide === "right";
  const buildRows = buildIsRight ? right : left;
  const probeRows = buildIsRight ? left : right;
  const emit = (probe, match) => (buildIsRight ? { left: probe, right: match } : { left: match, right: probe });
  const stats = emptyStats(partitions);
  stats.broadcastRecords = buildRows.length * partitions;
  stats.shuffledRecords = probeRows.length;
  const buildByKey = new Map();
  for (const row of buildRows) {
    const key = keyOf(row, joinKey);
    if (key === null) continue;
    if (!buildByKey.has(key)) buildByKey.set(key, []);
    buildByKey.get(key).push(row);
  }
  const buckets = mapAndShuffle(probeRows, "probe", joinKey, partitions);
  const rows = [];
  for (let p = 0; p < partitions; p++) {
    let produced = 0;
    for (const tagged of buckets[p]) {
      const key = keyOf(tagged.row, joinKey);
      const matches = key === null ? undefined : buildByKey.get(key);
      if (matches === undefined) {
        // Outer padding applies to whichever side is being probed.
        if (mode === "left" && buildIsRight) { rows.push({ left: tagged.row, right: null }); produced++; }
        if (mode === "right" && !buildIsRight) { rows.push({ left: null, right: tagged.row }); produced++; }
        continue;
      }
      for (const match of matches) { rows.push(emit(tagged.row, match)); produced++; }
    }
    const shuffledCount = buckets[p].length;
    stats.perPartition[p].left = buildIsRight ? shuffledCount : buildRows.length;
    stats.perPartition[p].right = buildIsRight ? buildRows.length : shuffledCount;
    stats.perPartition[p].matches = produced;
  }
  stats.matches = rows.length;
  return { rows, stats };
}

/** Brute-force reference: nested loops with SQL null-key semantics (inner/left/right). */
function nestedLoopJoin(left, right, joinKey, mode = "inner") {
  const out = [];
  for (const l of left) {
    let matched = false;
    if (keyOf(l, joinKey) !== null) {
      for (const r of right) {
        if (keyOf(r, joinKey) === keyOf(l, joinKey)) {
          out.push({ left: l, right: r });
          matched = true;
        }
      }
    }
    if (!matched && mode === "left") out.push({ left: l, right: null });
  }
  if (mode === "right") {
    for (const r of right) {
      const key = keyOf(r, joinKey);
      const matched = key !== null && left.some((l) => keyOf(l, joinKey) === key);
      if (!matched) out.push({ left: null, right: r });
    }
  }
  return out;
}

const canonical = (rows) => rows.map((r) => JSON.stringify(r)).sort();

/** Partition sizes of the shuffled left side, for skew reporting. */
function skewReport(stats) {
  const sizes = stats.perPartition.map((p) => p.left + p.right);
  const total = sizes.reduce((a, b) => a + b, 0);
  const average = total / sizes.length;
  return { max: Math.max(...sizes), average, ratio: average === 0 ? 1 : Math.max(...sizes) / average };
}

module.exports = {
  hashKey, mapAndShuffle, shuffleJoin, broadcastJoin, nestedLoopJoin, joinPartition,
  skewReport, canonical, makeRng, validatePartitions,
};

if (require.main === module) {
  // -- hand-checked case: 3 left rows x 2 right rows sharing a key ----------------
  const smallLeft = [{ k: "a", v: 1 }, { k: "a", v: 2 }, { k: "b", v: 3 }];
  const smallRight = [{ k: "a", w: 9 }, { k: "z", w: 8 }];
  const manual = shuffleJoin(smallLeft, smallRight, "k", { partitions: 2 });
  strictEqual(manual.rows.length, 2, "two left rows share key a, which has one right row");
  deepStrictEqual(canonical(manual.rows), canonical(nestedLoopJoin(smallLeft, smallRight, "k")));
  strictEqual(manual.stats.matches, 2);
  strictEqual(shuffleJoin(smallLeft, [{ k: "a", w: 9 }, { k: "a", w: 10 }], "k").rows.length, 4,
    "duplicate keys on both sides become a cross product");

  // -- shuffle accounting: nothing lost, nothing duplicated ---------------------
  const rng = makeRng(20251016);
  const leftRows = Array.from({ length: 240 }, (_, i) =>
    ({ id: i, k: i % 5 === 0 ? "hot" : `k${rng(60)}`, payload: rng(1000) }));
  const rightRows = Array.from({ length: 90 }, (_, i) =>
    ({ rid: i, k: i % 3 === 0 ? "hot" : `k${rng(60)}`, attrs: rng(1000) }));
  for (const partitions of [1, 3, 7]) {
    const join = shuffleJoin(leftRows, rightRows, "k", { partitions });
    const reference = nestedLoopJoin(leftRows, rightRows, "k");
    deepStrictEqual(canonical(join.rows), canonical(reference), `inner join must match nested loop (P=${partitions})`);
    const summedLeft = join.stats.perPartition.reduce((s, p) => s + p.left, 0);
    const summedRight = join.stats.perPartition.reduce((s, p) => s + p.right, 0);
    strictEqual(summedLeft, leftRows.length, "every left record reaches exactly one partition");
    strictEqual(summedRight, rightRows.length, "every right record reaches exactly one partition");
    strictEqual(join.stats.shuffledRecords, leftRows.length + rightRows.length);
    // partition locality: each key is present in exactly one partition, on both sides
    const placement = new Map();
    for (let p = 0; p < partitions; p++) {
      for (const entry of join.leftBuckets[p]) {
        const key = entry.row.k;
        strictEqual(placement.get(key) ?? p, p, `key ${key} must be bucketed deterministically`);
        placement.set(key, p);
      }
      for (const entry of join.rightBuckets[p]) {
        strictEqual(hashKey(entry.row.k) % partitions, p, "each record sits in hash(key) % P");
      }
    }
    // the concatenation of per-partition results is the global result
    let perPartitionTotal = 0;
    for (const partition of join.stats.perPartition) perPartitionTotal += partition.matches;
    strictEqual(perPartitionTotal, join.rows.length, "per-partition matches must sum to the output size");
    // redistribute and confirm the answer does not depend on the partition count
    const other = shuffleJoin(leftRows, rightRows, "k", { partitions: partitions + 2 });
    deepStrictEqual(canonical(other.rows), canonical(reference));
  }

  // -- left outer join, including unmatched and null keys -----------------------
  const lonely = [{ k: "none", v: 1 }, { v: 2 }, { k: "hot", v: 3 }];
  const innerLonely = shuffleJoin(lonely, rightRows, "k", { partitions: 3 });
  const leftLonely = shuffleJoin(lonely, rightRows, "k", { partitions: 3, mode: "left" });
  deepStrictEqual(canonical(innerLonely.rows), canonical(nestedLoopJoin(lonely, rightRows, "k")));
  deepStrictEqual(canonical(leftLonely.rows), canonical(nestedLoopJoin(lonely, rightRows, "k", "left")));
  const padded = leftLonely.rows.filter((r) => r.right === null);
  strictEqual(padded.length, 2, "the unmatched key and the null key are both padded");
  ok(innerLonely.rows.every((r) => r.right !== null), "an inner join never emits a null side");
  strictEqual(shuffleJoin(lonely, rightRows, "k", { mode: "left" }).rows.some((r) => r.left.v === 2), true,
    "a row without the join key still survives a left join");

  // -- empty inputs -------------------------------------------------------------
  strictEqual(shuffleJoin([], rightRows, "k").rows.length, 0, "an empty left side yields nothing");
  strictEqual(shuffleJoin(leftRows, [], "k").rows.length, 0, "an empty right side yields nothing");
  strictEqual(shuffleJoin(leftRows, [], "k", { mode: "left" }).rows.length, leftRows.length,
    "a left join against an empty right side pads every row");
  strictEqual(shuffleJoin([], [], "k").stats.shuffledRecords, 0);
  throws(() => shuffleJoin(leftRows, rightRows, "k", { partitions: 0 }), RangeError,
    "a zero partition count must be rejected");

  // -- map-side broadcast path agrees with the shuffle path ---------------------
  const broadcast = broadcastJoin(leftRows, rightRows, "k", { partitions: 3 });
  deepStrictEqual(canonical(broadcast.rows), canonical(nestedLoopJoin(leftRows, rightRows, "k")),
    "the broadcast join must produce the same answer");
  strictEqual(broadcast.stats.broadcastRecords, rightRows.length * 3,
    "the build side is replicated once per partition");
  strictEqual(broadcast.stats.shuffledRecords, leftRows.length, "only the probe side is shuffled");
  const broadcastLeft = broadcastJoin(leftRows, rightRows, "k", { partitions: 4, mode: "left" });
  deepStrictEqual(canonical(broadcastLeft.rows), canonical(nestedLoopJoin(leftRows, rightRows, "k", "left")));
  const broadcastOnLeft = broadcastJoin(leftRows, rightRows, "k", { partitions: 5, buildSide: "left" });
  deepStrictEqual(canonical(broadcastOnLeft.rows), canonical(nestedLoopJoin(leftRows, rightRows, "k")),
    "building on the left side works too");
  strictEqual(broadcastOnLeft.stats.broadcastRecords, leftRows.length * 5);
  const broadcastRightOuter = broadcastJoin(leftRows, rightRows, "k",
    { partitions: 5, buildSide: "left", mode: "right" });
  deepStrictEqual(canonical(broadcastRightOuter.rows),
    canonical(nestedLoopJoin(leftRows, rightRows, "k", "right")),
    "a right outer broadcast join pads the probed side");
  throws(() => broadcastJoin(leftRows, rightRows, "k", { buildSide: "middle" }), RangeError,
    "an unknown build side must be rejected");

  // -- skew visibility ----------------------------------------------------------
  const hot = Array.from({ length: 280 }, (_, i) => ({ k: i < 200 ? "hot" : `k${i}`, v: i }));
  const hotJoin = shuffleJoin(hot, [{ k: "hot", w: 1 }], "k", { partitions: 8 });
  const hotSkew = skewReport(hotJoin.stats);
  ok(hotSkew.ratio > 2, `a hot key must show up as skew, ratio ${hotSkew.ratio.toFixed(2)}`);
  strictEqual(hotJoin.stats.matches, 200, "exactly the hot rows match");
  const uniform = Array.from({ length: 280 }, (_, i) => ({ k: `u${i}`, v: i }));
  const uniformSkew = skewReport(shuffleJoin(uniform, [{ k: "u0" }], "k", { partitions: 8 }).stats);
  ok(uniformSkew.ratio < 2, `uniform keys must stay balanced, ratio ${uniformSkew.ratio.toFixed(2)}`);

  // -- determinism --------------------------------------------------------------
  const again = shuffleJoin(leftRows, rightRows, "k", { partitions: 7 });
  deepStrictEqual(canonical(again.rows), canonical(shuffleJoin(leftRows, rightRows, "k", { partitions: 7 }).rows));
  deepStrictEqual(again.stats, shuffleJoin(leftRows, rightRows, "k", { partitions: 7 }).stats);
  strictEqual(typeof hashKey(12345), "number");
  ok(hashKey("a") !== hashKey("b"), "distinct keys should hash apart");
  const distribution = new Array(4).fill(0);
  for (let i = 0; i < 400; i++) distribution[hashKey(`key${i}`) % 4]++;
  ok(Math.min(...distribution) > 40, "hashing must spread keys over the partitions");

  console.log(`shuffle: P=7 left=${leftRows.length} right=${rightRows.length} matches=${again.stats.matches} ` +
    `hotSkew=${hotSkew.ratio.toFixed(2)} uniformSkew=${uniformSkew.ratio.toFixed(2)} ` +
    `broadcastReplicated=${broadcast.stats.broadcastRecords}`);
  console.log("PASS");
}
