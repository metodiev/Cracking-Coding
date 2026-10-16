/** Task 992 - Implement reproducible random-seed protocol for benchmarking fairness.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Defines how every measurement in a benchmark run gets its random seed, and how the run order is
 * arranged so no implementation is ever systematically measured first. All seeds for one suite are
 * derived from a single master seed by hashing the tuple (master, benchmark, implementation,
 * repetition), so seeds are reproducible, independent per axis and never shared between two
 * implementations. The schedule is a rotated (Latin-square) traversal of the implementation list:
 * over a full number of rounds every implementation occupies every slot equally often.
 *
 * What is simplified: the suite is a schedule of descriptors rather than a real process pool, and
 * "measuring" means deriving the seed the implementation would use.
 * Verified: seed distinctness against a brute-force pairwise oracle, single-axis sensitivity of the
 * derivation, exact slot balance, tamper detection during audit, master-seed separation, and the
 * degenerate one-implementation/one-round and empty suits.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the file's only randomness source, so fixtures are byte-reproducible. */
function splitmix64(seed) {
  let s = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    s = BigInt.asUintN(64, s + 0x9e3779b97f4a7c15n);
    let z = BigInt.asUintN(64, (s ^ (s >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}

/** Finalizer that spreads one bit of input over the whole word, so near-identical keys decorrelate. */
function mix32(x) {
  let v = x >>> 0;
  v = Math.imul(v ^ (v >>> 16), 0x45d9f3b);
  v = Math.imul(v ^ (v >>> 16), 0x45d9f3b);
  return (v ^ (v >>> 16)) >>> 0;
}

/** The protocol: one seed per (master, benchmark, implementation, repetition) tuple. */
const keyedSeed = (master, bench, impl, rep) => mix32(fnv1a(`${master}:${bench}:${impl}:${rep}`));

/**
 * Latin-square schedule: round `r` shifts the implementation list by `r`, so position `k` is taken
 * by a different implementation each round and, over `impls.length` rounds, by every one of them.
 */
function schedule(impls, rounds) {
  const plan = [];
  for (let r = 0; r < rounds; r += 1) {
    for (let slot = 0; slot < impls.length; slot += 1) {
      plan.push({ round: r, slot, impl: impls[(slot + r) % impls.length] });
    }
  }
  return plan;
}

/** Enumerate the whole seed matrix for a suite; the matrix is a pure function of its arguments. */
function seedMatrix(master, benches, impls, rounds) {
  const rows = [];
  for (const bench of benches) for (const impl of impls) {
    for (let rep = 0; rep < rounds; rep += 1) {
      rows.push({ bench, impl, rep, seed: keyedSeed(master, bench, impl, rep) });
    }
  }
  return { master, rounds, rows, digest: fnv1a(JSON.stringify(rows)) };
}

/** Independent oracle: O(n^2) pairwise comparison, no Set, reporting the first duplicate pair. */
function firstSeedCollision(rows) {
  for (let i = 0; i < rows.length; i += 1) {
    for (let j = i + 1; j < rows.length; j += 1) {
      if (rows[i].seed === rows[j].seed) return [rows[i], rows[j]];
    }
  }
  return null;
}

/** Audit a recording: every recorded seed must equal the one the protocol derives for its key. */
function audit(recording, master) {
  const bad = recording.filter((r) => keyedSeed(master, r.bench, r.impl, r.rep) !== r.seed);
  return { ok: bad.length === 0, bad, checked: recording.length };
}

/** Record a suite by deriving and storing the seeds in schedule order. */
function recordRun(master, benches, impls, rounds) {
  return schedule(impls, rounds).flatMap((slot) => benches.map((bench) => ({ bench, impl: slot.impl,
    rep: slot.round, slot: slot.slot, seed: keyedSeed(master, bench, slot.impl, slot.round) })));
}

const BENCHES = ["sort", "search", "graph"];
const IMPLS = ["js-loop", "js-merge", "js-fast", "js-ref"];

function main() {
  const matrix = seedMatrix(20240992, BENCHES, IMPLS, 4);
  // 1. Happy path: the matrix has one row per tuple, all seeds are 32-bit and distinct.
  assert.strictEqual(matrix.rows.length, BENCHES.length * IMPLS.length * 4, "one seed per tuple");
  assert.ok(matrix.rows.every((r) => Number.isInteger(r.seed) && r.seed >= 0 && r.seed <= 0xffffffff),
    "seeds are unsigned 32-bit values");
  // 2. Oracle: brute-force pairwise distinctness, plus the Set check, must agree that none collide.
  assert.deepStrictEqual([firstSeedCollision(matrix.rows), new Set(matrix.rows.map((r) => r.seed)).size],
    [null, matrix.rows.length], "no two tuples share a seed");
  // 3. Single-axis sensitivity: changing exactly one component changes the seed, on every axis.
  const base = keyedSeed(1, "sort", "js-loop", 0);
  const changed = [keyedSeed(2, "sort", "js-loop", 0), keyedSeed(1, "search", "js-loop", 0),
    keyedSeed(1, "sort", "js-merge", 0), keyedSeed(1, "sort", "js-loop", 1)];
  assert.strictEqual(changed.every((s) => s !== base), true, "each axis perturbs the seed");
  assert.strictEqual(new Set(changed).size, changed.length, "the four perturbations differ from each other");
  // 4. Fairness: over a whole number of rounds every implementation occupies every slot equally often.
  const balanced = schedule(IMPLS, IMPLS.length);
  const counts = new Map();
  for (const item of balanced) {
    const key = `${item.impl}@${item.slot}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  assert.strictEqual(counts.size, IMPLS.length * IMPLS.length, "every implementation visited every slot");
  assert.strictEqual(balanced.filter((x) => x.slot === 0).length, IMPLS.length, "slot 0 used once per round");
  assert.strictEqual(Array.from(counts.values()).every((c) => c === 1), true, "each (impl, slot) pair exactly once");
  const twoRounds = schedule(IMPLS, 2 * IMPLS.length);
  assert.strictEqual(Array.from(twoRounds.reduce((m, x) => m.set(`${x.impl}@${x.slot}`,
    (m.get(`${x.impl}@${x.slot}`) || 0) + 1), new Map()).values()).every((c) => c === 2), true,
    "two full rotations stay balanced");
  // 5. Boundary: a single implementation, a single round and an empty implementation list.
  assert.deepStrictEqual([schedule(["only"], 1), schedule(IMPLS, 1).length, schedule([], 3).length],
    [[{ round: 0, slot: 0, impl: "only" }], IMPLS.length, 0], "degenerate schedules");
  assert.strictEqual(seedMatrix(1, [], IMPLS, 2).rows.length, 0, "no benchmarks means an empty matrix");
  // 6. Tamper detection: an honest recording audits clean, a single edited seed is named.
  const recording = recordRun(20240992, BENCHES, IMPLS, IMPLS.length);
  assert.deepStrictEqual([audit(recording, 20240992).ok, audit(recording, 20240992).checked],
    [true, recording.length], "honest recording audits clean");
  const tampered = recording.map((r, i) => (i === 5 ? { ...r, seed: (r.seed ^ 1) >>> 0 } : r));
  const report = audit(tampered, 20240992);
  assert.deepStrictEqual([report.ok, report.bad.map((r) => [r.bench, r.impl, r.rep])],
    [false, [[recording[5].bench, recording[5].impl, recording[5].rep]]], "only the edited row fails the audit");
  assert.strictEqual(audit(recording, 20240993).ok, false, "a different master seed invalidates the recording");
  // 7. Repetitions use fresh entropy: a repeated (bench, impl) pair never reuses a seed.
  const repeated = recording.filter((r) => r.bench === BENCHES[0] && r.impl === IMPLS[0]).map((r) => r.seed);
  assert.strictEqual(new Set(repeated).size, repeated.length, "each repetition gets its own seed");
  // 8. Determinism: the same master reproduces the matrix byte for byte, and two masters never overlap.
  assert.strictEqual(seedMatrix(20240992, BENCHES, IMPLS, 4).digest, matrix.digest, "matrix replays byte for byte");
  const other = seedMatrix(20240993, BENCHES, IMPLS, 4);
  const mine = new Set(matrix.rows.map((r) => r.seed));
  assert.strictEqual(other.rows.some((r) => mine.has(r.seed)), false, "seed spaces of two masters are disjoint");
  console.log(`seeds=${matrix.rows.length} rounds=${IMPLS.length} slots=${IMPLS.length} digest=${matrix.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, mix32, keyedSeed, schedule, seedMatrix, firstSeedCollision, audit,
  recordRun, BENCHES, IMPLS };
