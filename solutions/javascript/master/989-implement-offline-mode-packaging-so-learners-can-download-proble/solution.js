/** Task 989 - Implement offline mode packaging so learners can download problem sets + tests.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Bundles a selection of problems into self-contained download chunks bounded by a byte budget. A
 * problem and its tests are atomic -- they never straddle two bundles -- while shared assets (the
 * runner, the judge harness) are copied into every bundle that needs them and are counted once per
 * bundle in the manifest. A problem larger than the budget is reported as unpackable instead of
 * silently dropped.
 *
 * What is simplified: sizes are bytes, a "download" is an in-memory manifest, and packing uses
 * first-fit-decreasing rather than an optimal bin-packing search (an exact search is used below as an
 * oracle on a tiny instance).
 * Verified: full coverage, per-bundle capacity, shared-asset presence, per-bundle dedup, rejection of
 * oversized problems, tight-fit and empty selections, and greedy-vs-exact packing on a small instance.
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

/** One problem: its own files plus the shared assets its tests need, all measured in bytes. */
const problemSize = (p) => p.size + (p.tests || []).reduce((a, t) => a + t.size, 0);
const payloadOf = (p) => [{ path: `problems/${p.id}.md`, size: p.size },
  ...(p.tests || []).map((t) => ({ path: `tests/${p.id}/${t.name}`, size: t.size }))];

/**
 * First-fit-decreasing over a deterministic ordering: problems are sorted by descending payload and
 * then by id, so the produced bundles (and their ids) are a pure function of the input set.
 */
function packBundles(problems, limitBytes, shared = []) {
  const unpackable = problems.filter((p) => problemSize(p) > limitBytes).map((p) => p.id).sort();
  const packable = problems.filter((p) => problemSize(p) <= limitBytes)
    .slice().sort((a, b) => problemSize(b) - problemSize(a) || (a.id < b.id ? -1 : 1));
  const bundles = [];
  for (const p of packable) {
    /** Assets the bundle already holds are not charged a second time. */
    const charge = (b) => problemSize(p) + (p.uses || []).reduce((a, name) => {
      const s = shared.find((x) => x.name === name);
      return a + (s && !b.assets.some((x) => x.path === s.path) ? s.size : 0);
    }, 0);
    let target = bundles.find((b) => b.used + charge(b) <= limitBytes);
    if (!target) {
      target = { id: `bundle-${String(bundles.length + 1).padStart(2, "0")}`, assets: [], problems: [], used: 0 };
      bundles.push(target);
    }
    for (const name of p.uses || []) {
      const s = shared.find((x) => x.name === name);
      if (s && !target.assets.some((x) => x.path === s.path)) target.assets.push({ path: s.path, size: s.size });
    }
    for (const asset of payloadOf(p)) target.assets.push(asset);
    target.problems.push(p.id);
    target.used = target.assets.reduce((a, x) => a + x.size, 0);
  }
  const manifest = bundles.map((b) => ({ id: b.id, problems: b.problems.slice().sort(),
    assets: b.assets.slice().sort((x, y) => (x.path < y.path ? -1 : 1)), bytes: b.used }));
  return { limitBytes, bundles: manifest, unpackable, unpacked: unpackable,
    digest: fnv1a(JSON.stringify(manifest)) };
}

/** Exact minimum bin count for a tiny instance: a depth-first search with pruning. */
function exactBinCount(sizes, limit) {
  const sorted = sizes.slice().sort((a, b) => b - a);
  let best = sorted.length;
  const bins = [];
  const place = (i) => {
    if (bins.length >= best) return;
    if (i === sorted.length) { best = Math.min(best, bins.length); return; }
    for (let b = 0; b < bins.length; b += 1) {
      if (bins[b] + sorted[i] > limit) continue;
      bins[b] += sorted[i];
      place(i + 1);
      bins[b] -= sorted[i];
    }
    bins.push(sorted[i]);
    place(i + 1);
    bins.pop();
  };
  if (sorted.length === 0) return 0;
  if (Math.max(...sorted) > limit) return Number.POSITIVE_INFINITY;
  place(0);
  return best;
}

const SHARED = [{ name: "runner", path: "shared/runner.js", size: 12 },
  { name: "judge", path: "shared/judge.js", size: 8 }];
const PROBLEMS = [
  { id: "p001", size: 30, tests: [{ name: "t1.json", size: 10 }], uses: ["runner"] },
  { id: "p002", size: 25, tests: [{ name: "t1.json", size: 5 }], uses: ["runner", "judge"] },
  { id: "p003", size: 15, tests: [{ name: "t1.json", size: 5 }], uses: ["judge"] },
  { id: "p004", size: 55, tests: [{ name: "t1.json", size: 5 }], uses: [] },
  { id: "p005", size: 10, tests: [], uses: ["runner"] },
];
/** Seeded fixture with a bounded size distribution so every problem stays packable. */
function seededProblems(seed, count) {
  const rng = splitmix64(seed);
  return Array.from({ length: count }, (_, i) => ({ id: `p${String(i).padStart(3, "0")}`,
    size: 5 + rng.int(40), tests: Array.from({ length: rng.int(3) }, (_, t) => ({ name: `t${t}.json`, size: 1 + rng.int(6) })),
    uses: SHARED.filter(() => rng.int(2) === 0).map((s) => s.name) }));
}

function main() {
  const plan = packBundles(PROBLEMS, 100, SHARED);
  // 1. Happy path: every packable problem lands in exactly one bundle and nothing is over budget.
  const packed = plan.bundles.flatMap((b) => b.problems).sort();
  assert.deepStrictEqual([packed.length, new Set(packed).size], [packed.length, packed.length],
    "each selected problem appears exactly once");
  assert.ok(plan.bundles.every((b) => b.bytes <= 100), "no bundle exceeds the budget");
  // 2. Coverage: the manifest covers the selection minus the explicitly unpackable ids.
  assert.deepStrictEqual(packed.concat(plan.unpackable).sort(), PROBLEMS.map((p) => p.id).sort(),
    "manifest plus unpackable ids covers the selection");
  // 3. Shared assets: every bundle containing a problem that uses an asset ships that asset once.
  for (const b of plan.bundles) {
    for (const p of PROBLEMS.filter((x) => b.problems.includes(x.id))) {
      for (const name of p.uses) assert.ok(b.assets.some((a) => a.path === `shared/${name}.js`),
        `${name} shipped with ${p.id}`);
    }
    assert.strictEqual(new Set(b.assets.map((a) => a.path)).size, b.assets.length, "no duplicate asset in a bundle");
  }
  // 4. Boundary: an empty selection packs to nothing, and a set that fills the budget exactly is one
  // bundle rather than two.
  assert.deepStrictEqual([packBundles([], 100).bundles, packBundles([], 100).unpackable], [[], []],
    "empty selection produces no bundles");
  const tight = packBundles([{ id: "p900", size: 60, tests: [{ name: "t.json", size: 40 }], uses: [] }], 100);
  assert.deepStrictEqual([tight.bundles.length, tight.bundles[0].bytes, tight.bundles[0].problems],
    [1, 100, ["p900"]], "exact-fit problem fills one bundle");
  // 5. Negative: a problem larger than the budget is reported, not silently dropped or truncated.
  const oversized = packBundles([...PROBLEMS, { id: "p999", size: 500, tests: [], uses: [] }], 100, SHARED);
  assert.deepStrictEqual([oversized.unpackable, oversized.unpacked], [["p999"], ["p999"]], "oversized problem reported");
  assert.ok(!oversized.bundles.some((b) => b.problems.includes("p999")), "oversized problem is not packed");
  // 6. Oracle: on a tiny instance greedy bin count must equal the exact minimum from a search.
  const tiny = seededProblems(0x989, 7);
  const exact = exactBinCount(tiny.map((p) => problemSize(p)), 60);
  const greedy = packBundles(tiny, 60).bundles.length;
  assert.strictEqual(greedy, exact, "first-fit-decreasing matches the exact packing on the tiny instance");
  assert.ok(greedy >= exactBinCount(tiny.map((p) => problemSize(p)), 60), "greedy never beats the optimum");
  // 7. Determinism: shuffling the selection cannot change the bundles or the digest.
  const shuffled = packBundles(PROBLEMS.slice().reverse(), 100, SHARED.slice().reverse());
  assert.deepStrictEqual([shuffled.bundles, shuffled.digest], [plan.bundles, plan.digest],
    "packing is input-order independent");
  // 8. Scale: a 40-problem fixture has monotone byte accounting and reproduces byte for byte.
  const fixture = packBundles(seededProblems(0x989, 40), 120, SHARED);
  assert.deepStrictEqual([fixture.bundles.every((b) => b.bytes <= 120),
    fixture.bundles.flatMap((b) => b.problems).length, packBundles(seededProblems(0x989, 40), 120, SHARED).digest],
    [true, 40, fixture.digest], "fixture packs fully and reproduces");
  console.log(`bundles=${plan.bundles.length} bytes=${plan.bundles.reduce((a, b) => a + b.bytes, 0)} ` +
    `unpackable=${plan.unpackable.length} digest=${plan.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, problemSize, packBundles, exactBinCount, seededProblems, SHARED };
