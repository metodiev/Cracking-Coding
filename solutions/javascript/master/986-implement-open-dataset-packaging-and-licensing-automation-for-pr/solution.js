/** Task 986 - Implement open dataset packaging and licensing automation for practice problems.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Packages a problem set into a distributable dataset and derives the license obligations rather
 * than trusting a hand-written header. Every problem declares an SPDX license; the packager folds
 * those declarations into one aggregate license (public-domain < permissive < copyleft <
 * proprietary), carries share-alike duties forward, collects the attribution that the involved
 * licenses actually require and refuses datasets it cannot legally ship.
 *
 * What is simplified: no archive is written; the "package" is an in-memory manifest with FNV
 * checksums, and the license policy is a small curated table instead of a full SPDX graph.
 * Verified: aggregate selection, share-alike propagation, attribution collection, rejection of
 * proprietary/unknown/empty datasets, manifest digest stability under reordering, tamper detection,
 * and an independent compatibility oracle over every problem at once.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the file's only randomness source, so fixtures are byte-reproducible. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
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

/** Ordered by obligation strength: an aggregate must satisfy the strictest member. */
const RANK = { "CC0-1.0": 0, Unlicense: 0, MIT: 1, "Apache-2.0": 1, "BSD-3-Clause": 1,
  "GPL-3.0": 2, "AGPL-3.0": 3, "All-Rights-Reserved": 4 };
const CANONICAL = { 0: "CC0-1.0", 1: "MIT", 2: "GPL-3.0", 3: "AGPL-3.0", 4: "All-Rights-Reserved" };
const REQUIRES_ATTRIBUTION = new Set(["MIT", "Apache-2.0", "BSD-3-Clause", "GPL-3.0", "AGPL-3.0"]);
const SHARE_ALIKE = new Set(["GPL-3.0", "AGPL-3.0"]);

const rankOf = (license) => (RANK[license] === undefined ? -1 : RANK[license]);
const isKnown = (license) => rankOf(license) >= 0;

/** Stable serialisation: sorted keys, so a digest never depends on insertion order. */
function stableStringify(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(stableStringify).join(",") + "]";
  return "{" + Object.keys(value).sort()
    .map((k) => JSON.stringify(k) + ":" + stableStringify(value[k])).join(",") + "}";
}

/**
 * Fold the members' licenses into one aggregate. `worst` is the strictest rank seen, which is what
 * makes the operation associative and commutative: order of the problems cannot change the result.
 */
function aggregateLicense(problems) {
  if (!Array.isArray(problems) || problems.length === 0) {
    return { ok: false, reason: "empty-dataset", license: null, shareAlike: false, rank: -1 };
  }
  let worst = -1;
  const offenders = [];
  for (const p of problems) {
    const rank = rankOf(p.license);
    if (rank < 0) { offenders.push({ id: p.id, license: p.license, reason: "unknown-license" }); continue; }
    if (rank === 4) offenders.push({ id: p.id, license: p.license, reason: "proprietary" });
    if (rank > worst) worst = rank;
  }
  if (offenders.length > 0) {
    return { ok: false, reason: offenders.some((o) => o.reason === "proprietary") ? "proprietary" : "unknown-license",
      license: CANONICAL[worst < 0 ? 0 : worst], shareAlike: SHARE_ALIKE.has(CANONICAL[worst < 0 ? 0 : worst]),
      rank: worst, offenders };
  }
  const license = CANONICAL[worst];
  return { ok: true, reason: "packaged", license, shareAlike: SHARE_ALIKE.has(license), rank: worst, offenders };
}

/** Attribution lines required by the members' licenses, de-duplicated and sorted. */
function attribution(problems) {
  const lines = new Set();
  for (const p of problems) {
    for (const l of [p.license].concat(p.derivedFrom || [])) {
      if (REQUIRES_ATTRIBUTION.has(l) && p.author) lines.add(`${p.author} (${l}) - ${p.title}`);
    }
  }
  return Array.from(lines).sort();
}

/** The package itself: problems, aggregate license, attribution block and a per-file checksum list. */
function packageDataset(problems) {
  const ordered = problems.slice().sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const aggregate = aggregateLicense(ordered);
  const files = ordered.map((p) => ({ path: `problems/${p.id}.md`, digest: fnv1a(`${p.id}|${p.title}|${p.body || ""}`) }));
  const manifest = {
    files,
    license: aggregate.license,
    shareAlike: aggregate.shareAlike,
    rank: aggregate.rank,
    ok: aggregate.ok,
    reason: aggregate.reason,
    attribution: attribution(ordered),
    problemCount: ordered.length,
  };
  manifest.digest = fnv1a(stableStringify({ files, license: manifest.license, attribution: manifest.attribution }));
  manifest.offenders = aggregate.offenders || [];
  return manifest;
}

/** Integrity check: every listed file must still hash to its recorded digest. */
function verifyIntegrity(manifest, problemsById) {
  const bad = [];
  for (const f of manifest.files) {
    const id = f.path.slice("problems/".length, -".md".length);
    const p = problemsById.get(id);
    const digest = p ? fnv1a(`${p.id}|${p.title}|${p.body || ""}`) : 0;
    if (digest !== f.digest) bad.push(f.path);
  }
  return { ok: bad.length === 0, bad };
}

const LICENSES = Object.keys(RANK);
const ORACLES = ["MIT", "Apache-2.0", "GPL-3.0", "AGPL-3.0", "CC0-1.0", "All-Rights-Reserved", "WTFPL-unknown"];
/** Fixtures draw from a license pool; the default pool is shippable, the bad pool is not. */
function seededProblems(seed, count, pool = ORACLES.slice(0, 5)) {
  const rng = splitmix64(seed);
  const problems = [];
  for (let i = 0; i < count; i += 1) {
    problems.push({ id: `p${String(i).padStart(3, "0")}`, title: `Problem ${i}`,
      author: `author-${rng.int(4)}`, license: pool[rng.int(pool.length)], body: `body ${i}` });
  }
  return problems;
}

function main() {
  // 1. Happy path: a permissive-only set packages under the strictest member license.
  const permissive = [
    { id: "p001", title: "Two Sum", author: "ada", license: "MIT", body: "a" },
    { id: "p002", title: "LRU Cache", author: "bob", license: "MIT", body: "b" },
    { id: "p003", title: "Graph Walk", author: "chen", license: "Apache-2.0", body: "c" },
  ];
  const pack = packageDataset(permissive);
  assert.strictEqual(pack.ok, true, "permissive set must package");
  assert.strictEqual(pack.license, "MIT", "MIT and Apache-2.0 share the permissive rank, canonicalised to MIT");
  assert.strictEqual(pack.shareAlike, false, "permissive aggregate is not share-alike");
  assert.deepStrictEqual(pack.attribution,
    ["ada (MIT) - Two Sum", "bob (MIT) - LRU Cache", "chen (Apache-2.0) - Graph Walk"], "attribution block");

  // 2. Copyleft propagates: one GPL problem lifts the whole dataset, not just its own file.
  const withCopyleft = packageDataset(permissive.concat([
    { id: "p004", title: "Scheduler", author: "dana", license: "GPL-3.0", body: "d" }]));
  assert.strictEqual(withCopyleft.license, "GPL-3.0", "copyleft propagates to the aggregate");
  assert.strictEqual(withCopyleft.shareAlike, true, "share-alike duty carried forward");
  assert.strictEqual(withCopyleft.ok, true, "copyleft datasets are still distributable");

  // 3. Boundary: an empty dataset is refused instead of silently producing a license-less manifest.
  assert.deepStrictEqual(aggregateLicense([]).reason, "empty-dataset", "empty dataset rejected");
  assert.strictEqual(packageDataset([]).license, null, "empty package carries no license");

  // 4. A single problem is atomic: the aggregate is exactly its own license.
  const single = packageDataset([{ id: "p009", title: "Solo", author: "erin", license: "CC0-1.0", body: "s" }]);
  assert.strictEqual(single.license, "CC0-1.0", "public-domain aggregate stays public domain");
  assert.deepStrictEqual(single.attribution, [], "CC0 requires no attribution");

  // 5. Negative: proprietary or unknown licenses are refused, and the offenders are named.
  const mixed = packageDataset([
    { id: "p005", title: "Locked", author: "fay", license: "All-Rights-Reserved", body: "x" },
    { id: "p006", title: "Mystery", author: "gus", license: "WTFPL-unknown", body: "y" }]);
  assert.strictEqual(mixed.ok, false, "unshippable dataset refused");
  assert.strictEqual(mixed.reason, "proprietary", "proprietary wins over unknown");
  assert.deepStrictEqual(mixed.offenders.map((o) => [o.id, o.reason]),
    [["p005", "proprietary"], ["p006", "unknown-license"]], "offenders enumerated after id sort");

  // 6. Boundary: the whole dataset being proprietary is refused with no canonical license claim.
  assert.strictEqual(packageDataset([{ id: "p007", title: "T", author: "hal", license: "AGPL-3.0", body: "z" },
    { id: "p008", title: "U", author: "hal", license: "All-Rights-Reserved", body: "z" }]).ok, false, "no AGPL escape hatch");

  // 7. Digest stability: reordering problems must not change the manifest digest.
  const shuffled = permissive.slice().reverse().concat([{ id: "p000", title: "Zero", author: "zed",
    license: "MIT", body: "0" }]);
  const shuffledPack = packageDataset(shuffled);
  const straightPack = packageDataset(shuffled.slice().reverse());
  assert.strictEqual(shuffledPack.digest, straightPack.digest, "manifest digest is order independent");
  assert.deepStrictEqual(shuffledPack.files, straightPack.files, "file list sorted by id");

  // 8. Tamper detection: changing one problem body invalidates only that file's digest.
  const byId = new Map(permissive.map((p) => [p.id, { ...p }]));
  assert.deepStrictEqual(verifyIntegrity(pack, byId).bad, [], "untouched package verifies");
  byId.set("p002", { ...byId.get("p002"), body: "b-tampered" });
  assert.deepStrictEqual(verifyIntegrity(pack, byId).bad, ["problems/p002.md"], "only the edited file fails");

  // 9. Oracle: the aggregate must be compatible with every member, checked pairwise and independently
  // of the rank fold. On the shippable fixture the aggregate rank must be the maximum member rank.
  const problems = seededProblems(0x986, 40);
  const seeded = packageDataset(problems);
  const expectedRank = Math.max.apply(null, problems.map((p) => rankOf(p.license)));
  assert.strictEqual(seeded.ok, true, "fixture of known licenses packages");
  assert.strictEqual(seeded.rank, expectedRank, "aggregate rank is the maximum member rank");
  for (const p of problems) {
    assert.ok(rankOf(seeded.license) >= rankOf(p.license), `aggregate is at least as strict as ${p.license}`);
  }
  assert.strictEqual(seeded.shareAlike, expectedRank >= 2, "share-alike flag follows the aggregate rank");

  // The unshippable fixture must name every offender exactly once with the right reason.
  const bad = packageDataset(seededProblems(0x986, 40, ORACLES.slice(5)));
  assert.strictEqual(bad.ok, false, "fixture with proprietary/unknown licenses is refused");
  const expect = {};
  for (const p of seededProblems(0x986, 40, ORACLES.slice(5))) {
    const r = rankOf(p.license);
    if (r === 4) expect.proprietary = (expect.proprietary || 0) + 1;
    else expect["unknown-license"] = (expect["unknown-license"] || 0) + 1;
  }
  assert.strictEqual(bad.offenders.length, expect.proprietary + expect["unknown-license"],
    "every unshippable member is reported exactly once");
  assert.deepStrictEqual(bad.offenders.reduce((acc, o) => {
    acc[o.reason] = (acc[o.reason] || 0) + 1; return acc;
  }, {}), expect, "offender reason histogram matches the input licenses");

  // 10. Determinism: two independent builds of the same fixture agree byte for byte.
  assert.strictEqual(packageDataset(seededProblems(0x986, 40)).digest, seeded.digest, "rebuild digest stable");

  console.log(`packaged=${seeded.ok} license=${seeded.license} files=${seeded.files.length} digest=${seeded.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = { splitmix64, fnv1a, aggregateLicense, attribution, packageDataset, verifyIntegrity,
  stableStringify, RANK, CANONICAL };
