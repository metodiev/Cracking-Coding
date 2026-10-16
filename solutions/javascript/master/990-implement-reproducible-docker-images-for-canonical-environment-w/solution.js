/** Task 990 - Implement reproducible Docker images for canonical environment with JDK/test harness.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Builds the canonical exercise environment as a content-addressed chain of layers: every layer
 * digest is H(parent digest || step text), so the image digest is a Merkle root over the whole build
 * recipe. That is what makes the image reproducible (one recipe -> one digest), makes a change to an
 * early step invalidate every later layer, and lets a rebuild reuse an unchanged prefix from the
 * cache. Bases pinned only by tag are refused, and a lint pass flags the steps that would break
 * bit-for-bit reproducibility.
 *
 * What is simplified: no daemon and no filesystem -- a "layer" is a hash and the cache is a Map.
 * Verified: deterministic digests, Merkle sensitivity, cache reuse of the common prefix, refusal of
 * unpinned or tag-based bases, lint findings, and a seeded batch of recipes that must rebuild equal.
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

const PINNED = /^sha256:[0-9a-f]{16,}$/;
/** Steps that cannot produce identical bytes twice; every one is reported by the lint pass. */
const NONDETERMINISTIC = [
  { pattern: /apt-get\s+(update|upgrade)/i, reason: "package index is a moving target" },
  { pattern: /:latest\b/i, reason: "mutable tag" },
  { pattern: /\bnow\(\)|date \+|--build-arg\s+timestamp/i, reason: "wall-clock input" },
  { pattern: /--no-cache\b/i, reason: "unpinned fetch" },
  { pattern: /\bnpm\s+install\s*$|\bpip\s+install\s+[a-z0-9-]+\s*$/i, reason: "unpinned dependency" },
];
/** One layer digest: a chain hash, so no layer can be reordered without changing every successor. */
const layerDigest = (parent, step) => fnv1a(`${parent}|${step}`);

/**
 * Build the image. The cache is keyed by layer digest only, which is exactly the reproducible-build
 * cache: a hit proves the whole prefix (base digest plus every earlier step) is bit-identical.
 */
function buildImage(spec, cache = new Map()) {
  if (!PINNED.test(spec.baseDigest || "")) {
    return { ok: false, reason: "unpinned-base", digest: null, layers: [], cacheHits: 0, reused: 0 };
  }
  let parent = spec.baseDigest;
  const layers = [];
  let hits = 0;
  for (const step of spec.steps || []) {
    const digest = layerDigest(parent, step);
    if (cache.has(digest)) hits += 1;
    cache.set(digest, parent);
    layers.push({ step, parent, digest });
    parent = digest;
  }
  return { ok: true, reason: "built", digest: parent, layers, cacheHits: hits, reused: hits };
}

/** Independent recomputation of the Merkle root, without the build loop or the cache. */
function replayDigest(spec) {
  return (spec.steps || []).reduce((parent, step) => layerDigest(parent, step), spec.baseDigest);
}

/** Lint: reproducibility findings plus a 0..1 score over the recipe's steps. */
function lintSpec(spec) {
  const findings = [];
  for (const step of spec.steps || []) {
    for (const rule of NONDETERMINISTIC) if (rule.pattern.test(step)) findings.push({ step, reason: rule.reason });
  }
  const steps = (spec.steps || []).length;
  return { ok: findings.length === 0, findings,
    score: steps === 0 ? 1 : (steps - new Set(findings.map((f) => f.step)).size) / steps };
}

const BASE = "sha256:9f2c4a7be1d05c3388aabbccddeeff00";
const GOOD = ["FROM pinned", "COPY solution.js /app/solution.js", "RUN javac /app/solution.js",
  "COPY tests /app/tests", "ENV LANG=C.UTF-8", "ENTRYPOINT [\"node\", \"/app/solution.js\"]"];
const BAD = ["RUN apt-get update", "RUN curl -sL node:latest", "RUN pip install requests",
  "RUN date +%s > /stamp", "RUN npm install"];
/** Seeded recipes built only from pinned, lint-clean steps, so builds succeed by construction. */
function seededSpecs(seed, count) {
  const rng = splitmix64(seed);
  return Array.from({ length: count }, (_, i) => ({ baseDigest: BASE,
    steps: Array.from({ length: 1 + rng.int(5) }, () => GOOD[rng.int(GOOD.length)]), tag: `ex-${i}` }));
}

function main() {
  const spec = { baseDigest: BASE, steps: GOOD };
  const image = buildImage(spec);
  // 1. Happy path: one layer per step, and the image digest is the hash chain root.
  assert.deepStrictEqual([image.ok, image.layers.length, image.layers.map((l) => l.step)],
    [true, GOOD.length, GOOD], "one layer per step in recipe order");
  assert.strictEqual(image.digest, replayDigest(spec), "image digest equals the independent replay");
  assert.ok(image.layers.every((l, i) => l.parent === (i === 0 ? BASE : image.layers[i - 1].digest)),
    "each layer chains onto its parent");
  // 2. Reproducibility: a second build with a cold cache produces the identical digest.
  const rebuild = buildImage(spec);
  assert.deepStrictEqual([rebuild.digest, rebuild.layers.map((l) => l.digest)], [image.digest, image.layers.map((l) => l.digest)],
    "rebuild is bit-identical");
  // 3. Merkle sensitivity: editing the *first* step changes every later layer; editing the last step
  // leaves the prefix intact but still changes the image digest.
  const warm = new Map();
  buildImage(spec, warm);
  const front = buildImage({ baseDigest: BASE, steps: ["FROM pinned v2", ...GOOD.slice(1)] }, new Map(warm));
  const back = buildImage({ baseDigest: BASE, steps: [...GOOD.slice(0, -1), "ENTRYPOINT [\"node\", \"/a\"]"] }, new Map(warm));
  assert.strictEqual(front.layers.filter((l, i) => l.digest !== image.layers[i].digest).length, GOOD.length,
    "leading edit invalidates every layer");
  assert.deepStrictEqual(back.layers.slice(0, -1).map((l) => l.digest), image.layers.slice(0, -1).map((l) => l.digest),
    "trailing edit keeps the prefix");
  assert.notStrictEqual(back.digest, image.digest, "trailing edit still changes the image digest");
  // 4. Cache: rebuilding a recipe that diverges after step k must reuse exactly k layers.
  const cache = new Map();
  buildImage(spec, cache);
  const diverged = buildImage({ baseDigest: BASE, steps: [...GOOD.slice(0, 3), "RUN javac -Xlint /app"] }, cache);
  assert.deepStrictEqual([diverged.cacheHits, diverged.reused], [3, 3], "common prefix is served from cache");
  assert.strictEqual(buildImage(spec, cache).cacheHits, GOOD.length, "an unchanged recipe is a full cache hit");
  assert.strictEqual(diverged.layers[3].parent, image.layers[2].digest, "the new layer forks off the cached prefix");
  // 5. Negative: a mutable base tag is refused, and nothing is built.
  const floating = buildImage({ baseDigest: "node:latest", steps: GOOD });
  assert.deepStrictEqual([floating.ok, floating.reason, floating.digest, floating.layers], [false, "unpinned-base", null, []],
    "tag-based base is refused");
  // 6. Boundary: a recipe with no steps is just its base image, and step order is part of identity.
  assert.strictEqual(buildImage({ baseDigest: BASE, steps: [] }).digest, BASE, "no steps means the base digest");
  const swapped = buildImage({ baseDigest: BASE, steps: [GOOD[1], GOOD[0], ...GOOD.slice(2)] });
  assert.notStrictEqual(swapped.digest, image.digest, "reordering steps changes the image");
  // 7. Lint: every non-deterministic step is named, and the clean recipe scores 1.
  const lint = lintSpec({ baseDigest: BASE, steps: BAD });
  assert.deepStrictEqual([lint.ok, lint.findings.length, lint.score], [false, BAD.length, 0], "all bad steps are flagged");
  assert.deepStrictEqual([lintSpec(spec).ok, lintSpec({ baseDigest: BASE, steps: [] }).score], [true, 1],
    "clean recipe lints clean, empty recipe scores 1");
  // 8. Seeded batch: 12 recipes must build, lint clean, and rebuild to the same digest on a second pass.
  const specs = seededSpecs(0x990, 12);
  const digests = specs.map((s) => buildImage(s).digest);
  const replayed = specs.map((s) => replayDigest(s));
  assert.deepStrictEqual(digests, replayed, "batch builds match the independent replay");
  assert.strictEqual(new Set(digests).size, digests.length, "distinct recipes give distinct images");
  assert.deepStrictEqual(specs.map((s) => buildImage(s).digest), digests, "batch rebuilds are stable");
  assert.strictEqual(specs.every((s) => lintSpec(s).ok), true, "generated recipes are lint clean");
  console.log(`layers=${image.layers.length} cacheHits=${diverged.cacheHits} findings=${lint.findings.length} ` +
    `digest=${image.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, layerDigest, buildImage, replayDigest, lintSpec, seededSpecs };
