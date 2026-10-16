/** Task 987 - Implement migration tools for older problem formats to new canonical schema.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A versioned migration chain: every step is a pure record->record transform declaring the version it
 * consumes and produces, so the runner can plan a path between any two versions (or refuse one), apply
 * the steps in order and validate the result against the canonical schema. Steps are guarded by the
 * record's own `version`, which makes migration idempotent: re-running the chain over an already
 * migrated document applies zero steps and returns it unchanged.
 *
 * What is simplified: documents are plain objects and the validator is a hand-written field checker
 * rather than JSON Schema. Verified: path planning and ordering, refusal of unknown/backwards versions,
 * idempotence, per-step version and identity invariants, tag normalisation, validation errors with
 * field paths, and equivalence with an independent one-shot mapper over a seeded fixture.
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

const DIFFICULTIES = ["easy", "medium", "hard"];
const CANONICAL_VERSION = 4;
/** v1 "level" values predate the canonical scale. */
const normalizeDifficulty = (level) => {
  const v = String(level === undefined || level === null ? "" : level).trim().toLowerCase();
  if (DIFFICULTIES.includes(v)) return v;
  if (["e", "1", "low", "intro"].includes(v)) return "easy";
  if (["h", "3", "high", "expert"].includes(v)) return "hard";
  return "medium";
};
/** Tags arrive as a comma separated string in v1 and as an array later; both normalise the same way. */
function normalizeTags(tags) {
  const raw = Array.isArray(tags) ? tags : String(tags || "").split(",");
  const out = [];
  for (const t of raw) {
    const v = String(t).trim().toLowerCase();
    if (v.length > 0 && out.indexOf(v) === -1) out.push(v);
  }
  return out;
}

const slugify = (text) => String(text).trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

const toCases = (tests) => (tests || []).map((pair) => ({ input: pair[0], expected: pair[1] }));
const limitsOf = (r) => ({ timeMs: (r.limits && r.limits.timeMs) || r.timeoutMs || 2000,
  memoryMb: (r.limits && r.limits.memoryMb) || r.memoryMb || 256 });
/**
 * Pure transforms keyed by the version they upgrade. Each builds the next schema explicitly instead
 * of spreading its input, so legacy-only keys can never leak into a canonical document, and each
 * preserves `id` so a record can be traced across the whole chain.
 */
const STEPS = {
  1: (r) => ({ version: 2, id: r.id, title: r.name, statement: r.desc, cases: toCases(r.tests),
    difficulty: normalizeDifficulty(r.level), tags: normalizeTags(r.tags), timeoutMs: r.timeoutMs || 2000 }),
  2: (r) => ({ version: 3, id: r.id, slug: r.slug || slugify(r.title), title: r.title, statement: r.statement,
    difficulty: r.difficulty, tags: r.tags, cases: r.cases, limits: limitsOf(r),
    licenses: r.license ? [r.license] : (r.licenses || []) }),
  3: (r) => ({ version: 4, id: r.id, slug: r.slug || slugify(r.title), title: r.title, statement: r.statement,
    cases: r.cases, difficulty: normalizeDifficulty(r.difficulty), tags: normalizeTags(r.tags),
    limits: limitsOf(r), licenses: r.licenses || [] }),
};

const pathCache = new Map();
/** Plan the ascending chain of versions from `from` to `to`; empty when already there. */
function migrationPath(from, to) {
  const key = `${from}->${to}`;
  if (pathCache.has(key)) return pathCache.get(key);
  const result = (() => {
    if (!Number.isInteger(from) || !Number.isInteger(to)) return { ok: false, reason: "unknown-version", steps: [] };
    if (from === to) return { ok: true, reason: "no-op", steps: [] };
    if (from > to) return { ok: false, reason: "backwards-migration", steps: [] };
    if (from < 1 || to > CANONICAL_VERSION) return { ok: false, reason: "unsupported-version", steps: [] };
    const steps = [];
    for (let v = from; v < to; v += 1) steps.push({ from: v, to: v + 1, apply: STEPS[v] });
    return { ok: true, reason: "planned", steps };
  })();
  pathCache.set(key, result);
  return result;
}
/** Apply the planned chain. Unknown versions leave the record byte-identical and report the reason. */
function migrate(record, target = CANONICAL_VERSION) {
  const plan = migrationPath(record.version, target);
  if (!plan.ok) return { ok: false, reason: plan.reason, record, applied: [] };
  let current = record;
  const applied = [];
  for (const step of plan.steps) {
    if (current.version !== step.from) return { ok: false, reason: "version-skew", record: current, applied };
    current = step.apply(current);
    applied.push(`${step.from}->${step.to}`);
  }
  return { ok: true, reason: "migrated", record: current, applied };
}
/** Canonical schema check reporting every failure as a `field` path. */
function validateCanonical(rec) {
  const issues = [];
  const need = (path, ok, message) => { if (!ok) issues.push({ field: path, message }); };
  if (rec.version !== CANONICAL_VERSION) issues.push({ field: "version", message: `expected ${CANONICAL_VERSION}` });
  need("id", typeof rec.id === "string" && rec.id.length > 0, "id must be a non-empty string");
  need("slug", typeof rec.slug === "string" && /^[a-z0-9-]+$/.test(rec.slug || ""), "slug must be kebab-case");
  need("title", typeof rec.title === "string" && rec.title.length > 0, "title required");
  need("statement", typeof rec.statement === "string" && rec.statement.length > 0, "statement required");
  need("difficulty", DIFFICULTIES.includes(rec.difficulty), "difficulty must be easy|medium|hard");
  need("tags", Array.isArray(rec.tags), "tags must be an array");
  need("cases", Array.isArray(rec.cases) && rec.cases.every((c) => c && "input" in c && "expected" in c),
    "cases must be {input, expected} records");
  need("limits.timeMs", rec.limits && rec.limits.timeMs > 0, "timeMs must be positive");
  need("limits.memoryMb", rec.limits && rec.limits.memoryMb > 0, "memoryMb must be positive");
  need("licenses", Array.isArray(rec.licenses), "licenses must be an array");
  return { ok: issues.length === 0, issues };
}
/** Independent oracle: map any legacy shape straight to canonical in one pass, no step machinery. */
function oracleCanonical(rec) {
  const title = rec.title || rec.name;
  const statement = rec.statement || rec.desc;
  const rawCases = rec.cases || (rec.tests || []).map((p) => ({ input: p[0], expected: p[1] }));
  const timeMs = (rec.limits && rec.limits.timeMs) || rec.timeoutMs || 2000;
  const memoryMb = (rec.limits && rec.limits.memoryMb) || rec.memoryMb || 256;
  return { version: CANONICAL_VERSION, id: rec.id, slug: rec.slug || slugify(title), title, statement,
    difficulty: normalizeDifficulty(rec.difficulty || rec.level),
    tags: normalizeTags(rec.tags), cases: rawCases,
    limits: { timeMs, memoryMb }, licenses: rec.license ? [rec.license] : (rec.licenses || []) };
}
const stableStringify = (value) => {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return "[" + value.map(stableStringify).join(",") + "]";
  return "{" + Object.keys(value).sort().map((k) => JSON.stringify(k) + ":" + stableStringify(value[k])).join(",") + "}";
};
const TITLES = ["Two Sum", "Merge Intervals", "Word Ladder", "Coin Change", "LRU Cache"];const TAGSETS = ["arrays,strings", "dp, memoization ,dp", "graphs,bfs", ""];
const LEVELS = ["easy", "1", "Hard", "expert", "unknown-level"];
/** Seeded fixture mixing all three legacy versions, including a record with an unknown level. */
function seededLegacy(seed, count) {
  const rng = splitmix64(seed);
  const records = [];
  for (let i = 0; i < count; i += 1) {
    const version = 1 + rng.int(3);
    const title = TITLES[rng.int(TITLES.length)] + " " + i;
    const common = { version, id: `p${String(i).padStart(3, "0")}`, cases: [{ input: `${i}`, expected: `${i * 2}` }] };
    if (version === 1) {
      records.push({ ...common, name: title, desc: `desc ${i}`, level: LEVELS[rng.int(LEVELS.length)],
        tags: TAGSETS[rng.int(TAGSETS.length)], tests: [[String(i), String(i * 2)]], timeoutMs: 1000 });
    } else if (version === 2) {
      records.push({ ...common, title, statement: `statement ${i}`, difficulty: LEVELS[rng.int(LEVELS.length)],
        tags: TAGSETS[rng.int(TAGSETS.length)], timeoutMs: 1500, license: "MIT" });
    } else {
      records.push({ ...common, title, slug: slugify(title), statement: `statement ${i}`,
        difficulty: LEVELS[rng.int(LEVELS.length)], tags: TAGSETS[rng.int(TAGSETS.length)],
        limits: { timeMs: 3000, memoryMb: 128 }, licenses: ["BSD-3-Clause"] });
    }
  }
  return records;
}
function main() {
  // 1. Happy path: a v1 record walks the whole chain and lands on a valid canonical document.
  const v1 = { version: 1, id: "p001", name: "Two Sum", desc: "Find the pair.", level: "1",
    tags: "Arrays, hash-map ,arrays", tests: [[[1, 2], [3]]], timeoutMs: 1000 };
  const result = migrate(v1);
  assert.deepStrictEqual(result.applied, ["1->2", "2->3", "3->4"], "three steps applied in order");
  assert.strictEqual(result.record.version, 4, "record upgraded to canonical");
  assert.strictEqual(result.record.title, "Two Sum", "name is renamed to title");
  assert.strictEqual(result.record.statement, "Find the pair.", "desc is renamed to statement");
  assert.deepStrictEqual(result.record.tags, ["arrays", "hash-map"], "tags split, trimmed, deduped, lowercased");
  assert.strictEqual(result.record.difficulty, "easy", "legacy numeric level maps to easy");
  assert.strictEqual(result.record.slug, "two-sum", "slug derived from the title");
  assert.deepStrictEqual(result.record.limits, { timeMs: 1000, memoryMb: 256 }, "timeout kept, memory defaulted");
  assert.strictEqual(validateCanonical(result.record).ok, true, "canonical output validates");
  // 2. Boundary: an already canonical record is untouched and costs zero steps (idempotence).
  const again = migrate(result.record);
  assert.deepStrictEqual(again.applied, [], "no steps for an up-to-date record");
  assert.deepStrictEqual(again.record, result.record, "record unchanged");
  assert.strictEqual(stableStringify(again.record), stableStringify(result.record), "byte-identical re-migration");
  // 3. Path planning: prefix targets stop early, backwards and unknown versions are refused.
  assert.deepStrictEqual(migrationPath(1, 2).steps.map((s) => s.to), [2], "path to v2 is one step");
  assert.deepStrictEqual(migrationPath(2, 4).steps.map((s) => s.from), [2, 3], "path from v2 runs two steps");
  assert.deepStrictEqual(migrationPath(3, 3), { ok: true, reason: "no-op", steps: [] }, "no-op path");
  const backwards = migrate({ version: 4, id: "x" }, 2);
  assert.strictEqual(backwards.reason, "backwards-migration", "no downgrade available");
  const unknown = migrate({ version: 0, id: "x" });
  assert.strictEqual(unknown.reason, "unsupported-version", "unknown version refused");
  assert.deepStrictEqual(unknown.record, { version: 0, id: "x" }, "refused record left byte-identical");
  assert.deepStrictEqual(unknown.applied, [], "nothing applied to a refused record");
  // 4. Validation reports every problem with a field path, not just the first.
  const broken = validateCanonical({ version: 4, id: "", slug: "Not Kebab", title: "", statement: "s",
    difficulty: "impossible", tags: "arrays", cases: [{}], limits: { timeMs: 0, memoryMb: 256 }, licenses: [] });
  assert.strictEqual(broken.ok, false, "broken document rejected");
  assert.deepStrictEqual(broken.issues.map((i) => i.field),
    ["id", "slug", "title", "difficulty", "tags", "cases", "limits.timeMs"], "one issue per offending field");
  // 5. Oracle: over a seeded fixture of v1/v2/v3 records the chain must equal a one-shot mapper.
  const legacy = seededLegacy(0x987, 45);
  const digests = [];
  for (const rec of legacy) {
    const migrated = migrate(rec);
    assert.deepStrictEqual(migrated.record, oracleCanonical(rec), `oracle agrees for ${rec.id} (v${rec.version})`);
    assert.strictEqual(validateCanonical(migrated.record).ok, true, `migrated ${rec.id} validates`);
    digests.push(fnv1a(stableStringify(migrated.record)));
  }
  // 6. Invariant after every step: the version advanced by one and identity fields survived.
  for (const rec of legacy) {
    let current = rec;
    for (const step of migrationPath(rec.version, 4).steps) {
      const next = step.apply(current);
      assert.strictEqual(next.version, step.to, "step produces the version it declares");
      assert.strictEqual(next.id, rec.id, "id is preserved across every step");
      current = next;
    }
  }
  // 7. Idempotence and determinism across the whole fixture.
  const twice = legacy.map((rec) => fnv1a(stableStringify(migrate(migrate(rec).record).record)));
  assert.deepStrictEqual(twice, digests, "migrating twice equals migrating once for every record");
  const reversed = legacy.slice().reverse().map((rec) => fnv1a(stableStringify(migrate(rec).record)));
  assert.deepStrictEqual(reversed.slice().reverse(), digests, "digests do not depend on record order");
  const all = fnv1a(digests.join("|"));
  assert.strictEqual(legacy.map((rec) => fnv1a(stableStringify(migrate(rec).record))).join("|"),
    digests.join("|"), "fixture digest reproduces in a second independent pass");
  // 8. Unknown levels are normalised rather than rejected: the fixture exercises that branch.
  const unknownLevel = legacy.filter((r) => r.version === 1 && !DIFFICULTIES.includes(String(r.level)));
  assert.ok(unknownLevel.length > 0, "fixture contains legacy levels needing normalisation");
  for (const rec of unknownLevel) {
    assert.strictEqual(migrate(rec).record.difficulty, normalizeDifficulty(rec.level), "level normalised");
  }

  console.log(`migrated=${legacy.length} steps=${migrationPath(1, 4).steps.length} digest=${all.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = { splitmix64, fnv1a, migrationPath, migrate, validateCanonical, oracleCanonical,
  normalizeTags, normalizeDifficulty, slugify, CANONICAL_VERSION };
