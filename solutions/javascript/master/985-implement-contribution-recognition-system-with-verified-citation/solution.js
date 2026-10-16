/** Task 985 - Implement contribution recognition system with verified citations for authors.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Credits only work that can be traced to a citation record, so recognition cannot be gamed by
 * self-reported activity. A contribution is *accepted* only when it carries at least one distinct
 * citation that (a) exists in the registry, (b) cites the very artifact the contribution claims and
 * (c) has a citation kind compatible with the contribution kind; otherwise the ledger records a
 * rejection reason and the contribution never reaches an author's score.
 *
 * What is simplified: the registry is an in-memory map rather than a DOI/URL resolver, so
 * "verification" is a structural check instead of a network fetch. Verified: accept/reject reasons,
 * citation dedup, tie-broken ranking, insertion-order independence, an incremental index equal to a
 * full recompute after every mutation, a brute-force scoring oracle and the audit histogram.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the file's only randomness source, so every fixture is byte-reproducible. */
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

const KIND_WEIGHT = { commit: 3, review: 2, issue: 1, docs: 1 };
/** A contribution kind may only be credited by a citation of a compatible kind. */
const COMPATIBLE = { commit: ["commit"], review: ["review"], issue: ["issue"], docs: ["docs", "commit"] };
const MAX_CITATION_BONUS = 2;

/** Registry of the citation records a contribution may point at. */
class CitationRegistry {
  constructor() { this.byId = new Map(); }

  register(rec) {
    assert.ok(typeof rec.id === "string" && rec.id.length > 0, "citation id required");
    assert.ok(KIND_WEIGHT[rec.kind] !== undefined, `unknown citation kind ${rec.kind}`);
    assert.ok(typeof rec.subject === "string" && rec.subject.length > 0, "citation subject required");
    assert.ok(!this.byId.has(rec.id), `duplicate citation ${rec.id}`);
    this.byId.set(rec.id, { id: rec.id, kind: rec.kind, subject: rec.subject });
    return this;
  }

  has(id) { return this.byId.has(id); }
  get(id) { return this.byId.get(id); }
  get size() { return this.byId.size; }
}

/** Accepts verified contributions and maintains per-author and per-kind totals. */
class RecognitionLedger {
  constructor(registry) {
    this.registry = registry;
    this.contributions = [];
    this.rejections = [];
    this.byAuthor = new Map();          // author -> { score, accepted }
    this.byKind = new Map();            // contribution kind -> accepted count
  }

  /** Never throws on a malformed contribution; returns the verdict entry it stored. */
  add(contribution) {
    const c = contribution;
    const verdict = RecognitionLedger.verify(this.registry, c);
    const entry = { author: c.author, kind: c.kind, artifact: c.artifact, weight: c.weight,
      accepted: verdict.ok, reason: verdict.reason, distinctCitations: verdict.distinct,
      score: verdict.ok ? verdict.score : 0 };
    this.contributions.push(entry);
    const cur = this.byAuthor.get(c.author) || { score: 0, accepted: 0 };
    if (verdict.ok) {
      cur.score += verdict.score;
      cur.accepted += 1;
      this.byKind.set(c.kind, (this.byKind.get(c.kind) || 0) + 1);
    } else {
      this.rejections.push(entry);
    }
    this.byAuthor.set(c.author, cur);
    return entry;
  }

  /** Structural citation check; pure, so the oracle can re-run it independently of ledger state. */
  static verify(registry, c) {
    const ids = [];
    for (const id of c.citations || []) if (ids.indexOf(id) === -1) ids.push(id);
    if (ids.length === 0) return { ok: false, reason: "no-citation", distinct: 0 };
    if (!(c.weight > 0)) return { ok: false, reason: "non-positive-weight", distinct: ids.length };
    for (const id of ids) {
      const rec = registry.get(id);
      if (!rec) return { ok: false, reason: "unknown-citation", distinct: ids.length };
      if (rec.subject !== c.artifact) return { ok: false, reason: "subject-mismatch", distinct: ids.length };
      if (!COMPATIBLE[c.kind].includes(rec.kind)) {
        return { ok: false, reason: "kind-mismatch", distinct: ids.length };
      }
    }
    const bonus = Math.min(MAX_CITATION_BONUS, ids.length - 1);
    return { ok: true, reason: "accepted", distinct: ids.length,
      score: c.weight * KIND_WEIGHT[c.kind] + bonus };
  }

  scoreOf(author) { return (this.byAuthor.get(author) || { score: 0 }).score; }

  /** Deterministic ranking: score desc, verified count desc, then author name asc. */
  ranking() {
    return Array.from(this.byAuthor.entries())
      .map(([author, s]) => ({ author, score: s.score, accepted: s.accepted }))
      .filter((row) => row.accepted > 0)
      .sort((a, b) => b.score - a.score || b.accepted - a.accepted || (a.author < b.author ? -1 : 1));
  }

  /** Share of contributions whose citations verified, plus a per-reason histogram and ranking digest. */
  audit() {
    const reasons = new Map();
    for (const r of this.rejections) reasons.set(r.reason, (reasons.get(r.reason) || 0) + 1);
    return { total: this.contributions.length, accepted: this.contributions.length - this.rejections.length,
      verifiedRate: this.contributions.length === 0
        ? 1 : (this.contributions.length - this.rejections.length) / this.contributions.length,
      reasons: Object.fromEntries(Array.from(reasons.entries()).sort()),
      digest: fnv1a(JSON.stringify(this.ranking())) };
  }
}

/** Independent oracle: a naive second scoring pass with no shared cache with the ledger. */
function bruteScore(registry, contributions, author) {
  let total = 0;
  for (const c of contributions) {
    if (c.author !== author) continue;
    const verdict = RecognitionLedger.verify(registry, c);
    if (verdict.ok) total += verdict.score;
  }
  return total;
}

const SUBJECTS = ["qs-1", "qs-2", "qs-3"];
/** Seeded fixture: ~70% well-formed citations, the rest split over the four rejection reasons. */
function seededFixture(seed, count) {
  const rng = splitmix64(seed);
  const kinds = Object.keys(KIND_WEIGHT);
  const authors = ["ada", "brian", "chen", "dana", "erin", "farid"];
  const registry = new CitationRegistry();
  const contributions = [];
  let seq = 0;
  const mint = (kind, subject) => { const id = `c${seq}`; seq += 1; registry.register({ id, kind, subject }); return id; };
  for (let i = 0; i < count; i += 1) {
    const author = authors[rng.int(authors.length)];
    const kind = kinds[rng.int(kinds.length)];
    const artifact = SUBJECTS[rng.int(SUBJECTS.length)];
    const roll = rng.int(12);
    const citations = [];
    if (roll < 8) { // verified, occasionally with a repeated id that must dedup
      citations.push(mint(COMPATIBLE[kind][0], artifact));
      if (rng.int(2) === 1) citations[citations.length] = citations[0];
    } else if (roll < 9) { citations.push(`missing-${i}`); }
    else if (roll < 10) { citations.push(mint(COMPATIBLE[kind][0], "qs-999")); }
    else if (roll < 11) { citations.push(mint("review" === kind ? "issue" : "review", artifact)); }
    else { citations.push(mint("commit", artifact)); }
    contributions.push({ author, kind, artifact, weight: 1 + rng.int(5), citations });
  }
  return { registry, contributions, authors };
}

function main() {
  // Hand-written fixture: ada's duplicate citation earns the bonus, bob's single citation does not.
  const registry = new CitationRegistry();
  registry.register({ id: "c1", kind: "commit", subject: "qs-1" });
  registry.register({ id: "c2", kind: "commit", subject: "qs-1" });
  registry.register({ id: "r1", kind: "review", subject: "qs-2" });
  registry.register({ id: "i1", kind: "issue", subject: "qs-3" });

  const ledger = new RecognitionLedger(registry);
  const ada = ledger.add({ author: "ada", kind: "commit", artifact: "qs-1", weight: 4, citations: ["c1", "c2", "c1"] });
  const bob = ledger.add({ author: "bob", kind: "commit", artifact: "qs-1", weight: 4, citations: ["c1"] });
  const chen1 = ledger.add({ author: "chen", kind: "commit", artifact: "qs-1", weight: 4, citations: ["missing"] });
  ledger.add({ author: "chen", kind: "review", artifact: "qs-2", weight: 5, citations: ["r1"] });
  ledger.add({ author: "dana", kind: "issue", artifact: "qs-3", weight: 2, citations: ["i1"] });
  const erin = ledger.add({ author: "erin", kind: "review", artifact: "qs-1", weight: 3, citations: ["c1"] });
  const fay = ledger.add({ author: "fay", kind: "docs", artifact: "qs-1", weight: 0, citations: ["c1"] });
  const gus = ledger.add({ author: "gus", kind: "issue", artifact: "qs-3", weight: 2, citations: [] });

  // 1. Happy path: the multi-citation bonus separates ada from the otherwise identical bob.
  assert.strictEqual(ada.accepted, true, "ada accepted");
  assert.strictEqual(ada.distinctCitations, 2, "repeated citation ids collapse to 2 distinct");
  assert.strictEqual(ada.score, 4 * KIND_WEIGHT.commit + 1, "weighted score plus one bonus point");
  assert.strictEqual(bob.score, 4 * KIND_WEIGHT.commit, "one verified citation scores the base weight");
  assert.ok(ledger.scoreOf("ada") > ledger.scoreOf("bob"), "ada outranks bob");
  // 2. Every rejection reason is reported exactly, and none of them scores.
  assert.deepStrictEqual([chen1.reason, erin.reason, fay.reason, gus.reason],
    ["unknown-citation", "kind-mismatch", "non-positive-weight", "no-citation"], "rejection reasons");
  assert.strictEqual(ledger.scoreOf("chen"), 5 * KIND_WEIGHT.review, "only chen's review counts");
  assert.deepStrictEqual([ledger.scoreOf("erin"), ledger.scoreOf("fay"), ledger.scoreOf("gus")], [0, 0, 0],
    "rejected contributions contribute nothing");
  // 3. Boundary: an author with no contributions scores 0 and is absent from the ranking; an empty
  // ledger is a valid degenerate state whose audit is vacuously verified.
  assert.strictEqual(ledger.scoreOf("grace"), 0, "unknown author scores zero");
  assert.strictEqual(ledger.ranking().some((r) => r.author === "grace"), false, "unknown author not ranked");
  assert.strictEqual(ledger.ranking()[0].author, "ada", "ada leads the ranking");
  const empty = new RecognitionLedger(registry);
  assert.deepStrictEqual(empty.ranking(), [], "empty ledger ranks nobody");
  assert.strictEqual(empty.audit().verifiedRate, 1, "empty audit is fully verified");
  // 4. Audit histogram and rate agree with the eight contributions added above.
  const audit = ledger.audit();
  assert.strictEqual(audit.total, 8, "eight contributions recorded");
  assert.strictEqual(audit.accepted, 4, "four accepted");
  assert.deepStrictEqual(audit.reasons,
    { "kind-mismatch": 1, "no-citation": 1, "non-positive-weight": 1, "unknown-citation": 1 }, "reason histogram");
  assert.strictEqual(audit.verifiedRate, 4 / 8, "verified rate is accepted over total");
  // 5. Oracle: on the seeded fixture the index must equal a brute-force recompute for every author.
  const { registry: reg2, contributions, authors } = seededFixture(0x985, 60);
  const ledger2 = new RecognitionLedger(reg2);
  for (const c of contributions) ledger2.add(c);
  for (const a of authors) {
    assert.strictEqual(ledger2.scoreOf(a), bruteScore(reg2, contributions, a), `oracle score for ${a}`);
  }
  // 6. Invariant after mutations: the running index equals a recompute of the processed prefix.
  const inc = new RecognitionLedger(reg2);
  for (let i = 0; i < contributions.length; i += 1) {
    inc.add(contributions[i]);
    if (i % 13 === 0 || i === contributions.length - 1) {
      for (const a of authors) {
        assert.strictEqual(inc.scoreOf(a), bruteScore(reg2, contributions.slice(0, i + 1), a),
          `prefix recompute mismatch for ${a} at ${i}`);
      }
    }
  }
  // 7. Insertion order must not change the ranking (ties fall back to the author name).
  const reversed = new RecognitionLedger(reg2);
  for (const c of contributions.slice().reverse()) reversed.add(c);
  assert.deepStrictEqual(reversed.ranking(), inc.ranking(), "ranking is insertion-order independent");
  assert.strictEqual(reversed.audit().digest, inc.audit().digest, "ranking digest is stable");
  // 8. The fixture is discriminating: it must contain both outcomes.
  assert.ok(ledger2.audit().accepted > 0 && ledger2.audit().accepted < ledger2.audit().total,
    "fixture exercises acceptances and rejections");

  console.log(`fixture accepted=${ledger2.audit().accepted}/${ledger2.audit().total} ` +
    `ranked=${ledger2.ranking().length} digest=${ledger2.audit().digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = { splitmix64, fnv1a, CitationRegistry, RecognitionLedger, bruteScore, seededFixture,
  KIND_WEIGHT, COMPATIBLE };
