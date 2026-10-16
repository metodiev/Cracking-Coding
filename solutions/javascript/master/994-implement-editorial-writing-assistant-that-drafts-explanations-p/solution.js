/** Task 994 - Implement editorial writing assistant that drafts explanations per solution.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Turns the structured metadata of a solution (title, approach, complexity, pitfalls, tags) into an
 * editorial draft with a fixed section skeleton. The phrasing of the Approach section is chosen from
 * a small table keyed by the task's tags, and everything is assembled by pure string joining, so the
 * same metadata yields a byte-identical draft. A reviewer pass then enforces the house rules -- the
 * required sections, a word budget, a complexity claim and a ban on hand-waving phrasing -- and a
 * trimmer can shorten a draft without ever dropping a heading.
 *
 * What is simplified: prose comes from templates rather than a language model, which is what makes
 * the output deterministic and checkable. Verified: section skeleton, tag-driven phrasing, placeholder
 * handling for missing metadata, reviewer findings, budget trimming that preserves headings, a word
 * counter cross-checked by a natural-language-free oracle, and draft reproducibility.
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

const REQUIRED = ["Overview", "Approach", "Complexity", "Pitfalls", "Testing"];
const PLACEHOLDER = "[to be filled in by the editor]";
const BANNED = ["obviously", "trivially", "just use", "everyone knows"];
const DEFAULT_BUDGET = 220;

/** Tag-driven opening sentence: the first matching tag wins, otherwise a generic fallback. */
const APPROACH_PHRASES = [
  { tag: "hash-map", text: "Index the input in a hash map so each lookup is constant time, then sweep once." },
  { tag: "dp", text: "Define the subproblem state, order the states so dependencies come first, and fill them bottom up." },
  { tag: "graph", text: "Model the input as a graph and traverse it, marking nodes as they are consumed." },
  { tag: "two-pointer", text: "Move two indices inward from the ends, shrinking the range on the side that cannot improve." },
  { tag: "binary-search", text: "Binary search the answer space instead of the array, using a monotone predicate." },
];

const approachText = (meta) => {
  const hit = APPROACH_PHRASES.find((p) => (meta.tags || []).includes(p.tag));
  const opening = hit ? hit.text : "Reduce the problem to a smaller instance and recurse on the difference.";
  const notes = (meta.notes || []).length > 0 ? ` ${meta.notes.join(" ")}` : "";
  return `${opening}${notes}`;
};
const complexityText = (meta) => {
  const time = meta.complexity && meta.complexity.time ? meta.complexity.time : null;
  const space = meta.complexity && meta.complexity.space ? meta.complexity.space : null;
  if (!time && !space) return `Time complexity ${PLACEHOLDER} Space complexity ${PLACEHOLDER}`;
  return `Time complexity ${time || PLACEHOLDER} Space complexity ${space || PLACEHOLDER}.`;
};
const pitfallsText = (meta) => ((meta.pitfalls || []).length === 0
  ? `No known pitfalls yet. ${PLACEHOLDER}`
  : (meta.pitfalls || []).map((p, i) => `${i + 1}. ${p}`).join(" "));
const testingText = (meta) => ((meta.cases || []).length === 0
  ? "Exercise the boundary cases: empty input, a single element and the largest allowed size."
  : `Cover ${(meta.cases || []).length} documented cases, starting with ${JSON.stringify((meta.cases || [])[0])} `
    + "and finishing with the largest allowed size.");

/** Assemble the draft; the section order is fixed by REQUIRED, not by the metadata's key order. */
function draft(meta) {
  const title = meta.title || "Untitled solution";
  const bodies = { Overview: `${title} is solved as follows.`, Approach: approachText(meta),
    Complexity: complexityText(meta), Pitfalls: pitfallsText(meta), Testing: testingText(meta) };
  const sections = REQUIRED.map((heading) => ({ heading, body: bodies[heading] }));
  const text = sections.map((s) => `## ${s.heading}\n\n${s.body}`).join("\n\n");
  return { title, sections, text, words: countWords(text), digest: fnv1a(text) };
}

const countWords = (text) => text.split(/\s+/).filter((w) => w.replace(/[^\w]/g, "").length > 0).length;

/** Independent oracle for the word count: manual whitespace tokenising, no split and no regex. */
function bruteWordCount(text) {
  const isWordChar = (ch) => (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z")
    || (ch >= "0" && ch <= "9") || ch === "_";
  const isSpace = (ch) => ch === " " || ch === "\n" || ch === "\t" || ch === "\r";
  let words = 0, token = "";
  const flush = () => {
    for (const ch of token) if (isWordChar(ch)) { words += 1; break; }
    token = "";
  };
  for (const ch of text) { if (isSpace(ch)) flush(); else token += ch; }
  flush();
  return words;
}

/** Reviewer pass: the house rules a draft must satisfy before it can be published. */
function review(draftResult, { budget = DEFAULT_BUDGET } = {}) {
  const findings = [];
  const headings = draftResult.sections.map((s) => s.heading);
  for (const required of REQUIRED) {
    if (!headings.includes(required)) findings.push({ section: required, reason: "missing-section" });
  }
  if (headings.join("|") !== REQUIRED.join("|")) findings.push({ section: "order", reason: "section-order" });
  const complexity = draftResult.sections.find((s) => s.heading === "Complexity");
  if (!/O\(/.test(complexity.body)) findings.push({ section: "Complexity", reason: "missing-complexity-claim" });
  if (draftResult.words > budget) findings.push({ section: "all", reason: "over-budget" });
  const lower = draftResult.text.toLowerCase();
  for (const phrase of BANNED) if (lower.includes(phrase)) findings.push({ section: "all", reason: `banned-phrase:${phrase}` });
  return { ok: findings.length === 0, findings };
}

/** Trim to a word budget by shortening bodies round-robin; headings are never dropped. */
function trimToBudget(draftResult, budget) {
  const sections = draftResult.sections.map((s) => ({ heading: s.heading, words: s.body.split(/\s+/) }));
  const total = () => sections.reduce((a, s) => a + s.words.length, 0) + sections.length * 2;
  let cursor = 0;
  while (total() > budget) {
    const section = sections[cursor % sections.length];
    if (section.words.length > 1) section.words.pop();
    cursor += 1;
    if (sections.every((s) => s.words.length <= 1)) break;
  }
  const rebuilt = sections.map((s) => ({ heading: s.heading, body: s.words.join(" ") }));
  const text = rebuilt.map((s) => `## ${s.heading}\n\n${s.body}`).join("\n\n");
  return { ...draftResult, sections: rebuilt, text, words: countWords(text), digest: fnv1a(text) };
}

const TAGS = ["hash-map", "dp", "graph", "two-pointer", "binary-search", "unknown-tag"];
/** Seeded metadata fixtures covering every tag and both empty and populated pitfalls. */
function seededMeta(seed, count) {
  const rng = splitmix64(seed);
  return Array.from({ length: count }, (_, i) => ({ title: `Task ${i}: Solve It`,
    tags: [TAGS[rng.int(TAGS.length)]],
    complexity: rng.int(5) === 0 ? null : { time: `O(n^${1 + rng.int(3)})`, space: rng.int(2) ? "O(1)" : "O(n)" },
    pitfalls: Array.from({ length: rng.int(3) }, (_, k) => `Watch out for case ${k} of task ${i}.`),
    cases: Array.from({ length: rng.int(3) }, (_, k) => [k, k * k]) }));
}

function main() {
  const meta = { title: "Two Sum", tags: ["hash-map"], complexity: { time: "O(n)", space: "O(n)" },
    pitfalls: ["Do not reuse an index twice.", "Handle duplicate values."], cases: [[[2, 7], 9]] };
  const result = draft(meta);
  // 1. Happy path: the fixed skeleton is present in order, with the tag-driven approach sentence.
  assert.deepStrictEqual(result.sections.map((s) => s.heading), REQUIRED, "all required sections in order");
  assert.ok(result.text.includes("Index the input in a hash map"), "hash-map tag picks its phrasing");
  assert.ok(result.text.includes("Time complexity O(n)") && result.text.includes("Space complexity O(n)"),
    "complexity claims are rendered");
  assert.strictEqual(review(result).ok, true, "a well-formed draft passes review");
  // 2. Tag-driven phrasing: each known tag yields a distinct approach, unknown tags fall back.
  const approaches = TAGS.map((tag) => draft({ ...meta, tags: [tag] }).sections.find((s) => s.heading === "Approach").body);
  assert.strictEqual(new Set(approaches).size, TAGS.length, "every tag produces its own approach text");
  assert.ok(approaches[TAGS.length - 1].includes("Reduce the problem to a smaller instance"), "unknown tag falls back");
  // 3. Boundary: metadata with nothing but a title still yields every section with placeholders.
  const bare = draft({});
  assert.deepStrictEqual([bare.sections.map((s) => s.heading), bare.text.includes(PLACEHOLDER),
    review(bare).findings.map((f) => f.reason)], [REQUIRED, true, ["missing-complexity-claim"]],
    "empty metadata degrades to placeholders");
  // 4. Reviewer: hand-waving phrasing and an over-long draft are both refused, with reasons.
  const sloppy = review(draft({ ...meta, notes: ["This is obviously trivial."] }));
  const long = draft({ ...meta, notes: Array.from({ length: 60 }, () => "padding words here") });
  assert.deepStrictEqual([sloppy.ok, sloppy.findings.map((f) => f.reason)], [false, ["banned-phrase:obviously"]],
    "banned phrasing is flagged");
  assert.deepStrictEqual([review(long).ok, review(long).findings[0].reason], [false, "over-budget"],
    "over-budget draft is flagged");
  // 5. Trimming: the budget is met without dropping a single heading.
  const trimmed = trimToBudget(long, 120);
  assert.deepStrictEqual([trimmed.words <= 120, trimmed.sections.map((s) => s.heading)], [true, REQUIRED],
    "trimming respects the budget and keeps every heading");
  assert.strictEqual(trimToBudget(result, 1000).digest, result.digest, "a draft inside budget is unchanged");
  // 6. Oracle: the word counter agrees with an independent character-scan implementation on every
  // generated draft.
  for (const fixture of seededMeta(0x994, 12)) {
    const generated = draft(fixture);
    assert.strictEqual(generated.words, bruteWordCount(generated.text), `word count agrees for ${fixture.title}`);
  }
  // 7. Determinism: drafts are byte-identical on a rebuild and independent of the metadata key order.
  const records = seededMeta(0x994, 20);
  const digests = records.map((m) => draft(m).digest);
  assert.deepStrictEqual(records.map((m) => draft({ ...m }).digest), digests, "drafts reproduce byte for byte");
  const reordered = records.map((m) => ({ cases: m.cases, pitfalls: m.pitfalls, complexity: m.complexity,
    tags: m.tags, title: m.title }));
  assert.deepStrictEqual(reordered.map((m) => draft(m).digest), digests, "metadata key order is irrelevant");
  // 8. Distinctness and sanity: different metadata gives different drafts, and each draft passes or
  // fails review deterministically.
  assert.ok(new Set(digests).size > digests.length / 2, "drafts are mostly distinct");
  const verdicts = records.map((m) => review(draft(m)).ok);
  assert.deepStrictEqual(records.map((m) => review(draft(m)).ok), verdicts, "review verdicts are stable");
  console.log(`sections=${result.sections.length} words=${result.words} trimmed=${trimmed.words} digest=${result.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, draft, countWords, bruteWordCount, review, trimToBudget, seededMeta,
  REQUIRED, BANNED };
