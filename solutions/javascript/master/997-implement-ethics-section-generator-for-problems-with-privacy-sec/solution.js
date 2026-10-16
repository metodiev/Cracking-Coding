/** Task 997 - Implement ethics section generator for problems with privacy/security implications.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Classifies a problem's tags into privacy/security implications, scores them, and emits the ethics
 * section the problem owes its learners. Each implication signal carries a weight and the note it
 * mandates, so the required subsections are derived from the signals rather than hand-picked, and the
 * risk score is just the sum of the weights of the signals present. A compliance pass then checks the
 * declared data handling -- consent for personal data, a bounded retention period, an encryption
 * statement where cryptography is involved -- and escalates anything that does not add up.
 *
 * What is simplified: the policy table is a small curated list instead of a legal framework.
 * Verified: signal-driven subsection completeness, risk levels at their exact thresholds, the consent
 * and retention gates in both directions, escalation of high risk, an independent recomputation of
 * the score and the findings, and byte-stable sections under tag reordering.
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

/** One row per implication: the weight it contributes and the subsection it makes mandatory. */
const SIGNALS = [
  { tag: "privacy", weight: 2, note: "data-minimisation" },
  { tag: "pii", weight: 3, note: "consent" },
  { tag: "security", weight: 2, note: "security-controls" },
  { tag: "auth", weight: 1, note: "security-controls" },
  { tag: "telemetry", weight: 1, note: "retention" },
  { tag: "crypto", weight: 1, note: "key-management" },
];
const NOTE_TITLES = { "data-minimisation": "Data minimisation", consent: "Consent and lawful basis",
  "security-controls": "Security controls", retention: "Retention limits", "key-management": "Key management" };
const LEVELS = [{ level: "high", at: 5 }, { level: "medium", at: 2 }, { level: "low", at: 0 }];
const MAX_RETENTION_DAYS = 400;
const NOTES = {
  "data-minimisation": "Collect only the fields the exercise needs; justify every extra field in the statement.",
  consent: "Personal data may only be processed when the learner has given explicit consent or a lawful basis applies.",
  "security-controls": "State the threat model and the controls (input validation, least privilege, rate limiting) that answer it.",
  retention: "Declare how long the data lives and what deletes it; keep the window as short as the exercise allows.",
  "key-management": "Never embed keys in the problem; describe rotation and where secrets are expected to live.",
  none: "This problem carries no privacy or security implications; no learner data leaves the sandbox.",
};

/** Signals present in the tag list, ordered by the policy table so the output is tag-order independent. */
const signalsOf = (tags) => SIGNALS.filter((s) => (tags || []).includes(s.tag));
const riskScore = (tags) => signalsOf(tags).reduce((a, s) => a + s.weight, 0);
const riskLevel = (score) => LEVELS.find((l) => score >= l.at).level;

/** Compliance findings for the declared data handling, each tied to the rule that produced it. */
function validateHandling(problem) {
  const tags = problem.tags || [];
  const findings = [];
  if (tags.includes("pii") && problem.consent !== true) {
    findings.push({ rule: "consent", reason: "personal data without a declared lawful basis" });
  }
  if (tags.includes("pii") && !(problem.retentionDays > 0)) {
    findings.push({ rule: "retention", reason: "personal data without a retention window" });
  }
  if (problem.retentionDays > MAX_RETENTION_DAYS) {
    findings.push({ rule: "retention", reason: `retention ${problem.retentionDays}d exceeds the ${MAX_RETENTION_DAYS}d policy` });
  }
  if ((tags.includes("security") || tags.includes("crypto")) && problem.encrypted !== true) {
    findings.push({ rule: "encryption", reason: "cryptography is implied but no protection is stated" });
  }
  return findings;
}

/**
 * Generate the ethics section. Subsection order is fixed (Scope, then policy-table order, then
 * Escalation), so the section is a pure function of the problem's signals and findings.
 */
function ethicsSection(problem) {
  const signals = signalsOf(problem.tags);
  const score = signals.reduce((a, s) => a + s.weight, 0);
  const level = riskLevel(score);
  const findings = validateHandling(problem);
  const notes = signals.length === 0 ? ["none"] : signals.map((s) => s.note);
  const subsections = [{ heading: "Scope", body: `${problem.title || "Untitled problem"} touches ` +
    `${signals.length === 0 ? "no" : signals.map((s) => s.tag).join(", ")} implication(s); risk ${level} (score ${score}).` }];
  for (const note of notes) {
    subsections.push({ heading: NOTE_TITLES[note] || "Other duties", body: NOTES[note] });
  }
  const needsReview = level === "high" || findings.length > 0;
  if (needsReview) {
    subsections.push({ heading: "Escalation", body: `Send to the ethics reviewer: ${findings.length > 0
      ? findings.map((f) => f.reason).join("; ") : "risk score above the auto-publish threshold"}.` });
  }
  const text = subsections.map((s) => `### ${s.heading}\n${s.body}`).join("\n");
  return { riskScore: score, riskLevel: level, findings, needsReview, subsections, text,
    digest: fnv1a(text) };
}

/** Independent oracle: recompute score and required notes with an explicit double loop. */
function bruteExpectations(problem) {
  const tags = problem.tags || [];
  let score = 0;
  const required = [];
  for (const signal of SIGNALS) {
    let present = false;
    for (const tag of tags) if (tag === signal.tag) present = true;
    if (!present) continue;
    score += signal.weight;
    if (required.indexOf(signal.note) === -1) required.push(signal.note);
  }
  return { score, required: required.length === 0 ? ["none"] : required };
}

const TAGS = ["privacy", "pii", "security", "auth", "telemetry", "crypto", "arrays", "dp"];
/** Seeded problems spanning every signal combination the generator can be asked about. */
function seededProblems(seed, count) {
  const rng = splitmix64(seed);
  return Array.from({ length: count }, (_, i) => {
    const tags = TAGS.filter(() => rng.int(3) === 0);
    const hasPii = tags.includes("pii");
    return { id: `p${i}`, title: `Ethics ${i}`, tags,
      consent: hasPii ? rng.int(2) === 0 : undefined,
      retentionDays: hasPii ? 30 * (1 + rng.int(20)) : undefined,
      encrypted: tags.includes("security") || tags.includes("crypto") ? rng.int(2) === 0 : undefined };
  });
}

function main() {
  const pii = { title: "User Profile Store", tags: ["pii", "security"], consent: true, retentionDays: 90, encrypted: true };
  const section = ethicsSection(pii);
  // 1. Happy path: the mandatory subsections come from the signals, in policy-table order.
  assert.deepStrictEqual(section.subsections.map((s) => s.heading),
    ["Scope", "Consent and lawful basis", "Security controls", "Escalation"],
    "signal-driven subsections in policy order, escalation last");
  const medium = ethicsSection({ title: "Telemetry Only", tags: ["privacy", "telemetry"] });
  assert.deepStrictEqual([medium.subsections.map((s) => s.heading), medium.needsReview],
    [["Scope", "Data minimisation", "Retention limits"], false], "medium risk with clean handling does not escalate");
  assert.deepStrictEqual([section.riskScore, section.riskLevel, section.needsReview, section.findings],
    [5, "high", true, []], "clean high-risk problem needs review but has no findings");
  // 2. Risk thresholds are inclusive at the boundary value.
  assert.deepStrictEqual([riskLevel(4), riskLevel(5), riskLevel(2), riskLevel(1), riskLevel(0)],
    ["medium", "high", "medium", "low", "low"], "levels switch exactly at their thresholds");
  assert.deepStrictEqual([riskScore(["pii"]), riskScore(["privacy", "pii"]), riskScore(["arrays"])], [3, 5, 0],
    "score is the sum of the present weights");
  // 3. Boundary: a problem with no implications gets the minimal section and never escalates.
  const none = ethicsSection({ title: "Binary Search", tags: ["arrays", "binary-search"] });
  assert.deepStrictEqual([none.subsections.map((s) => s.heading), none.riskScore, none.needsReview, none.findings],
    [["Scope", "Other duties"], 0, false, []], "no implications means no duties and no escalation");
  const untagged = ethicsSection({ title: "Empty Tags", tags: [] });
  assert.deepStrictEqual([untagged.subsections.length, untagged.text.includes("no privacy or security implications")],
    [2, true], "an untagged problem falls back to the none-note");
  // 4. Consent gate in both directions: personal data without a lawful basis is escalated.
  const noConsent = ethicsSection({ title: "Leaderboard", tags: ["pii"], consent: false, retentionDays: 30 });
  const withConsent = ethicsSection({ title: "Leaderboard", tags: ["pii"], consent: true, retentionDays: 30 });
  assert.deepStrictEqual([noConsent.needsReview, withConsent.needsReview,
    noConsent.findings.map((f) => f.rule)], [true, false, ["consent"]], "consent gate is two-sided");
  // 5. Retention gate: an undeclared window and an over-long window are both findings, the policy
  // maximum itself is not.
  assert.deepStrictEqual([ethicsSection({ title: "R", tags: ["pii"], consent: true }).findings.map((f) => f.rule),
    ethicsSection({ title: "R", tags: ["pii"], consent: true, retentionDays: MAX_RETENTION_DAYS }).findings.length,
    ethicsSection({ title: "R", tags: ["telemetry"], retentionDays: MAX_RETENTION_DAYS + 1 }).findings.map((f) => f.rule)],
    [["retention"], 0, ["retention"]], "retention rules at and beyond the boundary");
  // 6. Encryption finding: cryptography implied without a stated protection is flagged.
  assert.deepStrictEqual([ethicsSection({ title: "C", tags: ["crypto"] }).findings.map((f) => f.reason),
    ethicsSection({ title: "C", tags: ["crypto"], encrypted: true }).findings.length],
    [["cryptography is implied but no protection is stated"], 0], "encryption rule fires only when unstated");
  // 7. Oracle: subsections and score match an independent double-loop recomputation, and the count of
  // mandated notes equals the number of distinct signals.
  for (const problem of seededProblems(0x997, 20)) {
    const generated = ethicsSection(problem);
    const expected = bruteExpectations(problem);
    assert.strictEqual(generated.riskScore, expected.score, `score agrees for ${problem.id}`);
    assert.deepStrictEqual(generated.subsections.slice(1, 1 + expected.required.length).map((s) => s.heading),
      expected.required.map((n) => NOTE_TITLES[n] || "Other duties"), `required notes agree for ${problem.id}`);
    assert.strictEqual(generated.findings.length, validateHandling(problem).length, "findings are complete");
    assert.strictEqual(generated.needsReview, generated.riskLevel === "high" || generated.findings.length > 0,
      "escalation rule is consistent");
  }
  // 8. Determinism: tag order is irrelevant and the section reproduces byte for byte.
  const reordered = ethicsSection({ ...pii, tags: ["security", "pii"] });
  assert.deepStrictEqual([reordered.digest, ethicsSection(pii).digest], [section.digest, section.digest],
    "tag order does not change the section");
  const fixtures = seededProblems(0x997, 20);
  assert.deepStrictEqual(fixtures.map((p) => ethicsSection(p).digest), fixtures.map((p) => ethicsSection({ ...p }).digest),
    "sections reproduce byte for byte");
  console.log(`subsections=${section.subsections.length} risk=${section.riskScore}/${section.riskLevel} ` +
    `escalations=${fixtures.filter((p) => ethicsSection(p).needsReview).length} digest=${section.digest.toString(16)}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { splitmix64, fnv1a, signalsOf, riskScore, riskLevel, validateHandling, ethicsSection,
  bruteExpectations, seededProblems, SIGNALS, MAX_RETENTION_DAYS };
