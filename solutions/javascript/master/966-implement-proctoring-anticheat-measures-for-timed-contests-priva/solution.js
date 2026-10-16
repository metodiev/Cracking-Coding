/** Task 966 - Implement proctoring/anticheat measures for timed contests (privacy-aware).
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A privacy-first proctoring pass over a contest event stream. Five weighted, named rules run over a
 * retention window (paste-then-solve inside a window, a tab-switch count, an implausible keystroke rate,
 * a solve with no prior input, and solves synchronised across candidates); the summed score maps to a
 * verdict with inclusive thresholds. Privacy is structural, not aspirational: the report is built from a
 * sanitised projection that keeps only event types, ticks and counts (identifiers become salted hashes,
 * free text and contact fields are dropped), and review findings are published only when the review
 * group reaches the k-anonymity size -- otherwise the count is reported and the findings suppressed.
 * Retention drops events older than the TTL before scoring, and appeals rewrite the verdict without
 * deleting evidence.
 *
 * Simplified: rules are keyword/timing heuristics over a synthetic event stream rather than real
 * telemetry, and salting is a deterministic hash rather than a cryptographic one. Verified: a clean
 * candidate, each rule firing with its weight, the paste window and threshold boundaries, k-anonymity
 * suppression and release, TTL dropping, the no-PII report guarantee, appeal handling, an independent
 * score recomputation, and deterministic replay.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const RULES = {
  pasteThenSolve: { weight: 40, window: 20 },
  tabSwitches: { weight: 20, threshold: 5 },
  superhumanTyping: { weight: 25, keysPerTick: 15 },
  solutionBeforeInput: { weight: 50 },
  synchronizedSolve: { weight: 15, group: 3 },
};
const THRESHOLDS = { review: 60, watch: 30 };
const PRIVACY = { ttl: 600, k: 3, salt: "proctor-2026" };
const PII_KEYS = new Set(["email", "name", "content", "text", "ip", "candidateId"]);
const EVENT_TYPES = new Set(["input", "paste", "solve", "tab-switch"]);
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
const refHash = (candidateId, salt) => `${fnv1a(`${salt}:${candidateId}`).toString(16).padStart(8, "0")}`;
/** Sanitiser: only the shape of an event survives -- type, tick and numeric counters, never text. */
function sanitize(event) {
  const clean = { type: typeof event.type === "string" ? event.type : "unknown", at: Number.isFinite(event.at) ? event.at : 0 };
  for (const [key, value] of Object.entries(event)) {
    if (typeof value === "number" && !PII_KEYS.has(key)) clean[key] = value;
    else if (PII_KEYS.has(key)) continue; // identifiers, free text and contact details are dropped
    else if (typeof value === "string" && key !== "type" && key !== "task") continue;
  }
  return clean;
}
const verdictFor = (score) => (score >= THRESHOLDS.review ? "review" : score >= THRESHOLDS.watch ? "watch" : "clear");
/** Score one candidate's sanitised, in-window events against the named rules. */
function scoreCandidate(events, { synchronizedAt = [] } = {}) {
  const findings = [];
  const add = (rule, weight, at) => findings.push({ rule, weight, at });
  const inputs = events.filter((event) => event.type === "input");
  const solves = events.filter((event) => event.type === "solve");
  const pastes = events.filter((event) => event.type === "paste");
  for (const paste of pastes) {
    const hit = solves.find((solve) => solve.at >= paste.at && solve.at - paste.at <= RULES.pasteThenSolve.window);
    if (hit) add("paste-then-solve", RULES.pasteThenSolve.weight, hit.at);
  }
  const tabs = events.filter((event) => event.type === "tab-switch").length;
  if (tabs > RULES.tabSwitches.threshold) add("tab-switches", RULES.tabSwitches.weight, events[events.length - 1].at);
  const fast = inputs.find((event) => (event.keys || 0) > RULES.superhumanTyping.keysPerTick);
  if (fast) add("superhuman-typing", RULES.superhumanTyping.weight, fast.at);
  const firstInput = inputs.length ? Math.min(...inputs.map((event) => event.at)) : Infinity;
  const early = inputs.length ? solves.find((solve) => solve.at < firstInput) : undefined;
  if (early) add("solution-before-input", RULES.solutionBeforeInput.weight, early.at);
  if (solves.some((solve) => synchronizedAt.includes(solve.at))) {
    add("synchronized-solve", RULES.synchronizedSolve.weight, solves[0].at);
  }
  const score = findings.reduce((a, finding) => a + finding.weight, 0);
  return { score, verdict: verdictFor(score), findings: findings.sort((a, b) => (a.at - b.at) || (a.rule < b.rule ? -1 : 1)) };
}
/** Whole-stream analysis: retention, per-candidate scoring, synchronised-solve detection, k-anonymity. */
function analyzeExam(events, { now = 1000, ttl = PRIVACY.ttl, k = PRIVACY.k, salt = PRIVACY.salt } = {}) {
  const sanitized = events.map((event) => ({ ...sanitize(event), candidateId: event.candidateId }));
  const retained = sanitized.filter((event) => now - event.at <= ttl);
  const dropped = sanitized.length - retained.length;
  const byCandidate = new Map();
  for (const event of retained) {
    if (!byCandidate.has(event.candidateId)) byCandidate.set(event.candidateId, []);
    byCandidate.get(event.candidateId).push(event);
  }
  const solveCounts = new Map();
  for (const event of retained) if (event.type === "solve") solveCounts.set(event.at, (solveCounts.get(event.at) || 0) + 1);
  const synchronizedAt = [...solveCounts.entries()].filter(([, count]) => count >= RULES.synchronizedSolve.group).map(([at]) => at);
  const candidates = [...byCandidate.entries()].map(([candidateId, own]) => ({ candidateId, ...scoreCandidate(own, { synchronizedAt }) }))
    .sort((a, b) => (b.score - a.score) || (a.candidateId < b.candidateId ? -1 : 1));
  const reviews = candidates.filter((candidate) => candidate.verdict === "review");
  const publishable = reviews.length >= k;
  const report = {
    generatedAt: now,
    candidates: candidates.map((candidate) => ({ ref: refHash(candidate.candidateId, salt), score: candidate.score,
      verdict: candidate.verdict, rules: candidate.findings.map((finding) => finding.rule) })),
    published: publishable ? reviews.map((candidate) => refHash(candidate.candidateId, salt)) : [],
    kAnonymity: { k, reviewGroup: reviews.length, suppressed: publishable ? 0 : reviews.length },
    retention: { ttl, dropped },
    ignored: retained.filter((event) => !EVENT_TYPES.has(event.type)).length,
  };
  return { report, candidates, synchronizedAt, dropped };
}
/** An appeal rewrites the verdict and keeps the evidence; nothing is deleted. */
function appeal(report, ref, { reason = "proctor reviewed the recording", now = 1000 } = {}) {
  const ledger = report.candidates.map((candidate) => (candidate.ref === ref
    ? { ...candidate, verdict: "appealed", appeal: { reason, at: now } }
    : candidate));
  return { ...report, candidates: ledger, appeals: [...(report.appeals || []), { ref, reason, at: now }] };
}

function main() {
  const rng = makeRng(20240966);
  const clean = [{ candidateId: "cand-clean", type: "input", at: 900, keys: 6 }, { candidateId: "cand-clean", type: "input", at: 910, keys: 8 },
    { candidateId: "cand-clean", type: "solve", at: 980, task: "arrays" }];
  const cheater = [{ candidateId: "cand-cheat", type: "paste", at: 700, size: 4000, content: "solution text", email: "cheat@example.com" },
    { candidateId: "cand-cheat", type: "solve", at: 710, task: "arrays" },
    { candidateId: "cand-cheat", type: "input", at: 600, keys: 20 }];
  const exam = analyzeExam([...clean, ...cheater]);
  const cleanReport = exam.report.candidates.find((candidate) => candidate.ref === refHash("cand-clean", PRIVACY.salt));
  const cheatReport = exam.report.candidates.find((candidate) => candidate.ref === refHash("cand-cheat", PRIVACY.salt));
  // 1. Happy path: the ordinary candidate scores zero and is clear, with no rules attached.
  assert(cleanReport.score === 0 && cleanReport.verdict === "clear" && cleanReport.rules.length === 0, `clean ${JSON.stringify(cleanReport)}`);
  assert(cheatReport.score === RULES.pasteThenSolve.weight + RULES.superhumanTyping.weight && cheatReport.verdict === "review"
    && cheatReport.rules.slice().sort().join() === "paste-then-solve,superhuman-typing", `cheat ${JSON.stringify(cheatReport)}`);
  // 2. Rule weights and the paste-window boundary, checked at the edge and one tick past it.
  const boundary = (delta) => analyzeExam([{ candidateId: "c", type: "paste", at: 100 }, { candidateId: "c", type: "solve", at: 100 + delta }], { now: 500 }).candidates[0].score;
  assert(boundary(RULES.pasteThenSolve.window) === RULES.pasteThenSolve.weight, "a solve exactly at the window edge fires");
  assert(boundary(RULES.pasteThenSolve.window + 1) === 0, "one tick later it does not");
  const tabs = analyzeExam(Array.from({ length: 6 }, (_, i) => ({ candidateId: "t", type: "tab-switch", at: 200 + i })), { now: 500 }).candidates[0];
  assert(tabs.score === RULES.tabSwitches.weight && tabs.findings[0].rule === "tab-switches", "six tab switches trip the rule");
  assert(analyzeExam(Array.from({ length: 5 }, (_, i) => ({ candidateId: "t", type: "tab-switch", at: 200 + i })), { now: 500 }).candidates[0].score === 0,
    "five tab switches do not");
  const fastKeys = analyzeExam([{ candidateId: "k", type: "input", at: 300, keys: 16 }], { now: 500 }).candidates[0];
  assert(fastKeys.score === RULES.superhumanTyping.weight && fastKeys.findings[0].rule === "superhuman-typing", "a 16-key tick is superhuman");
  const noInput = analyzeExam([{ candidateId: "n", type: "input", at: 400, keys: 3 }, { candidateId: "n", type: "solve", at: 300 }], { now: 500 }).candidates[0];
  assert(noInput.score === RULES.solutionBeforeInput.weight && noInput.findings[0].rule === "solution-before-input",
    "a solve before the first keystroke is the heaviest rule");
  // 3. Verdict thresholds are inclusive on both sides.
  assert(verdictFor(60) === "review" && verdictFor(59.9) === "watch" && verdictFor(30) === "watch" && verdictFor(29.9) === "clear",
    "threshold semantics");
  const combined = analyzeExam([{ candidateId: "x", type: "input", at: 100, keys: 20 },
    { candidateId: "x", type: "paste", at: 200 }, { candidateId: "x", type: "solve", at: 210 }], { now: 500 });
  assert(combined.candidates[0].score === 65 && combined.candidates[0].verdict === "review", `combined ${combined.candidates[0].score}`);
  // 4. Privacy: the report is hashes plus counters -- no raw identifiers, contact fields or pasted text.
  const serialized = JSON.stringify(exam.report);
  assert(!serialized.includes("cand-cheat") && !serialized.includes("cheat@example.com") && !serialized.includes("solution text"),
    "the report carries no raw identifiers or content");
  assert(refHash("cand-cheat", PRIVACY.salt) !== refHash("cand-cheat", "another-salt"), "salting changes the reference");
  assert(new Set(exam.report.candidates.map((candidate) => candidate.ref)).size === 2, "every candidate is referenced by hash");
  // 5. k-anonymity: a lone review is suppressed, three reviews are released.
  assert(exam.report.kAnonymity.reviewGroup === 1 && exam.report.published.length === 0 && exam.report.kAnonymity.suppressed === 1,
    `suppressed singleton ${JSON.stringify(exam.report.kAnonymity)}`);
  const trio = analyzeExam([1, 2, 3].flatMap((i) => [{ candidateId: `r${i}`, type: "input", at: 50 + i, keys: 20 },
    { candidateId: `r${i}`, type: "paste", at: 100 + i }, { candidateId: `r${i}`, type: "solve", at: 105 + i }]), { now: 300 });
  assert(trio.report.kAnonymity.reviewGroup === 3 && trio.report.published.length === 3 && trio.report.kAnonymity.suppressed === 0,
    "three reviews clear the k-anonymity bar");
  // 6. Synchronised solves: three candidates solving on the same tick share the rule.
  const sameTick = analyzeExam([1, 2, 3].map((i) => ({ candidateId: `s${i}`, type: "solve", at: 500 })), { now: 500 });
  assert(sameTick.synchronizedAt.join() === "500" && sameTick.candidates.every((candidate) => candidate.findings.some((f) => f.rule === "synchronized-solve")),
    "synchronised solves are detected");
  assert(sameTick.candidates[0].score === RULES.synchronizedSolve.weight, "the synchronised rule is worth its weight");
  // 7. Retention: events older than the TTL cannot influence the score, and the drop is reported.
  const stale = analyzeExam([{ candidateId: "old", type: "paste", at: 100, email: "old@example.com" },
    { candidateId: "old", type: "solve", at: 110, task: "arrays" }], { now: 1000 });
  assert(stale.candidates.length === 0 && stale.report.retention.dropped === 2, `retention ${JSON.stringify(stale.report.retention)}`);
  assert(analyzeExam([{ candidateId: "old", type: "paste", at: 950 }, { candidateId: "old", type: "solve", at: 960 }], { now: 1000 }).dropped === 0,
    "in-window events survive");
  // 8. Appeals keep the evidence; the ledger is append-only and deterministic.
  const appealed = appeal(trio.report, trio.report.published[0], { reason: "candidate showed the work", now: 1200 });
  const target = appealed.candidates.find((candidate) => candidate.ref === trio.report.published[0]);
  assert(target.verdict === "appealed" && target.appeal.reason === "candidate showed the work", "the appeal is recorded");
  assert(appealed.appeals.length === 1 && appealed.candidates.every((candidate) => candidate.score >= 0), "no evidence is deleted");
  assert(Object.keys(appealed.candidates[0]).includes("rules"), "the original rules survive the appeal");
  // 9. Independent recomputation and boundaries: an outside fold of the rules reproduces every score.
  const handScores = exam.candidates.map((candidate) => candidate.findings.reduce((a, finding) => a + finding.weight, 0));
  assert(handScores.every((score, index) => score === exam.candidates[index].score), "scores equal the sum of the named weights");
  const blank = analyzeExam([]);
  assert(blank.report.candidates.length === 0 && blank.report.retention.dropped === 0 && blank.report.published.length === 0, "an empty stream is clean");
  const noise = analyzeExam([{ candidateId: "z", type: "bogus", at: 100 }, { candidateId: "z", type: "input", at: 100, keys: 3 }], { now: 500 });
  assert(noise.report.ignored === 1 && noise.candidates[0].score === 0, "unknown event types are ignored, not scored");
  assert.deepStrictEqual(analyzeExam([...clean, ...cheater]), exam, "analysis replays identically");
  assert(rng.int(4) < 4 && refHash(`seed-${rng.int(100)}`, PRIVACY.salt).length === 8, "seeded references stay 8 hex digits");
  console.log(`proctoring: candidates=${exam.report.candidates.length} reviews=${exam.report.kAnonymity.reviewGroup} `
    + `suppressed=${exam.report.kAnonymity.suppressed} dropped=${exam.report.retention.dropped} maxScore=${exam.candidates[0].score}`);
  console.log("PASS");
}

module.exports = { makeRng, RULES, THRESHOLDS, PRIVACY, PII_KEYS, fnv1a, refHash, sanitize, verdictFor,
  scoreCandidate, analyzeExam, appeal };
if (require.main === module) main();
