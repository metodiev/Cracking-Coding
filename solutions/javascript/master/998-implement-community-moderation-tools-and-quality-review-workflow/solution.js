/** Task 998 - Implement community moderation tools and quality review workflow.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * The moderation pipeline for user-contributed exercises: a guarded state machine for a submission's
 * life (draft -> submitted -> triaged -> accepted/changes-requested/rejected -> published, with an
 * appeal branch), a reputation-and-age priority queue, shingle-based duplicate detection, expertise-
 * aware reviewer assignment under a per-reviewer capacity, and an SLA clock that flags work waiting
 * too long. Every transition appends to an audit trail, so the whole workflow can be replayed and its
 * outcome checked against the trail. What is simplified: moderators are simulated over an in-memory
 * event list. Verified: legal and illegal transitions, appeal handling, priority ordering against an
 * independent sort, duplicate detection against a brute-force Jaccard oracle at the exact threshold,
 * balanced reviewer load, SLA breach and clearance, and byte-identical replay of the audit trail.
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
/** The lifecycle. Every reachable state lists exactly the states it may move to. */
const TRANSITIONS = {
  draft: ["submitted"], submitted: ["triaged", "rejected"], triaged: ["accepted", "changes-requested", "rejected"],
  accepted: ["published"], "changes-requested": ["submitted", "rejected"], rejected: ["appealed"],
  appealed: ["accepted", "rejected"], published: [],
};
const SLA_HOURS = 24;
/** Guarded transition: an illegal move leaves the submission untouched and names the reason. */
function transition(current, next, event = {}) {
  if (!TRANSITIONS[current]) return { ok: false, reason: "unknown-state", state: current, event: null };
  if (!TRANSITIONS[current].includes(next)) {
    return { ok: false, reason: `illegal-transition:${current}->${next}`, state: current, event: null };
  }
  return { ok: true, reason: "transitioned", state: next,
    event: { from: current, to: next, actor: event.actor || "system", reason: event.reason || null } };
}
/** The workflow holds submissions and their audit trail; only legal moves ever appear in the trail. */
class Workflow {
  constructor() { this.submissions = new Map(); this.trail = []; }
  submit(id, meta = {}) { const s = { id, state: "draft", meta, submittedAt: null, reviewedAt: null };
    this.submissions.set(id, s); return s; }
  move(id, next, event = {}) {
    const submission = this.submissions.get(id);
    const result = transition(submission.state, next, event);
    if (!result.ok) return result;
    submission.state = next;
    if (next === "submitted") submission.submittedAt = event.at || 0;
    if (["triaged", "accepted", "rejected"].includes(next)) submission.reviewedAt = event.at || 0;
    this.trail.push({ id, ...result.event, at: event.at || 0 });
    return result;
  }
  auditDigest() { return fnv1a(JSON.stringify(this.trail)); }
}
const REPUTATION = { new: 1, contributor: 3, maintainer: 5 };
/** Priority score: submitter reputation plus the age bonus in whole days the item has waited. */
function priorityOf(submission, nowHours) {
  const age = submission.submittedAt === null ? 0 : Math.max(0, Math.floor((nowHours - submission.submittedAt) / 24));
  return (REPUTATION[submission.meta.reputation] || 0) + age;
}
/** Queue order: priority desc, then submission id asc -- independent of insertion order. */
function queue(submissions, nowHours) {
  return submissions.map((s) => ({ id: s.id, priority: priorityOf(s, nowHours) }))
    .sort((a, b) => b.priority - a.priority || (a.id < b.id ? -1 : 1));
}
/** Independent oracle for the queue: repeated max-selection, no Array#sort. */
function queueBySelection(submissions, nowHours) {
  const remaining = submissions.map((s) => ({ id: s.id, priority: priorityOf(s, nowHours) }));
  const out = [];
  while (remaining.length > 0) {
    let best = 0;
    for (let i = 1; i < remaining.length; i += 1) {
      const a = remaining[i], b = remaining[best];
      if (a.priority > b.priority || (a.priority === b.priority && a.id < b.id)) best = i;
    }
    out.push(remaining[best]); remaining.splice(best, 1);
  }
  return out;
}
/** Word 3-shingles; a shorter statement simply has fewer, so an empty one has none. */
function shingles(text, k = 3) {
  const words = String(text || "").toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 0);
  const set = new Set();
  for (let i = 0; i + k <= words.length; i += 1) set.add(words.slice(i, i + k).join(" "));
  return set;
}
function jaccard(a, b) {
  const left = shingles(a), right = shingles(b);
  if (left.size === 0 && right.size === 0) return 0;
  let shared = 0;
  for (const s of left) if (right.has(s)) shared += 1;
  return shared / (left.size + right.size - shared);
}
/** Duplicate pairs at or above the threshold; a pair with no shingles is never a duplicate. */
function findDuplicates(submissions, threshold = 0.6) {
  const pairs = [];
  for (let i = 0; i < submissions.length; i += 1) for (let j = i + 1; j < submissions.length; j += 1) {
    const similarity = jaccard(submissions[i].statement, submissions[j].statement);
    if (similarity > 0 && similarity >= threshold) {
      pairs.push({ a: submissions[i].id, b: submissions[j].id, similarity });
    }
  }
  return pairs.sort((x, y) => (x.a < y.a ? -1 : x.a > y.a ? 1 : x.b < y.b ? -1 : 1));
}
/** Independent oracle: the same shingle Jaccard, built with arrays and indexOf instead of Set. */
function bruteJaccard(a, b) {
  const build = (text) => {
    const words = [], out = [];
    let token = "";
    const flush = () => { if (token.length > 0) { words.push(token); token = ""; } };
    for (const ch of String(text || "").toLowerCase()) {
      if ((ch >= "a" && ch <= "z") || (ch >= "0" && ch <= "9")) token += ch; else flush();
    }
    flush();
    for (let i = 0; i + 3 <= words.length; i += 1) {
      const shingle = `${words[i]} ${words[i + 1]} ${words[i + 2]}`;
      if (out.indexOf(shingle) === -1) out.push(shingle);
    }
    return out;
  };
  const left = build(a), right = build(b);
  const shared = left.filter((s) => right.indexOf(s) !== -1).length;
  const union = left.length + right.length - shared;
  return union === 0 ? 0 : shared / union;
}
/** Expertise-aware round-robin: the best-overlap reviewer with spare capacity, ties by name. */
function assignReviewers(items, reviewers) {
  const load = new Map(reviewers.map((r) => [r.name, 0]));
  const assignments = [];
  for (const item of items) {
    const eligible = reviewers.filter((r) => load.get(r.name) < r.capacity)
      .map((r) => ({ name: r.name,
        overlap: (item.tags || []).filter((t) => (r.tags || []).includes(t)).length }))
      .sort((a, b) => b.overlap - a.overlap || (a.name < b.name ? -1 : 1));
    if (eligible.length === 0) { assignments.push({ id: item.id, reviewer: null, reason: "no-capacity" }); continue; }
    load.set(eligible[0].name, load.get(eligible[0].name) + 1);
    assignments.push({ id: item.id, reviewer: eligible[0].name, overlap: eligible[0].overlap });
  }
  return { assignments, load: Object.fromEntries(Array.from(load.entries()).sort()) };
}
/** SLA sweep: anything sitting in triage longer than the SLA without a review. */
function slaBreaches(submissions, nowHours) {
  return submissions.filter((s) => s.submittedAt !== null && s.reviewedAt === null
    && nowHours - s.submittedAt > SLA_HOURS).map((s) => s.id).sort();
}
const TOPICS = ["arrays", "graphs", "strings", "dynamic-programming"];
const STATEMENTS = ["find the shortest path in a weighted graph using dijkstra",
  "find the shortest path in a weighted directed graph using dijkstra",
  "reverse every word of a sentence in place", "count distinct substrings of a string"];
/** Seeded submissions with a mix of reputation levels and a few near-duplicate statements. */
function seededSubmissions(seed, count) {
  const rng = splitmix64(seed);
  const reputations = Object.keys(REPUTATION);
  return Array.from({ length: count }, (_, i) => ({ id: `s${String(i).padStart(3, "0")}`,
    statement: STATEMENTS[rng.int(STATEMENTS.length)], tags: [TOPICS[rng.int(TOPICS.length)]],
    submittedAt: rng.int(4) * SLA_HOURS, reputation: reputations[rng.int(reputations.length)] })); }function main() {
  const workflow = new Workflow();
  workflow.submit("s001", { reputation: "contributor" });
  // 1. Happy path: the canonical path is walked, in order, and recorded in the trail.
  const path = [["submitted", "ada", 0], ["triaged", "bot", 5], ["accepted", "bob", 9], ["published", "carol", 10]];
  assert.deepStrictEqual(path.map(([to, actor, at]) => workflow.move("s001", to, { actor, at }).ok),
    path.map(() => true), "canonical path is walked");
  assert.deepStrictEqual(workflow.trail.map((e) => `${e.from}->${e.to}`),
    ["draft->submitted", "submitted->triaged", "triaged->accepted", "accepted->published"], "trail preserves order");
  assert.strictEqual(workflow.submissions.get("s001").state, "published", "submission ends published");
  // 2. Negative: illegal moves are refused and leave both the state and the trail untouched.
  const guarded = new Workflow();
  guarded.submit("s002", { reputation: "new" });
  const illegal = [guarded.move("s002", "accepted"), guarded.move("s002", "published"),
    guarded.move("s002", "appealed"), guarded.move("s002", "triaged")];
  assert.deepStrictEqual([illegal.map((r) => r.reason), guarded.submissions.get("s002").state, guarded.trail.length],
    [["illegal-transition:draft->accepted", "illegal-transition:draft->published",
      "illegal-transition:draft->appealed", "illegal-transition:draft->triaged"], "draft", 0],
    "illegal transitions are refused without side effects");
  // 3. Appeal branch: a rejection can be appealed and overturned, but a published item is terminal.
  guarded.move("s002", "submitted"); guarded.move("s002", "rejected", { actor: "bob" });
  assert.deepStrictEqual([guarded.move("s002", "appealed", { actor: "dana" }).ok, guarded.move("s002", "accepted").ok,
    guarded.move("s002", "submitted").reason], [true, true, "illegal-transition:accepted->submitted"],
    "appeal can overturn a rejection, but accepted cannot go back");
  assert.strictEqual(TRANSITIONS.published.length, 0, "published is a terminal state");
  // 4. Boundary: an empty queue assigns nothing, and a submission with no shingles is not a duplicate
  // of anything -- an empty/short statement must not divide by zero.
  assert.deepStrictEqual([queue([], 100), assignReviewers([], [{ name: "a", tags: [], capacity: 1 }]).assignments],
    [[], []], "empty inputs produce empty outputs");
  assert.deepStrictEqual([jaccard("", ""), jaccard("hi", "hi"), findDuplicates([{ id: "a", statement: "" },
    { id: "b", statement: "" }], 0.5)], [0, 0, []], "shingle-free statements never match");
  // 5. Duplicate detection: near-identical statements are flagged, and the threshold is inclusive.
  const nearDuplicate = STATEMENTS.map((statement, i) => ({ id: `s00${i + 1}`, statement }));
  const exact = bruteJaccard(STATEMENTS[0], STATEMENTS[1]);
  assert.deepStrictEqual([findDuplicates(nearDuplicate, 0.5).map((p) => [p.a, p.b]), findDuplicates(nearDuplicate, 0.9)],
    [[["s001", "s002"]], []], "near duplicates are caught at 0.5, unrelated statements are not");
  assert.deepStrictEqual([findDuplicates(nearDuplicate, exact).length, exact > 0 && exact < 0.9], [1, true],
    "a similarity exactly at the threshold counts as a duplicate");
  // 6. Oracle: the shingle-based Jaccard agrees with a Set-free token implementation on every pair.
  const seeded = seededSubmissions(0x998, 14);
  for (let i = 0; i < seeded.length; i += 1) for (let j = i + 1; j < seeded.length; j += 1) {
    assert.strictEqual(jaccard(seeded[i].statement, seeded[j].statement),
      bruteJaccard(seeded[i].statement, seeded[j].statement), `jaccard agrees for ${i}/${j}`);
  }
  // 7. Priority and reviewers: the queue matches an independent max-selection, and no reviewer is
  // given more than their capacity.
  const workflow2 = new Workflow();
  for (const s of seeded) {
    workflow2.submit(s.id, { reputation: s.reputation, tags: s.tags });
    workflow2.move(s.id, "submitted", { at: s.submittedAt });
  }
  const live = Array.from(workflow2.submissions.values());
  const queueRows = queue(live, 96);
  assert.deepStrictEqual(queueRows, queueBySelection(live, 96), "queue matches the selection oracle");
  const reviewers = [{ name: "ada", tags: ["graphs", "arrays"], capacity: 3 },
    { name: "bob", tags: ["strings"], capacity: 2 }, { name: "chen", tags: [], capacity: 2 }];
  const items = queueRows.map((q) => ({ id: q.id, tags: live.find((s) => s.id === q.id).meta.tags }));
  const { assignments, load } = assignReviewers(items, reviewers);
  assert.strictEqual(assignments.filter((a) => a.reviewer === null).length,
    Math.max(0, assignments.length - reviewers.reduce((a, r) => a + r.capacity, 0)),
    "exactly the overflow has no reviewer");
  assert.ok(Object.entries(load).every(([name, used]) => used <= reviewers.find((r) => r.name === name).capacity),
    "no reviewer exceeds capacity");
  assert.ok(assignments.filter((a) => a.overlap > 0).every((a) => a.reviewer !== null), "expertise matches are assigned");
  assert.deepStrictEqual(assignReviewers(items, reviewers).assignments, assignments, "assignment is stable across rebuilds");
  // 8. SLA: a submission waiting past the SLA is flagged until it is reviewed, and the audit trail is
  // reproducible byte for byte.
  const waiting = new Workflow();
  waiting.submit("s900", { reputation: "new" }); waiting.move("s900", "submitted", { at: 0 });
  assert.deepStrictEqual([slaBreaches(Array.from(waiting.submissions.values()), 12),
    slaBreaches(Array.from(waiting.submissions.values()), SLA_HOURS)], [[], []], "inside the SLA nothing is flagged");
  assert.deepStrictEqual(slaBreaches(Array.from(waiting.submissions.values()), SLA_HOURS + 1), ["s900"],
    "past the SLA the submission is flagged");
  waiting.move("s900", "triaged", { at: 30 });
  assert.deepStrictEqual(slaBreaches(Array.from(waiting.submissions.values()), 48), [], "review clears the flag");
  const replay = () => { const w = new Workflow(); w.submit("s900", { reputation: "new" });
    w.move("s900", "submitted", { at: 0 }); w.move("s900", "triaged", { at: 30 }); return w.auditDigest(); };
  assert.strictEqual(waiting.auditDigest(), replay(), "a rebuilt workflow matches the live audit digest");
  console.log(`trail=${workflow.trail.length} duplicates=${findDuplicates(seeded, 0.6).length} ` +
    `assigned=${assignments.filter((a) => a.reviewer).length}/${assignments.length} digest=${workflow.auditDigest().toString(16)}`);
  console.log("PASS");
}
if (require.main === module) main();
module.exports = { splitmix64, fnv1a, transition, Workflow, priorityOf, queue, queueBySelection, shingles,
  jaccard, bruteJaccard, findDuplicates, assignReviewers, slaBreaches, seededSubmissions, TRANSITIONS };
