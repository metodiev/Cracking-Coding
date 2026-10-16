/** Task 961 - Implement mentoring workflow pairing novices with maintainers for PR review.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * Novice-proposing Gale-Shapley with mentor capacities (the classic residents/hospitals variant). Both
 * sides' preference lists are derived from a pair score -- shared languages dominate, then timezone
 * proximity, then a mild seniority/level gap penalty -- with the pair id as the tie-break, so every list
 * is a strict total order and the run is deterministic. Novices propose down their list; a mentor keeps
 * the best `capacity` proposals and rejects the rest, which keeps proposing; the loop drains when every
 * novice is matched or has exhausted the list. The result is verified, not trusted: an independent
 * blocking-pair scan re-checks the mentor-optimal stability condition, and unmatched novices are
 * reported when the total capacity is too small.
 *
 * Simplified: mentors are modelled by language set, timezone offset, seniority and capacity instead of
 * a real review queue, and a "pair score" stands in for human preference. Verified: the stable matching
 * against a brute-force blocking-pair oracle, capacity limits, preference lists being total orders, a
 * hand-computed ranking, the under-capacity case with a correct unmatched set, the empty/zero-capacity
 * boundaries, input-order independence, and the load/report arithmetic.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const MENTORS = [
  { id: "m-ada", languages: ["javascript", "python"], timezone: 1, seniority: 9, capacity: 2 },
  { id: "m-hopper", languages: ["java", "python"], timezone: -5, seniority: 8, capacity: 2 },
  { id: "m-lovelace", languages: ["csharp", "java"], timezone: 0, seniority: 7, capacity: 1 },
  { id: "m-turing", languages: ["javascript", "go"], timezone: 0, seniority: 6, capacity: 2 },
];
const NOVICES = [
  { id: "n-a", languages: ["javascript", "python"], timezone: 1, level: 3 },
  { id: "n-b", languages: ["java"], timezone: -5, level: 2 },
  { id: "n-c", languages: ["csharp"], timezone: 0, level: 1 },
  { id: "n-d", languages: ["go", "javascript"], timezone: 0, level: 2 },
  { id: "n-e", languages: ["python"], timezone: -4, level: 4 },
  { id: "n-f", languages: ["java", "csharp"], timezone: 2, level: 1 },
];
/** Pair score: shared languages first, then timezone proximity, minus a capped seniority/level gap. */
function pairScore(mentor, novice) {
  const shared = novice.languages.filter((language) => mentor.languages.includes(language)).length;
  const proximity = Math.max(0, 3 - Math.abs(mentor.timezone - novice.timezone));
  return 4 * shared + 2 * proximity - Math.min(Math.abs(mentor.seniority - novice.level), 4);
}
/** Strict total order by score, ties broken by id, so no two candidates ever compare equal. */
const orderBy = (items, scoreOf) => items.slice()
  .sort((a, b) => (scoreOf(b) - scoreOf(a)) || (a.id < b.id ? -1 : 1)).map((item) => item.id);
function preferenceLists(mentors, novices) {
  const novicePrefs = new Map(novices.map((n) => [n.id, orderBy(mentors, (m) => pairScore(m, n))]));
  const mentorPrefs = new Map(mentors.map((m) => [m.id, orderBy(novices, (n) => pairScore(m, n))]));
  return { novicePrefs, mentorPrefs };
}
/** Resident-proposing matching with mentor capacities; rejected novices re-enter the queue. */
function match(mentors, novices, { capacityOf = (mentor) => mentor.capacity } = {}) {
  const prefs = preferenceLists(mentors, novices);
  const capacity = new Map(mentors.map((mentor) => [mentor.id, capacityOf(mentor)]));
  const held = new Map(mentors.map((mentor) => [mentor.id, []]));
  const heldBy = new Map();
  const cursor = new Map(novices.map((novice) => [novice.id, 0]));
  const queue = novices.map((novice) => novice.id);
  let proposals = 0;
  const rankOf = (mentorId, noviceId) => prefs.mentorPrefs.get(mentorId).indexOf(noviceId);
  while (queue.length > 0) {
    const noviceId = queue.shift();
    if (heldBy.has(noviceId)) continue; // already matched: stop proposing
    const list = prefs.novicePrefs.get(noviceId);
    if (cursor.get(noviceId) >= list.length) continue; // list exhausted: this novice stays unmatched
    const mentorId = list[cursor.get(noviceId)];
    cursor.set(noviceId, cursor.get(noviceId) + 1);
    proposals += 1;
    const roster = held.get(mentorId);
    roster.push(noviceId);
    heldBy.set(noviceId, mentorId);
    roster.sort((a, b) => rankOf(mentorId, a) - rankOf(mentorId, b));
    while (roster.length > capacity.get(mentorId)) {
      const rejected = roster.pop();
      heldBy.delete(rejected);
      queue.push(rejected);
    }
  }
  const assignments = [...held.entries()].flatMap(([mentor, list]) => list.map((novice) => ({ novice, mentor })))
    .sort((a, b) => (a.novice < b.novice ? -1 : 1));
  const matched = new Set(assignments.map((a) => a.novice));
  const loads = Object.fromEntries(mentors.map((mentor) => [mentor.id, held.get(mentor.id).length]));
  return { prefs, assignments, loads, proposals, capacities: Object.fromEntries(capacity),
    unmatched: novices.map((n) => n.id).filter((id) => !matched.has(id)).sort() };
}
/** Independent stability oracle: brute-force every (novice, mentor) pair against the condition. */
function blockingPairs(mentors, novices, result) {
  const current = new Map(result.assignments.map((a) => [a.novice, a.mentor]));
  const blockers = [];
  for (const novice of novices) {
    for (const mentor of mentors) {
      const mine = current.get(novice.id);
      const noviceRank = result.prefs.novicePrefs.get(novice.id).indexOf(mentor.id);
      const wants = mine === undefined || noviceRank < result.prefs.novicePrefs.get(novice.id).indexOf(mine);
      if (!wants) continue;
      const holds = result.assignments.filter((a) => a.mentor === mentor.id).map((a) => a.novice);
      const spared = holds.length < result.capacities[mentor.id];
      const outranksSomebody = holds.some((held) => result.prefs.mentorPrefs.get(mentor.id).indexOf(novice.id)
        < result.prefs.mentorPrefs.get(mentor.id).indexOf(held));
      if (spared || outranksSomebody) blockers.push({ novice: novice.id, mentor: mentor.id });
    }
  }
  return blockers;
}

function main() {
  const rng = makeRng(20240961);
  const result = match(MENTORS, NOVICES);
  // 1. Happy path: enough capacity for everybody, loads inside the limits, proposals >= novices.
  assert(result.unmatched.length === 0 && result.assignments.length === NOVICES.length,
    `unmatched ${result.unmatched.join(",")}`);
  assert(MENTORS.every((mentor) => result.loads[mentor.id] <= mentor.capacity), `loads ${JSON.stringify(result.loads)}`);
  assert(result.proposals >= NOVICES.length && result.proposals <= NOVICES.length * MENTORS.length, `proposals ${result.proposals}`);
  // 2. Stability oracle: no blocking pair exists, checked by a brute-force scan of all pairs.
  assert.deepStrictEqual(blockingPairs(MENTORS, NOVICES, result), [], "the matching is stable");
  // 3. Preference lists are strict total orders on both sides.
  const mentorIds = MENTORS.map((m) => m.id).sort();
  const noviceIds = NOVICES.map((n) => n.id).sort();
  assert([...result.prefs.novicePrefs.values()].every((list) => list.length === MENTORS.length
    && new Set(list).size === MENTORS.length && list.slice().sort().join() === mentorIds.join()), "novice lists are permutations");
  assert([...result.prefs.mentorPrefs.values()].every((list) => list.length === NOVICES.length
    && new Set(list).size === NOVICES.length && list.slice().sort().join() === noviceIds.join()), "mentor lists are permutations");
  const adaTop = result.prefs.mentorPrefs.get("m-ada");
  assert(adaTop[0] === "n-a", `mentor preference starts with the strongest match (${adaTop[0]})`);
  // 4. Hand-checked ranking: the novice fluent in javascript and python must rank the mentor sharing
  //    both languages ahead of every mentor sharing at most one, whatever the timezone.
  const sharedBoth = result.prefs.novicePrefs.get("n-a")[0];
  assert(sharedBoth === "m-ada", `top preference for n-a is ${sharedBoth}`);
  assert(pairScore(MENTORS[0], NOVICES[0]) > pairScore(MENTORS[2], NOVICES[0]), "two shared languages outrank one");
  // 5. Under-capacity: with a single slot per mentor only four novices can be served, stably.
  const tight = match(MENTORS, NOVICES, { capacityOf: () => 1 });
  assert(tight.assignments.length === MENTORS.length && tight.unmatched.length === NOVICES.length - MENTORS.length,
    `tight matching served ${tight.assignments.length}, unmatched ${tight.unmatched.join(",")}`);
  assert.deepStrictEqual(blockingPairs(MENTORS, NOVICES, tight), [], "the capacity-limited matching is still stable");
  assert(Object.values(tight.loads).every((load) => load <= 1), `tight loads ${JSON.stringify(tight.loads)}`);
  // 6. Boundaries: no novices, no mentors, and mentors that can take nobody.
  const noNovices = match(MENTORS, []);
  assert(noNovices.assignments.length === 0 && noNovices.unmatched.length === 0 && noNovices.proposals === 0, "empty cohort");
  const noMentors = match([], NOVICES);
  assert(noMentors.assignments.length === 0 && noMentors.unmatched.length === NOVICES.length, "no mentors leaves everyone unmatched");
  const zeroCapacity = match(MENTORS, NOVICES, { capacityOf: () => 0 });
  assert(zeroCapacity.assignments.length === 0 && zeroCapacity.unmatched.length === NOVICES.length, "zero capacity serves nobody");
  assert(zeroCapacity.proposals === NOVICES.length * MENTORS.length, `every list is exhausted (${zeroCapacity.proposals})`);
  // 7. Input order must not matter: Gale-Shapley yields the proposer-optimal matching for any queue order.
  const shuffled = NOVICES.slice().reverse();
  const reversed = match(MENTORS.slice().reverse(), shuffled);
  assert.deepStrictEqual(reversed.assignments, result.assignments, "input order does not change the matching");
  assert.strictEqual(reversed.proposals, result.proposals, "nor the proposal count");
  assert.deepStrictEqual(match(MENTORS, NOVICES), result, "the whole run replays identically");
  // 8. Report arithmetic: loads, assignments and the unmatched list partition the cohort.
  const total = Object.values(result.loads).reduce((a, load) => a + load, 0);
  assert(total === result.assignments.length && result.assignments.length + result.unmatched.length === NOVICES.length,
    `partition check ${total} + ${result.unmatched.length} != ${NOVICES.length}`);
  assert(new Set(result.assignments.map((a) => a.novice)).size === result.assignments.length, "no novice is paired twice");
  // 9. Seeded cohorts: every generated instance must still come out stable.
  const languages = ["javascript", "python", "java", "go", "csharp"];
  let stableRuns = 0;
  for (let trial = 0; trial < 10; trial += 1) {
    const mentors = Array.from({ length: 1 + rng.int(3) }, (_, i) => ({ id: `f-m${i}`,
      languages: [languages[rng.int(5)], languages[rng.int(5)]], timezone: rng.int(7) - 3,
      seniority: 1 + rng.int(8), capacity: 1 + rng.int(2) }));
    const novices = Array.from({ length: 1 + rng.int(6) }, (_, i) => ({ id: `f-n${i}`, languages: [languages[rng.int(5)]],
      timezone: rng.int(7) - 3, level: 1 + rng.int(6) }));
    if (blockingPairs(mentors, novices, match(mentors, novices)).length === 0) stableRuns += 1;
  }
  assert.strictEqual(stableRuns, 10, `seeded cohorts stable: ${stableRuns}/10`);
  console.log(`mentoring: novices=${NOVICES.length} mentors=${MENTORS.length} matched=${result.assignments.length} `
    + `proposals=${result.proposals} loads=${JSON.stringify(result.loads)}`);
  console.log("PASS");
}

module.exports = { makeRng, MENTORS, NOVICES, pairScore, orderBy, preferenceLists, match, blockingPairs };
if (require.main === module) main();
