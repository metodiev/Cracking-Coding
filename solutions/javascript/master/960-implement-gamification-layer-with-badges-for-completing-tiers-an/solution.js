/** Task 960 - Implement gamification layer with badges for completing tiers and streaks.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * An event log of solves (task id, tier, logical day) is folded into a profile: streaks measured in
 * logical days with optional "freeze" tokens that may bridge a single missed day each, badges for
 * completing every task of a tier in the catalogue, for reaching streak milestones (3/7/30 days), for
 * solving a volume threshold, and for coming back after a gap; then points -- tier-weighted solve
 * points plus bonuses that are derived from the badges themselves. Re-solving a task is idempotent: the
 * first solve of a task is the one that counts, so replaying or duplicating the log cannot inflate the
 * score, and the badge ledger is unique and ordered by (day, badge id).
 *
 * Simplified: days are plain integers rather than timestamps (no time zone or calendar rules), the
 * catalogue is small, and bridge tokens are allocated by dynamic programming. Verified: the optimal
 * streak against a day-by-day brute force with no tokens and against an exhaustive gap-subset oracle
 * with tokens, the empty/single-solve boundaries, milestone threshold semantics, tier completion,
 * duplicate idempotence, the comeback threshold, an independent points recomputation, and replay.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const CATALOG = { beginner: ["b1", "b2", "b3"], intermediate: ["i1", "i2"], advanced: ["a1", "a2"] };
const TIER_POINTS = { beginner: 1, intermediate: 2, advanced: 3, expert: 5, master: 8 };
const TIER_BONUS = 25;
const STREAK_BONUS = 5;
const COMEBACK_BONUS = 15;
const STREAK_MILESTONES = [3, 7, 30];
const VOLUME_MILESTONE = 5;
const COMEBACK_GAP = 30;

/** Optimal streaks: dp[i][u] = longest run ending at solve day i using at most u bridge tokens.
 *  A one-day hole costs one token and contributes two days (the bridged day plus the new solve), so
 *  the DP considers every token allocation instead of spending tokens greedily in day order. */
function streaks(days, freezes = 0) {
  const sorted = [...new Set(days)].sort((a, b) => a - b);
  if (sorted.length === 0) return { longest: 0, current: 0, freezesUsed: 0, longestRun: null };
  const f = sorted.map(() => new Array(freezes + 1).fill(1));
  let champion = { length: 1, used: 0, end: sorted[0] };
  for (let i = 0; i < sorted.length; i += 1) {
    for (let used = 0; used <= freezes; used += 1) {
      let length = 1;
      if (i > 0) {
        const gap = sorted[i] - sorted[i - 1] - 1;
        if (gap === 0) length = Math.max(length, f[i - 1][used] + 1);
        else if (gap === 1 && used >= 1) length = Math.max(length, f[i - 1][used - 1] + 2);
      }
      if (used > 0) length = Math.max(length, f[i][used - 1]);
      f[i][used] = length;
      const better = length > champion.length
        || (length === champion.length && (used < champion.used || (used === champion.used && sorted[i] < champion.end)));
      if (better) champion = { length, used, end: sorted[i] };
    }
  }
  const lastIndex = sorted.length - 1;
  let current = 0;
  for (let used = 0; used <= freezes; used += 1) current = Math.max(current, f[lastIndex][used]);
  return { longest: champion.length, current: sorted.length ? current : 0, freezesUsed: champion.used,
    longestRun: { length: champion.length, used: champion.used, end: champion.end,
      startDay: champion.end - champion.length + 1 - champion.used, days: sorted } };
}
/** Brute force: mark the chosen gap days covered, then walk every day in range counting runs. */
function longestRunCovering(sortedDays, covered) {
  const set = new Set([...sortedDays, ...covered]);
  const lo = Math.min(...set), hi = Math.max(...set);
  let best = 0, run = 0;
  for (let day = lo; day <= hi; day += 1) { run = set.has(day) ? run + 1 : 0; best = Math.max(best, run); }
  return best;
}
/** Exhaustive oracle: try every subset of at most `freezes` one-day gaps and take the best run. */
function streakOracle(days, freezes = 0) {
  const sorted = [...new Set(days)].sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const gapDays = [];
  for (let i = 1; i < sorted.length; i += 1) if (sorted[i] - sorted[i - 1] === 2) gapDays.push(sorted[i] - 1);
  let best = longestRunCovering(sorted, []);
  const choose = (start, left, covered) => {
    if (left === 0) { best = Math.max(best, longestRunCovering(sorted, covered)); return; }
    for (let i = start; i < gapDays.length; i += 1) choose(i + 1, left - 1, [...covered, gapDays[i]]);
  };
  for (let size = 1; size <= freezes; size += 1) choose(0, size, []);
  return best;
}
/** Fold the log into streaks, badges and points. The first solve of a task is the only one counted. */
function evaluate(events, { freezes = 0, catalog = CATALOG } = {}) {
  const firstSolve = new Map();
  for (const event of events) {
    const known = firstSolve.get(event.taskId);
    if (!known || event.day < known.day) firstSolve.set(event.taskId, event);
  }
  const ordered = [...firstSolve.values()].sort((a, b) => (a.day - b.day) || (a.taskId < b.taskId ? -1 : 1));
  const streak = streaks(ordered.map((e) => e.day), freezes);
  const badges = [];
  for (const [tier, taskIds] of Object.entries(catalog)) {
    const solved = taskIds.map((id) => firstSolve.get(id));
    if (solved.every(Boolean)) badges.push({ id: `tier:${tier}`, day: Math.max(...solved.map((e) => e.day)), detail: `${taskIds.length} tasks` });
  }
  for (const milestone of STREAK_MILESTONES) {
    if (streak.longestRun && streak.longestRun.length >= milestone) {
      badges.push({ id: `streak:${milestone}`, day: streak.longestRun.startDay + milestone - 1, detail: `${milestone} days` });
    }
  }
  if (ordered.length >= VOLUME_MILESTONE) badges.push({ id: `volume:${VOLUME_MILESTONE}`, day: ordered[VOLUME_MILESTONE - 1].day, detail: `${ordered.length} solves` });
  const comebackIndex = ordered.findIndex((event, index) => index > 0 && event.day - ordered[index - 1].day > COMEBACK_GAP);
  if (comebackIndex > 0) {
    const gap = ordered[comebackIndex].day - ordered[comebackIndex - 1].day;
    badges.push({ id: "comeback", day: ordered[comebackIndex].day, detail: `${gap} day gap` });
  }
  const ledger = badges.sort((a, b) => (a.day - b.day) || (a.id < b.id ? -1 : 1));
  const breakdown = {
    solvePoints: ordered.reduce((a, event) => a + (TIER_POINTS[event.tier] || 0), 0),
    tierBonus: ledger.filter((b) => b.id.startsWith("tier:")).length * TIER_BONUS,
    streakBonus: ledger.filter((b) => b.id.startsWith("streak:")).reduce((a, b) => a + STREAK_BONUS * Number(b.id.split(":")[1]), 0),
    comebackBonus: ledger.some((b) => b.id === "comeback") ? COMEBACK_BONUS : 0,
  };
  return { streak, ledger, solves: ordered.length, breakdown,
    points: breakdown.solvePoints + breakdown.tierBonus + breakdown.streakBonus + breakdown.comebackBonus };
}

function main() {
  const rng = makeRng(20240960);
  const solve = (taskId, tier, day) => ({ taskId, tier, day });
  // 1. Oracle (no tokens): the greedy streak equals a day-by-day brute force on seeded logs.
  let noTokenAgreements = 0;
  for (let trial = 0; trial < 60; trial += 1) {
    const days = [...new Set(Array.from({ length: 1 + rng.int(14) }, () => rng.int(40)))];
    if (streaks(days, 0).longest === streakOracle(days, 0)) noTokenAgreements += 1;
  }
  assert.strictEqual(noTokenAgreements, 60, `streak oracle disagreements ${60 - noTokenAgreements}`);
  // 2. Oracle (with tokens): greedy equals the exhaustive subset search over one-day gaps.
  let tokenAgreements = 0;
  for (let trial = 0; trial < 30; trial += 1) {
    const days = [...new Set(Array.from({ length: 1 + rng.int(10) }, () => rng.int(30)))];
    for (let freezes = 1; freezes <= 2; freezes += 1) {
      if (streaks(days, freezes).longest === streakOracle(days, freezes)) tokenAgreements += 1;
    }
  }
  assert.strictEqual(tokenAgreements, 60, `token oracle disagreements ${60 - tokenAgreements}`);
  // 3. Boundary: nothing solved, then exactly one solve.
  const empty = evaluate([]);
  assert(empty.solves === 0 && empty.streak.longest === 0 && empty.points === 0 && empty.ledger.length === 0, "empty log is all zeros");
  const single = evaluate([solve("b1", "beginner", 5)]);
  assert(single.streak.longest === 1 && single.streak.current === 1 && single.ledger.length === 0, "one solve is no badge");
  assert(single.points === 1 && single.breakdown.solvePoints === 1, "one beginner solve is one point");
  // 4. Milestone thresholds: six days earns streak:3, seven days earns streak:7 as well.
  const sixDays = evaluate(Array.from({ length: 6 }, (_, i) => solve(`x${i}`, "expert", 100 + i)));
  const sevenDays = evaluate(Array.from({ length: 7 }, (_, i) => solve(`x${i}`, "expert", 100 + i)));
  assert(sixDays.ledger.map((b) => b.id).join(",") === "streak:3,volume:5", `six days: ${sixDays.ledger.map((b) => b.id)}`);
  assert(sevenDays.ledger.map((b) => b.id).join(",") === "streak:3,volume:5,streak:7", "seven days adds the 7-day badge");
  assert(sevenDays.ledger[2].day === 106, `the 7-day badge lands on the seventh day (${sevenDays.ledger[2].day})`);
  assert(!sixDays.ledger.some((b) => b.id === "streak:7"), "six days is one short of the 7-day milestone");
  // 5. Tier completion: only the whole catalogue of a tier awards the tier badge and its bonus.
  const partial = evaluate([solve("b1", "beginner", 1), solve("b2", "beginner", 2)]);
  const complete = evaluate([solve("b1", "beginner", 1), solve("b2", "beginner", 2), solve("b3", "beginner", 40)]);
  assert(!partial.ledger.some((b) => b.id.startsWith("tier:")) && partial.points === 2, "two of three tasks is no tier badge");
  assert(complete.ledger.some((b) => b.id === "tier:beginner") && complete.breakdown.tierBonus === TIER_BONUS, "the completed tier pays once");
  assert(complete.ledger.filter((b) => b.id === "tier:beginner").length === 1, "the tier badge is not duplicated");
  // 6. Idempotence: duplicates and replayed logs change nothing.
  const log = [solve("b1", "beginner", 1), solve("b2", "beginner", 2), solve("b3", "beginner", 3), solve("a1", "advanced", 3)];
  const replayed = evaluate([...log, ...log, solve("b1", "beginner", 1)]);
  assert.deepStrictEqual(replayed, evaluate(log), "duplicated events fold to the same profile");
  assert.strictEqual(replayed.solves, 4, `duplicates do not inflate the solve count (${replayed.solves})`);
  assert.strictEqual(replayed.points, 1 + 1 + 1 + 3 + TIER_BONUS + 5 * 3, `points ${replayed.points}`);
  // 7. Comeback threshold: a gap strictly greater than 30 days is a comeback, 30 is not.
  const gap30 = evaluate([solve("b1", "beginner", 1), solve("b2", "beginner", 31)]);
  const gap31 = evaluate([solve("b1", "beginner", 1), solve("b2", "beginner", 32)]);
  assert(!gap30.ledger.some((b) => b.id === "comeback") && gap31.ledger.some((b) => b.id === "comeback"),
    "a 30-day gap stays quiet, a 31-day gap fires the comeback");
  assert(gap31.breakdown.comebackBonus === COMEBACK_BONUS, "the comeback bonus is applied once");
  // 8. Independent points recomputation straight from the events, plus ledger hygiene.
  const double = evaluate([solve("b1", "beginner", 1), solve("b2", "beginner", 2), solve("b3", "beginner", 3),
    solve("b1", "beginner", 4), solve("i1", "intermediate", 5)]);
  const expected = new Set(["b1", "b2", "b3", "i1"]);
  const solvePoints = [...expected].reduce((a, id) => a + TIER_POINTS[id === "i1" ? "intermediate" : "beginner"], 0);
  assert(double.points === solvePoints + TIER_BONUS + 5 * (double.streak.longest >= 3 ? 3 : 0) + 0,
    `points ${double.points} recomputed by hand`);
  assert(new Set(double.ledger.map((b) => b.id)).size === double.ledger.length, "badge ids are unique");
  assert(double.ledger.every((b, i) => i === 0 || double.ledger[i - 1].day <= b.day), "the ledger is ordered by day");
  // 9. Freeze accounting and determinism.
  const holed = evaluate([solve("b1", "beginner", 1), solve("b2", "beginner", 2), solve("b3", "beginner", 4)], { freezes: 2 });
  assert(holed.streak.longest === 4 && holed.streak.freezesUsed === 1, `freeze bridged the hole (${holed.streak.longest})`);
  assert(holed.ledger.some((b) => b.id === "streak:3"), "a bridged run still earns the milestone");
  assert.deepStrictEqual(evaluate(log), evaluate(log), "evaluation replays identically");
  console.log(`gamification: solves=${replayed.solves} points=${replayed.points} badges=${replayed.ledger.map((b) => b.id).join("/")} `
    + `streak=${double.streak.longest} oracleChecks=${noTokenAgreements + tokenAgreements}`);
  console.log("PASS");
}

module.exports = { makeRng, CATALOG, TIER_POINTS, TIER_BONUS, STREAK_BONUS, COMEBACK_BONUS, STREAK_MILESTONES,
  VOLUME_MILESTONE, COMEBACK_GAP, streaks, longestRunCovering, streakOracle, evaluate };
if (require.main === module) main();
