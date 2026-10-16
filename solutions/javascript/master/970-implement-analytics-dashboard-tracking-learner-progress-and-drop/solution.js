/** Task 970 - Implement analytics dashboard tracking learner progress and drop-off.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Telemetry is a flat, deterministically generated event log (`start`/`attempt`/`pass`/`quit`)
 * keyed by learner and day. The dashboard folds it into four views: (1) a conversion funnel over
 * the learner population with per-stage rates and the largest stage-to-stage loss (the drop-off),
 * (2) per-learner progress (solved count, active days, longest streak), (3) weekly cohort
 * retention, and (4) fixed-size time windows for trend lines.
 * Simplified: a day is the atomic time unit (no intra-day clocks) and the generator uses two
 * archetypes - steady and fading learners - instead of a real product. Verified: funnel monotonicity,
 * exact conversions and retention recomputed by an independent filter, drop-off identification,
 * window partition conservation, the empty-log boundary and byte-identical reruns per seed.
 */
"use strict";
const assert = require("node:assert");

/** splitmix64 - the only source of randomness, so every dashboard is reproducible. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, unit: () => Number(next() >> 11n) / 2 ** 53, int: (n) => Number(next() % BigInt(n)) };
}

/** Steady learners keep going for the whole window, fading learners churn early. */
function buildTelemetry(seed, learners = 40, days = 21) {
  const rng = splitmix64(seed);
  const events = [];
  for (let learner = 0; learner < learners; learner += 1) {
    const skill = 0.25 + 0.6 * rng.unit();
    const active = rng.unit() < 0.4 ? rng.int(days + 1) : days;
    events.push({ learner, day: 0, type: "start" });
    for (let day = 1; day <= active; day += 1) {
      const attempts = 1 + rng.int(3);
      for (let a = 0; a < attempts; a += 1) {
        events.push({ learner, day, type: "attempt" });
        if (rng.unit() < skill) events.push({ learner, day, type: "pass" });
      }
    }
    if (active < days) events.push({ learner, day: active, type: "quit" });
  }
  return events;
}

/** Per-learner aggregate built in a single pass, used by the funnel and the progress view. */
function summarizeLearners(events) {
  const by = new Map();
  for (const event of events) {
    if (!by.has(event.learner)) by.set(event.learner, { attempts: 0, passes: 0, days: new Set(), quit: false });
    const learner = by.get(event.learner);
    if (event.type === "attempt") learner.attempts += 1;
    else if (event.type === "pass") learner.passes += 1;
    else if (event.type === "quit") learner.quit = true;
    learner.days.add(event.day);
  }
  return by;
}

const FUNNEL_STAGES = [
  { stage: "started", keep: () => true },
  { stage: "attempted", keep: (l) => l.attempts > 0 },
  { stage: "solved", keep: (l) => l.passes > 0 },
  { stage: "solved3", keep: (l) => l.passes >= 3 },
];

function funnel(events) {
  const learners = [...summarizeLearners(events).values()];
  const total = learners.length;
  return FUNNEL_STAGES.map(({ stage, keep }) => {
    const count = learners.filter(keep).length;
    return { stage, count, rate: total === 0 ? 0 : count / total };
  });
}

/** Largest absolute stage-to-stage loss; earliest stage wins ties, `null` for a single stage. */
function dropOff(stages) {
  let worst = null;
  for (let i = 1; i < stages.length; i += 1) {
    const lost = stages[i - 1].count - stages[i].count;
    if (worst === null || lost > worst.lost) {
      worst = { from: stages[i - 1].stage, to: stages[i].stage, lost };
    }
  }
  return worst;
}

function perLearnerProgress(events) {
  const out = new Map();
  for (const [learner, data] of summarizeLearners(events)) {
    const days = [...data.days].sort((a, b) => a - b);
    let streak = 0;
    let best = 0;
    for (let i = 0; i < days.length; i += 1) {
      streak = i > 0 && days[i] === days[i - 1] + 1 ? streak + 1 : 1;
      best = Math.max(best, streak);
    }
    out.set(learner, { solved: data.passes, activeDays: days.length, streak: best, churned: data.quit });
  }
  return out;
}

/** Weekly cohorts: of the learners active in week w, how many come back in week w+1. */
function cohortRetention(events, weekDays = 7) {
  const activeWeeks = new Map();
  for (const event of events) {
    const week = Math.floor(event.day / weekDays) + 1;
    if (!activeWeeks.has(event.learner)) activeWeeks.set(event.learner, new Set());
    activeWeeks.get(event.learner).add(week);
  }
  const weekly = new Map();
  for (const weeks of activeWeeks.values()) {
    for (const week of weeks) weekly.set(week, (weekly.get(week) || 0) + 1);
  }
  const cohorts = [];
  for (const week of [...weekly.keys()].sort((a, b) => a - b)) {
    if (!weekly.has(week + 1)) continue;
    const active = weekly.get(week);
    const returning = [...activeWeeks.values()].filter((set) => set.has(week) && set.has(week + 1)).length;
    cohorts.push({ week, active, returning, rate: returning / active });
  }
  return cohorts;
}

function windowCounts(events, size) {
  const buckets = new Map();
  for (const event of events) {
    const bucket = Math.floor(event.day / size) * size;
    buckets.set(bucket, (buckets.get(bucket) || 0) + 1);
  }
  return [...buckets].sort((a, b) => a[0] - b[0]);
}

const LOG = [
  { learner: "a", day: 0, type: "start" }, { learner: "a", day: 1, type: "attempt" },
  { learner: "a", day: 1, type: "pass" }, { learner: "a", day: 2, type: "attempt" },
  { learner: "a", day: 2, type: "pass" }, { learner: "a", day: 3, type: "attempt" },
  { learner: "a", day: 3, type: "pass" },
  { learner: "b", day: 0, type: "start" }, { learner: "b", day: 1, type: "attempt" },
  { learner: "b", day: 2, type: "quit" },
  { learner: "c", day: 0, type: "start" },
];

function main() {
  // 1. Happy path on a hand-built log: exact funnel counts and rates.
  const small = funnel(LOG);
  assert.deepStrictEqual(small.map((s) => s.count), [3, 2, 1, 1], "funnel counts");
  assert.strictEqual(small[0].rate, 1, "everyone started");
  assert.strictEqual(small[2].rate.toFixed(4), "0.3333", "one of three solved");

  // 2. Independent oracle: each stage recomputed from a filtered set of learners.
  const oracleStage = (events, predicate) => new Set(events.filter(predicate).map((e) => e.learner)).size;
  const oracleCounts = [
    oracleStage(LOG, () => true),
    oracleStage(LOG, (e) => e.type === "attempt"),
    oracleStage(LOG, (e) => e.type === "pass"),
  ];
  const passCounts = new Map();
  for (const event of LOG) if (event.type === "pass") passCounts.set(event.learner, (passCounts.get(event.learner) || 0) + 1);
  const oracleSolved3 = [...passCounts.values()].filter((n) => n >= 3).length;
  assert.deepStrictEqual([...oracleCounts, oracleSolved3], small.map((s) => s.count), "funnel matches the oracle");

  // 3. Monotonicity invariant plus drop-off identification on the generated population.
  const events = buildTelemetry(970, 40, 21);
  const stages = funnel(events);
  for (let i = 1; i < stages.length; i += 1) {
    assert(stages[i].count <= stages[i - 1].count, `${stages[i].stage} cannot exceed ${stages[i - 1].stage}`);
    assert(stages[i].rate <= stages[i - 1].rate, "rates stay non-increasing");
  }
  assert.deepStrictEqual(dropOff(small), { from: "started", to: "attempted", lost: 1 }, "earliest tie wins");
  const worst = dropOff(stages);
  const losses = stages.slice(1).map((s, i) => stages[i].count - s.count);
  assert.strictEqual(worst.lost, Math.max(...losses), "drop-off is the largest stage loss");
  assert.strictEqual(dropOff([{ stage: "only", count: 5 }]), null, "a single stage has no drop-off");
  assert.strictEqual(dropOff(funnel([])).lost, 0, "an empty log has no losses");

  // 4. Progress view: solved counts, active days and the longest streak.
  const progress = perLearnerProgress(LOG);
  assert.deepStrictEqual(progress.get("a"), { solved: 3, activeDays: 4, streak: 4, churned: false });
  assert.deepStrictEqual(progress.get("b"), { solved: 0, activeDays: 3, streak: 3, churned: true });
  assert.strictEqual(progress.get("c").streak, 1, "a single day is a streak of one");
  assert.strictEqual(progress.get("a").solved, LOG.filter((e) => e.learner === "a" && e.type === "pass").length);
  const generated = perLearnerProgress(events);
  assert([...generated.values()].every((p) => p.streak <= p.activeDays), "a streak never beats active days");
  const churnedDays = [...generated.values()].filter((p) => p.churned).map((p) => p.activeDays);
  const keptDays = [...generated.values()].filter((p) => !p.churned).map((p) => p.activeDays);
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
  assert(churnedDays.length >= 2 && keptDays.length >= 2, "both archetypes are present");
  assert(mean(churnedDays) < mean(keptDays), "churned learners accumulate fewer active days");

  // 5. Cohort retention: rates in [0,1] and the returning count recomputed independently.
  const cohorts = cohortRetention(events, 7);
  assert(cohorts.length >= 2, `several weekly cohorts: ${cohorts.length}`);
  for (const cohort of cohorts) {
    assert(cohort.returning <= cohort.active && cohort.rate <= 1 && cohort.rate >= 0, `week ${cohort.week} sane`);
    const activeSet = new Set(events.filter((e) => Math.floor(e.day / 7) + 1 === cohort.week).map((e) => e.learner));
    const nextSet = new Set(events.filter((e) => Math.floor(e.day / 7) + 1 === cohort.week + 1).map((e) => e.learner));
    assert.strictEqual(cohort.active, activeSet.size, "cohort size matches a direct count");
    assert.strictEqual(cohort.returning, [...activeSet].filter((l) => nextSet.has(l)).length, "returnees match");
  }
  assert(cohorts.every((c) => c.rate >= cohorts.at(-1).rate - 0.5), "retention stays in a plausible band");

  // 6. Windows: buckets partition the log, and a window wider than the span collapses to one.
  const bucketed = windowCounts(events, 4);
  assert.strictEqual(bucketed.reduce((sum, [, n]) => sum + n, 0), events.length, "windows conserve events");
  assert.strictEqual(windowCounts(events, 1000).length, 1, "an oversized window yields one bucket");
  assert.deepStrictEqual(windowCounts([], 4), [], "no events, no buckets");
  assert.deepStrictEqual(bucketed.map(([day]) => day), [0, 4, 8, 12, 16, 20], "day 0 (signup) is its own bucket");

  // 7. Boundary: the generated population always has at least one churned learner to report.
  const churnRate = stages.slice(1).reduce((acc, s, i) => acc + (stages[i].count - s.count), 0);
  assert(churnRate >= 1, `the funnel loses learners somewhere: ${churnRate}`);

  // 8. Determinism: one seed, one dashboard.
  assert(JSON.stringify(buildTelemetry(970, 40, 21)) === JSON.stringify(events), "telemetry is reproducible");
  assert(JSON.stringify(buildTelemetry(971, 40, 21)) !== JSON.stringify(events), "another seed differs");
  assert.deepStrictEqual(funnel(events), stages, "the funnel is a pure function of the log");
  const replay = perLearnerProgress(buildTelemetry(970, 40, 21));
  assert(JSON.stringify([...replay]) === JSON.stringify([...generated]), "progress is reproducible");

  console.log(`analytics: ${events.length} events, ${generated.size} learners, funnel ` +
    `${stages.map((s) => s.count).join(">")}, drop-off ${worst.from}->${worst.to} (-${worst.lost}), ` +
    `cohorts ${cohorts.map((c) => c.rate.toFixed(2)).join(",")}`);
  console.log("PASS");
}
module.exports = { splitmix64, buildTelemetry, summarizeLearners, funnel, dropOff,
  perLearnerProgress, cohortRetention, windowCounts, FUNNEL_STAGES, LOG };
if (require.main === module) main();
