/** Task 969 - Implement curriculum mapping to academic topics and pacing suggestions.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * The curriculum is a DAG of topics (each with prerequisite topics and estimated study hours) plus
 * a problem bank whose items are tagged with a topic and a difficulty weight. The mapper folds the
 * bank into per-topic load, flags items whose tag has no topic and detects prerequisite cycles;
 * the pacer performs a deterministic Kahn topological sort and a first-fit scheduling pass that
 * never starts a topic before every prerequisite has finished and never exceeds the weekly hours
 * budget; the recommender then offers the unlocked, not-yet-mastered topics in curriculum order.
 * Simplified: hours are integers and the calendar has 52 fixed weeks (no holidays, no learner
 * speed-up), which keeps the schedule reproducible. Verified: load conservation, prerequisite and
 * capacity invariants, an independent longest-path (critical-path) oracle for the makespan lower
 * bound, cycle rejection, the empty-curriculum boundary and byte-identical determinism.
 */
"use strict";
const assert = require("node:assert");

const TOPICS = {
  arrays: { hours: 4, prereqs: [] },
  strings: { hours: 3, prereqs: [] },
  hashing: { hours: 5, prereqs: ["arrays"] },
  sorting: { hours: 4, prereqs: ["arrays"] },
  twoPointers: { hours: 3, prereqs: ["arrays", "sorting"] },
  graphs: { hours: 8, prereqs: ["hashing"] },
  dynamicProgramming: { hours: 10, prereqs: ["hashing", "sorting"] },
  greedy: { hours: 5, prereqs: ["sorting"] },
  bitManipulation: { hours: 3, prereqs: ["arrays"] },
};
const BANK = [
  { id: "p01", topic: "arrays", difficulty: 2 }, { id: "p02", topic: "arrays", difficulty: 3 },
  { id: "p03", topic: "strings", difficulty: 1 }, { id: "p04", topic: "hashing", difficulty: 4 },
  { id: "p05", topic: "sorting", difficulty: 3 }, { id: "p06", topic: "twoPointers", difficulty: 3 },
  { id: "p07", topic: "graphs", difficulty: 5 }, { id: "p08", topic: "graphs", difficulty: 4 },
  { id: "p09", topic: "dynamicProgramming", difficulty: 5 }, { id: "p10", topic: "greedy", difficulty: 2 },
  { id: "p11", topic: "bitManipulation", difficulty: 2 }, { id: "p12", topic: "geometry", difficulty: 2 },
];

/** Map the problem bank onto topics; unknown tags are reported instead of silently dropped. */
function mapProblems(bank, topics) {
  const byTopic = new Map(Object.keys(topics).map((name) => [name, []]));
  const unmapped = [];
  for (const item of bank) {
    if (!byTopic.has(item.topic)) unmapped.push({ ...item, action: `create topic "${item.topic}"` });
    else byTopic.get(item.topic).push(item.id);
  }
  const load = new Map([...byTopic].map(([name, ids]) => [name, ids.reduce(
    (sum, id) => sum + bank.find((item) => item.id === id).difficulty, 0)]));
  return { byTopic, load, unmapped };
}

/** Kahn topological sort with alphabetical tie-breaking, so the order is reproducible. */
function topoOrder(topics) {
  const indegree = new Map(Object.keys(topics).map((n) => [n, topics[n].prereqs.length]));
  const dependents = new Map(Object.keys(topics).map((n) => [n, []]));
  for (const [name, topic] of Object.entries(topics)) {
    for (const pre of topic.prereqs) {
      if (!(pre in topics)) throw new Error(`${name} depends on unknown topic ${pre}`);
      dependents.get(pre).push(name);
    }
  }
  const ready = [...indegree].filter(([, d]) => d === 0).map(([n]) => n).sort();
  const order = [];
  while (ready.length) {
    const name = ready.shift();
    order.push(name);
    for (const next of dependents.get(name).sort()) {
      indegree.set(next, indegree.get(next) - 1);
      if (indegree.get(next) === 0) { ready.push(next); ready.sort(); }
    }
  }
  if (order.length !== Object.keys(topics).length) throw new Error("prerequisite cycle detected");
  return order;
}

/** Longest-path DP: total hours on the critical path *through* each topic (oracle input). */
function criticalHours(topics, load) {
  const depth = new Map();
  for (const name of topoOrder(topics)) {
    const own = topics[name].hours + (load.get(name) || 0);
    const pre = Math.max(0, ...topics[name].prereqs.map((p) => depth.get(p)));
    depth.set(name, own + pre);
  }
  return depth;
}

/** First-fit pacing: topics in topological order, each filling week slots within the budget. */
function schedule(topics, load, weeklyHours, maxWeeks = 52) {
  const order = topoOrder(topics);
  const finish = new Map();
  const used = new Map();
  const assigned = new Map();
  const plan = [];
  for (const name of order) {
    const earliest = 1 + Math.max(0, ...topics[name].prereqs.map((p) => finish.get(p) || 0));
    let remaining = topics[name].hours + (load.get(name) || 0);
    let week = earliest;
    let startWeek = null;
    while (remaining > 0) {
      if (week > maxWeeks) throw new Error(`${name} cannot finish within ${maxWeeks} weeks`);
      const free = weeklyHours - (used.get(week) || 0);
      const take = Math.min(remaining, free);
      if (take > 0) {
        used.set(week, (used.get(week) || 0) + take);
        assigned.set(name, (assigned.get(name) || 0) + take);
        remaining -= take;
        if (startWeek === null) startWeek = week;
      }
      if (remaining > 0) week += 1;
    }
    finish.set(name, week);
    plan.push({ topic: name, startWeek, endWeek: week, hours: topics[name].hours + (load.get(name) || 0) });
  }
  const weeks = [...new Set(plan.flatMap((entry) => [...Array(entry.endWeek - entry.startWeek + 1)]
    .map((_, i) => entry.startWeek + i)))].sort((a, b) => a - b);
  return { plan, weeks, used, assigned, finish };
}

/** Topics whose prerequisites are all mastered, in curriculum order, capped by `limit`. */
function suggestNext(topics, load, masteredIds, limit = 3) {
  const mastered = new Set(masteredIds);
  return topoOrder(topics)
    .filter((name) => !mastered.has(name) && topics[name].prereqs.every((p) => mastered.has(p)))
    .map((name) => ({ topic: name, hours: topics[name].hours + (load.get(name) || 0) }))
    .slice(0, limit);
}

function main() {
  const mapped = mapProblems(BANK, TOPICS);
  const order = topoOrder(TOPICS);

  // 1. Mapping: known tags land in their bucket, the unknown tag is surfaced with an action.
  assert.strictEqual(mapped.unmapped.length, 1, "one problem has no matching topic");
  assert.strictEqual(mapped.unmapped[0].id, "p12");
  assert(mapped.unmapped[0].action.includes("geometry"), "unmapped items suggest a new topic");
  assert.deepStrictEqual(mapped.byTopic.get("graphs"), ["p07", "p08"], "tagged problems stay grouped");
  assert.deepStrictEqual(mapped.byTopic.get("geometry"), undefined, "unknown topics get no bucket");

  // 2. Independent oracle: per-topic load recomputed by a direct filter over the bank.
  for (const [name, ids] of mapped.byTopic) {
    const expected = BANK.filter((item) => item.topic === name).reduce((s, item) => s + item.difficulty, 0);
    assert.strictEqual(mapped.load.get(name), expected, `load for ${name} must equal the plain sum`);
  }
  const totalLoad = [...mapped.load.values()].reduce((a, b) => a + b, 0);
  assert.strictEqual(totalLoad, BANK.reduce((s, i) => s + (mapped.byTopic.has(i.topic) ? i.difficulty : 0), 0));

  // 3. Topological order: every prerequisite precedes its dependents, and the order is total.
  assert.strictEqual(order.length, Object.keys(TOPICS).length, "the order covers every topic");
  const position = new Map(order.map((name, i) => [name, i]));
  for (const [name, topic] of Object.entries(TOPICS)) {
    for (const pre of topic.prereqs) assert(position.get(pre) < position.get(name), `${pre} before ${name}`);
  }
  assert.deepStrictEqual(order.slice(0, 2), ["arrays", "bitManipulation"], "alphabetical tie-break after arrays");
  assert.deepStrictEqual(topoOrder(TOPICS), order, "the topological order is reproducible");

  // 4. Pacing invariants: prerequisite ordering, capacity budget and span bound per topic.
  const weekly = 6;
  const paced = schedule(TOPICS, mapped.load, weekly);
  for (const entry of paced.plan) {
    assert(entry.startWeek >= 1 && entry.endWeek <= 52, `${entry.topic} lands inside the calendar`);
    for (const pre of TOPICS[entry.topic].prereqs) {
      assert(paced.finish.get(pre) < entry.startWeek, `${pre} finishes before ${entry.topic} starts`);
    }
    assert(paced.assigned.get(entry.topic) === entry.hours, `${entry.topic} is allocated its full ${entry.hours}h`);
    assert(entry.startWeek >= 1, `${entry.topic} starts inside the calendar`);
  }
  for (const [week, hours] of paced.used) assert(hours <= weekly, `week ${week} stays within ${weekly}h`);
  assert(paced.plan[0].topic === "arrays" && paced.plan[0].startWeek === 1, "the schedule starts at a root in week 1");

  // 5. Oracle: the makespan cannot beat the critical-path lower bound, nor waste a whole week.
  const critical = criticalHours(TOPICS, mapped.load);
  const lowerBound = Math.ceil(Math.max(...critical.values()) / weekly);
  const totalHours = [...mapped.load.entries()].reduce((s, [n, l]) => s + TOPICS[n].hours + l, 0);
  const upperBound = Math.ceil(totalHours / weekly) + Math.max(...paced.plan.map((e) => e.endWeek - e.startWeek));
  assert(paced.weeks.length >= lowerBound, `makespan ${paced.weeks.length} >= lower bound ${lowerBound}`);
  assert(paced.weeks.length <= upperBound, "makespan stays inside the greedy upper bound");
  assert(critical.get("dynamicProgramming") > critical.get("hashing"), "DP sits deeper than hashing");

  // 6. Round-trip: total scheduled hours equal the curriculum plus problem load.
  const scheduled = paced.plan.reduce((s, e) => s + e.hours, 0);
  assert.strictEqual(scheduled, totalHours, "no hour is lost or duplicated");

  // 7. Recommendations plus boundaries: locked topics are never suggested, empty input is fine.
  const fresh = suggestNext(TOPICS, mapped.load, []);
  assert.deepStrictEqual(fresh.map((s) => s.topic), ["arrays", "strings"], "only unlocked roots are offered");
  const afterArrays = suggestNext(TOPICS, mapped.load, ["arrays"]);
  assert(!afterArrays.some((s) => s.topic === "arrays"), "mastered topics are excluded");
  assert(afterArrays.some((s) => s.topic === "hashing") && !afterArrays.some((s) => s.topic === "twoPointers"),
    "twoPointer stays locked until sorting is mastered");
  assert.deepStrictEqual(suggestNext(TOPICS, mapped.load, Object.keys(TOPICS)), [], "nothing left to suggest");
  assert.deepStrictEqual(schedule({}, new Map(), weekly).plan, [], "empty curriculum schedules nothing");
  assert.throws(() => topoOrder({ a: { hours: 1, prereqs: ["b"] }, b: { hours: 1, prereqs: ["a"] } }),
    /cycle/, "cyclic prerequisites are rejected");
  assert.throws(() => topoOrder({ a: { hours: 1, prereqs: ["ghost"] } }), /unknown topic/, "dangling prereq");
  assert.throws(() => schedule(TOPICS, mapped.load, weekly, 3), /within 3 weeks/, "an impossible calendar fails");

  // 8. Determinism and a tighter budget: the same inputs always give the same plan.
  const again = schedule(TOPICS, mapped.load, weekly);
  assert(JSON.stringify(again.plan) === JSON.stringify(paced.plan), "schedules are reproducible");
  const tight = schedule(TOPICS, mapped.load, 15);
  assert(tight.weeks.length < paced.weeks.length, "a bigger weekly budget compresses the schedule");

  console.log(`curriculum: ${order.length} topics, ${BANK.length} problems, ${paced.weeks.length} weeks at ${weekly}h ` +
    `(critical lower bound ${lowerBound}), unmapped ${mapped.unmapped.length}`);
  console.log("PASS");
}
module.exports = { TOPICS, BANK, mapProblems, topoOrder, criticalHours, schedule, suggestNext };
if (require.main === module) main();
