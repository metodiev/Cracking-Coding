/** Task 971 - Implement spaced-repetition scheduling for problem revisit recommendations.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * An SM-2 style scheduler: every solved problem becomes a card carrying an ease factor, the last
 * interval, the repetition streak and a due day. A review applies a 0-5 grade: grades below 3 are
 * lapses (interval resets to one day, ease drops 0.2), otherwise the interval becomes 1, then 6,
 * then `round(previous interval * ease before the update)`, and the ease moves by the grade's delta
 * inside [1.3, 2.8]. A learner simulator drives a seeded grade stream day by day, and the
 * recommendation view is the due queue ordered by overdue-ness then card id.
 * Simplified: the day is the finest time unit, the grade stream is synthetic, and only the classic
 * SM-2 core is modelled (no FSRS difficulty/stability terms, no same-day re-learning steps).
 * Verified: the interval ladder against a hand-computed reference sequence, lapse resets, ease
 * clamps, per-mutation due invariants, queue ordering, daily review budgets and determinism.
 */
"use strict";
const assert = require("node:assert");

const MIN_EASE = 1.3;
const MAX_EASE = 2.8;
const clampEase = (value) => Math.min(MAX_EASE, Math.max(MIN_EASE, value));

function newCard(id, ease = 2.5) {
  return { id, ease, interval: 0, reps: 0, lapses: 0, due: 0, reviews: 0, history: [] };
}

/** One review: a lapse resets the streak, otherwise the interval follows the SM-2 ladder. */
function review(card, grade, day) {
  assert(Number.isInteger(grade) && grade >= 0 && grade <= 5, `grade ${grade} is outside 0..5`);
  const lapsed = grade < 3;
  const reps = lapsed ? 0 : card.reps + 1;
  const interval = lapsed ? 1 : reps === 1 ? 1 : reps === 2 ? 6 : Math.round(card.interval * card.ease);
  const delta = lapsed ? -0.2 : grade === 5 ? 0.1 : grade === 4 ? 0 : -0.14;
  const ease = clampEase(card.ease + delta);
  return { ...card, ease, interval, reps, lapses: card.lapses + (lapsed ? 1 : 0), due: day + interval,
    reviews: card.reviews + 1, history: [...card.history, { day, grade, interval, ease }] };
}

/** Apply a grade sequence and collect the resulting intervals (used for the reference check). */
function intervalPlan(grades, ease = 2.5) {
  let card = newCard("probe", ease);
  return grades.map((grade, day) => {
    card = review(card, grade, day);
    return card.interval;
  });
}

/** Hand-computed SM-2 ladder for five perfect reviews (ease 2.5 -> 2.8, capped). */
const PERFECT_SEQUENCE = [1, 6, 16, 45, 126];

/** splitmix64 - the only randomness in the simulator, so every schedule is reproducible. */
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
const nextGrade = (rng) => {
  const roll = rng.unit();
  return roll < 0.12 ? rng.int(3) : roll < 0.55 ? 4 : 5;
};

/** Recommendation queue: most overdue first, ties by card id so the order is stable. */
function dueQueue(cards, day, limit = Infinity) {
  return cards
    .filter((card) => card.due <= day)
    .sort((a, b) => (a.due - b.due) || (a.id < b.id ? -1 : 1))
    .slice(0, limit);
}

/** Day-by-day study session: each day reviews up to `dailyLimit` recommended cards. */
function simulate({ cards, days = 60, dailyLimit = 8, seed = 971 }) {
  const rng = splitmix64(seed);
  const state = new Map(cards.map((card) => [card.id, card]));
  const trace = [];
  const daily = [];
  for (let day = 1; day <= days; day += 1) {
    const backlog = [...state.values()].filter((card) => card.due <= day).length;
    const queue = dueQueue([...state.values()], day, dailyLimit);
    daily.push({ day, backlog, reviewed: queue.length });
    for (const card of queue) {
      const next = review(card, nextGrade(rng), day);
      state.set(card.id, next);
      trace.push({ day, id: card.id, grade: next.history.at(-1).grade, interval: next.interval,
        ease: next.ease, due: next.due });
    }
  }
  return { cards: [...state.values()].sort((a, b) => (a.id < b.id ? -1 : 1)), trace, daily };
}

const buildCards = (count) => Array.from({ length: count }, (_, i) => newCard(`c${String(i + 1).padStart(2, "0")}`));

function main() {
  // 1. Happy path: the interval ladder grows 1, 6, then previous * ease.
  let card = newCard("x");
  card = review(card, 4, 0);
  assert.strictEqual(card.interval, 1, "first successful review waits one day");
  assert.strictEqual(card.ease, 2.5, "grade 4 leaves the ease untouched");
  card = review(card, 4, 1);
  assert.strictEqual(card.interval, 6, "second review waits six days");
  const third = review(card, 4, 7);
  assert.strictEqual(third.interval, Math.round(6 * 2.5), "third interval uses the ease in force");
  assert.strictEqual(third.reps, 3, "three successful reps");

  // 2. Independent oracle: a hand-computed ladder for perfect grades.
  assert.deepStrictEqual(intervalPlan([5, 5, 5, 5, 5]), PERFECT_SEQUENCE, "perfect-grade reference ladder");
  assert.deepStrictEqual(intervalPlan([4, 4]), [1, 6], "grade 4 keeps the reference ease of 2.5");

  // 3. Lapses: grade below 3 resets the streak, shortens the interval and costs 0.2 ease.
  const matured = [5, 5, 5, 5].reduce((acc, grade, day) => review(acc, grade, day), newCard("m"));
  assert.strictEqual(matured.interval, 45, "four perfect reviews reach 45 days");
  const lapsed = review(matured, 2, 100);
  assert.strictEqual(lapsed.interval, 1, "a lapse resets to one day");
  assert.strictEqual(lapsed.reps, 0, "a lapse resets the streak");
  assert.strictEqual(lapsed.lapses, 1, "the lapse counter increments");
  assert.strictEqual(lapsed.ease.toFixed(2), String(clampEase(matured.ease - 0.2).toFixed(2)), "ease drops 0.2");
  assert.strictEqual(lapsed.due, 101, "the card returns tomorrow");
  assert(lapsed.history.length === matured.history.length + 1, "the lapse is recorded");

  // 4. Bounds: the ease factor is clamped, so intervals stay finite and never negative.
  const hammered = intervalPlan(Array(25).fill(3));
  const perfect = intervalPlan(Array(25).fill(5));
  assert.strictEqual(intervalPlan(Array(25).fill(3)).length, 25, "every grade produces an interval");
  assert(hammered.every((i) => i >= 1) && perfect.every((i) => i >= 1), "intervals are never below one day");
  assert(perfect.at(-1) > 1e6, "perfect grades explode the interval");
  const floored = Array(25).fill(3).reduce((c, _, i) => review(c, 3, i), newCard("f"));
  const ceiled = Array(25).fill(5).reduce((c, _, i) => review(c, 5, i), newCard("c"));
  assert.strictEqual(floored.ease, MIN_EASE, "ease bottoms out at 1.3");
  assert.strictEqual(ceiled.ease, MAX_EASE, "ease tops out at 2.8");
  assert.throws(() => review(newCard("z"), 7, 0), /outside 0\.\.5/, "illegal grades are rejected");

  // 5. Invariant after every mutation: due is exactly the review day plus the new interval.
  let probe = newCard("p");
  const grades = [5, 4, 3, 2, 5, 5, 0, 5, 4, 4, 3, 5];
  grades.forEach((grade, day) => {
    probe = review(probe, grade, day * 3);
    assert.strictEqual(probe.due, day * 3 + probe.interval, `due after grade ${grade} on day ${day * 3}`);
    assert(probe.ease >= MIN_EASE && probe.ease <= MAX_EASE, "ease stays clamped after every review");
    assert.strictEqual(probe.history.length, probe.reviews, "history length tracks the review count");
  });

  // 6. Recommendation queue: overdue first, then id; limits and empty days behave.
  const queue = dueQueue(buildCards(5), 0);
  assert.deepStrictEqual(queue.map((c) => c.id), ["c01", "c02", "c03", "c04", "c05"], "day 0: all due, by id");
  const mixed = [{ id: "b", due: 5 }, { id: "a", due: 5 }, { id: "c", due: 99 }];
  assert.deepStrictEqual(dueQueue(mixed, 10).map((c) => c.id), ["a", "b"], "id breaks due-date ties");
  assert.deepStrictEqual(dueQueue(mixed, 200).map((c) => c.id), ["a", "b", "c"], "the far card joins later");
  assert.strictEqual(dueQueue(buildCards(5), -1).length, 0, "no card is due before its first review");
  assert.strictEqual(dueQueue(buildCards(5), 0, 0).length, 0, "a zero limit recommends nothing");

  // 7. Simulation: daily budgets, no repeat inside a day, full coverage and a reference count.
  const run = simulate({ cards: buildCards(25), days: 60, dailyLimit: 8, seed: 971 });
  assert(run.trace.every((step) => step.due === step.day + step.interval && step.interval >= 1), "due invariant");
  assert(run.trace.every((step) => step.grade >= 0 && step.grade <= 5), "grades stay in range");
  assert(run.daily.every((d) => d.reviewed <= 8), "the daily budget is respected");
  assert(run.daily.every((d) => d.reviewed <= d.backlog), "never review more than the backlog");
  for (const day of run.daily) {
    const ids = run.trace.filter((s) => s.day === day.day).map((s) => s.id);
    assert(new Set(ids).size === ids.length, `day ${day.day} reviews each card at most once`);
  }
  assert.strictEqual(new Set(run.trace.map((s) => s.id)).size, 25, "every card is revisited");
  assert.strictEqual(run.trace.length, run.cards.reduce((sum, c) => sum + c.reviews, 0), "trace matches history");
  const firstPass = run.cards.filter((c) => c.lapses === 0);
  assert(firstPass.length > 0, "some cards are never lapsed");
  assert(firstPass.every((c) => c.reps === c.reviews), "without a lapse every review is a repetition");
  assert(run.daily.some((d) => d.backlog > d.reviewed), "the backlog sometimes exceeds the daily budget");

  // 8. Determinism: one seed, one schedule.
  const again = simulate({ cards: buildCards(25), days: 60, dailyLimit: 8, seed: 971 });
  assert(JSON.stringify(again.trace) === JSON.stringify(run.trace), "the same seed replays the schedule");
  assert(JSON.stringify(simulate({ cards: buildCards(25), days: 60, dailyLimit: 8, seed: 972 }).trace)
    !== JSON.stringify(run.trace), "another seed changes the grades");
  assert.strictEqual(simulate({ cards: [], days: 5, seed: 1 }).trace.length, 0, "an empty collection is stable");

  const due = run.cards.filter((c) => c.due > 60).length;
  console.log(`spaced repetition: ${run.trace.length} reviews over 60 days, ${run.cards.length} cards, ` +
    `${due} pushed beyond the horizon, mean ease ${(run.cards.reduce((s, c) => s + c.ease, 0) / 25).toFixed(2)}`);
  console.log("PASS");
}
module.exports = { MIN_EASE, MAX_EASE, PERFECT_SEQUENCE, newCard, review, intervalPlan,
  splitmix64, nextGrade, dueQueue, simulate, buildCards };
if (require.main === module) main();
