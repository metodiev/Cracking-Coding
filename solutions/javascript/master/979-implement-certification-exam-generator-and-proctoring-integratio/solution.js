/** Task 979 - Implement certification exam generator and proctoring integration.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A blueprint fixes the form length and the share of each domain; the assembler apportions seats
 * with the largest-remainder method, then draws items per domain preferring low exposure and a
 * balanced difficulty mix, skipping retired items. Proctoring consumes a session's event stream
 * (tab blur, clipboard use, concurrent sessions, identity mismatch), caps the credit per event type
 * so repeated noise cannot run away, and maps the weighted anomaly score to clear / review /
 * invalidate. The two halves meet in the certification rule: an invalidated session is never
 * certified, however high the raw score.
 * Simplified: cohorts, items and events are generated from a seeded model (no real item bank or
 * proctoring client) and the anomaly weights are hand-tuned constants. Verified: blueprint counts
 * against an independent apportionment, no duplicates or retired items, exposure control, the
 * anomaly score against a direct recomputation, verdict thresholds, the certification rule and
 * determinism.
 */
"use strict";
const assert = require("node:assert");

const BLUEPRINT = { length: 60, passMark: 0.7,
  domains: { arrays: 0.4, graphs: 0.3, dp: 0.2, strings: 0.1 } };
const BANDS = ["easy", "medium", "hard"];
const EVENT_WEIGHTS = { blur: 2, clipboard: 6, concurrent: 12, identity: 25 };
const EVENT_CAPS = { blur: 3, clipboard: 2, concurrent: 1, identity: 1 };
const VERDICTS = [{ verdict: "invalidate", min: 25 }, { verdict: "review", min: 8 }, { verdict: "clear", min: 0 }];

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

/** Item pool: ids, domain, difficulty band, an exposure counter and a retirement flag. */
function makePool(seed, perDomain = 90) {
  const rng = splitmix64(seed);
  const pool = [];
  for (const domain of Object.keys(BLUEPRINT.domains)) {
    for (let i = 0; i < perDomain; i += 1) {
      pool.push({
        id: `${domain}-${String(i).padStart(3, "0")}`,
        domain,
        band: BANDS[i % BANDS.length],
        exposure: i < 4 ? 12 : rng.int(6),
        retired: i % 45 === 44,
      });
    }
  }
  return pool;
}

/** Largest-remainder apportionment of `seats` over the blueprint weights. */
function apportion(seats, domains) {
  const total = Object.values(domains).reduce((sum, weight) => sum + weight, 0);
  const exact = Object.entries(domains).map(([domain, weight]) => ({ domain, exact: (seats * weight) / total }));
  const seatsGiven = new Map(exact.map((row) => [row.domain, Math.floor(row.exact)]));
  let left = seats - [...seatsGiven.values()].reduce((sum, value) => sum + value, 0);
  const byRemainder = [...exact].sort((a, b) => (b.exact - Math.floor(b.exact)) - (a.exact - Math.floor(a.exact))
    || (a.domain < b.domain ? -1 : 1));
  for (const row of byRemainder) {
    if (left <= 0) break;
    seatsGiven.set(row.domain, seatsGiven.get(row.domain) + 1);
    left -= 1;
  }
  return seatsGiven;
}

/** Assemble a form: per domain, cheapest exposure first, rotating through the difficulty bands. */
function assembleForm({ pool, blueprint = BLUEPRINT, seed = 979, exposureLimit = 8 }) {
  const rng = splitmix64(seed);
  const seats = apportion(blueprint.length, blueprint.domains);
  const form = [];
  for (const [domain, count] of seats) {
    const eligible = pool
      .filter((item) => item.domain === domain && !item.retired && item.exposure <= exposureLimit)
      .sort((a, b) => (a.exposure - b.exposure) || (a.id < b.id ? -1 : 1));
    if (eligible.length < count) throw new Error(`domain ${domain} has only ${eligible.length} usable items for ${count} seats`);
    const chosen = [];
    for (let band = 0; chosen.length < count; band += 1) {
      const wanted = BANDS[band % BANDS.length];
      const next = eligible.find((item) => item.band === wanted && !chosen.includes(item));
      chosen.push(next ?? eligible.find((item) => !chosen.includes(item)));
    }
    form.push(...chosen);
  }
  for (let i = form.length - 1; i > 0; i -= 1) {
    const j = rng.int(i + 1);
    [form[i], form[j]] = [form[j], form[i]];
  }
  return { blueprint, seats, items: form };
}

/** Proctoring: cap each event type, weight it, and map the total to a verdict. */
function scoreSession(events, { weights = EVENT_WEIGHTS, caps = EVENT_CAPS } = {}) {
  const counts = new Map();
  const unknown = [];
  for (const event of events) {
    if (!(event.type in weights)) { unknown.push(event.type); continue; }
    counts.set(event.type, (counts.get(event.type) || 0) + 1);
  }
  const contributions = [...counts].map(([type, count]) => ({ type, count,
    counted: Math.min(count, caps[type]), points: Math.min(count, caps[type]) * weights[type] }))
    .sort((a, b) => (b.points - a.points) || (a.type < b.type ? -1 : 1));
  const score = contributions.reduce((sum, row) => sum + row.points, 0);
  const verdict = VERDICTS.find((row) => score >= row.min).verdict;
  return { score, verdict, contributions, unknown,
    evidence: contributions.filter((row) => row.points > 0).map((row) => `${row.type}x${row.counted}`) };
}

/** Certification rule: pass mark plus a proctoring verdict that is not "invalidate". */
function certify({ correct, total, proctoring, passMark = BLUEPRINT.passMark }) {
  const ratio = total === 0 ? 0 : correct / total;
  if (proctoring.verdict === "invalidate") return { certified: false, ratio, reason: "proctoring invalidated the session" };
  if (ratio < passMark) return { certified: false, ratio, reason: `score ${ratio.toFixed(2)} below the pass mark` };
  if (proctoring.verdict === "review") return { certified: false, ratio, reason: "held for proctor review" };
  return { certified: true, ratio, reason: "passed cleanly" };
}

function main() {
  const pool = makePool(979);
  const form = assembleForm({ pool, seed: 979 });

  // 1. Happy path: the form is exactly blueprint-sized, unique and free of retired items.
  assert.strictEqual(form.items.length, 60, "the form has the blueprint length");
  assert.strictEqual(new Set(form.items.map((item) => item.id)).size, 60, "no item appears twice");
  assert(form.items.every((item) => !item.retired), "retired items are never used");
  assert(form.items.every((item) => item.exposure <= 8), "over-exposed items are skipped");

  // 2. Oracle: an independent apportionment (exact share rounding) reproduces the seat counts.
  const oracleSeats = new Map();
  for (const [domain, weight] of Object.entries(BLUEPRINT.domains)) {
    oracleSeats.set(domain, Math.round(60 * weight));
  }
  assert.strictEqual([...oracleSeats.values()].reduce((a, b) => a + b, 0), 60, "weights are exact for this form");
  for (const [domain, count] of oracleSeats) assert.strictEqual(form.seats.get(domain), count, `seats for ${domain}`);
  for (const [domain, count] of form.seats) {
    const actual = form.items.filter((item) => item.domain === domain).length;
    assert.strictEqual(actual, count, `${domain} items match its seats`);
  }

  // 3. Exposure control: the assembler is strictly cheaper than a uniformly random draw.
  const meanExposure = (items) => items.reduce((sum, item) => sum + item.exposure, 0) / items.length;
  const rng = splitmix64(1);
  const randomDraw = Array.from({ length: 60 }, () => pool[rng.int(pool.length)]);
  assert(meanExposure(form.items) < meanExposure(randomDraw), "cheap items are preferred");
  assert(form.items.every((item) => item.exposure <= 2), "chosen items are low-exposure");
  for (const item of form.items) {
    const rivals = pool.filter((rival) => rival.domain === item.domain && rival.band === item.band
      && !rival.retired && !form.items.includes(rival));
    assert(Math.min(...rivals.map((rival) => rival.exposure)) >= item.exposure,
      `${item.id} is the freshest of its band`);
  }

  // 4. Proctoring: the weighted score equals a direct recomputation from the raw events.
  const session = ["blur", "blur", "blur", "blur", "blur", "clipboard", "clipboard", "clipboard", "identity"];
  const proctoring = scoreSession(session.map((type) => ({ type })));
  const direct = Math.min(5, 3) * 2 + Math.min(3, 2) * 6 + 1 * 25;
  assert.strictEqual(proctoring.score, direct, `score ${proctoring.score} vs direct ${direct}`);
  assert.strictEqual(proctoring.verdict, "invalidate", "an identity mismatch invalidates");
  assert.deepStrictEqual(proctoring.contributions.map((row) => row.type), ["identity", "clipboard", "blur"],
    "contributions are ordered by weight");
  const capped = scoreSession(Array.from({ length: 50 }, () => ({ type: "blur" })));
  assert.strictEqual(capped.score, 3 * 2, "repeated blur events stop at the cap");
  const unknown = scoreSession([{ type: "blur" }, { type: "mouse-jiggle" }]);
  assert.deepStrictEqual(unknown.unknown, ["mouse-jiggle"], "unknown event types are reported, not scored");
  assert.strictEqual(unknown.score, 2, "unknown events add nothing");

  // 5. Verdict thresholds: the boundaries are inclusive at the lower edge.
  assert.strictEqual(scoreSession([]).verdict, "clear", "no events is clear");
  assert.strictEqual(scoreSession([{ type: "blur" }, { type: "blur" }, { type: "blur" }, { type: "blur" }]).verdict,
    "clear", "3 counted blurs (6 points) stay clear");
  assert.strictEqual(scoreSession([{ type: "clipboard" }, { type: "clipboard" }]).verdict, "review",
    "12 points asks for a review");
  assert.strictEqual(scoreSession([{ type: "concurrent" }, { type: "clipboard" }, { type: "clipboard" }]).score, 24,
    "24 points is just below the line");
  assert.strictEqual(scoreSession([{ type: "concurrent" }, { type: "clipboard" }, { type: "clipboard" },
    { type: "blur" }, { type: "blur" }, { type: "blur" }]).verdict, "invalidate", "30 points invalidates");
  assert.strictEqual(scoreSession([{ type: "identity" }]).score, 25, "identity alone invalidates");

  // 6. Certification rule: an invalidated session never certifies, a reviewed one is held.
  assert.deepStrictEqual(certify({ correct: 55, total: 60, proctoring: scoreSession([]) }),
    { certified: true, ratio: 55 / 60, reason: "passed cleanly" });
  const invalidated = certify({ correct: 60, total: 60, proctoring });
  assert(!invalidated.certified && invalidated.reason.includes("proctoring"), "a perfect score cannot rescue it");
  const reviewed = certify({ correct: 60, total: 60, proctoring: scoreSession([{ type: "clipboard" }, { type: "clipboard" }]) });
  assert(!reviewed.certified && reviewed.reason.includes("review"), "review holds the result");
  assert(!certify({ correct: 40, total: 60, proctoring: scoreSession([]) }).certified, "below the pass mark");
  assert(!certify({ correct: 0, total: 0, proctoring: scoreSession([]) }).certified, "an empty session cannot pass");

  // 7. Boundaries: a domain without usable items, and a blueprint smaller than the domain count.
  assert.throws(() => assembleForm({ pool: pool.filter((item) => item.domain !== "dp"), seed: 3 }),
    /has only 0 usable items/, "a starved domain is reported");
  const tiny = assembleForm({ pool, blueprint: { length: 4, domains: BLUEPRINT.domains }, seed: 3 });
  assert.strictEqual(tiny.items.length, 4, "a four-seat form is still blueprint-sized");
  assert.deepStrictEqual([...tiny.seats].map(([domain, seats]) => [domain, seats]),
    [["arrays", 2], ["graphs", 1], ["dp", 1], ["strings", 0]], "largest remainder decides the odd seats");
  const small = assembleForm({ pool, blueprint: { length: 4, domains: { strings: 1 } }, seed: 3 });
  assert.strictEqual(small.items.length, 4, "a four-seat single-domain form works");
  const tight = apportion(2, { a: 1, b: 1, c: 1 });
  assert.deepStrictEqual([...tight.values()].sort(), [0, 1, 1], "largest remainder drops a domain when seats run out");

  // 8. Determinism: the seed pins the form; a different seed gives another valid form.
  assert(JSON.stringify(assembleForm({ pool, seed: 979 })) === JSON.stringify(form), "forms are reproducible");
  const other = assembleForm({ pool, seed: 980 });
  assert(JSON.stringify(other.items) !== JSON.stringify(form.items), "another seed shuffles differently");
  assert.deepStrictEqual([...other.seats], [...form.seats], "the blueprint does not depend on the seed");

  console.log(`certification: ${form.items.length}-item form from ${pool.length} items ` +
    `(${[...form.seats].map(([d, n]) => `${d}:${n}`).join(" ")}), proctoring score ${proctoring.score} ` +
    `-> ${proctoring.verdict}, evidence ${proctoring.evidence.join(",")}`);
  console.log("PASS");
}
module.exports = { BLUEPRINT, BANDS, EVENT_WEIGHTS, EVENT_CAPS, VERDICTS, splitmix64, makePool, apportion,
  assembleForm, scoreSession, certify };
if (require.main === module) main();
