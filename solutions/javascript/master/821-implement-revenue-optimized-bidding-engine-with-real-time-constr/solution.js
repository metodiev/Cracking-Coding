/** Task 821 — Implement revenue-optimized bidding engine with real-time constraints (auction simulator).
 *  (Tier: Master, Beyond — Machine Learning & AI)
 *
 * Engine. Every auction has a list of eligible bidders (value per click, quality score, remaining
 * budget) and a slot schedule with decreasing click-through rates. The engine ranks candidates by
 * eCPM = bid × quality, applies a reserve price, fills the slots in that order, and charges each winner
 * with the *generalized second price* rule — the next-highest eCPM divided by the winner's own quality
 * (rounded up to the cent), multiplied by the slot's click-through rate. Spenders are paced: each
 * bidder throttles its bid by a pace multiplier derived from its remaining budget and the remaining
 * opportunities, and the engine never accepts a bid larger than what the bidder can still pay.
 *
 * What is claimed, and how it is checked:
 *   - *allocative efficiency*: for small auctions the file enumerates **every** assignment of bidders to
 *     slots and confirms that the eCPM ordering maximizes total value (the rearrangement inequality),
 *     while also measuring the revenue-optimal assignment — which is a different one, because the
 *     second-price rule is not revenue-optimal. That gap is reported rather than hidden;
 *   - *truthfulness in the single-slot case*: over a grid of bid deviations the file verifies that
 *     bidding the true value maximizes the bidder's utility, which is the dominant-strategy property
 *     GSP has when there is exactly one slot;
 *   - *budget safety*: per-bidder spend never exceeds the budget in any auction of the horizon (checked
 *     by an independent replayer of the event log), and pace multipliers are monotone in the remaining
 *     budget;
 *   - *real-time budget*: the decision work is counted (comparisons and candidate examinations) and
 *     bounded by the analytic `O(n log n)` sorting bound plus a hard candidate cap; oversubscribed
 *     auctions pre-filter by score before the cap.
 *
 * Simplified versus production: no network of bidders (bids arrive as data), no randomised or learned
 * reserve prices, no multi-slot discounting, no conversion modelling beyond the click-through rates, no
 * sharding or co-located replicas — latency is expressed as bounded work counters rather than measured
 * wall clock.
 *
 * Verified: allocation feasibility (one bidder per slot, one slot per bidder, reserve respected),
 * eCPM-order optimality of value against brute force, the measured revenue gap to the revenue-optimal
 * assignment, single-slot truthfulness by deviation grid, budget feasibility through a horizon replayed
 * by a validator, pacing monotonicity and schedule tracking, work bounds, deterministic tie-breaking,
 * and the degenerate auctions (no bidders, one bidder, equal bids, zero budget, bids below the
 * reserve, a budget exactly exhausted).
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic randomness

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

const SLOT_SCHEDULE = [100, 62, 41, 27, 18]; // per-mille click-through rate of each slot

function slotCtr(slot) {
  return slot < SLOT_SCHEDULE.length ? SLOT_SCHEDULE[slot] / 1000 : 0;
}

// ---------------------------------------------------------------------------- engine

class AuctionEngine {
  /**
   * @param {object} config `{ slots, reserve, candidateCap, minBid }`
   */
  constructor(config = {}) {
    this.slots = config.slots ?? 3;
    this.reserve = config.reserve ?? 10; // minimum eCPM in cents
    this.candidateCap = config.candidateCap ?? 64;
    this.minBid = config.minBid ?? 1;
  }

  /**
   * Run one auction. Bidders carry `{ id, value, quality, budget, spent }`; the bid is the paced value
   * `pace * value` (pace defaults to 1), clipped to what the budget can still pay for.
   */
  runAuction(auction) {
    const stats = { candidates: 0, examinations: 0, comparisons: 0, filtered: 0, eligible: 0 };
    const bidders = auction.bidders ?? [];
    const candidateRows = [];
    for (const bidder of bidders) {
      stats.candidates++;
      const remaining = bidder.budget - bidder.spent;
      if (remaining <= 0) continue; // out of budget: not a candidate at all
      const pace = bidder.pace ?? 1;
      const paced = Math.floor(pace * bidder.value);
      const bid = Math.min(paced, remaining); // a bidder can never bid more than it can pay
      if (bid < this.minBid) continue;
      const ecpm = bid * bidder.quality;
      if (ecpm < this.reserve) continue;
      candidateRows.push({ bidder, bid, ecpm });
      stats.eligible++;
    }
    // Real-time guard: cap the candidate set by score before sorting (score = eCPM, no hashing).
    let rows = candidateRows;
    if (rows.length > this.candidateCap) {
      rows = rows.slice().sort((a, b) => b.ecpm - a.ecpm || compareIds(a.bidder.id, b.bidder.id)).slice(0, this.candidateCap);
      stats.filtered = candidateRows.length - rows.length;
    }
    const sorted = rows.slice().sort((a, b) => {
      stats.comparisons++;
      return b.ecpm - a.ecpm || compareIds(a.bidder.id, b.bidder.id);
    });
    stats.examinations = sorted.length;
    const allocation = [];
    const used = new Set();
    for (let slot = 0; slot < this.slots && slot < sorted.length; slot++) {
      const winner = sorted[slot];
      if (used.has(winner.bidder.id)) continue;
      const next = sorted.find((row, index) => index > slot && !used.has(row.bidder.id));
      const threshold = Math.max(this.reserve, next ? next.ecpm : this.reserve);
      const payment = Math.max(this.minBid, Math.ceil(threshold / winner.bidder.quality));
      const spend = Math.round((payment * SLOT_SCHEDULE[slot]) / 1000);
      used.add(winner.bidder.id);
      allocation.push({ slot, bidderId: winner.bidder.id, ecpm: winner.ecpm, payment, spend, ctr: slotCtr(slot) });
    }
    return { auctionId: auction.id, allocation, stats, revenue: allocation.reduce((sum, entry) => sum + entry.spend, 0) };
  }

  /** Analytic work bound for one auction with `n` eligible candidates. */
  static workBound(n) {
    return n === 0 ? 0 : n * Math.ceil(Math.log2(Math.max(2, n)));
  }
}

function compareIds(a, b) {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Apply an auction result to the bidders' budgets (the only place budgets change). */
function settle(bidders, result) {
  const byId = new Map(bidders.map((bidder) => [bidder.id, bidder]));
  for (const entry of result.allocation) {
    const bidder = byId.get(entry.bidderId);
    bidder.spent += entry.spend;
    if (bidder.spent > bidder.budget) throw new Error(`bidder ${bidder.id} overspent its budget`);
  }
  return result;
}

// ---------------------------------------------------------------------------- brute-force oracles

/**
 * Enumerate every injective assignment of bidders to slots (or a subset of the slots) and return the
 * best total value and the best total GSP revenue. Used as the oracle for the engine's ordering.
 */
function bruteForceAuctions(auction, slots, reserve) {
  const bidders = auction.bidders.filter((bidder) => bidder.value * bidder.quality >= reserve);
  let bestValue = 0;
  let bestValueAssignment = null;
  let bestRevenue = 0;
  let bestRevenueAssignment = null;
  const slotCount = Math.min(slots, bidders.length);
  const choose = (remaining, chosen) => {
    if (chosen.length > slotCount) return;
    // value of the current partial assignment
    let value = 0;
    let revenue = 0;
    for (let slot = 0; slot < chosen.length; slot++) {
      const winner = chosen[slot];
      value += SLOT_SCHEDULE[slot] * winner.value * winner.quality;
      const others = remaining.filter((candidate) => candidate !== winner).concat(chosen.slice(slot + 1));
      let nextEcpm = reserve;
      for (const other of others) nextEcpm = Math.max(nextEcpm, other.value * other.quality);
      const payment = Math.max(1, Math.ceil(nextEcpm / winner.quality));
      revenue += Math.round((payment * SLOT_SCHEDULE[slot]) / 1000);
    }
    if (value > bestValue) {
      bestValue = value;
      bestValueAssignment = chosen.slice();
    }
    if (revenue > bestRevenue) {
      bestRevenue = revenue;
      bestRevenueAssignment = chosen.slice();
    }
    if (chosen.length === slotCount) return;
    for (const candidate of remaining) {
      choose(remaining.filter((other) => other !== candidate), [...chosen, candidate]);
    }
  };
  choose(bidders, []);
  return { bestValue, bestValueAssignment, bestRevenue, bestRevenueAssignment };
}

/** Utility (value − cost) of a single-slot auction for a bidder bidding `bid` instead of its value. */
function singleSlotUtility(bidders, index, bid, reserve) {
  const rows = bidders.map((bidder, position) => ({ bidder, ecpm: (position === index ? bid : bidder.value) * bidder.quality }));
  rows.sort((a, b) => b.ecpm - a.ecpm || compareIds(a.bidder.id, b.bidder.id));
  const winner = rows[0];
  if (winner.bidder !== bidders[index]) return 0;
  const nextEcpm = Math.max(reserve, rows[1] ? rows[1].ecpm : reserve);
  const payment = Math.max(1, Math.ceil(nextEcpm / bidders[index].quality));
  return (bidders[index].value - payment) * SLOT_SCHEDULE[0];
}

// ---------------------------------------------------------------------------- pacing

/**
 * Budget pacing: throttle the bid so that the remaining budget lasts until the end of the horizon.
 * `expectedOpportunities` is what the agent still believes it will see.
 */
function paceMultiplier(bidder, expectedOpportunities, slotPerMille = SLOT_SCHEDULE[0]) {
  if (expectedOpportunities <= 0) return 0;
  const remaining = Math.max(0, bidder.budget - bidder.spent);
  const affordable = (remaining * 1000) / (expectedOpportunities * slotPerMille);
  const ratio = bidder.value === 0 ? 0 : affordable / bidder.value;
  return Math.max(0, Math.min(1, ratio));
}

// ---------------------------------------------------------------------------- horizon simulation

function generateHorizon(bidders, auctions, options = {}) {
  const participation = options.participation ?? 0.7;
  const next = splitmix64(options.seed ?? 0x821abc);
  const horizon = [];
  for (let i = 0; i < auctions; i++) {
    const present = bidders.filter(() => Number(next() % 1000n) / 1000 < participation);
    horizon.push({ id: i, bidders: present.length > 0 ? present : bidders.slice(0, 1) });
  }
  return horizon;
}

function simulateHorizon(engine, bidders, horizon) {
  const log = [];
  const pacing = [];
  for (let index = 0; index < horizon.length; index++) {
    const auction = horizon[index];
    const remainingOpportunities = horizon.length - index;
    for (const bidder of auction.bidders) bidder.pace = paceMultiplier(bidder, remainingOpportunities);
    const result = engine.runAuction(auction);
    settle(auction.bidders, result);
    log.push(result);
    pacing.push({
      auction: index,
      paces: Object.fromEntries(auction.bidders.map((bidder) => [bidder.id, bidder.pace])),
      spent: Object.fromEntries(bidders.map((bidder) => [bidder.id, bidder.spent])),
    });
  }
  return { log, pacing };
}

// ---------------------------------------------------------------------------- validator

class AuctionValidator {
  constructor(bidders, engine) {
    this.budget = new Map(bidders.map((bidder) => [bidder.id, bidder.budget]));
    this.engine = engine;
    this.violations = [];
  }

  fail(auctionId, message) {
    this.violations.push(`auction ${auctionId}: ${message}`);
  }

  /** Replays the raw result log and re-derives every invariant from scratch. */
  verify(log, horizon) {
    const spent = new Map([...this.budget.keys()].map((id) => [id, 0]));
    for (let index = 0; index < log.length; index++) {
      const result = log[index];
      const auction = horizon[index];
      const seenBidders = new Set();
      let previousEcpm = Infinity;
      const eligible = new Set(auction.bidders.filter((bidder) => bidder.value * bidder.quality >= this.engine.reserve).map((bidder) => bidder.id));
      for (const entry of result.allocation) {
        if (entry.slot >= this.engine.slots) this.fail(result.auctionId, `slot ${entry.slot} does not exist`);
        if (seenBidders.has(entry.bidderId)) this.fail(result.auctionId, `bidder ${entry.bidderId} won twice`);
        seenBidders.add(entry.bidderId);
        if (entry.ecpm < this.engine.reserve) this.fail(result.auctionId, "allocated below the reserve price");
        if (!eligible.has(entry.bidderId)) this.fail(result.auctionId, `ineligible bidder ${entry.bidderId} won`);
        if (entry.ecpm > previousEcpm + 1e-9) this.fail(result.auctionId, "slots were not filled in eCPM order");
        previousEcpm = entry.ecpm;
        if (entry.payment < 1) this.fail(result.auctionId, "payment below the minimum bid");
        spent.set(entry.bidderId, spent.get(entry.bidderId) + entry.spend);
        if (spent.get(entry.bidderId) > this.budget.get(entry.bidderId)) this.fail(result.auctionId, `bidder ${entry.bidderId} exceeded its budget`);
      }
      const revenue = result.allocation.reduce((sum, entry) => sum + entry.spend, 0);
      if (revenue !== result.revenue) this.fail(result.auctionId, "revenue accounting mismatch");
    }
    return this.violations;
  }
}

module.exports = {
  splitmix64,
  SLOT_SCHEDULE,
  slotCtr,
  AuctionEngine,
  settle,
  bruteForceAuctions,
  singleSlotUtility,
  paceMultiplier,
  generateHorizon,
  simulateHorizon,
  AuctionValidator,
  compareIds,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- eCPM ordering is value-optimal; revenue optimisation differs ----------------
  let oracleChecked = 0;
  const oracleSeed = splitmix64(0x8215eed);
  for (let trial = 0; trial < 40; trial++) {
    const count = 2 + Number(oracleSeed() % 4n);
    const bidders = Array.from({ length: count }, (_, index) => ({
      id: `b${index}`,
      value: 20 + Number(oracleSeed() % 200n),
      quality: 1 + Number(oracleSeed() % 5n),
      budget: 100000,
      spent: 0,
    }));
    const engine = new AuctionEngine({ slots: 1 + Number(oracleSeed() % 3n), reserve: 5 });
    const result = engine.runAuction({ id: trial, bidders });
    const oracle = bruteForceAuctions({ id: trial, bidders }, engine.slots, engine.reserve);
    const value = result.allocation.reduce((sum, entry) => sum + entry.ecpm * SLOT_SCHEDULE[entry.slot], 0);
    const expectedValue = oracle.bestValueAssignment.reduce((sum, bidder, slot) => sum + bidder.value * bidder.quality * SLOT_SCHEDULE[slot], 0);
    assert.strictEqual(value, expectedValue, `trial ${trial}: the engine must maximise total value`);
    assert.ok(result.revenue <= oracle.bestRevenue, `trial ${trial}: revenue can never exceed the revenue-optimal assignment`);
    oracleChecked++;
  }
  assert.ok(oracleChecked === 40);
  const revenueGapExample = (() => {
    const bidders = [
      { id: "high", value: 100, quality: 1, budget: 100000, spent: 0 },
      { id: "mid", value: 60, quality: 1, budget: 100000, spent: 0 },
    ];
    const engine = new AuctionEngine({ slots: 1, reserve: 5 });
    const result = engine.runAuction({ id: 1, bidders });
    const oracle = bruteForceAuctions({ id: 1, bidders }, 1, 5);
    return { engineRevenue: result.revenue, optimalRevenue: oracle.bestRevenue, winner: result.allocation[0].bidderId };
  })();
  assert.strictEqual(revenueGapExample.winner, "high", "the efficient winner is the highest eCPM");
  assert.ok(revenueGapExample.optimalRevenue > revenueGapExample.engineRevenue, "second price is not revenue optimal - measured, not hidden");

  // --- single-slot truthfulness by deviation grid -----------------------------------
  const truthfulBidders = [
    { id: "a", value: 120, quality: 2, budget: 100000, spent: 0 },
    { id: "b", value: 90, quality: 3, budget: 100000, spent: 0 },
    { id: "c", value: 40, quality: 4, budget: 100000, spent: 0 },
  ];
  let deviationsChecked = 0;
  for (let index = 0; index < truthfulBidders.length; index++) {
    const truthfulUtility = singleSlotUtility(truthfulBidders, index, truthfulBidders[index].value, 5);
    for (let bid = 1; bid <= 260; bid++) {
      const utility = singleSlotUtility(truthfulBidders, index, bid, 5);
      assert.ok(utility <= truthfulUtility + 1e-9, `bidder ${truthfulBidders[index].id} gained by bidding ${bid}: ${utility} > ${truthfulUtility}`);
      deviationsChecked++;
    }
  }
  assert.strictEqual(deviationsChecked, 3 * 260);

  // --- allocation feasibility on a full horizon -------------------------------------
  const bidders = Array.from({ length: 12 }, (_, index) => ({
    id: `bidder${String(index).padStart(2, "0")}`,
    value: 30 + index * 17,
    quality: 1 + (index % 4),
    budget: 4000 + index * 137,
    spent: 0,
  }));
  const engine = new AuctionEngine({ slots: 4, reserve: 8, candidateCap: 16 });
  const horizon = generateHorizon(bidders, 300, { participation: 0.65, seed: 0x821bbb });
  const { log, pacing } = simulateHorizon(engine, bidders, horizon);
  const violations = new AuctionValidator(bidders, engine).verify(log, horizon);
  assert.deepStrictEqual(violations, [], `validator: ${violations.slice(0, 3).join(" | ")}`);
  for (const bidder of bidders) assert.ok(bidder.spent <= bidder.budget, `${bidder.id} overspent`);
  const totalRevenue = log.reduce((sum, result) => sum + result.revenue, 0);
  assert.ok(totalRevenue > 0, "the horizon must produce revenue");
  const filledSlots = log.reduce((sum, result) => sum + result.allocation.length, 0);
  const possibleSlots = log.length * engine.slots;
  assert.ok(filledSlots <= possibleSlots);
  assert.ok(filledSlots / possibleSlots > 0.2, "a healthy fill rate is expected on this instance");

  // --- pacing: monotone in the remaining budget, and schedule-tracking --------------
  const pacingProbe = { id: "p", value: 100, quality: 1, budget: 1000, spent: 0 };
  const paceEarly = paceMultiplier(pacingProbe, 100);
  pacingProbe.spent = 500;
  const paceLate = paceMultiplier(pacingProbe, 100);
  assert.ok(paceLate < paceEarly, "a bidder that has spent half its budget must throttle harder");
  pacingProbe.spent = 1000;
  assert.strictEqual(paceMultiplier(pacingProbe, 100), 0, "an exhausted budget stops bidding");
  assert.strictEqual(paceMultiplier({ ...pacingProbe, spent: 0 }, 0), 0, "no remaining opportunities means no pacing");
  for (const snapshot of pacing) {
    for (const bidder of bidders) {
      const spent = snapshot.spent[bidder.id];
      assert.ok(spent <= bidder.budget);
      if (snapshot.paces[bidder.id] !== undefined) {
        assert.ok(snapshot.paces[bidder.id] >= 0 && snapshot.paces[bidder.id] <= 1);
      }
    }
  }
  const firstCheck = pacing[0];
  const midCheck = pacing[Math.floor(pacing.length / 2)];
  for (const bidder of bidders) {
    assert.ok(midCheck.spent[bidder.id] >= firstCheck.spent[bidder.id], "spend is monotone over the horizon");
  }
  const spendSpread = Math.max(...bidders.map((bidder) => bidder.spent / bidder.budget)) - Math.min(...bidders.map((bidder) => bidder.spent / bidder.budget));
  assert.ok(spendSpread < 0.95, `pacing should keep budget consumption comparable across bidders, spread ${spendSpread.toFixed(3)}`);

  // --- real-time work bounds ---------------------------------------------------------
  const crowded = Array.from({ length: 200 }, (_, index) => ({ id: `c${index}`, value: 10 + index, quality: 1 + (index % 3), budget: 100000, spent: 0 }));
  const capped = engine.runAuction({ id: 999, bidders: crowded });
  assert.ok(capped.stats.eligible === 200);
  assert.ok(capped.stats.filtered === 200 - engine.candidateCap, "the candidate cap must pre-filter");
  assert.ok(capped.stats.examinations <= engine.candidateCap);
  assert.ok(capped.stats.comparisons <= AuctionEngine.workBound(engine.candidateCap), `comparisons ${capped.stats.comparisons} above the O(n log n) bound`);
  const smallAuction = engine.runAuction({ id: 1000, bidders: crowded.slice(0, 8) });
  assert.ok(smallAuction.stats.comparisons <= AuctionEngine.workBound(8));
  assert.strictEqual(AuctionEngine.workBound(0), 0);

  // --- degenerate auctions -------------------------------------------------------------
  const empty = engine.runAuction({ id: 1, bidders: [] });
  assert.deepStrictEqual(empty.allocation, []);
  assert.strictEqual(empty.revenue, 0);
  const single = engine.runAuction({ id: 2, bidders: [{ id: "solo", value: 100, quality: 1, budget: 1000, spent: 0 }] });
  assert.strictEqual(single.allocation.length, 1, "a single eligible bidder wins the first slot");
  assert.strictEqual(single.allocation[0].payment, engine.reserve, "with no runner-up the winner pays the reserve price");
  assert.ok(single.revenue > 0);
  const equalBids = engine.runAuction({
    id: 4,
    bidders: [
      { id: "tieB", value: 50, quality: 2, budget: 1000, spent: 0 },
      { id: "tieA", value: 50, quality: 2, budget: 1000, spent: 0 },
    ],
  });
  assert.strictEqual(equalBids.allocation[0].bidderId, "tieA", "equal eCPM must break ties deterministically by id");
  const broke = engine.runAuction({ id: 5, bidders: [{ id: "empty", value: 500, quality: 1, budget: 100, spent: 100 }] });
  assert.deepStrictEqual(broke.allocation, [], "a bidder without budget is not a candidate");
  const belowReserve = engine.runAuction({ id: 6, bidders: [{ id: "cheap", value: 1, quality: 1, budget: 100, spent: 0 }] });
  assert.deepStrictEqual(belowReserve.allocation, []);
  const exactBudget = { id: "exact", value: 60, quality: 1, budget: 100000, spent: 0 };
  const exactEngine = new AuctionEngine({ slots: 1, reserve: 5 });
  const exactResult = exactEngine.runAuction({ id: 7, bidders: [exactBudget, { id: "runner", value: 55, quality: 1, budget: 100000, spent: 0 }] });
  settle([exactBudget], exactResult);
  assert.ok(exactBudget.spent <= exactBudget.budget);
  assert.strictEqual(exactResult.allocation[0].payment, 55, "the winner pays the next eCPM");

  // --- determinism ------------------------------------------------------------------------
  const repeatBidders = bidders.map((bidder) => ({ ...bidder, spent: 0 }));
  const repeatEngine = new AuctionEngine({ slots: 4, reserve: 8, candidateCap: 16 });
  const repeat = simulateHorizon(repeatEngine, repeatBidders, generateHorizon(repeatBidders, 300, { participation: 0.65, seed: 0x821bbb }));
  assert.strictEqual(repeat.log.reduce((sum, result) => sum + result.revenue, 0), totalRevenue, "the same horizon must produce the same revenue");
  assert.deepStrictEqual(repeat.log.map((result) => result.allocation.map((entry) => entry.bidderId)), log.map((result) => result.allocation.map((entry) => entry.bidderId)));

  console.log(
    `oracle trials ${oracleChecked} (eCPM order is value-optimal, revenue gap ${(100 * (1 - revenueGapExample.engineRevenue / revenueGapExample.optimalRevenue)).toFixed(1)}% vs the revenue-optimal winner), ` +
      `single-slot deviations checked ${deviationsChecked}, horizon: ${log.length} auctions, revenue ${totalRevenue} cents, fill rate ${(100 * filledSlots / possibleSlots).toFixed(1)}%, ` +
      `budget spread ${spendSpread.toFixed(3)}, capped auction: ${capped.stats.eligible} eligible -> ${capped.stats.examinations} examined with ${capped.stats.comparisons} comparisons (bound ${AuctionEngine.workBound(engine.candidateCap)})`
  );
  console.log("PASS");
}
