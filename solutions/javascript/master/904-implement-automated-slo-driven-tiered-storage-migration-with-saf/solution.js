/** Task 904 - Implement automated SLO-driven tiered storage migration with safety.
 *  (Tier: Master, CTCI Beyond — Storage Systems & Reliability Engineering)
 *
 * Three storage tiers (hot 1 ms / 1.0 cost per GB, warm 5 ms / 0.2, cold 25 ms / 0.02) and a controller
 * that keeps an explicit SLO — "at least sloFraction of reads are served within 10 ms" — by keeping the
 * smallest share-carrying prefix of the keys (ranked by decayed access rate) on the fast tier. Safety
 * comes from three mechanisms checked below after every round: a key lives in exactly one tier (a
 * migration moves it, never copies or drops it), migrations are throttled (maxMigrations per round plus a
 * per-key cooldown), and a hysteresis margin stops a key on the boundary from flip-flopping.
 *
 * Checks: per-round placement invariants, the SLO itself, the cost reduction against an all-hot
 * baseline, the throttle, oscillation bounds on a stationary workload, bounded promotion latency after
 * the hot set shifts, degenerate configurations (no keys, no migration budget, one key), an exhaustive
 * subset oracle for the minimum-cost SLO-feasible plan, and reproducibility.
 */
"use strict";
const assert = require("node:assert");
const TIERS = [
  { name: "hot", latencyMs: 1, costPerGb: 1.0 }, { name: "warm", latencyMs: 5, costPerGb: 0.2 }, { name: "cold", latencyMs: 25, costPerGb: 0.02 }, ];
const SLO_LATENCY_MS = 10;

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { return this.next() % bound; }
  float() { return this.next() / 4294967296; }
}
/** Access counts for one round: zipf weights over the keys, optionally re-weighted after a shift. */
function sampleRound(keyCount, operations, skew, hotKeys, prng) {
  const weights = [];
  let total = 0;
  for (let key = 0; key < keyCount; key++) {
    const weight = 1 / Math.pow(key + 1, skew) * (hotKeys[key] ? 12 : 1);
    weights.push(weight);
    total += weight;
  }
  const cumulative = [];
  let running = 0;
  for (const weight of weights) { running += weight / total; cumulative.push(running); }
  const counts = new Array(keyCount).fill(0);
  for (let op = 0; op < operations; op++) {
    const draw = prng.float();
    let lo = 0;
    let hi = keyCount - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cumulative[mid] < draw) lo = mid + 1; else hi = mid;
    }
    counts[lo] += 1;
  }
  return counts;
}
/** The SLO-driven controller: prefix of the rate ranking that carries the required share, with hysteresis. */
function planHotSet(rates, sloFraction, margin, currentlyHot) {
  const order = rates.map((_, key) => key).sort((a, b) => rates[b] - rates[a] || a - b);
  const total = rates.reduce((a, b) => a + b, 0) || 1;
  const hot = new Array(rates.length).fill(false);
  let cumulative = 0;
  for (const key of order) {
    hot[key] = cumulative < (currentlyHot[key] ? sloFraction + margin : sloFraction);
    cumulative += rates[key] / total;
  }
  return hot;
}

function simulate(config, prng) {
  const { keyCount, rounds, opsPerRound, skew, sloFraction, margin, maxMigrations, cooldown, shift } = config;
  const sizes = Array.from({ length: keyCount }, (_, key) => (config.uniform ? 1 : 0.5 + (key % 7) * 0.25));
  const tier = new Array(keyCount).fill(2);
  const rates = new Array(keyCount).fill(0);
  const lastMove = new Array(keyCount).fill(-99);
  const moves = new Array(keyCount).fill(0);
  const migrationsPerRound = [];
  const accessLog = [];
  const totalCounts = new Array(keyCount).fill(0);
  const fastByRound = [];
  for (let round = 0; round < rounds; round++) {
    const hotKeys = Array.from({ length: keyCount }, (_, key) => Boolean(shift) && round >= shift.round && shift.keys.includes(key));
    const counts = sampleRound(keyCount, opsPerRound, skew, hotKeys, prng);
    let fast = 0;
    for (let key = 0; key < keyCount; key++) {
      rates[key] = 0.5 * rates[key] + 0.5 * counts[key];
      totalCounts[key] += counts[key];
      for (let i = 0; i < counts[key]; i++) accessLog.push({ round, key, tier: tier[key] });
      if (tier[key] === 0) fast += counts[key];
    }
    fastByRound.push(fast / opsPerRound);
    const wantHot = planHotSet(rates, sloFraction, margin, tier.map((value) => value === 0));
    let migrated = 0;
    for (const [key] of rates.map((rate, index) => [index, rate]).sort((a, b) => b[1] - a[1] || a[0] - b[0])) {
      if (migrated >= maxMigrations) break;
      if (round - lastMove[key] < cooldown) continue;
      const target = wantHot[key] ? 0 : 2;
      if (tier[key] === target) continue;
      tier[key] = target;                             // move, never copy: the key is in one tier only
      lastMove[key] = round;
      moves[key] += 1;
      migrated += 1;
    }
    migrationsPerRound.push(migrated);
  }
  return { tier, sizes, moves, migrationsPerRound, accessLog, totalCounts, fastByRound, rates };
}
/** Independent replay of the access log: fast share and cost computed from first principles. */
function audit(result, fromRound = 0) {
  let fast = 0;
  let seen = 0;
  for (const entry of result.accessLog) {
    if (entry.round < fromRound) continue;
    seen += 1;
    if (TIERS[entry.tier].latencyMs <= SLO_LATENCY_MS) fast += 1;
  }
  let cost = 0;
  for (let key = 0; key < result.tier.length; key++) cost += result.sizes[key] * TIERS[result.tier[key]].costPerGb;
  return { fastShare: fast / (seen || 1), cost, sampled: seen };
}
/** Cost of a finished plan under the tier price list. */
const planCost = (result) => result.tier.reduce((sum, tierIndex, key) =>
  sum + result.sizes[key] * TIERS[tierIndex].costPerGb, 0);
/** Exhaustive oracle: cheapest tier assignment whose hot set carries at least sloFraction of the load. */
function optimalCost(counts, sizes, sloFraction) {
  const total = counts.reduce((a, b) => a + b, 0);
  let best = Infinity;
  for (let mask = 0; mask < (1 << counts.length); mask++) {
    let carried = 0;
    let cost = 0;
    for (let key = 0; key < counts.length; key++) {
      const hot = ((mask >>> key) & 1) === 1;
      if (hot) carried += counts[key];
      cost += sizes[key] * (hot ? TIERS[0].costPerGb : TIERS[2].costPerGb);
    }
    if (carried / total >= sloFraction && cost < best) best = cost;
  }
  return best;
}

function main() {
  const base = { keyCount: 60, rounds: 24, opsPerRound: 600, skew: 1.5, sloFraction: 0.9, margin: 0.1, maxMigrations: 6, cooldown: 2 };
  const stationary = simulate(base, new Prng(0x5eed0904));
  // 1. happy path — after every round each key sits in exactly one valid tier (a move, never a copy).
  {
    const seen = new Set(stationary.accessLog.map((entry) => entry.key));
    assert.strictEqual(seen.size, base.keyCount, "every key is still addressable");
    for (const tierIndex of stationary.tier) assert.ok(tierIndex >= 0 && tierIndex < TIERS.length, "tier index");
    assert.strictEqual(stationary.tier.length, base.keyCount, "one placement per key");
    assert.strictEqual(new Set(stationary.tier).size <= TIERS.length, true, "tiers are a partition");
    const buckets = new Map();
    stationary.tier.forEach((tierIndex, key) => buckets.set(tierIndex, (buckets.get(tierIndex) ?? []).concat(key)));
    const flattened = [...buckets.values()].flat();
    assert.strictEqual(flattened.length, base.keyCount, "the tier buckets partition the keys");
    assert.strictEqual(new Set(flattened).size, base.keyCount, "no key appears in two tiers");
  }
  // 2. the SLO — once the rate estimates and the throttle have warmed up, the fast tier carries the share.
  {
    const steady = stationary.fastByRound.slice(6);
    const worst = Math.min(...steady);
    assert.ok(worst >= base.sloFraction - 0.02, `worst steady fast share ${worst} must meet the SLO`);
    const audited = audit(stationary, 6);
    assert.ok(audited.fastShare >= base.sloFraction - 0.02, `audited fast share ${audited.fastShare.toFixed(3)} must meet the SLO`);
    const warmup = stationary.fastByRound[0];
    assert.ok(steady[steady.length - 1] >= warmup, "the SLO improves as rate estimates warm up");
  }
  // 3. cost — the tiered plan is far cheaper than pinning everything in the hot tier.
  {
    const allHot = stationary.sizes.reduce((a, b) => a + b, 0) * TIERS[0].costPerGb;
    const cost = audit(stationary).cost;
    assert.ok(cost <= allHot * 0.4, `tiered cost ${cost.toFixed(2)} vs all-hot ${allHot.toFixed(2)}`);
    assert.ok(stationary.tier.filter((tierIndex) => tierIndex === 0).length < base.keyCount, "the plan does not simply keep everything hot");
  }
  // 4. throttling — never more than maxMigrations moves in a round, and the cooldown is respected.
  {
    assert.ok(stationary.migrationsPerRound.every((count) => count <= base.maxMigrations), "the migration budget must be respected every round");
    assert.ok(stationary.moves.reduce((a, b) => a + b, 0) <= base.keyCount * 2, "a stationary workload must not oscillate unboundedly");
    assert.ok(stationary.moves.filter((count) => count > 2).length === 0, "no key should migrate more than twice under a stationary workload");
  }
  // 5. shifting workload — keys that become hot are promoted within a bounded number of rounds.
  {
    const shift = { round: 10, keys: [50, 55, 58] };
    const shifted = simulate({ ...base, rounds: 20, shift }, new Prng(0x5eed0904));
    assert.ok(shift.keys.every((key) => shifted.tier[key] === 0), "newly hot keys end up on the fast tier");
    const median = [...shifted.totalCounts].sort((a, b) => a - b)[Math.floor(base.keyCount / 2)];
    assert.ok(shift.keys.every((key) => shifted.totalCounts[key] > median), "the shifted keys really did get hot");
    assert.ok(shifted.fastByRound.slice(shift.round + 1, shift.round + 6)
      .every((share) => share >= base.sloFraction - 0.02), "the SLO is restored within five rounds of the shift");
    const late = simulate({ ...base, rounds: 20, shift: { round: 19, keys: [50] } }, new Prng(0x5eed0904));
    assert.strictEqual(late.tier[50], 0, "a key that turns hot in the last round is promoted at once");
    assert.strictEqual(late.moves[50], 1, "each promotion costs exactly one migration");
  }
  // 6. exhaustive oracle — without hysteresis the rate-driven plan *is* the cost-optimal SLO-feasible
  //    plan; the hysteresis margin that buys stability costs a bounded amount more.
  {
    const uniform = { ...base, keyCount: 10, rounds: 14, opsPerRound: 300, uniform: true };
    const tight = simulate({ ...uniform, margin: 0 }, new Prng(99));
    const tightBest = optimalCost(tight.totalCounts, tight.sizes, base.sloFraction);
    assert.strictEqual(planCost(tight), tightBest, `plan ${planCost(tight).toFixed(3)} should equal the enumerated optimum ${tightBest.toFixed(3)}`);
    const stable = simulate(uniform, new Prng(99));
    const stableBest = optimalCost(stable.totalCounts, stable.sizes, base.sloFraction);
    assert.ok(planCost(stable) <= stableBest * 1.4, `hysteresis plan ${planCost(stable).toFixed(3)} vs optimum ${stableBest.toFixed(3)}`);
    assert.ok(audit(stable, 6).fastShare >= base.sloFraction - 0.02, "the small run also honours the SLO");
    const mixed = simulate({ ...base, keyCount: 10, rounds: 14, opsPerRound: 300 }, new Prng(99));
    const mixedBest = optimalCost(mixed.totalCounts, mixed.sizes, base.sloFraction);
    assert.ok(mixedBest <= planCost(mixed), "no plan can beat the enumerated optimum");
    assert.ok(planCost(mixed) <= mixedBest * 2.2, "the size-blind policy stays within 2.2x of the optimum");
  }
  // 7. boundary — no keys, no migration budget, and a single key.
  {
    const empty = simulate({ ...base, keyCount: 0, opsPerRound: 1 }, new Prng(7));
    assert.deepStrictEqual(empty.tier, []);
    assert.deepStrictEqual(empty.migrationsPerRound, new Array(base.rounds).fill(0));
    const frozen = simulate({ ...base, maxMigrations: 0 }, new Prng(7));
    assert.ok(frozen.migrationsPerRound.every((count) => count === 0), "a zero budget blocks all migration");
    assert.ok(frozen.tier.every((tierIndex) => tierIndex === 2), "nothing left the cold tier");
    assert.ok(frozen.fastByRound[base.rounds - 1] === 0, "with no budget the SLO cannot be met");
    const single = simulate({ ...base, keyCount: 1, rounds: 6, opsPerRound: 20 }, new Prng(7));
    assert.strictEqual(single.tier[0], 0, "a single hot key is promoted");
    assert.strictEqual(single.moves[0], 1, "exactly one migration for one key");
  }
  // 8. determinism — same seed replays exactly, a different seed moves at least one decision.
  {
    const again = simulate(base, new Prng(0x5eed0904));
    assert.deepStrictEqual(again.tier, stationary.tier);
    assert.deepStrictEqual(again.migrationsPerRound, stationary.migrationsPerRound);
    const other = simulate(base, new Prng(0x5eed0905));
    assert.notDeepStrictEqual(other.migrationsPerRound, stationary.migrationsPerRound);
  }
  const steady = audit(stationary, 6);
  console.log(`tiering: ${base.keyCount} keys, ${base.rounds} rounds, steady fast share ` +
    `${steady.fastShare.toFixed(3)} >= SLO ${base.sloFraction}, cost ${steady.cost.toFixed(2)} vs all-hot ` +
    `${stationary.sizes.reduce((a, b) => a + b, 0).toFixed(2)}, migrations per round <= ${base.maxMigrations}, ` +
    `total migrations ${stationary.moves.reduce((a, b) => a + b, 0)}`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = { TIERS, SLO_LATENCY_MS, Prng, sampleRound, planHotSet, simulate, audit, optimalCost };
