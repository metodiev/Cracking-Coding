/** Task 940 - Implement high-fidelity load generator with correlated user behavior models.
 * (Tier: Master, CTCI Ch 11 - Testing)
 *
 * An open-loop workload generator: sessions arrive as a Poisson process (exponential inter-arrivals
 * from a seeded splitmix64 stream), so offered load is independent of session length - the property
 * closed-loop generators cannot give. Each session walks a Markov chain over user actions
 * (browse/search/product/cart/checkout) and its dwell times are *correlated*: one latent engagement
 * draw per session scales every think time of that session, which produces the bursty per-user pacing
 * real traffic shows. Simplified: exponential think times, sessions end on the absorbing state and
 * nothing is sent over a socket. Verified: the offered rate, the exponential inter-arrival CDF against
 * the closed form, transition frequencies against the configured matrix, the analytic visit
 * distribution of a two-state model, dwell-time correlation, Little's law, well-formedness,
 * reproducibility and degenerate inputs.
 */
"use strict";
const assert = require("node:assert");
/** splitmix64 - the only source of randomness; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}
const unit = (rng) => Number(rng.next() >> 11n) / 2 ** 53;
/** Inverse transform sampling of an exponential with the given mean. */
const exponential = (rng, meanDraw) => -Math.log(1 - unit(rng)) * meanDraw;
const DEFAULT_MODEL = {
  entry: "browse",
  dwell: { browse: [5, 15], search: [3, 9], product: [10, 30], cart: [20, 60], checkout: [30, 90], exit: [0, 0] },
  transitions: {
    browse: { browse: 0.3, search: 0.4, product: 0.2, exit: 0.1 },
    search: { search: 0.15, browse: 0.15, product: 0.5, exit: 0.2 },
    product: { product: 0.2, cart: 0.4, browse: 0.3, exit: 0.1 },
    cart: { cart: 0.15, checkout: 0.55, product: 0.2, exit: 0.1 },
    checkout: { exit: 1 },
    exit: {},
  },
};
/** One weighted draw; the last key absorbs rounding, so every row sums to one. */
function pick(rng, row) {
  const keys = Object.keys(row);
  let r = unit(rng);
  for (const key of keys) {
    r -= row[key];
    if (r <= 0) return key;
  }
  return keys[keys.length - 1];
}
function simulateSession(rng, model, startTime, id) {
  const engagement = model.correlated === false ? 0.5 : unit(rng);
  const scale = model.correlated === false ? 1 : Math.exp(1.6 * (2 * engagement - 1));
  const actions = [];
  let state = model.entry;
  let at = startTime;
  for (;;) {
    const [lo, hi] = model.dwell[state] ?? [0, 0];
    const dwell = exponential(rng, ((lo + hi) / 2) * scale);
    actions.push({ state, at, dwell });
    at += dwell;
    const row = model.transitions[state];
    if (!row || Object.keys(row).length === 0) break;
    state = pick(rng, row);
  }
  return { id, start: startTime, end: at, engagement, actions, think: actions.map((a) => a.dwell) };
}
/** Open-loop generator: arrivals are Poisson(rate) and every arrival starts one session. */
function generateStream({ seed, duration, rate, model = DEFAULT_MODEL, maxSessions = 4000 }) {
  const rng = splitmix64(seed);
  const sessions = [];
  let t = 0;
  while (sessions.length < maxSessions) {
    t += exponential(rng, 1 / rate);
    if (t > duration) break;
    sessions.push(simulateSession(rng, model, t, sessions.length + 1));
  }
  return { sessions, duration, rate, model };
}
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const variance = (xs) => mean(xs.map((x) => (x - mean(xs)) ** 2));
const arrivals = (stream) => stream.sessions.map((s) => s.start);
const interArrivals = (stream) => { const a = arrivals(stream); return a.slice(1).map((v, i) => v - a[i]); };
const dwellPairs = (stream) => stream.sessions.flatMap((s) => s.think.slice(0, -1).map((d, i) => [d, s.think[i + 1]]));
/** Consecutive think times of the *same* action, grouped by action: only engagement varies there. */
function sameStatePairs(stream) {
  const groups = new Map();
  for (const session of stream.sessions) {
    for (let i = 0; i + 1 < session.actions.length; i += 1) {
      const first = session.actions[i];
      if (session.actions[i + 1].state !== first.state) continue;
      groups.set(first.state, [...(groups.get(first.state) ?? []), [first.dwell, session.actions[i + 1].dwell]]);
    }
  }
  return groups;
}
/**
 * Mean correlation inside one action state. Pooling states would mix their mean dwell times into the
 * covariance, so the within-state estimates are combined weighted by the number of pairs instead.
 */
function meanWithinStateCorrelation(groups) {
  let weighted = 0;
  let weight = 0;
  for (const pairs of groups.values()) {
    if (pairs.length < 30) continue;
    weighted += correlation(pairs) * pairs.length;
    weight += pairs.length;
  }
  return weight ? weighted / weight : 0;
}
function correlation(pairs) {
  const xs = pairs.map((p) => p[0]);
  const ys = pairs.map((p) => p[1]);
  const mx = mean(xs);
  const my = mean(ys);
  const cov = mean(xs.map((x, i) => (x - mx) * (ys[i] - my)));
  return cov / Math.sqrt(variance(xs) * variance(ys));
}
/** Empirical transition counts, the oracle for the configured matrix. */
function transitionCounts(stream) {
  const counts = new Map();
  for (const session of stream.sessions) {
    for (let i = 0; i + 1 < session.actions.length; i += 1) {
      const key = `${session.actions[i].state}->${session.actions[i + 1].state}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return counts;
}
function occupancy(stream) {
  const counts = new Map();
  let total = 0;
  for (const session of stream.sessions) {
    for (const action of session.actions) {
      counts.set(action.state, (counts.get(action.state) ?? 0) + 1);
      total += 1;
    }
  }
  return { counts, total, share: (state) => (counts.get(state) ?? 0) / Math.max(1, total) };
}
/**
 * Independent closed form: expected visits per session is sum_k P^k e_entry over the transient
 * states (a geometric series that converges because the exit probability keeps the spectral radius
 * below one), computed by direct iteration instead of by solving the linear system.
 */
function expectedVisits(model, states) {
  const index = new Map(states.map((s, i) => [s, i]));
  let vector = states.map((s) => (s === model.entry ? 1 : 0));
  const totals = [...vector];
  for (let step = 0; step < 500; step += 1) {
    const next = states.map(() => 0);
    states.forEach((from, i) => {
      for (const [to, p] of Object.entries(model.transitions[from] ?? {})) {
        if (index.has(to)) next[index.get(to)] += vector[i] * p;
      }
    });
    vector = next;
    vector.forEach((v, i) => { totals[i] += v; });
    if (vector.every((v) => v < 1e-12)) break;
  }
  return new Map(states.map((s, i) => [s, totals[i]]));
}
/** Independent Little's-law estimate: sweep the busy horizon and average the number of live sessions. */
function averageInFlight(stream, samples = 4000) {
  const starts = stream.sessions.map((s) => s.start).sort((a, b) => a - b);
  const ends = stream.sessions.map((s) => s.end).sort((a, b) => a - b);
  const horizon = Math.max(stream.duration, ends.at(-1) ?? 0);
  let i = 0;
  let j = 0;
  let active = 0;
  let total = 0;
  for (let k = 0; k < samples; k += 1) {
    const t = (k * horizon) / samples;
    while (i < starts.length && starts[i] <= t) { active += 1; i += 1; }
    while (j < ends.length && ends[j] <= t) { active -= 1; j += 1; }
    total += active;
  }
  return total / samples;
}
function main() {
  const stream = generateStream({ seed: 940, duration: 600, rate: 2 }); const gaps = interArrivals(stream);
  // 1. Offered load: the empirical arrival rate matches the configured Poisson rate.
  assert(stream.sessions.length > 900, `expected roughly 1200 sessions, got ${stream.sessions.length}`);
  assert(Math.abs(stream.sessions.length / 600 - 2) / 2 < 0.06, `arrival rate within 6%, got ${stream.sessions.length / 600}`);
  // 2. Poisson property: counts per bucket are equidispersed (index of dispersion near one).
  const buckets = new Array(20).fill(0);
  for (const start of arrivals(stream)) buckets[Math.min(19, Math.floor(start / 30))] += 1;
  assert(Math.abs(variance(buckets) / mean(buckets) - 1) < 0.25, `dispersion ${variance(buckets) / mean(buckets)}`);
  // 3. Independent closed form: the inter-arrival CDF is exponential.
  assert(Math.abs(mean(gaps) - 0.5) < 0.03, `mean gap ${mean(gaps)} should be 1/rate`);
  const cdfAtMean = gaps.filter((g) => g <= 0.5).length / gaps.length;
  assert(Math.abs(cdfAtMean - (1 - Math.exp(-1))) < 0.05, `empirical CDF at the mean ${cdfAtMean}`);
  // 4. Behaviour model: empirical transitions match the configured matrix within 3 points.
  const counts = transitionCounts(stream);
  const totals = new Map();
  for (const [key, n] of counts) totals.set(key.split("->")[0], (totals.get(key.split("->")[0]) ?? 0) + n);
  for (const [from, row] of Object.entries(DEFAULT_MODEL.transitions)) {
    if (!totals.get(from)) continue;
    for (const [to, p] of Object.entries(row)) {
      const observed = (counts.get(`${from}->${to}`) ?? 0) / totals.get(from);
      assert(Math.abs(observed - p) < 0.03, `P(${from}->${to}) observed ${observed} vs model ${p}`);
    }
  }
  // 5. Closed-form oracle: a two-state model visits states as the geometric series predicts.
  const twoState = { entry: "a", correlated: false, dwell: { a: [10, 10], b: [10, 10], exit: [0, 0] },
    transitions: { a: { a: 0.8, b: 0.15, exit: 0.05 }, b: { a: 0.3, b: 0.65, exit: 0.05 }, exit: {} } };
  const two = generateStream({ seed: 941, duration: 4000, rate: 1, model: twoState });
  const visits = expectedVisits(twoState, ["a", "b"]);
  const seen = occupancy(two);
  const empiricalA = seen.share("a") / (seen.share("a") + seen.share("b"));
  const analyticA = visits.get("a") / (visits.get("a") + visits.get("b"));
  assert(Math.abs(empiricalA - analyticA) < 0.02, `two-state visit share ${empiricalA} vs closed form ${analyticA}`);
  const predictedLength = visits.get("a") + visits.get("b") + 1;
  assert(Math.abs(mean(two.sessions.map((s) => s.actions.length)) - predictedLength) < 1.5, `mean session length vs ${predictedLength}`);
  // 6. Correlated pacing: successive think times in a session are positively correlated, and only
  //    the latent engagement factor explains the correlation between equal actions.
  const rho = correlation(dwellPairs(stream)); const sameState = meanWithinStateCorrelation(sameStatePairs(stream));
  assert(rho > 0.12, `dwell times must be correlated within a session, got ${rho}`);
  assert(sameState > 0.2, `same-action pacing correlates through engagement, got ${sameState}`);
  const flatStream = generateStream({ seed: 941, duration: 600, rate: 2, model: { ...DEFAULT_MODEL, correlated: false } });
  assert(Math.abs(correlation(dwellPairs(flatStream))) < 0.12, "without engagement the pacing is uncorrelated");
  assert(Math.abs(meanWithinStateCorrelation(sameStatePairs(flatStream))) < 0.08, "equal actions are uncorrelated without engagement");
  // 7. Little's law over the busy horizon: the timeline sweep equals arrivals times mean sojourn.
  const dwelt = mean(stream.sessions.map((s) => s.end - s.start));
  const horizon = Math.max(stream.duration, ...stream.sessions.map((s) => s.end));
  const predicted = (stream.sessions.length / horizon) * dwelt;
  const swept = averageInFlight(stream);
  assert(Math.abs(swept - predicted) / predicted < 0.03, `Little's law: ${swept} vs ${predicted}`);
  // 8. Well-formedness: sessions start at the entry state and only follow model edges.
  const edges = (state) => DEFAULT_MODEL.transitions[state] ?? {};
  assert(stream.sessions.every((s) => s.actions[0].state === DEFAULT_MODEL.entry && s.end >= s.start && s.actions.every((a) => a.dwell >= 0)), "sessions start at the entry state with non-negative durations");
  assert(stream.sessions.every((s) => s.actions.every((a, i) => i === 0 || edges(s.actions[i - 1].state)[a.state] > 0)), "no session takes an undefined edge");
  // 9. Reproducibility and the degenerate windows.
  const streamAt = (options) => generateStream({ duration: 600, rate: 2, ...options });
  assert(JSON.stringify(streamAt({ seed: 940 }).sessions) === JSON.stringify(stream.sessions), "the same seed reproduces the stream byte for byte");
  assert(JSON.stringify(streamAt({ seed: 942 }).sessions) !== JSON.stringify(stream.sessions), "another seed changes the stream");
  const none = streamAt({ seed: 940, duration: 0 });
  assert(none.sessions.length === 0 && interArrivals(none).length === 0, "a zero-length window produces no sessions");
  const single = streamAt({ seed: 940, maxSessions: 1 });
  assert(single.sessions.length === 1 && single.sessions[0].actions.length >= 1, "the session cap is respected");
  const trickle = generateStream({ seed: 5, duration: 1, rate: 0.01 });
  assert(trickle.sessions.length <= 1 && Number.isFinite(mean(trickle.sessions.map((s) => s.end))), "a near-idle rate stays finite");
  console.log(`load-generator: sessions=${stream.sessions.length} rate=${(stream.sessions.length / 600).toFixed(3)} meanGap=${mean(gaps).toFixed(3)} rho=${rho.toFixed(2)}`);
  console.log("PASS");
}
module.exports = { splitmix64, generateStream, simulateSession, DEFAULT_MODEL, transitionCounts, occupancy, correlation, interArrivals };
if (require.main === module) main();
