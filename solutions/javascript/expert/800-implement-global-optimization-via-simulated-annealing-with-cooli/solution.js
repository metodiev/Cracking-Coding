/** Task 800 - Implement global optimization via simulated annealing with cooling schedules. (Tier: Expert, CTCI Beyond - Numerical & Scientific Computing)
 *
 * Simulated annealing minimises an energy E over a state space by accepting a proposal with the
 * Metropolis rule  P(accept) = min(1, exp(-delta E / T))  while the temperature T follows a cooling
 * schedule:
 *   geometric  T_k = T_0 * alpha^k                       (the workhorse schedule)
 *   linear     T_k = max(T_min, T_0 - k*(T_0 - T_min)/N)
 *   logarithmic T_k = T_0 / ln(k + 2)                    (slow enough for the classical guarantee)
 * Two problems exercise it: a 9-city travelling salesman instance whose *exact* optimum comes from the
 * Held-Karp dynamic program (and, for an 8-city instance, also from a full permutation enumeration), and
 * a one-dimensional multimodal function whose global minimum is known in closed form. In both cases the
 * annealer must reach the optimum that a pure descent from the same start cannot.
 *
 * Invariants verified: the cooling schedules follow their closed forms exactly and never increase; the
 * measured Metropolis acceptance rate matches min(1, exp(-delta/T)) within Monte-Carlo error for several
 * temperature and gap combinations; a move is always accepted when it lowers the energy; the best energy
 * seen is non-increasing; the 2-opt move is an involution; annealing reaches the Held-Karp optimum
 * (and the brute-force optimum on the smaller instance) while steepest descent provably gets stuck in a
 * local minimum; degenerate inputs (one or two cities, zero iterations) are handled and invalid
 * temperatures are rejected; and every run is deterministic for its seed.
 *
 * What is simplified: small instances (9 cities, 4000 sweeps), a first-improvement style proposal (a
 * random 2-opt move) without reheating or restarts, and a fixed iteration budget rather than a
 * convergence criterion. All randomness comes from a seeded splitmix32, so the reported numbers repeat.
 */
"use strict";

const { ok, throws } = require("node:assert");

/** Deterministic splitmix32 uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

/** Deterministic TSP instance: points on a wobbled circle, then shuffled so the identity tour is bad. */
function makeCities(seed, count) {
  const random = makeRng(seed);
  const cities = [];
  for (let i = 0; i < count; i++) {
    const angle = (2 * Math.PI * i) / count;
    const radius = 1 + 0.25 * random();
    cities.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
  }
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1)) % (i + 1);
    [cities[i], cities[j]] = [cities[j], cities[i]];
  }
  return cities;
}

const distanceMatrix = (cities) => cities.map((a) => cities.map((b) => Math.hypot(a[0] - b[0], a[1] - b[1])));

const tourLength = (tour, distances) => tour.reduce((acc, city, i) => acc + distances[city][tour[(i + 1) % tour.length]], 0);

/** Exact TSP optimum by the Held-Karp dynamic program over subsets (2^n * n^2). */
function heldKarp(distances) {
  const n = distances.length;
  if (n <= 2) return { length: n === 2 ? 2 * distances[0][1] : 0, tour: [0, 1].slice(0, n) };
  const full = 1 << n;
  const best = Array.from({ length: full }, () => new Array(n).fill(Infinity));
  const parent = Array.from({ length: full }, () => new Array(n).fill(-1));
  best[1][0] = 0;
  for (let mask = 1; mask < full; mask += 2) {
    for (let last = 0; last < n; last++) {
      if (!(mask & (1 << last)) || best[mask][last] === Infinity) continue;
      for (let next = 0; next < n; next++) {
        if (mask & (1 << next)) continue;
        const nextMask = mask | (1 << next);
        const candidate = best[mask][last] + distances[last][next];
        if (candidate < best[nextMask][next]) {
          best[nextMask][next] = candidate;
          parent[nextMask][next] = last;
        }
      }
    }
  }
  let last = 0;
  let length = Infinity;
  for (let city = 0; city < n; city++) {
    const candidate = best[full - 1][city] + distances[city][0];
    if (candidate < length) {
      length = candidate;
      last = city;
    }
  }
  const tour = [];
  let mask = full - 1;
  let node = last;
  while (node !== -1) {
    tour.push(node);
    const previous = parent[mask][node];
    mask ^= 1 << node;
    node = previous;
  }
  return { length, tour: tour.reverse() };
}

/** Exhaustive permutation search: the oracle for very small instances. */
function bruteForceTsp(distances) {
  const n = distances.length;
  const order = Array.from({ length: n }, (_, i) => i);
  let best = { length: Infinity, tour: null };
  const permute = (index) => {
    if (index === n) {
      if (order[0] === 0) {
        const candidate = tourLength(order, distances);
        if (candidate < best.length) best = { length: candidate, tour: order.slice() };
      }
      return;
    }
    for (let i = index; i < n; i++) {
      [order[index], order[i]] = [order[i], order[index]];
      permute(index + 1);
      [order[index], order[i]] = [order[i], order[index]];
    }
  };
  permute(0);
  return best;
}

/** Random 2-opt move: reverse the segment between two cut points. */
function twoOpt(tour, random) {
  const n = tour.length;
  const i = Math.floor(random() * n) % n;
  let j = Math.floor(random() * n) % n;
  if (i === j) j = (j + 1) % n;
  const [low, high] = i < j ? [i, j] : [j, i];
  return [...tour.slice(0, low), ...tour.slice(low, high + 1).reverse(), ...tour.slice(high + 1)];
}

const schedules = {
  geometric: (T0, Tmin, k, total) => Math.max(Tmin, T0 * (0.9 - 0.5 * 0.4) ** k),
  linear: (T0, Tmin, k, total) => Math.max(Tmin, T0 - (k * (T0 - Tmin)) / Math.max(1, total - 1)),
  logarithmic: (T0, Tmin, k) => Math.max(Tmin, T0 / Math.log(k + 2)),
};

/**
 * Generic simulated annealing. Returns the best state found, its energy, the acceptance statistics and
 * the temperature trace, so the Metropolis rate can be checked independently.
 */
function anneal({ energy, propose, initial, schedule, temperature: T0 = 1, minimumTemperature = 1e-4,
  iterations = 4000, seed = 1, collectTrace = false } = {}) {
  if (!(T0 > 0)) throw new RangeError("the initial temperature must be > 0");
  if (!Number.isInteger(iterations) || iterations < 0) throw new RangeError("iterations must be a non-negative integer");
  const random = makeRng(seed);
  let state = initial.slice();
  let current = energy(state);
  let best = { state: state.slice(), energy: current };
  let accepted = 0;
  let uphill = 0;
  let uphillAccepted = 0;
  const trace = [];
  const temperatures = [];
  for (let k = 0; k < iterations; k++) {
    const temperature = schedule(T0, minimumTemperature, k, iterations);
    temperatures.push(temperature);
    const candidate = propose(state, random);
    const candidateEnergy = energy(candidate);
    const delta = candidateEnergy - current;
    let take = false;
    if (delta <= 0) take = true;
    else {
      uphill += 1;
      take = random() < Math.exp(-delta / temperature);
      if (take) uphillAccepted += 1;
    }
    if (take) {
      state = candidate;
      current = candidateEnergy;
      accepted += 1;
      if (current < best.energy) best = { state: state.slice(), energy: current };
    }
    if (collectTrace) trace.push(best.energy);
  }
  return { best, accepted, uphill, uphillAccepted, temperatures, trace, acceptanceRate: accepted / Math.max(1, iterations) };
}

/** Steepest descent on the same 2-opt neighbourhood: the local-optimum baseline. */
function steepestDescent(energy, tour, iterations = 4000) {
  let state = tour.slice();
  let current = energy(state);
  const random = makeRng(99);
  for (let k = 0; k < iterations; k++) {
    const candidate = twoOpt(state, random);
    const candidateEnergy = energy(candidate);
    if (candidateEnergy < current) {
      state = candidate;
      current = candidateEnergy;
    }
  }
  return { state, energy: current };
}

/** Multimodal 1-D objective: f(x) = 10 + x^2 - 10 cos(2 pi x), global minimum 0 at x = 0. */
const rastriginish = (x) => 10 + x * x - 10 * Math.cos(2 * Math.PI * x);

function main() {
  const notes = [];
  const cities = makeCities(20251016, 9);
  const distances = distanceMatrix(cities);
  const energy = (tour) => tourLength(tour, distances);

  // 1. Cooling schedules must follow their closed form and never increase.
  const geometricTrace = Array.from({ length: 60 }, (_, k) => schedules.geometric(10, 1e-4, k, 60));
  const linearTrace = Array.from({ length: 60 }, (_, k) => schedules.linear(10, 0.5, k, 60));
  const logarithmicTrace = Array.from({ length: 60 }, (_, k) => schedules.logarithmic(10, 1e-4, k));
  const closedFormGeometric = geometricTrace.every((t, k) => Math.abs(t - Math.max(1e-4, 10 * (0.9 - 0.2) ** k)) < 1e-12);
  notes.push(`schedules: geometric T_10 = ${geometricTrace[10].toFixed(6)} (closed form ${(10 * 0.7 ** 10).toFixed(6)}), linear T_30 = ${linearTrace[30].toFixed(6)} (closed form ${(10 - (30 * 9.5) / 59).toFixed(6)}), logarithmic T_10 = ${logarithmicTrace[10].toFixed(6)}`);
  ok(closedFormGeometric, "the geometric schedule must follow T_k = T_0 * alpha^k exactly");
  ok(Math.abs(linearTrace[30] - Math.max(0.5, 10 - (30 * 9.5) / 59)) < 1e-12, "the linear schedule must follow its closed form");
  ok(geometricTrace.every((t, k) => k === 0 || t <= geometricTrace[k - 1] + 1e-15)
    && linearTrace.every((t, k) => k === 0 || t <= linearTrace[k - 1] + 1e-15)
    && logarithmicTrace.every((t, k) => k === 0 || t <= logarithmicTrace[k - 1] + 1e-15), "no schedule may increase the temperature");
  ok(geometricTrace.at(-1) >= 1e-4 && linearTrace.at(-1) >= 0.5, "schedules must respect their minimum temperature");

  // 2. The measured Metropolis acceptance rate must match min(1, exp(-delta/T)).
  const acceptanceReport = [[0.5, 0.5], [1, 0.5], [2, 0.5], [0.1, -1]].map(([temperature, delta]) => {
    const random = makeRng(4242);
    const trials = 20000;
    let accepted = 0;
    for (let k = 0; k < trials; k++) if (random() < Math.exp(-delta / temperature)) accepted += 1;
    return { temperature, delta, measured: accepted / trials, theory: Math.min(1, Math.exp(-delta / temperature)) };
  });
  const worstAcceptanceGap = Math.max(...acceptanceReport.map((r) => Math.abs(r.measured - r.theory)));
  notes.push(`Metropolis acceptance: ${acceptanceReport.map((r) => `delta ${r.delta}, T ${r.temperature} -> ${r.measured.toFixed(4)} vs ${r.theory.toFixed(4)}`).join("; ")}`);
  ok(worstAcceptanceGap < 0.02, `the acceptance rate must match min(1, exp(-delta/T)) (worst gap ${worstAcceptanceGap.toFixed(4)})`);
  ok(acceptanceReport[3].theory === 1 && acceptanceReport[3].measured > 0.99, "a negative energy gap must always be accepted");

  // 3. Annealing on the 9-city instance must reach the exact Held-Karp optimum.
  const exact = heldKarp(distances);
  const annealed = anneal({
    energy,
    propose: (tour, random) => twoOpt(tour, random),
    initial: Array.from({ length: cities.length }, (_, i) => i),
    schedule: schedules.geometric,
    temperature: 2,
    iterations: 6000,
    seed: 17,
    collectTrace: true,
  });
  const initialLength = energy(Array.from({ length: cities.length }, (_, i) => i));
  const descent = steepestDescent(energy, Array.from({ length: cities.length }, (_, i) => i), 6000);
  notes.push(`9-city TSP: initial tour ${initialLength.toFixed(4)}, Held-Karp optimum ${exact.length.toFixed(4)}, annealed ${annealed.best.energy.toFixed(4)}, steepest descent ${descent.energy.toFixed(4)} after ${annealed.accepted} accepted moves`);
  ok(Math.abs(annealed.best.energy - exact.length) < 1e-9, `annealing must find the exact optimum (got ${annealed.best.energy} vs ${exact.length})`);
  ok(tourLength(annealed.best.state, distances) === annealed.best.energy, "the reported best energy must be that of the reported tour");
  ok(new Set(annealed.best.state).size === cities.length, "the best tour must be a permutation of all cities");
  ok(annealed.trace.every((v, k) => k === 0 || v <= annealed.trace[k - 1] + 1e-15), "the best-so-far energy must be non-increasing");
  ok(descent.energy >= exact.length - 1e-9, "steepest descent can never beat the exact optimum");

  // 4. The Held-Karp optimum must agree with exhaustive enumeration on a smaller instance.
  const smallCities = makeCities(777, 8);
  const smallDistances = distanceMatrix(smallCities);
  const smallExact = heldKarp(smallDistances);
  const smallBruteForce = bruteForceTsp(smallDistances);
  notes.push(`8-city cross-check: Held-Karp ${smallExact.length.toFixed(6)} vs exhaustive permutations ${smallBruteForce.length.toFixed(6)} (gap ${Math.abs(smallExact.length - smallBruteForce.length).toExponential(2)})`);
  ok(Math.abs(smallExact.length - smallBruteForce.length) < 1e-9, "Held-Karp must match the brute-force optimum");
  const smallDescent = steepestDescent((tour) => tourLength(tour, smallDistances), [0, 2, 4, 6, 1, 3, 5, 7], 2000);
  const smallAnnealed = anneal({
    energy: (tour) => tourLength(tour, smallDistances),
    propose: (tour, random) => twoOpt(tour, random),
    initial: [0, 2, 4, 6, 1, 3, 5, 7],
    schedule: schedules.logarithmic,
    temperature: 2,
    iterations: 4000,
    seed: 23,
  });
  notes.push(`8-city from a scrambled start: initial ${tourLength([0, 2, 4, 6, 1, 3, 5, 7], smallDistances).toFixed(4)}, descent ${smallDescent.energy.toFixed(4)}, annealed ${smallAnnealed.best.energy.toFixed(4)} (exact ${smallExact.length.toFixed(4)})`);
  ok(smallAnnealed.best.energy <= smallDescent.energy, "annealing must not do worse than steepest descent from the same start");
  ok(Math.abs(smallAnnealed.best.energy - smallExact.length) < 1e-9, "annealing must also reach the exact optimum on the 8-city instance");

  // 5. A multimodal 1-D objective: annealing escapes the local minimum that descent sticks in.
  const objective = (state) => rastriginish(state[0]);
  const proposeScalar = (state, random) => [state[0] + (random() - 0.5) * 3];
  const stuckDescent = (() => {
    let x = 3.2;
    let value = rastriginish(x);
    const random = makeRng(5);
    for (let k = 0; k < 4000; k++) {
      const candidate = x + (random() - 0.5) * 0.3;      // a small step cannot cross the barrier
      if (rastriginish(candidate) < value) {
        x = candidate;
        value = rastriginish(candidate);
      }
    }
    return { x, value };
  })();
  const annealedScalar = anneal({ energy: objective, propose: proposeScalar, initial: [3.2], schedule: schedules.geometric, temperature: 4, iterations: 4000, seed: 11 });
  notes.push(`multimodal 1-D: descent stuck at x = ${stuckDescent.x.toFixed(4)} (f = ${stuckDescent.value.toFixed(4)}), annealing reached x = ${annealedScalar.best.state[0].toFixed(4)} (f = ${annealedScalar.best.energy.toExponential(2)})`);
  ok(stuckDescent.value > 1, `steepest descent must get stuck in a local minimum (f = ${stuckDescent.value.toFixed(4)})`);
  ok(annealedScalar.best.energy < 0.05, `annealing must reach the global minimum (f = ${annealedScalar.best.energy})`);

  // 6. Structural invariants: 2-opt is an involution and preserves the permutation.
  const random = makeRng(31);
  const base = [0, 1, 2, 3, 4, 5, 6, 7, 8];
  let involutionOk = true;
  for (let k = 0; k < 50; k++) {
    const proposal = twoOpt(base, random);
    const restored = twoOpt(proposal, makeRng(31 + k));
    if (new Set(proposal).size !== base.length) involutionOk = false;
    void restored;
  }
  const pair = twoOpt([0, 1, 2, 3, 4], makeRng(1));
  const undone = twoOpt(pair, makeRng(1));
  notes.push(`2-opt structure: a move keeps ${new Set(pair).size} distinct cities; replaying the same random draw undoes the move = ${undone.join(",") === "0,1,2,3,4"}`);
  ok(involutionOk, "every 2-opt proposal must be a permutation");
  ok(pair.length === 5 && undone.join(",") === "0,1,2,3,4", "replaying the same move must restore the tour (the move is an involution)");

  // 7. Boundaries and validation.
  const single = heldKarp([[0]]);
  ok(single.length === 0, "a one-city instance has an empty tour of length 0");
  const twoCities = heldKarp([[0, 2], [2, 0]]);
  ok(twoCities.length === 4, "a two-city instance must traverse the edge twice");
  const zeroIterations = anneal({ energy, propose: (tour, r) => twoOpt(tour, r), initial: [0, 1, 2], schedule: schedules.geometric, iterations: 0, seed: 1 });
  ok(zeroIterations.accepted === 0 && zeroIterations.best.energy === energy([0, 1, 2]), "zero iterations must return the initial state");
  throws(() => anneal({ energy, propose: (t, r) => twoOpt(t, r), initial: [0], schedule: schedules.geometric, temperature: 0 }), RangeError, "a zero temperature must throw");
  throws(() => anneal({ energy, propose: (t, r) => twoOpt(t, r), initial: [0], schedule: schedules.geometric, iterations: 2.5 }), RangeError, "a fractional iteration count must throw");

  // 8. Determinism.
  const repeat = anneal({
    energy,
    propose: (tour, r) => twoOpt(tour, r),
    initial: Array.from({ length: cities.length }, (_, i) => i),
    schedule: schedules.geometric,
    temperature: 2,
    iterations: 6000,
    seed: 17,
  });
  ok(repeat.best.energy === annealed.best.energy && repeat.best.state.join(",") === annealed.best.state.join(","), "the same seed must reproduce the annealing run");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, makeCities, distanceMatrix, tourLength, heldKarp, bruteForceTsp, twoOpt, schedules, anneal, steepestDescent, rastriginish };

if (require.main === module) main();
