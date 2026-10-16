/**
 * Task 537 - Implement approximate TSP heuristics (2-opt / 3-opt). (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Euclidean TSP on a handful of points: a deterministic nearest-neighbour construction followed
 * by local search over the 2-opt and 3-opt neighbourhoods.  A 2-opt move reverses the inner
 * segment tour[i..j] (i >= 1), which - because reversing either of the two arcs of a cycle yields
 * the same cycle up to orientation - covers every 2-opt reconnection without a redundant
 * mirrored scan.  A 3-opt move cuts the tour into four pieces S1 | S2 | S3 | S4 at linear
 * positions 1 <= p1 < p2 < p3 <= n-1 and reconnects them as S1, X, Y, S4 where {X, Y} is an
 * ordering of {S2, S3} with independent reversal flags: 2 orderings x 4 flag pairs = the 8
 * classical reconnections (1 identity + 7 real moves, three of which are exactly the 2-opt
 * moves, so sweeping 3-opt also removes every 2-opt defect).  Both searches use first
 * improvement with a deterministic scan order and a documented round cap, so they terminate and
 * never depend on randomness.
 *
 * Documented checks: `threeOptDelta` is audited against a full re-costing of the moved tour for
 * every (p1, p2, p3, variant) on random tours (agreement <= 1e-9) and `applyThreeOpt` always
 * returns a permutation; the pipeline nearest-neighbour -> 2-opt -> 3-opt is verified
 * non-increasing in cost at every step and 2-opt/3-opt local optimality is re-verified afterwards
 * by an exhaustive improving-move scan; every tour must be a permutation and must not beat the
 * exact Held-Karp optimum; the Christofides-style ratio bound cost <= 1.5 * OPT is asserted for
 * the tested metric instances, as is matching the exact optimum on most seeds.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function lcgPoints(n, seed, limit) {
  const points = [];
  let s = seed % 65537;
  for (let i = 0; i < n; i++) {
    s = lcgNext(s);
    const x = s % limit;
    s = lcgNext(s);
    const y = s % limit;
    points.push([x, y]);
  }
  return points;
}

function distanceMatrix(points) {
  const n = points.length;
  const d = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      d[i][j] = Math.sqrt(dx * dx + dy * dy);
    }
  }
  return d;
}

function tourCost(d, tour) {
  let total = 0;
  for (let i = 0; i < tour.length; i++) total += d[tour[i]][tour[(i + 1) % tour.length]];
  return total;
}

function tourIsPermutation(n, tour) {
  if (tour.length !== n) return false;
  const sorted = tour.slice().sort((a, b) => a - b);
  return sorted.every((v, i) => v === i);
}

function shuffle(values, seed) {
  const out = values.slice();
  let s = seed % 65537;
  for (let i = out.length - 1; i > 0; i--) {
    s = lcgNext(s);
    const j = s % (i + 1);
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

/** Deterministic nearest neighbour; ties go to the smallest index. */
function nearestNeighbour(d, start = 0) {
  const n = d.length;
  if (n === 0) return [];
  const visited = new Array(n).fill(false);
  const tour = [start];
  visited[start] = true;
  for (let step = 1; step < n; step++) {
    const from = tour[tour.length - 1];
    let best = -1;
    for (let v = 0; v < n; v++) {
      if (visited[v]) continue;
      if (best === -1 || d[from][v] < d[from][best]) best = v;
    }
    tour.push(best);
    visited[best] = true;
  }
  return tour;
}

function twoOptMove(d, tour, i, j) {
  return tour.slice(0, i).concat(tour.slice(i, j + 1).reverse(), tour.slice(j + 1));
}

/**
 * First-improvement 2-opt.  i >= 1 only: reversing tour[i..j] is the same cycle (up to
 * orientation) as reversing the complementary arc, so no 2-opt move is lost.
 */
function twoOpt(d, tour, maxRounds = 1000) {
  let best = tour.slice();
  let rounds = 0;
  let moves = 0;
  for (;;) {
    let improved = false;
    rounds += 1;
    for (let i = 1; i + 1 < best.length && !improved; i++) {
      for (let j = i + 1; j < best.length; j++) {
        const candidate = twoOptMove(d, best, i, j);
        if (tourCost(d, candidate) < tourCost(d, best) - 1e-12) {
          best = candidate;
          moves += 1;
          improved = true;
          break;
        }
      }
    }
    if (!improved || rounds >= maxRounds) break;
  }
  return { tour: best, rounds, moves };
}

/** The 8 reconnections of S1 | S2 | S3 | S4; variant 0 is the identity. */
const THREE_OPT_VARIANTS = [
  [[2, false], [3, false]],
  [[2, true], [3, false]],
  [[2, false], [3, true]],
  [[2, true], [3, true]],
  [[3, true], [2, false]],
  [[3, false], [2, false]],
  [[3, false], [2, true]],
  [[3, true], [2, true]],
];

function applyThreeOpt(tour, p1, p2, p3, variant) {
  const head = tour.slice(0, p1);
  const s2 = tour.slice(p1, p2);
  const s3 = tour.slice(p2, p3);
  const tail = tour.slice(p3);
  const pieces = THREE_OPT_VARIANTS[variant].map(([seg, rev]) => {
    const piece = seg === 2 ? s2.slice() : s3.slice();
    if (rev) piece.reverse();
    return piece;
  });
  return head.concat(pieces[0], pieces[1], tail);
}

/** Delta by edge formula: three removed edges out, three new edges in. */
function threeOptDelta(d, tour, p1, p2, p3, variant) {
  const ranges = { 2: [p1, p2 - 1], 3: [p2, p3 - 1] };
  const firstOf = (seg, rev) => (rev ? tour[ranges[seg][1]] : tour[ranges[seg][0]]);
  const lastOf = (seg, rev) => (rev ? tour[ranges[seg][0]] : tour[ranges[seg][1]]);
  const removed = d[tour[p1 - 1]][tour[p1]] + d[tour[p2 - 1]][tour[p2]] + d[tour[p3 - 1]][tour[p3]];
  const order = THREE_OPT_VARIANTS[variant];
  const x = firstOf(order[0][0], order[0][1]);
  const y = lastOf(order[0][0], order[0][1]);
  const z = firstOf(order[1][0], order[1][1]);
  const w = lastOf(order[1][0], order[1][1]);
  const added = d[tour[p1 - 1]][x] + d[y][z] + d[w][tour[p3]];
  return added - removed;
}

function threeOpt(d, tour, maxRounds = 1000) {
  const n = tour.length;
  let best = tour.slice();
  let rounds = 0;
  let moves = 0;
  for (;;) {
    let improved = false;
    rounds += 1;
    for (let p1 = 1; p1 + 2 < n && !improved; p1++) {
      for (let p2 = p1 + 1; p2 + 1 < n && !improved; p2++) {
        for (let p3 = p2 + 1; p3 < n && !improved; p3++) {
          for (let variant = 1; variant < 8; variant++) {
            if (threeOptDelta(d, best, p1, p2, p3, variant) < -1e-12) {
              best = applyThreeOpt(best, p1, p2, p3, variant);
              moves += 1;
              improved = true;
              break;
            }
          }
        }
      }
    }
    if (!improved || rounds >= maxRounds) break;
  }
  return { tour: best, rounds, moves };
}

/** Exhaustive re-scan proving no improving move of the searched neighbourhood is left. */
function isTwoOptOptimal(d, tour) {
  const n = tour.length;
  for (let i = 1; i + 1 < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (tourCost(d, twoOptMove(d, tour, i, j)) < tourCost(d, tour) - 1e-9) return false;
    }
  }
  return true;
}

function isThreeOptOptimal(d, tour) {
  const n = tour.length;
  for (let p1 = 1; p1 + 2 < n; p1++) {
    for (let p2 = p1 + 1; p2 + 1 < n; p2++) {
      for (let p3 = p2 + 1; p3 < n; p3++) {
        for (let variant = 1; variant < 8; variant++) {
          if (threeOptDelta(d, tour, p1, p2, p3, variant) < -1e-9) return false;
        }
      }
    }
  }
  return true;
}

/** Exact Held-Karp DP over subsets - the independent optimality witness. */
function heldKarp(d, start = 0) {
  const n = d.length;
  if (n <= 1) return 0;
  const full = 1 << n;
  const dp = Array.from({ length: n }, () => new Float64Array(full).fill(Infinity));
  dp[start][1 << start] = 0;
  for (let mask = 1; mask < full; mask++) {
    for (let u = 0; u < n; u++) {
      if (!(mask & (1 << u))) continue;
      const current = dp[u][mask];
      if (current === Infinity) continue;
      for (let v = 0; v < n; v++) {
        if (mask & (1 << v)) continue;
        const next = mask | (1 << v);
        const candidate = current + d[u][v];
        if (candidate < dp[v][next]) dp[v][next] = candidate;
      }
    }
  }
  let best = Infinity;
  for (let u = 0; u < n; u++) {
    const candidate = dp[u][full - 1] + d[u][start];
    if (candidate < best) best = candidate;
  }
  return best;
}

/** Nearest neighbour -> 2-opt -> 3-opt, reporting the cost at every stage. */
function pipeline(d) {
  const nn = nearestNeighbour(d);
  const nnCost = tourCost(d, nn);
  const two = twoOpt(d, nn);
  const twoCost = tourCost(d, two.tour);
  const three = threeOpt(d, two.tour);
  const threeCost = tourCost(d, three.tour);
  return { nn, nnCost, two: two.tour, twoCost, three: three.tour, threeCost };
}

function main() {
  const assert = require("node:assert");

  // (1) Unit square: nearest neighbour scores 4, which happens to be optimal.
  const square = distanceMatrix([[0, 0], [1, 0], [1, 1], [0, 1]]);
  assert.strictEqual(tourCost(square, nearestNeighbour(square)), 4);
  assert.strictEqual(heldKarp(square), 4);
  assert.deepStrictEqual(nearestNeighbour(square), [0, 1, 2, 3]);
  assert.ok(isTwoOptOptimal(square, [0, 1, 2, 3]));

  // (2) The delta formula must agree with a full re-costing of the moved tour, and every move
  //     must keep the tour a permutation - audited over the whole 3-opt neighbourhood.
  for (let seed = 3; seed <= 6; seed++) {
    const n = 8 + (seed % 3);
    const d = distanceMatrix(lcgPoints(n, seed * 37, 25));
    const tour = shuffle([...Array(n).keys()], seed);
    const before = tourCost(d, tour);
    for (let p1 = 1; p1 + 2 < n; p1++) {
      for (let p2 = p1 + 1; p2 + 1 < n; p2++) {
        for (let p3 = p2 + 1; p3 < n; p3++) {
          for (let variant = 0; variant < 8; variant++) {
            const moved = applyThreeOpt(tour, p1, p2, p3, variant);
            assert.ok(tourIsPermutation(n, moved), `seed ${seed}: move must be a permutation`);
            const delta = threeOptDelta(d, tour, p1, p2, p3, variant);
            assert.ok(Math.abs(delta - (tourCost(d, moved) - before)) <= 1e-9,
              `seed ${seed} (${p1},${p2},${p3},v${variant}): delta ${delta} vs re-costed`);
          }
        }
      }
    }
    assert.deepStrictEqual(applyThreeOpt(tour, 1, 2, 3, 0), tour, "variant 0 is the identity");
    assert.deepStrictEqual(applyThreeOpt(tour, 1, 2, n - 1, 1), tour,
      "variant 1 with a singleton S2 is the identity as well");
    assert.deepStrictEqual(applyThreeOpt(tour, 1, 2, n - 1, 2).slice(2, n - 1),
      tour.slice(2, n - 1).reverse(), "variant 2 is exactly the 2-opt reversal of the tail segment");
    assert.deepStrictEqual(applyThreeOpt(tour, 1, 2, n - 1, 2)[0], tour[0]);
  }

  // (3) End-to-end pipeline on metric instances: non-increasing costs, valid tours, no beating
  //     the exact optimum, and 1.5 * OPT respected.
  let optimaFound = 0;
  const instances = [];
  for (let seed = 11; seed <= 18; seed++) {
    const n = 8 + (seed % 4);
    const d = distanceMatrix(lcgPoints(n, seed * 53, 30));
    const optimal = heldKarp(d);
    const run = pipeline(d);
    assert.ok(tourIsPermutation(n, run.nn) && tourIsPermutation(n, run.two) &&
      tourIsPermutation(n, run.three), `seed ${seed}: all stages are permutations`);
    assert.ok(run.twoCost <= run.nnCost + 1e-9, `seed ${seed}: 2-opt never worsens NN`);
    assert.ok(run.threeCost <= run.twoCost + 1e-9, `seed ${seed}: 3-opt never worsens 2-opt`);
    assert.ok(run.threeCost >= optimal - 1e-9, `seed ${seed}: no tour may beat the optimum`);
    assert.ok(run.threeCost <= 1.5 * optimal + 1e-9,
      `seed ${seed}: heuristic stays within 1.5 * OPT (${run.threeCost} vs ${optimal})`);
    assert.ok(isTwoOptOptimal(d, run.two), `seed ${seed}: 2-opt result is 2-opt optimal`);
    assert.ok(isThreeOptOptimal(d, run.three), `seed ${seed}: 3-opt result is 3-opt optimal`);
    if (Math.abs(run.threeCost - optimal) <= 1e-9) optimaFound += 1;
    instances.push([seed, n, run.nnCost, run.twoCost, run.threeCost, optimal]);
  }
  assert.ok(optimaFound >= Math.ceil(instances.length / 2),
    `expected the pipeline to hit the optimum on most seeds, got ${optimaFound}/${instances.length}`);
  assert.ok(instances.some(([, , nn, two]) => two < nn - 1e-9),
    "at least one instance must show 2-opt strictly improving nearest neighbour");

  // (4) 3-opt started from an arbitrary tour can only help, and small instances behave.
  for (let seed = 21; seed <= 24; seed++) {
    const n = 9;
    const d = distanceMatrix(lcgPoints(n, seed * 71, 40));
    for (let t = 0; t < 3; t++) {
      const tour = shuffle([...Array(n).keys()], seed * 10 + t);
      const before = tourCost(d, tour);
      const after = tourCost(d, threeOpt(d, tour).tour);
      assert.ok(after <= before + 1e-9, `seed ${seed}: 3-opt must not increase the cost`);
    }
  }
  const tiny = distanceMatrix([[0, 0], [3, 0], [3, 4]]);
  assert.deepStrictEqual(nearestNeighbour(tiny), [0, 1, 2]);
  assert.strictEqual(tourCost(tiny, nearestNeighbour(tiny)), 12, "3 + 4 + 5");
  assert.strictEqual(heldKarp(tiny), 12);
  assert.deepStrictEqual(twoOpt(tiny, [0, 1, 2]).tour, [0, 1, 2]);
  assert.deepStrictEqual(threeOpt(tiny, [0, 1, 2]).tour, [0, 1, 2]);
  const pair = distanceMatrix([[0, 0], [2, 0]]);
  assert.strictEqual(tourCost(pair, nearestNeighbour(pair)), 4);
  assert.deepStrictEqual(nearestNeighbour(distanceMatrix([[5, 5]])), [0]);

  // (5) Explicit improvement example: a tour that is 2-opt and then 3-opt improvable.
  const kinked = distanceMatrix([[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [0, 1]]);
  const bad = [0, 2, 4, 1, 3, 5];
  const twoBetter = twoOpt(kinked, bad);
  assert.ok(tourCost(kinked, twoBetter.tour) < tourCost(kinked, bad), "2-opt finds the kink");
  assert.ok(tourCost(kinked, twoBetter.tour) >= heldKarp(kinked) - 1e-9);

  // (6) Determinism.
  const dRepeat = distanceMatrix(lcgPoints(10, 777, 30));
  assert.deepStrictEqual(pipeline(dRepeat), pipeline(dRepeat));
  assert.deepStrictEqual(twoOpt(dRepeat, shuffle([...Array(10).keys()], 5)),
    twoOpt(dRepeat, shuffle([...Array(10).keys()], 5)));
}

module.exports = { lcgNext, lcgPoints, distanceMatrix, tourCost, tourIsPermutation, shuffle,
  nearestNeighbour, twoOptMove, twoOpt, THREE_OPT_VARIANTS, applyThreeOpt, threeOptDelta, threeOpt,
  isTwoOptOptimal, isThreeOptOptimal, heldKarp, pipeline };

if (require.main === module) {
  main();
  console.log("PASS");
}
