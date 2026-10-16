/**
 * Task 670 - Implement resource allocation graph and Banker's algorithm simulator.
 * (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Two complementary views of the same deadlock-avoidance machinery:
 *
 *  1. Banker's algorithm on a state (total vector, allocation matrix, max-claim matrix). The derived
 *     vectors are `available = total - sum_i allocation[i]` and `need[i] = max[i] - allocation[i]`.
 *     `safetyCheck` runs the classic greedy completion: repeatedly pick any unfinished process whose
 *     whole need fits in the current work vector, let it run to completion and release everything it
 *     holds; the state is safe iff every process can be finished, and the order found is a safe
 *     sequence. `requestResources` implements the request step (request <= need, request <= available,
 *     and the tentatively granted state must still be safe) and reports the rejected reason.
 *     `bruteForceSafe` enumerates *all* n! completion orders and is the independent reference for the
 *     greedy test - the greedy order is valid iff some permutation is.
 *
 *  2. The resource allocation graph (RAG): request edges P->R and assignment edges R->P, with
 *     `ragCycle` doing a depth-first cycle hunt on that bipartite graph and `detectDeadlock` running
 *     the reduction algorithm (a process is finished when it holds nothing; it can be finished if its
 *     remaining requests fit in the currently free units). Both views are extracted from the same
 *     allocation/request matrices, so they are answers about one state.
 *
 * Verified: the textbook 5-process safe state (10/5/7 units) has the hand-computed available vector
 * [3,3,2], is safe, and the greedy sequence replays as a valid order that the exhaustive search also
 * finds; the textbook requests (1,0,2) for P1 is granted, whereas P0's (0,2,0) and P4's (3,3,0) are
 * refused because the resulting state is unsafe - the reason string says so; malformed requests
 * (negative, wrong arity, unknown process) and claim/availability violations are rejected with
 * distinct reasons; the invariant sum_i allocation[i][j] + available[j] == total[j] is re-checked
 * after every grant of a randomised request sequence; greedy and brute force agree on safe/unsafe for
 * dozens of random states; and the RAG shows the classic caveat that a cycle is necessary but not
 * sufficient once a resource has several instances - the identical graph is a deadlock with one unit
 * of R0 and merely reducible with two.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed670n;

/** Deterministic 64-bit splitmix64, used for the random states/requests. */
class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ------------------------------------------------------------------------------------ state layer
const state = (total, allocation, max) => ({ total, allocation, max });

function availableOf(s) {
  return s.total.map((units, j) => units - s.allocation.reduce((acc, row) => acc + row[j], 0));
}

function needOf(s) {
  return s.max.map((row, i) => row.map((claim, j) => claim - s.allocation[i][j]));
}

const fits = (a, b) => a.every((v, j) => v <= b[j]);
const zeros = (m) => new Array(m).fill(0);

/** Replay an order against the current state; returns whether it is a legal completion order. */
function replayOrder(s, order) {
  const need = needOf(s);
  const work = availableOf(s);
  for (const i of order) {
    if (!fits(need[i], work)) return false;
    for (let j = 0; j < work.length; j++) work[j] += s.allocation[i][j];
  }
  return true;
}
/**
 * Banker's safety algorithm: greedy completion. Picking any satisfiable process first is safe because
 * finishing a process only ever adds units back, so a greedy failure means no order works.
 */
function safetyCheck(s) {
  const need = needOf(s);
  const work = availableOf(s);
  const n = s.allocation.length;
  const finished = new Array(n).fill(false);
  const sequence = [];
  let progressed = true;
  while (progressed) {
    progressed = false;
    for (let i = 0; i < n; i++) {
      if (finished[i] || !fits(need[i], work)) continue;
      for (let j = 0; j < work.length; j++) work[j] += s.allocation[i][j];
      finished[i] = true;
      sequence.push(i);
      progressed = true;
    }
  }
  return { safe: finished.every(Boolean), sequence, finalWork: work,
    unfinished: finished.map((f, i) => (f ? null : i)).filter((i) => i !== null) };
}

/** Independent reference: try every one of the n! completion orders. */
function bruteForceSafe(s) {
  const items = [...Array(s.allocation.length).keys()];
  const orders = [];
  const build = (prefix, rest) => {
    if (rest.length === 0) {
      orders.push(prefix);
      return;
    }
    for (const [k, item] of rest.entries()) build([...prefix, item], rest.filter((_, i) => i !== k));
  };
  build([], items);
  for (const order of orders) {
    if (replayOrder(s, order)) return { safe: true, order, considered: orders.length };
  }
  return { safe: false, order: null, considered: orders.length };
}

/**
 * Banker's request step. Rejections are classified so the checks can assert *why* a request failed.
 * A granted request returns the new state; the original one is never mutated.
 */
function requestResources(s, p, request) {
  const m = s.total.length;
  if (!Number.isInteger(p) || p < 0 || p >= s.allocation.length) return { granted: false, reason: "invalid-process", state: s };
  if (!Array.isArray(request) || request.length !== m || request.some((v) => !Number.isInteger(v) || v < 0)) return { granted: false, reason: "invalid-request", state: s };
  const need = needOf(s);
  const available = availableOf(s);
  if (!fits(request, need[p])) return { granted: false, reason: "exceeds-claim", state: s };
  if (!fits(request, available)) return { granted: false, reason: "insufficient-available", state: s };
  const tentative = state(s.total, s.allocation.map((row, i) => (i === p ? row.map((v, j) => v + request[j]) : [...row])), s.max);
  const check = safetyCheck(tentative);
  if (!check.safe) return { granted: false, reason: "unsafe", state: s, tentative, sequence: check.sequence };
  return { granted: true, reason: "granted", state: tentative, sequence: check.sequence };
}

// ---------------------------------------------------------------------------------- graph layer
/**
 * Resource allocation graph: node i is process i, node n + j is resource j. Edges go P->R for a
 * pending request and R->P for an assignment. A cycle is found with the usual three-colour DFS.
 */
function ragCycle(allocation, requests) {
  const n = allocation.length;
  const m = allocation[0].length;
  const adj = Array.from({ length: n + m }, () => []);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      if (allocation[i][j] > 0) adj[n + j].push(i);
      if (requests[i][j] > 0) adj[i].push(n + j);
    }
  }
  const colour = new Array(n + m).fill(0);
  const path = [];
  let cycle = null;
  const visit = (u) => {
    colour[u] = 1;
    path.push(u);
    for (const v of adj[u]) {
      if (colour[v] === 1) {
        cycle = path.slice(path.indexOf(v)).concat(v);
        return true;
      }
      if (colour[v] === 0 && visit(v)) return true;
    }
    path.pop();
    colour[u] = 2;
    return false;
  };
  let hasCycle = false;
  for (let u = 0; u < n + m && !hasCycle; u++) {
    if (colour[u] === 0) hasCycle = visit(u);
  }
  return { hasCycle, cycle, edges: adj.reduce((acc, list) => acc + list.length, 0) };
}

/** Reduction algorithm: processes that can never be finished stay in the returned list. */
function detectDeadlock(allocation, requests, total) {
  const n = allocation.length;
  const m = total.length;
  const work = total.map((units, j) => units - allocation.reduce((acc, row) => acc + row[j], 0));
  const finished = allocation.map((row) => row.every((v) => v === 0));
  let progressed = true;
  while (progressed) {
    progressed = false;
    for (let i = 0; i < n; i++) {
      if (finished[i] || !fits(requests[i], work)) continue;
      for (let j = 0; j < m; j++) work[j] += allocation[i][j];
      finished[i] = true;
      progressed = true;
    }
  }
  return { deadlocked: finished.map((f, i) => (f ? null : i)).filter((i) => i !== null), fullyReduced: finished.every(Boolean), finalWork: work };
}

/** Matrix of an (allocation, request) pair as a Banker's state, for the textbook instances. */
const asState = (total, allocation, max) => state(total, allocation, max);

// ------------------------------------------------------------------------------------------ cases
/** Silberschatz' five-process example: 10 / 5 / 7 units of A, B, C. */
function textbook() {
  return asState([10, 5, 7], [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]], [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]);
}

/** Random Banker's state with n processes and m resources; claims always dominate allocations. */
function randomState(rng, n, m) {
  const total = Array.from({ length: m }, () => 2 + rng.nextInt(5));
  const allocation = [];
  const max = [];
  const free = [...total];
  for (let i = 0; i < n; i++) {
    const row = zeros(m);
    const claim = zeros(m);
    for (let j = 0; j < m; j++) {
      const take = free[j] > 0 ? rng.nextInt(free[j] + 1) : 0;
      row[j] = take;
      free[j] -= take;
      claim[j] = take + rng.nextInt(3);
    }
    allocation.push(row);
    max.push(claim);
  }
  return asState(total, allocation, max);
}

function main() {
  // (1) The textbook state: derived vectors, safety and the hand-computed safe sequence.
  const s = textbook();
  assert.deepStrictEqual(availableOf(s), [3, 3, 2], "available = total - sum allocation = [3,3,2]");
  assert.deepStrictEqual(needOf(s)[1], [1, 2, 2], "need[P1] = max - allocation = [1,2,2]");
  const safety = safetyCheck(s);
  assert.strictEqual(safety.safe, true, "the textbook state is safe");
  assert.strictEqual(safety.sequence.length, 5, "and the found sequence finishes all five processes");
  assert.deepStrictEqual(safety.unfinished, [], "nothing is left unfinished");
  assert.ok(replayOrder(s, safety.sequence), "replaying the greedy sequence is a legal order");
  assert.ok(replayOrder(s, [1, 3, 4, 0, 2]), "the textbook order P1,P3,P4,P0,P2 also replays");

  // (2) Brute force agrees, and reports the search it performed.
  const brute = bruteForceSafe(s);
  assert.strictEqual(brute.safe, true, "exhaustive search over 5! = 120 orders finds one");
  assert.strictEqual(brute.considered, 120, "and it really enumerated 120 permutations");
  assert.ok(replayOrder(s, brute.order));
  const unsafe = asState([10, 5, 7], [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [3, 3, 2]], [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]);
  assert.strictEqual(safetyCheck(unsafe).safe, false, "the textbook's unsafe state is rejected");
  assert.deepStrictEqual(availableOf(unsafe), [0, 0, 2], "it is exactly the state a grant of P4's (3,3,0) would create");
  assert.deepStrictEqual(safetyCheck(unsafe).unfinished, [0, 1, 2, 3, 4], "no need fits [0,0,2], so nothing finishes");
  assert.strictEqual(bruteForceSafe(unsafe).safe, false, "and no permutation can complete it");

  // (3) The textbook requests: P1's (1,0,2) is granted, P0's (0,2,0) and P4's (3,3,0) are not.
  const grant = requestResources(s, 1, [1, 0, 2]);
  assert.strictEqual(grant.granted, true, "P1's request (1,0,2) is safe and granted");
  assert.deepStrictEqual(availableOf(grant.state), [2, 3, 0], "leaving available = [2,3,0]");
  assert.deepStrictEqual(needOf(grant.state)[1], [0, 2, 0], "and need[P1] drops to [0,2,0]");
  assert.ok(replayOrder(grant.state, grant.sequence), "the recorded sequence is valid");
  assert.strictEqual(requestResources(grant.state, 0, [0, 2, 0]).reason, "unsafe", "P0's (0,2,0) fits there but leaves no safe sequence");
  assert.strictEqual(requestResources(s, 4, [3, 3, 0]).reason, "unsafe", "P4's (3,3,0) fails the safety test too");
  assert.deepStrictEqual(s.allocation[0], [0, 1, 0], "a refusal never mutates the original state");

  // (4) Rejections are classified, and each reason is reachable.
  assert.strictEqual(requestResources(s, 1, [3, 2, 2]).reason, "exceeds-claim");
  assert.strictEqual(requestResources(s, 2, [4, 0, 0]).reason, "insufficient-available", "P2 may claim 6 units of A but only 3 are free");
  assert.strictEqual(requestResources(s, 1, [-1, 0, 0]).reason, "invalid-request");
  assert.strictEqual(requestResources(s, 1, [0, 0]).reason, "invalid-request");
  assert.strictEqual(requestResources(s, 9, [0, 0, 0]).reason, "invalid-process");
  assert.strictEqual(requestResources(s, 1, [0, 0, 0]).granted, true, "a zero request is a never-failing no-op");

  // (5) Invariant after every grant: units are conserved resource by resource.
  const rng = new SplitMix64(DEFAULT_SEED);
  let running = textbook();
  let grants = 0;
  for (let step = 0; step < 60; step++) {
    const p = rng.nextInt(running.allocation.length);
    const avail = availableOf(running);
    const need = needOf(running);
    const request = running.total.map((_, j) => Math.min(avail[j], need[p][j], rng.nextInt(3)));
    const result = requestResources(running, p, request);
    if (result.granted) {
      running = result.state;
      grants += 1;
    }
    const available = availableOf(running);
    running.total.forEach((units, j) => {
      const held = running.allocation.reduce((acc, row) => acc + row[j], 0);
      assert.strictEqual(held + available[j], units, `resource ${j} is conserved after every step`);
    });
    assert.ok(running.allocation.every((row, i) => row.every((v, j) => v <= running.max[i][j])), "claims hold");
    assert.strictEqual(safetyCheck(running).safe, true, "granted states stay safe by construction");
  }
  assert.ok(grants >= 10, `${grants} of the 60 randomised requests were granted`);

  // (6) Greedy vs exhaustive on random states; both verdicts must occur.
  let safeStates = 0;
  for (let k = 0; k < 40; k++) {
    const random = randomState(rng, 4, 3);
    const greedy = safetyCheck(random);
    const exhaustive = bruteForceSafe(random);
    assert.strictEqual(greedy.safe, exhaustive.safe, `greedy and exhaustive disagree on ${JSON.stringify(random.allocation)}`);
    if (greedy.safe) {
      safeStates += 1;
      assert.ok(replayOrder(random, greedy.sequence), "a safe verdict comes with a valid order");
      assert.ok(replayOrder(random, exhaustive.order));
    } else {
      assert.ok(greedy.unfinished.length > 0, "an unsafe state always leaves someone stuck");
    }
  }
  assert.ok(safeStates > 3 && safeStates < 40, `${safeStates} of 40 random states are safe, so both branches are covered`);

  // (7) RAG: with single-unit resources a cycle is exactly a deadlock.
  const singleDead = { total: [1, 1], allocation: [[1, 0], [0, 1]], requests: [[0, 1], [1, 0]] };
  const dead = detectDeadlock(singleDead.allocation, singleDead.requests, singleDead.total);
  assert.deepStrictEqual(dead.deadlocked, [0, 1], "both processes are deadlocked with one unit each");
  assert.strictEqual(dead.fullyReduced, false);
  const cycle = ragCycle(singleDead.allocation, singleDead.requests);
  assert.strictEqual(cycle.hasCycle, true, "and the resource allocation graph has a cycle");
  assert.deepStrictEqual(cycle.cycle, [0, 3, 1, 2, 0], "P0 -> R1 -> P1 -> R0 -> P0 (node 2 = R0, node 3 = R1)");
  const acyclic = { total: [1, 1], allocation: [[1, 0], [0, 0]], requests: [[0, 0], [1, 0]] };
  assert.strictEqual(ragCycle(acyclic.allocation, acyclic.requests).hasCycle, false);
  assert.deepStrictEqual(detectDeadlock(acyclic.allocation, acyclic.requests, acyclic.total).deadlocked, [], "no cycle, nobody permanently blocked");

  // (8) The documented caveat: the same graph with two units of R0 reduces, so it is not a deadlock.
  const twoUnits = { total: [2, 1], allocation: [[1, 0], [0, 1]], requests: [[0, 1], [1, 0]] };
  assert.strictEqual(ragCycle(twoUnits.allocation, twoUnits.requests).hasCycle, true, "the graph still has the same cycle");
  const reduced = detectDeadlock(twoUnits.allocation, twoUnits.requests, twoUnits.total);
  assert.deepStrictEqual(reduced.deadlocked, [], "documented: with two units of R0 the cycle is not a deadlock");
  assert.strictEqual(reduced.fullyReduced, true, "every process can still run to completion");

  // (9) Determinism and a degenerate state.
  assert.deepStrictEqual(safetyCheck(textbook()), safetyCheck(textbook()));
  assert.deepStrictEqual(ragCycle(singleDead.allocation, singleDead.requests), cycle);
  const solo = asState([1], [[1]], [[1]]);
  assert.strictEqual(safetyCheck(solo).safe, true, "one process holding the only unit is safe");
  assert.deepStrictEqual(safetyCheck(solo).sequence, [0]);
  assert.strictEqual(requestResources(solo, 0, [1]).reason, "exceeds-claim", "nobody can ask beyond their claim");
}

module.exports = { SplitMix64, DEFAULT_SEED, state, availableOf, needOf, fits, replayOrder,
  safetyCheck, bruteForceSafe, requestResources, ragCycle, detectDeadlock, asState, textbook,
  randomState };

if (require.main === module) {
  main();
  console.log("PASS");
}
