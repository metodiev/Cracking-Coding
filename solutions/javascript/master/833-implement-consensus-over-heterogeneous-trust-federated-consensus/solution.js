/** Task 833 - Implement consensus over heterogeneous trust (federated consensus) simulation.
 * (Tier: Master, Beyond — Distributed Systems & Cryptography)
 *
 * Federated Byzantine agreement (Stellar-style) over a weighted, heterogeneous trust topology: every
 * node declares its own quorum set = (threshold, weighted validators). A set Q is a *quorum* for a node
 * v when the weight Q ∩ validators(v) reaches v's threshold, and Q is a quorum (period) when it is a
 * quorum for every node inside it - the recursive definition that makes trust federated rather than
 * uniform. Safety rests on the combinatorial property that two quorums cannot be disjoint (or, with
 * faults, that their intersection contains at least one correct node); liveness rests on quorum
 * availability in the absence of faults.
 *
 * Everything here is exhaustive on purpose: quorums are enumerated by brute force over the 2^N
 * subsets, so the intersection property is *proved by enumeration* for the topologies used, and every
 * Byzantine subset of the assumed size is enumerated when checking safety. The simulation itself is a
 * simplified ballot: each node votes once per round for the value it accepts, delivers votes after one
 * round (with an optional deterministic drop schedule), Byzantine nodes equivocate deterministically
 * (different values to different peers), and a node commits when the weight of the matching votes it
 * has seen reaches its own threshold.
 *
 * What is simplified: no nomination phase, no ballot counters, no message authentication (so a message
 * is trusted as sent), and the schedule is a seeded LCG rather than an adversary. What is verified is
 * safety (never two conflicting commits), liveness (all correct nodes commit within a bounded number of
 * rounds when a quorum is available and reachable) and the exact quorum combinatorics behind them.
 */
"use strict";

/** A topology: weights, and for each node a quorum set {threshold, validators:[{node,weight}]}. */
function makeTopology(spec) {
  const ids = Object.keys(spec.weights).map(Number).sort((a, b) => a - b);
  const weights = new Map(ids.map((id) => [id, spec.weights[id]]));
  const quorumSets = new Map(ids.map((id) => [id, { threshold: spec.sets[id].threshold,
    validators: spec.sets[id].validators.map((entry) => ({ node: entry.node, weight: entry.weight })) }]));
  return { ids, weights, quorumSets };
}

/** Weight of `set` (an array of node ids) inside v's validator list. */
function trustedWeight(topology, v, set) {
  const members = new Set(set);
  const quorumSet = topology.quorumSets.get(v);
  if (quorumSet === undefined) return 0;
  let total = 0;
  for (const entry of quorumSet.validators) if (members.has(entry.node)) total += entry.weight;
  return total;
}

const isQuorumFor = (topology, v, set) =>
  trustedWeight(topology, v, set) >= topology.quorumSets.get(v).threshold;

/** A set is a quorum when it is a quorum for each of its own members (the federated definition). */
const isQuorum = (topology, set) => set.every((v) => isQuorumFor(topology, v, set));

/** Brute-force enumeration of every quorum (2^N subsets). */
function allQuorums(topology) {
  const ids = topology.ids;
  const found = [];
  for (let mask = 1; mask < 2 ** ids.length; mask++) {
    const set = ids.filter((_, position) => (mask >> position) & 1);
    if (isQuorum(topology, set)) found.push(set);
  }
  return found;
}

/** The first pair of quorums with an empty intersection, or null. */
function disjointQuorumPair(topology) {
  const quorums = allQuorums(topology);
  for (let i = 0; i < quorums.length; i++) {
    for (let j = i + 1; j < quorums.length; j++) {
      const shared = quorums[i].filter((v) => quorums[j].includes(v));
      if (shared.length === 0) return [quorums[i], quorums[j]];
    }
  }
  return null;
}

/**
 * Safety condition with f faults: for every Byzantine set of size <= f there must be a pair of
 * intersecting quorums whose intersection still contains a correct node. Exhaustive over all subsets.
 */
function safetyCheck(topology, byzantineLimit) {
  const ids = topology.ids;
  const quorums = allQuorums(topology);
  const failures = [];
  for (let mask = 0; mask < 2 ** ids.length; mask++) {
    const byzantine = ids.filter((_, position) => (mask >> position) & 1);
    if (byzantine.length > byzantineLimit) continue;
    const faulty = new Set(byzantine);
    let witness = null;
    for (let i = 0; i < quorums.length && witness === null; i++) {
      for (let j = i + 1; j < quorums.length && witness === null; j++) {
        const correctShared = quorums[i].filter((v) => quorums[j].includes(v) && !faulty.has(v));
        if (correctShared.length === 0) witness = [quorums[i], quorums[j]];
      }
    }
    if (witness !== null) failures.push({ byzantine, witness });
  }
  return { quorums, failures };
}

/** Classic threshold topology: every node trusts the same set with the same threshold. */
function thresholdTopology(count, threshold) {
  const weights = {};
  const validators = Array.from({ length: count }, (_, id) => ({ node: id, weight: 1 }));
  const sets = {};
  for (let id = 0; id < count; id++) {
    weights[id] = 1;
    sets[id] = { threshold, validators: validators.map((entry) => ({ ...entry })) };
  }
  return makeTopology({ weights, sets });
}

const lcg = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
};

/**
 * Simplified ballot simulation. `proposals[v]` is node v's initial value (Byzantine nodes may hold
 * several). Delivery happens one round later, `dropSchedule` lists {from,to,round} messages to lose,
 * and a node commits once the weight of matching votes it has seen reaches its own threshold.
 */
function simulateConsensus({ topology, proposals, byzantine = [], rounds = 12, seed = 833,
  dropSchedule = [] }) {
  const rng = lcg(seed);
  const faulty = new Set(byzantine);
  const received = new Map(topology.ids.map((id) => [id, new Map()]));
  const committed = new Map();
  const inFlight = [];
  const trace = [];
  let messages = 0;
  for (let id of topology.ids) if (!faulty.has(id)) received.get(id).set(id, proposals[id]);
  for (let round = 1; round <= rounds; round++) {
    // determine what each node votes for this round
    const votes = new Map();
    for (const id of topology.ids) {
      if (faulty.has(id)) {
        // equivocation: pick a value per peer deterministically
        votes.set(id, null);
        continue;
      }
      const counts = new Map();
      for (const [source, value] of received.get(id)) counts.set(value, (counts.get(value) ?? 0) + 1);
      let best = proposals[id];
      let bestWeight = -1;
      for (const [value, count] of counts) {
        const weight = trustedWeight(topology, id, [...received.get(id).entries()]
          .filter(([, candidate]) => candidate === value).map(([source]) => source));
        void count;
        if (weight > bestWeight) {
          bestWeight = weight;
          best = value;
        }
      }
      votes.set(id, best);
    }
    // deliver messages queued for this round
    const due = inFlight.filter((message) => message.round === round);
    for (const message of due) {
      received.get(message.to).set(message.from, message.value);
    }
    for (let i = inFlight.length - 1; i >= 0; i--) if (inFlight[i].round === round) inFlight.splice(i, 1);
    // send votes to the peers in the sender's quorum set (Byzantine nodes send conflicting values)
    for (const id of topology.ids) {
      const quorumSet = topology.quorumSets.get(id);
      for (const entry of quorumSet.validators) {
        const to = entry.node;
        const dropped = dropSchedule.some((drop) => drop.from === id && drop.to === to && drop.round === round);
        if (dropped) continue;
        if (faulty.has(id)) {
          const conflicting = rng() < 0.5 ? 0 : 1;
          inFlight.push({ from: id, to, value: conflicting, round: round + 1 });
          messages += 1;
          continue;
        }
        inFlight.push({ from: id, to, value: votes.get(id), round: round + 1 });
        messages += 1;
      }
    }
    // commit check over the votes received so far
    for (const id of topology.ids) {
      if (faulty.has(id) || committed.has(id)) continue;
      const seen = received.get(id);
      const values = new Set(seen.values());
      for (const value of values) {
        const supporters = [...seen.entries()].filter(([, candidate]) => candidate === value).map(([source]) => source);
        if (trustedWeight(topology, id, supporters) >= topology.quorumSets.get(id).threshold) {
          committed.set(id, value);
          break;
        }
      }
    }
    trace.push({ round, committed: new Map(committed) });
  }
  return { committed, trace, messages, received };
}

module.exports = { makeTopology, trustedWeight, isQuorumFor, isQuorum, allQuorums,
  disjointQuorumPair, safetyCheck, thresholdTopology, simulateConsensus, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];

  // ---- 1. quorum enumeration on a hand-checkable topology -------------------------------------
  const tiny = thresholdTopology(3, 3);
  const tinyQuorums = allQuorums(tiny).map((set) => set.join("+")).sort();
  assert.deepStrictEqual(tinyQuorums, ["0+1+2"], "unanimity leaves exactly one quorum");
  assert.ok(isQuorumFor(tiny, 0, [0, 1, 2]));
  assert.ok(!isQuorumFor(tiny, 0, [0, 1]), "below the threshold is not a quorum");
  const majority = thresholdTopology(3, 2);
  const majorityQuorums = allQuorums(majority).map((set) => set.join("+")).sort();
  assert.deepStrictEqual(majorityQuorums, ["0+1", "0+1+2", "0+2", "1+2"], "majority quorums of three nodes");
  assert.ok(disjointQuorumPair(majority) === null, "majority quorums pairwise intersect");
  const split = makeTopology({
    weights: { 0: 1, 1: 1, 2: 1, 3: 1 },
    sets: {
      0: { threshold: 2, validators: [{ node: 0, weight: 1 }, { node: 1, weight: 1 }] },
      1: { threshold: 2, validators: [{ node: 0, weight: 1 }, { node: 1, weight: 1 }] },
      2: { threshold: 2, validators: [{ node: 2, weight: 1 }, { node: 3, weight: 1 }] },
      3: { threshold: 2, validators: [{ node: 2, weight: 1 }, { node: 3, weight: 1 }] },
    },
  });
  const disjoint = disjointQuorumPair(split);
  assert.ok(disjoint !== null, "a partitioned trust topology has disjoint quorums");
  assert.deepStrictEqual(disjoint[0].slice().sort((a, b) => a - b), [0, 1]);
  assert.deepStrictEqual(disjoint[1].slice().sort((a, b) => a - b), [2, 3]);

  // ---- 2. safety by exhaustive enumeration ------------------------------------------------------
  const medium = thresholdTopology(5, 4);
  const mediumSafety = safetyCheck(medium, 1);
  assert.ok(mediumSafety.quorums.length > 0);
  assert.strictEqual(mediumSafety.failures.length, 0, "5 nodes with a 4-of-5 quorum tolerate one fault");
  const tight = thresholdTopology(4, 2);
  const tightSafety = safetyCheck(tight, 1);
  assert.ok(tightSafety.failures.length > 0, "a 2-of-4 quorum cannot tolerate any fault");
  const classic = thresholdTopology(4, 3);
  const classicSafety = safetyCheck(classic, 1);
  assert.strictEqual(classicSafety.failures.length, 0, "3-of-4 tolerates one fault");
  assert.ok(safetyCheck(classic, 2).failures.length > 0, "3-of-4 cannot tolerate two faults");
  const weighted = makeTopology({
    weights: { 0: 3, 1: 3, 2: 2, 3: 2, 4: 1 },
    sets: Object.fromEntries([0, 1, 2, 3, 4].map((id) => [id, {
      threshold: 8,
      validators: [{ node: 0, weight: 3 }, { node: 1, weight: 3 }, { node: 2, weight: 2 },
        { node: 3, weight: 2 }, { node: 4, weight: 1 }],
    }])),
  });
  const weightedSafety = safetyCheck(weighted, 1);
  assert.strictEqual(weightedSafety.failures.length, 0, `weighted topology tolerates one fault (total weight ${11 - 1})`);
  assert.ok(safetyCheck(weighted, 2).failures.length > 0, "removing two heavy nodes breaks the intersection");

  // ---- 3. liveness and safety in the simulation --------------------------------------------------
  const nodeCount = 5;
  const proposals = { 0: 1, 1: 1, 2: 0, 3: 1, 4: 0 };
  const topology = thresholdTopology(nodeCount, 4);
  const run = simulateConsensus({ topology, proposals, rounds: 6, seed: 833 });
  assert.strictEqual(new Set(run.committed.values()).size, 1, "every commit agrees on one value");
  assert.strictEqual(run.committed.size, nodeCount, "all correct nodes commit");
  assert.strictEqual(run.committed.get(0), 1, "the majority value wins");
  let previous = 0;
  for (const step of run.trace) {
    assert.ok(step.committed.size >= previous, "commits are monotone");
    previous = step.committed.size;
    assert.strictEqual(new Set(step.committed.values()).size <= 1, true, "no conflicting commits at any round");
  }
  assert.ok(run.messages <= nodeCount * nodeCount * 6, "message bound");
  const equivocating = simulateConsensus({ topology, proposals, byzantine: [4], rounds: 6, seed: 7 });
  assert.strictEqual(new Set(equivocating.committed.values()).size, 1, "safety holds with an equivocator");
  assert.strictEqual(equivocating.committed.size, 4);
  assert.ok(equivocating.committed.get(0) === equivocating.committed.get(1), "correct nodes agree");

  const partitioned = simulateConsensus({ topology: split, proposals: { 0: 1, 1: 1, 2: 0, 3: 0 },
    rounds: 6, seed: 11 });
  assert.strictEqual(new Set(partitioned.committed.values()).size, 2,
    "disjoint quorums produce conflicting commits - exactly what the intersection property forbids");

  // ---- 4. liveness with one dropped link ---------------------------------------------------------
  const lossy = simulateConsensus({ topology, proposals, rounds: 8, seed: 833,
    dropSchedule: [{ from: 2, to: 0, round: 1 }, { from: 2, to: 1, round: 1 }] });
  assert.strictEqual(lossy.committed.size, nodeCount, "dropped votes are recovered in later rounds");
  assert.strictEqual(new Set(lossy.committed.values()).size, 1);

  // ---- 5. heterogeneous trust: heavy nodes matter ------------------------------------------------
  const heterogeneous = makeTopology({
    weights: { 0: 4, 1: 4, 2: 4, 3: 4, 4: 1 },
    sets: {
      0: { threshold: 11, validators: [{ node: 0, weight: 4 }, { node: 1, weight: 4 }, { node: 2, weight: 4 }, { node: 3, weight: 4 }, { node: 4, weight: 1 }] },
      1: { threshold: 11, validators: [{ node: 0, weight: 4 }, { node: 1, weight: 4 }, { node: 2, weight: 4 }, { node: 3, weight: 4 }, { node: 4, weight: 1 }] },
      2: { threshold: 11, validators: [{ node: 0, weight: 4 }, { node: 1, weight: 4 }, { node: 2, weight: 4 }, { node: 3, weight: 4 }, { node: 4, weight: 1 }] },
      3: { threshold: 11, validators: [{ node: 0, weight: 4 }, { node: 1, weight: 4 }, { node: 2, weight: 4 }, { node: 3, weight: 4 }, { node: 4, weight: 1 }] },
      4: { threshold: 1, validators: [{ node: 4, weight: 1 }] },
    },
  });
  const hQuorums = allQuorums(heterogeneous);
  assert.ok(hQuorums.some((set) => set.length === 1 && set[0] === 4), "node 4 trusts only itself");
  assert.strictEqual(disjointQuorumPair(heterogeneous) === null, false,
    "a self-trusting node can form a disjoint quorum");
  const fixed = makeTopology({
    weights: { 0: 4, 1: 4, 2: 4, 3: 4, 4: 4 },
    sets: Object.fromEntries([0, 1, 2, 3, 4].map((id) => [id, { threshold: 16,
      validators: [0, 1, 2, 3, 4].map((node) => ({ node, weight: 4 })) }])),
  });
  assert.strictEqual(disjointQuorumPair(fixed), null, "balanced weights with a 16/20 threshold intersect");
  assert.strictEqual(safetyCheck(fixed, 1).failures.length, 0, "and tolerate one fault");
  assert.strictEqual(safetyCheck(fixed, 2).failures.length, 0, "any two 16/20 quorums share three nodes, so two faults are still tolerated");
  assert.ok(safetyCheck(fixed, 3).failures.length > 0, "but not three");
  const weaker = makeTopology({
    weights: { 0: 4, 1: 4, 2: 4, 3: 4, 4: 4 },
    sets: Object.fromEntries([0, 1, 2, 3, 4].map((id) => [id, { threshold: 9,
      validators: [0, 1, 2, 3, 4].map((node) => ({ node, weight: 4 })) }])),
  });
  assert.strictEqual(safetyCheck(weaker, 0).failures.length, 0, "a 9/20 threshold is safe without faults");
  assert.ok(safetyCheck(weaker, 1).failures.length > 0,
    "but a single shared node can be the faulty one, so it tolerates no fault");
  const fixedRun = simulateConsensus({ topology: fixed, proposals: { 0: 0, 1: 0, 2: 1, 3: 0, 4: 1 },
    byzantine: [4], rounds: 6, seed: 21 });
  assert.strictEqual(new Set(fixedRun.committed.values()).size, 1);
  assert.strictEqual(fixedRun.committed.size, 4, "the faulty node never commits");

  // ---- 6. determinism and degenerate topologies --------------------------------------------------
  const again = simulateConsensus({ topology, proposals, rounds: 6, seed: 833 });
  assert.deepStrictEqual([...again.committed.entries()], [...run.committed.entries()]);
  assert.strictEqual(again.messages, run.messages);
  const solo = thresholdTopology(1, 1);
  const soloRun = simulateConsensus({ topology: solo, proposals: { 0: 7 }, rounds: 2, seed: 1 });
  assert.strictEqual(soloRun.committed.get(0), 7, "a single node commits its own value immediately");
  const allFaulty = simulateConsensus({ topology: thresholdTopology(3, 3), proposals: { 0: 1, 1: 1, 2: 1 },
    byzantine: [0, 1, 2], rounds: 3, seed: 2 });
  assert.strictEqual(allFaulty.committed.size, 0, "with every node faulty nothing commits");
  assert.throws(() => isQuorumFor(thresholdTopology(2, 1), 5, [0]), /undefined|quorumSets|threshold/);

  metrics.push(`quorums: unanimity 1, majority-of-3 4, 4-of-5 tolerates one fault (${mediumSafety.failures.length} violations), 3-of-4 tolerates one fault but not two`);
  metrics.push(`weighted trust 3,3,2,2,1 with threshold 8: no violation for f = 1, a violating pair for f = 2`);
  metrics.push(`simulation: ${run.committed.size} nodes committed in ${run.trace.length} rounds with ${run.messages} messages, one value only; equivocation and a dropped link keep safety and liveness`);
  metrics.push(`partitioned trust produces ${new Set(partitioned.committed.values()).size} conflicting commits, which is exactly what the quorum intersection property rules out`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
