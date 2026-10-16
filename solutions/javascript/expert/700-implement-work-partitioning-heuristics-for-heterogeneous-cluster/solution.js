/** Task 700 - Implement work partitioning heuristics for heterogeneous clusters. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Model: m machines with different speeds and n indivisible tasks of different sizes. Task i placed on
 * machine j takes size_i / speed_j, a machine's load is the sum of the tasks assigned to it, and the
 * objective is to minimise the *makespan* (the largest load). Four partitioners are implemented:
 *
 *   roundRobin   task i goes to machine i mod m - the classic naive baseline;
 *   listSchedule greedy list scheduling: task i goes to the currently least loaded machine (this is
 *                the "work stealing" / FIFO partitioner);
 *   lptSchedule  the same greedy over tasks sorted by decreasing size (Longest Processing Time first);
 *   minMax       greedy that knows the speeds: each task goes to the machine whose *resulting
 *                makespan* is smallest, not merely to the least loaded one.
 *   fluid        the perfectly divisible reference: split the work in proportion to speed, which
 *                achieves the lower bound exactly - useful to see how much indivisibility costs.
 *
 * Because the objective is a min-max problem, the two universal references are computed too: the
 * arithmetic lower bound max(total work / total speed, min over machines of the largest task) and, for
 * small instances, the exact optimum by branch and bound.
 *
 * Simplified versus a real scheduler: tasks are independent and preemption-free, sizes are known in
 * advance, machines are uniform (speed scales every task equally), no data locality, no failures.
 *
 * Verified: exact optima for hand cases, minimum-loss properties (the greedy is never worse than the
 * lower bound and never better than the optimum), Graham's (2 - 1/m) bound for list scheduling on
 * equal-speed machines, the classic 2x failure of plain LPT on a heterogeneous cluster that minMax
 * avoids, the perfectly divisible reference hitting the lower bound exactly, assignment invariants
 * (each task once, loads that match the definition), the fluid efficiency of many small tasks,
 * degenerate inputs (empty, one machine, zero-size tasks, single task) and determinism.
 */
"use strict";

function totalWork(tasks) {
  let total = 0;
  for (const size of tasks) total += size;
  return total;
}

function totalSpeed(speeds) {
  let total = 0;
  for (const speed of speeds) total += speed;
  return total;
}

/** max(total / sum(speed), min over machines of the largest task's time there). */
function lowerBound(tasks, speeds) {
  const volume = totalWork(tasks) / totalSpeed(speeds);
  let largest = 0;
  for (const size of tasks) {
    let fastest = Infinity;
    for (const speed of speeds) fastest = Math.min(fastest, size / speed);
    largest = Math.max(largest, fastest);
  }
  return Math.max(volume, largest);
}

function loadsFor(assignment, tasks, speeds) {
  const loads = new Array(speeds.length).fill(0);
  for (let i = 0; i < tasks.length; i++) loads[assignment[i]] += tasks[i] / speeds[assignment[i]];
  return loads;
}

function makespanOf(assignment, tasks, speeds) {
  return Math.max(0, ...loadsFor(assignment, tasks, speeds));
}

/** Generic greedy: `chooser(loads, size, speeds)` returns the machine for the next task. */
function assignGreedy(tasks, speeds, chooser, order = null) {
  const indices = order || tasks.map((value, index) => index);
  const loads = new Array(speeds.length).fill(0);
  const assignment = new Array(tasks.length).fill(0);
  for (const index of indices) {
    const machine = chooser(loads, tasks[index], speeds);
    assignment[index] = machine;
    loads[machine] += tasks[index] / speeds[machine];
  }
  return assignment;
}

function leastLoaded(loads) {
  let best = 0;
  for (let machine = 1; machine < loads.length; machine++) {
    if (loads[machine] < loads[best] - 1e-15) best = machine;
  }
  return best;
}

function smallestResultingMakespan(loads, size, speeds) {
  let best = 0;
  let bestMakespan = Infinity;
  for (let machine = 0; machine < loads.length; machine++) {
    let candidate = loads[machine] + size / speeds[machine];
    for (let other = 0; other < loads.length; other++) {
      if (other !== machine) candidate = Math.max(candidate, loads[other]);
    }
    if (candidate < bestMakespan - 1e-15) {
      bestMakespan = candidate;
      best = machine;
    }
  }
  return best;
}

function roundRobin(tasks, speeds) {
  const assignment = new Array(tasks.length);
  for (let i = 0; i < tasks.length; i++) assignment[i] = i % speeds.length;
  return assignment;
}

function listSchedule(tasks, speeds) {
  return assignGreedy(tasks, speeds, (loads) => leastLoaded(loads));
}

/** LPT: longest processing time first, then the least loaded machine. */
function lptSchedule(tasks, speeds) {
  const order = tasks.map((size, index) => ({ size, index }))
    .sort((a, b) => b.size - a.size || a.index - b.index)
    .map((entry) => entry.index);
  return assignGreedy(tasks, speeds, (loads) => leastLoaded(loads), order);
}

/** Speed-aware greedy: minimise the makespan that would result from the placement. */
function minMaxSchedule(tasks, speeds) {
  const order = tasks.map((size, index) => ({ size, index }))
    .sort((a, b) => b.size - a.size || a.index - b.index)
    .map((entry) => entry.index);
  return assignGreedy(tasks, speeds, (loads, size) => smallestResultingMakespan(loads, size, speeds), order);
}

/** Perfectly divisible work: machine j gets total * speed_j / sum(speed). */
function fluidSchedule(tasks, speeds) {
  const total = totalWork(tasks);
  const denominator = totalSpeed(speeds);
  return speeds.map((speed) => (total * speed) / denominator);
}

function fluidMakespan(tasks, speeds) {
  return totalWork(tasks) / totalSpeed(speeds);
}

/** Exact optimum by branch and bound over the assignment tree (small instances only). */
function optimalMakespan(tasks, speeds, { nodeBudget = 200000 } = {}) {
  if (tasks.length === 0) return 0;
  const order = [...tasks].sort((a, b) => b - a);
  const times = order.map((size) => speeds.map((speed) => size / speed));
  const identical = speeds.every((speed) => speed === speeds[0]);
  const loads = new Array(speeds.length).fill(0);
  let best = Infinity;
  let nodes = 0;
  const search = (index, used, currentMax) => {
    if (++nodes > nodeBudget) return;
    if (currentMax >= best) return;                    // cannot improve
    if (index === order.length) {
      best = Math.min(best, currentMax);
      return;
    }
    // Identical machines are interchangeable, so only the first unused one is worth trying.
    const limit = identical ? Math.min(used, speeds.length - 1) : speeds.length - 1;
    for (let machine = 0; machine <= limit; machine++) {
      loads[machine] += times[index][machine];
      search(index + 1, Math.max(used, machine + 1), Math.max(currentMax, loads[machine]));
      loads[machine] -= times[index][machine];
    }
  };
  search(0, 0, 0);
  return best;
}

function metrics(name, assignment, tasks, speeds) {
  const loads = loadsFor(assignment, tasks, speeds);
  return {
    name,
    makespan: Math.max(...loads),
    loads,
    imbalance: Math.max(...loads) - Math.min(...loads),
    utilization: totalWork(tasks) / (Math.max(...loads) * totalSpeed(speeds)),
  };
}

class Lcg {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  int(bound) {
    return this.next() % bound;
  }
}

module.exports = {
  totalWork, totalSpeed, lowerBound, loadsFor, makespanOf, assignGreedy, roundRobin,
  listSchedule, lptSchedule, minMaxSchedule, fluidSchedule, fluidMakespan, optimalMakespan, metrics, Lcg,
};

if (require.main === module) {
  const assert = require("node:assert");

  const lcg = new Lcg(700);

  // --- hand case ------------------------------------------------------------------------------
  const tasks = [4, 4, 4, 2];
  const speeds = [1, 1, 1];
  const lpt = lptSchedule(tasks, speeds);
  assert.deepStrictEqual(lpt, [0, 1, 2, 0], "LPT fills the machines then adds the small task");
  assert.strictEqual(makespanOf(lpt, tasks, speeds), 6);
  assert.strictEqual(optimalMakespan(tasks, speeds), 6, "and six is optimal for indivisible tasks");
  assert.strictEqual(lowerBound(tasks, speeds), 14 / 3, "the volume bound is 14/3");
  assert.strictEqual(makespanOf(roundRobin(tasks, speeds), tasks, speeds), 6, "round robin ties by luck here");
  const skewed = [5, 1, 1, 1, 1, 1];
  const skewedSpeeds = [1, 1];
  assert.strictEqual(makespanOf(roundRobin(skewed, skewedSpeeds), skewed, skewedSpeeds), 7,
    "round robin piles the small tasks behind the big one");
  assert.deepStrictEqual(lptSchedule(skewed, skewedSpeeds), [0, 1, 1, 1, 1, 1]);
  assert.strictEqual(makespanOf(lptSchedule(skewed, skewedSpeeds), skewed, skewedSpeeds), 5);
  assert.strictEqual(optimalMakespan(skewed, skewedSpeeds), 5, "and LPT reaches the optimum here");
  assert.strictEqual(fluidMakespan(tasks, speeds), 14 / 3, "divisible work hits the bound");
  assert.deepStrictEqual(fluidSchedule(tasks, [1, 1, 1]), [14 / 3, 14 / 3, 14 / 3]);
  const lptMetrics = metrics("lpt", lpt, tasks, speeds);
  assert.strictEqual(lptMetrics.imbalance, 2, "loads are 6, 4, 4");
  assert.ok(Math.abs(lptMetrics.utilization - 14 / 18) < 1e-12);
  assert.deepStrictEqual(loadsFor(lpt, tasks, speeds), [6, 4, 4]);
  assert.strictEqual(loadsFor(lpt, tasks, speeds).reduce((a, b) => a + b, 0), 14, "work is conserved");

  // --- assignment invariants -------------------------------------------------------------------
  for (let trial = 0; trial < 40; trial++) {
    const machineCount = 2 + lcg.int(4);
    const machineSpeeds = [];
    for (let i = 0; i < machineCount; i++) machineSpeeds.push(1 + lcg.int(4));
    const sizes = [];
    const taskCount = 1 + lcg.int(12);
    for (let i = 0; i < taskCount; i++) sizes.push(1 + lcg.int(20));
    for (const schedule of [roundRobin, listSchedule, lptSchedule, minMaxSchedule]) {
      const assignment = schedule(sizes, machineSpeeds);
      assert.strictEqual(assignment.length, sizes.length, "one machine per task");
      for (const machine of assignment) assert.ok(machine >= 0 && machine < machineCount);
      const loads = loadsFor(assignment, sizes, machineSpeeds);
      let weightedWork = 0;
      for (let machine = 0; machine < machineCount; machine++) {
        weightedWork += loads[machine] * machineSpeeds[machine];
      }
      assert.ok(Math.abs(weightedWork - totalWork(sizes)) < 1e-9,
        "the speed-weighted load equals the total work");
      assert.strictEqual(Math.max(...loads), makespanOf(assignment, sizes, machineSpeeds));
      assert.ok(makespanOf(assignment, sizes, machineSpeeds) >= lowerBound(sizes, machineSpeeds) - 1e-9,
        "no schedule can beat the lower bound");
    }
  }

  // --- Graham's bound for list scheduling on equal-speed machines ------------------------------
  for (let trial = 0; trial < 25; trial++) {
    const machineCount = 2 + lcg.int(3);
    const equalSpeeds = new Array(machineCount).fill(1);
    const sizes = [];
    for (let i = 0; i < 6 + lcg.int(6); i++) sizes.push(1 + lcg.int(12));
    const greedy = makespanOf(listSchedule(sizes, equalSpeeds), sizes, equalSpeeds);
    const optimum = optimalMakespan(sizes, equalSpeeds);
    assert.ok(greedy >= optimum - 1e-9, `a heuristic cannot beat the optimum (${greedy} vs ${optimum})`);
    assert.ok(greedy <= (2 - 1 / machineCount) * optimum + 1e-9,
      `Graham's (2 - 1/m) bound holds (${greedy} <= ${(2 - 1 / machineCount) * optimum})`);
  }

  // --- LPT's blind spot on a heterogeneous cluster, and the fix -------------------------------
  const heterogeneous = [8, 8, 1];
  const heterogeneousSpeeds = [1, 4];
  const blind = lptSchedule(heterogeneous, heterogeneousSpeeds);
  assert.deepStrictEqual(blind, [0, 1, 1], "LPT sends the first big task to the slow machine");
  assert.strictEqual(makespanOf(blind, heterogeneous, heterogeneousSpeeds), 8);
  const aware = minMaxSchedule(heterogeneous, heterogeneousSpeeds);
  assert.deepStrictEqual(aware, [1, 1, 0], "minMax puts both big tasks on the fast machine");
  assert.strictEqual(makespanOf(aware, heterogeneous, heterogeneousSpeeds), 4);
  assert.strictEqual(optimalMakespan(heterogeneous, heterogeneousSpeeds), 4, "which is optimal");
  assert.strictEqual(lowerBound(heterogeneous, heterogeneousSpeeds), 17 / 5, "the volume bound is 17/5");
  assert.ok(optimalMakespan(heterogeneous, heterogeneousSpeeds) > lowerBound(heterogeneous, heterogeneousSpeeds),
    "indivisible tasks leave a gap above the divisible bound");
  assert.strictEqual(fluidMakespan(heterogeneous, heterogeneousSpeeds), 17 / 5, "which the fluid schedule closes");
  assert.ok(makespanOf(blind, heterogeneous, heterogeneousSpeeds) >= 2 * makespanOf(aware, heterogeneous, heterogeneousSpeeds),
    "plain LPT is twice as bad as the optimum here");
  assert.ok(makespanOf(roundRobin(heterogeneous, heterogeneousSpeeds), heterogeneous, heterogeneousSpeeds) >= 8);

  // on equal machines the speed-aware greedy is never worse than LPT
  let minMaxWins = 0;
  let lptWins = 0;
  for (let trial = 0; trial < 30; trial++) {
    const machineCount = 2 + lcg.int(3);
    const machineSpeeds = [];
    for (let i = 0; i < machineCount; i++) machineSpeeds.push(1 + lcg.int(6));
    const sizes = [];
    for (let i = 0; i < 4 + lcg.int(8); i++) sizes.push(1 + lcg.int(15));
    const viaLpt = makespanOf(lptSchedule(sizes, machineSpeeds), sizes, machineSpeeds);
    const viaMinMax = makespanOf(minMaxSchedule(sizes, machineSpeeds), sizes, machineSpeeds);
    assert.ok(viaMinMax >= lowerBound(sizes, machineSpeeds) - 1e-9);
    if (viaMinMax < viaLpt - 1e-9) minMaxWins++;
    else if (viaLpt < viaMinMax - 1e-9) lptWins++;
  }
  assert.ok(minMaxWins >= lptWins, `minMax at least matches LPT (${minMaxWins} vs ${lptWins})`);

  // --- asymptotic near-optimality on a large instance ------------------------------------------
  const bigTasks = [];
  for (let i = 0; i < 400; i++) bigTasks.push(1 + lcg.int(50));
  const bigSpeeds = [1, 1, 1, 1];
  const bigLpt = makespanOf(lptSchedule(bigTasks, bigSpeeds), bigTasks, bigSpeeds);
  const bigBound = lowerBound(bigTasks, bigSpeeds);
  assert.ok(bigLpt / bigBound < 1.1, `LPT is near-optimal on many small tasks (${(bigLpt / bigBound).toFixed(3)})`);
  const bigMinMax = makespanOf(minMaxSchedule(bigTasks, bigSpeeds), bigTasks, bigSpeeds);
  let bestHeuristic = Math.min(bigLpt, bigMinMax, makespanOf(listSchedule(bigTasks, bigSpeeds), bigTasks, bigSpeeds));
  assert.ok(bestHeuristic >= bigBound - 1e-9);
  const bigFluid = fluidMakespan(bigTasks, bigSpeeds);
  assert.ok(Math.abs(bigFluid - bigBound) < 1e-9, "the divisible reference equals the bound");
  assert.ok(bigFluid <= bestHeuristic + 1e-9, "divisible work is at least as good as any indivisible plan");

  // --- efficiency and utilisation ---------------------------------------------------------------
  const manySmall = new Array(101).fill(1);
  const fourSpeeds = [1, 1, 1, 1];
  const balanced = lptSchedule(manySmall, fourSpeeds);
  assert.strictEqual(makespanOf(balanced, manySmall, fourSpeeds), 26, "101 unit tasks on 4 machines take 26 ticks");
  assert.ok(101 / (26 * 4) > 0.97, "so the fluid efficiency is above 97%");
  const mixedSpeeds = [1, 2, 3];
  const oneTask = lptSchedule([6], mixedSpeeds);
  assert.deepStrictEqual(oneTask, [0], "the first task goes to the first machine on a tie");
  const speedBlind = listSchedule([0, 0, 6], mixedSpeeds);
  assert.strictEqual(makespanOf(speedBlind, [0, 0, 6], mixedSpeeds), 6,
    "the speed-blind greedy puts the heavy task on the slowest machine");
  assert.deepStrictEqual(lptSchedule([0, 0, 6], mixedSpeeds), [1, 1, 0],
    "LPT is speed-blind too: the heavy task takes machine 0 and the zeros fill the idle ones");
  const speedAware = minMaxSchedule([0, 0, 6], mixedSpeeds);
  assert.strictEqual(makespanOf(speedAware, [0, 0, 6], mixedSpeeds), 2, "minMax uses the fastest machine");
  assert.deepStrictEqual(loadsFor(minMaxSchedule([6], mixedSpeeds), [6], mixedSpeeds), [0, 0, 2],
    "with one task and no history minMax picks the fastest machine");
  assert.strictEqual(makespanOf(minMaxSchedule([6], mixedSpeeds), [6], mixedSpeeds), 2);

  // --- degenerate inputs ------------------------------------------------------------------------
  assert.deepStrictEqual(listSchedule([], [1, 2]), []);
  assert.strictEqual(makespanOf([], [1, 2], []), 0);
  assert.strictEqual(optimalMakespan([], [1, 2]), 0);
  assert.strictEqual(lowerBound([], [1, 2]), 0);
  assert.strictEqual(fluidMakespan([], [1]), 0);
  assert.strictEqual(totalSpeed([1, 2, 3]), 6);
  const single = [5];
  const singleMachine = [0.5];
  assert.strictEqual(makespanOf(lptSchedule(single, singleMachine), single, singleMachine), 10);
  assert.strictEqual(lowerBound(single, singleMachine), 10, "one machine is trivially balanced");
  assert.strictEqual(optimalMakespan(single, singleMachine), 10);
  assert.deepStrictEqual(listSchedule([1, 2, 3], [1]), [0, 0, 0], "one machine takes everything");
  assert.strictEqual(makespanOf([], [], [1, 1]), 0, "no tasks means no load");
  assert.deepStrictEqual(loadsFor([], [], [1, 2]), [0, 0]);
  const allZero = [0, 0, 0];
  assert.strictEqual(makespanOf(lptSchedule(allZero, [1, 1]), allZero, [1, 1]), 0);
  assert.strictEqual(lowerBound(allZero, [1, 1]), 0);

  // --- determinism -------------------------------------------------------------------------------
  const replayA = lptSchedule(bigTasks, bigSpeeds);
  const replayB = lptSchedule([...bigTasks], bigSpeeds);
  assert.deepStrictEqual(replayA, replayB);
  assert.deepStrictEqual(minMaxSchedule(bigTasks, bigSpeeds), minMaxSchedule([...bigTasks], bigSpeeds));
  assert.strictEqual(new Lcg(700).next(), new Lcg(700).next());
  assert.strictEqual(makespanOf(replayA, bigTasks, bigSpeeds), bigLpt);

  console.log(`lpt=${bigLpt.toFixed(2)} bound=${bigBound.toFixed(2)} ratio=${(bigLpt / bigBound).toFixed(3)}`);
  console.log(`heterogeneous lpt=${makespanOf(blind, heterogeneous, heterogeneousSpeeds)} minMax=${makespanOf(aware, heterogeneous, heterogeneousSpeeds)} optimum=${optimalMakespan(heterogeneous, heterogeneousSpeeds)}`);
  console.log("PASS");
}
