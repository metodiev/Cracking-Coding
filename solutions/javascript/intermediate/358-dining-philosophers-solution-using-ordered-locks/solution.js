/**
 * Task 358 - Dining philosophers solution using ordered locks. (Tier: Intermediate, CTCI Ch 15)
 *
 * JavaScript runs single-threaded, so this is a deterministic async simulation in the
 * documented task-195 style. A Fork is an async mutex whose promise waiters form a FIFO
 * handoff queue: on release the longest-waiting philosopher is woken with ownership
 * already transferred, which rules out the unfair reacquisition starvation a free-for-all
 * scheduler could cause. Each philosopher awaits a yield while "eating" between courses,
 * so the five loops genuinely interleave and a philosopher holding one fork blocks in the
 * queue for the second. The ordered-lock solution (each philosopher locks the
 * lower-numbered of its two forks first, so only philosopher 4 takes fork 0 before fork 4)
 * is proven again here: every philosopher eats exactly MEALS times. The naive grab order
 * is also demonstrated with a scripted interleaving that reaches the classic circular-wait
 * state where all five philosophers are blocked - exactly what the ordering prevents.
 */
"use strict";

const PHILOSOPHERS = 5;
const MEALS = 40;
const STEP_CAP = 5 * 40 * 1000;                   // termination guard for the simulation

class Fork {
  constructor(index) {
    this.index = index;
    this.owner = null;
    this.waiters = [];                             // FIFO of { philosopher, resolve }
    this.acquisitions = 0;
  }

  /** Fast path when free and uncontended; otherwise suspend in FIFO order. */
  acquire(philosopher) {
    if (this.owner === null && this.waiters.length === 0) {
      this.owner = philosopher;
      this.acquisitions += 1;
      return null;                                 // acquired synchronously
    }
    return new Promise((resolve) => this.waiters.push({ philosopher, resolve }));
  }

  release() {
    if (this.waiters.length > 0) {
      const next = this.waiters.shift();           // FIFO handoff: no starvation
      this.owner = next.philosopher;
      this.acquisitions += 1;
      next.resolve();
    } else {
      this.owner = null;
    }
  }
}

class Table {
  constructor() {
    this.forks = [];
    for (let i = 0; i < PHILOSOPHERS; i++) this.forks.push(new Fork(i));
    this.holders = new Array(PHILOSOPHERS).fill(0);
    this.maxHolders = new Array(PHILOSOPHERS).fill(0);
    this.meals = new Array(PHILOSOPHERS).fill(0);
    this.violations = 0;
    this.waitEvents = 0;
    this.steps = 0;
  }

  recordAcquire(philosopher, forkIndex) {
    const fork = this.forks[forkIndex];
    if (fork.owner !== philosopher) this.violations += 1;   // would mean the fork was shared
    this.holders[forkIndex] += 1;
    this.maxHolders[forkIndex] = Math.max(this.maxHolders[forkIndex], this.holders[forkIndex]);
  }

  recordRelease(philosopher, forkIndex) {
    const fork = this.forks[forkIndex];
    if (fork.owner !== philosopher) this.violations += 1;
    this.holders[forkIndex] -= 1;
  }
}

/** Documented lock ordering: the lower-numbered of the two forks always comes first. */
function forkOrder(philosopher) {
  const left = philosopher;
  const right = (philosopher + 1) % PHILOSOPHERS;
  return [Math.min(left, right), Math.max(left, right)];
}

async function philosopherLoop(table, philosopher) {
  const [first, second] = forkOrder(philosopher);
  for (let meal = 0; meal < MEALS; meal++) {
    table.steps += 1;
    if (table.steps > STEP_CAP) throw new Error("philosophers did not finish (deadlock?)");

    const waitFirst = table.forks[first].acquire(philosopher);
    if (waitFirst !== null) {
      table.waitEvents += 1;
      await waitFirst;
    }
    table.recordAcquire(philosopher, first);

    const waitSecond = table.forks[second].acquire(philosopher);
    if (waitSecond !== null) {
      table.waitEvents += 1;
      await waitSecond;
    }
    table.recordAcquire(philosopher, second);

    table.meals[philosopher] += 1;
    await Promise.resolve();                        // "eating takes time": let others run

    table.recordRelease(philosopher, second);
    table.forks[second].release();
    table.recordRelease(philosopher, first);
    table.forks[first].release();
  }
}

/** Scripted naive interleaving: everyone grabs fork i, then reaches for fork (i+1)%5. */
function naiveDeadlockBlockedSet() {
  const owner = new Array(PHILOSOPHERS).fill(null);
  for (let i = 0; i < PHILOSOPHERS; i++) owner[i] = i;   // first grabs all succeed
  const blocked = [];
  for (let i = 0; i < PHILOSOPHERS; i++) {
    const wanted = (i + 1) % PHILOSOPHERS;
    if (owner[wanted] !== null) blocked.push(i);         // circular wait: no progress possible
  }
  return blocked;
}

async function main() {
  const assert = require("node:assert");

  // Hand-checked ordering table: only philosopher 4 wraps around, taking fork 0 first.
  assert.deepStrictEqual([0, 1, 2, 3, 4].map(forkOrder),
    [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]], "documented increasing fork order");

  const table = new Table();
  const diners = [];
  for (let philosopher = 0; philosopher < PHILOSOPHERS; philosopher++) {
    diners.push(philosopherLoop(table, philosopher));
  }
  await Promise.all(diners);

  assert.strictEqual(table.meals.reduce((a, b) => a + b, 0), PHILOSOPHERS * MEALS,
    "total meals == 5 * MEALS");
  assert.ok(table.meals.every((count) => count === MEALS), "every philosopher ate MEALS times");
  assert.strictEqual(table.violations, 0, "per-fork single-holder invariant never violated");
  assert.ok(table.forks.every((fork) => fork.owner === null), "every fork released at the end");
  assert.ok(table.holders.every((count) => count === 0), "no residual holders");
  assert.ok(table.maxHolders.every((count) => count === 1), "one holder at a time per fork");
  assert.ok(table.forks.every((fork) => fork.acquisitions === 2 * MEALS),
    "each fork acquired by exactly its two neighbours");
  assert.ok(table.waitEvents >= 1, "the simulation really interleaved (someone queued)");

  // The naive order under the same forced interleaving deadlocks; ordering prevents it.
  assert.deepStrictEqual(naiveDeadlockBlockedSet(), [0, 1, 2, 3, 4],
    "naive order: all five philosophers block forever (circular wait)");
}

module.exports = { Fork, Table, forkOrder, naiveDeadlockBlockedSet };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
