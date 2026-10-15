/**
 * Task 351 - Implement thread pool basic with worker threads and task queue. (Tier: Intermediate, CTCI Ch 15)
 *
 * JavaScript runs single-threaded here (Node's event loop; real worker_threads would need a
 * separate module per worker plus message passing). Following the repo's task-195 pattern,
 * this is a deterministic async simulation: NUM_WORKERS virtual workers repeatedly await
 * TaskQueue.take(), whose promise waiters stand in for a blocking condition variable. The
 * event loop serialises every critical section, so each assertion is exact, and shutdown is
 * a documented sentinel: one queue close ends every worker after it drains the backlog.
 */
"use strict";

const NUM_WORKERS = 4;
const NUM_TASKS = 200;

function taskFunction(taskId) {
  const seed = (taskId * 75 + 74) % 65537;
  return taskId * taskId + seed;
}

class TaskQueue {
  constructor() {
    this.items = [];
    this.waiters = [];
    this.closed = false;
  }

  enqueue(item) {
    this.items.push(item);
    const resolve = this.waiters.shift();
    if (resolve) resolve();
  }

  async take() {
    while (this.items.length === 0 && !this.closed) await this._wait();
    if (this.items.length === 0) return null;        // graceful shutdown sentinel
    return this.items.shift();
  }

  close() {
    this.closed = true;
    const pending = this.waiters;
    this.waiters = [];
    for (const resolve of pending) resolve();
  }

  _wait() {
    return new Promise((resolve) => this.waiters.push(resolve));
  }
}

class ThreadPool {
  constructor(numWorkers, taskFunction) {
    if (numWorkers <= 0) throw new Error("numWorkers must be positive");
    this.taskFunction = taskFunction;
    this.queue = new TaskQueue();
    this.runCounts = new Map();
    this.results = new Map();
    this.workerTaskCounts = new Array(numWorkers).fill(0);
    this.workers = [];
    for (let workerId = 0; workerId < numWorkers; workerId++) {
      this.workers.push((async () => {
        for (;;) {
          const taskId = await this.queue.take();
          if (taskId === null) return;               // sentinel reached: worker exits
          const value = this.taskFunction(taskId);
          this.runCounts.set(taskId, (this.runCounts.get(taskId) || 0) + 1);
          this.results.set(taskId, value);
          this.workerTaskCounts[workerId] += 1;
        }
      })());
    }
  }

  submit(taskId) {
    this.queue.enqueue(taskId);
  }

  async shutdownAndWait() {
    this.queue.close();
    await Promise.all(this.workers);
    return this.queue.items.length === 0;
  }
}

async function main() {
  const assert = require("node:assert");

  const pool = new ThreadPool(NUM_WORKERS, taskFunction);
  for (let taskId = 0; taskId < NUM_TASKS; taskId++) pool.submit(taskId);
  assert.strictEqual(await pool.shutdownAndWait(), true, "queue fully drained");

  assert.strictEqual(pool.results.size, NUM_TASKS, "one result per task");
  let totalRuns = 0;
  for (const count of pool.runCounts.values()) totalRuns += count;
  assert.strictEqual(totalRuns, NUM_TASKS, "total completions exact");
  for (let taskId = 0; taskId < NUM_TASKS; taskId++) {
    assert.strictEqual(pool.runCounts.get(taskId), 1, `task ${taskId} ran exactly once`);
    assert.strictEqual(pool.results.get(taskId), taskFunction(taskId), `task ${taskId} result`);
  }
  for (const count of pool.workerTaskCounts) {
    assert.ok(count >= 1, "every worker consumed at least one task");
  }

  const pool2 = new ThreadPool(2, taskFunction);
  pool2.submit(5);
  pool2.submit(5);
  pool2.submit(7);
  assert.strictEqual(await pool2.shutdownAndWait(), true, "second queue drained");
  assert.strictEqual(pool2.runCounts.get(5), 2, "duplicate tasks are both counted");
  assert.strictEqual(pool2.runCounts.get(7), 1, "single submission counted once");
  assert.strictEqual(pool2.results.get(5), taskFunction(5), "duplicate results");
  assert.strictEqual(pool2.results.get(7), taskFunction(7), "single result");
}

module.exports = { TaskQueue, ThreadPool };

if (require.main === module) {
  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
