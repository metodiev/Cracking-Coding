/**
 * Task 349 - Reusable barrier synchronization primitive. (Tier: Intermediate, CTCI Ch 15)
 *
 * Generation-based barrier: the last arriving participant resets the counter, flips the generation
 * and releases everyone, so the object is instantly reusable for the next round.  A barrier with
 * one participant is a no-op.
 *
 * Node runs a single-threaded event loop, so this is the documented async simulation (the task 195
 * pattern): suspended participants are stored as promise resolvers.  Four interleaved async tasks
 * run five rounds, and a global event log proves that nobody departs a round before every arrival
 * of that round happened - the assertions are exact, not timing-based.
 */
"use strict";

class Barrier {
  constructor(parties) {
    if (parties <= 0) throw new Error("parties must be positive");
    this.parties = parties;
    this.count = 0;
    this.generation = 0;
    this.waiters = [];
  }

  wait() {
    this.count++;
    if (this.count === this.parties) {
      this.count = 0;
      this.generation++;
      const waiters = this.waiters;
      this.waiters = [];
      for (const resolve of waiters) resolve();
      return Promise.resolve();
    }
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  state() {
    return [this.count, this.generation];
  }
}

module.exports = { Barrier };

if (require.main === module) {
  const main = async () => {
    const assert = require("node:assert");

    const rounds = 5;
    const barrier = new Barrier(4);
    const events = [];
    const record = (kind, round, name) => events.push({ seq: events.length, kind, round, name });

    const participant = async (name) => {
      for (let round = 0; round < rounds; round++) {
        record("arrive", round, name);
        await barrier.wait();
        record("depart", round, name);
      }
    };
    const names = ["P0", "P1", "P2", "P3"];
    await Promise.all(names.map((name) => participant(name)));

    assert.strictEqual(events.filter((e) => e.kind === "arrive").length, 4 * rounds, "20 arrivals");
    assert.strictEqual(events.filter((e) => e.kind === "depart").length, 4 * rounds, "20 departures");

    for (let round = 0; round < rounds; round++) {
      const arrivals = events.filter((e) => e.round === round && e.kind === "arrive").map((e) => e.seq);
      const departures = events.filter((e) => e.round === round && e.kind === "depart").map((e) => e.seq);
      assert.strictEqual(arrivals.length, 4, `round ${round} arrivals`);
      assert.strictEqual(departures.length, 4, `round ${round} departures`);
      assert.ok(Math.max(...arrivals) < Math.min(...departures), `round ${round} released early`);
    }

    for (const name of names) {
      const arrivals = events.filter((e) => e.name === name && e.kind === "arrive").map((e) => e.round);
      const departures = events.filter((e) => e.name === name && e.kind === "depart").map((e) => e.round);
      assert.deepStrictEqual(arrivals, Array.from({ length: rounds }, (_, r) => r), `${name} arrival rounds`);
      assert.deepStrictEqual(departures, Array.from({ length: rounds }, (_, r) => r), `${name} departure rounds`);
    }

    assert.deepStrictEqual(barrier.state(), [0, rounds], "barrier reset after every round");

    // single-participant barrier is a no-op and reusable
    const solo = new Barrier(1);
    for (let i = 0; i < 3; i++) await solo.wait();
    assert.deepStrictEqual(solo.state(), [0, 3], "solo generation advanced once per wait");

    // a second barrier confirms the release is simultaneous per round
    const pair = new Barrier(2);
    const order = [];
    await Promise.all([
      (async () => { order.push("A-before"); await pair.wait(); order.push("A-after"); })(),
      (async () => { order.push("B-before"); await pair.wait(); order.push("B-after"); })(),
    ]);
    assert.deepStrictEqual(order.slice(0, 2).sort(), ["A-before", "B-before"], "both arrived first");
    assert.deepStrictEqual(pair.state(), [0, 1]);
  };

  main().then(() => console.log("PASS"), (err) => {
    console.error(err);
    process.exit(1);
  });
}
