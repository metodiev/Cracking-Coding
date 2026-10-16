/**
 * Task 678 - Implement multiprocessor memory model litmus tests (TSO).
 * (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * An exhaustive litmus-test explorer over two memory models, sharing one state-space search:
 *
 *   TSO  - every thread has a FIFO store buffer. A store is buffered (visible to others only when it is
 *          propagated), a load is forwarded from the thread's own buffer when it matches, otherwise it
 *          reads memory, and `mfence` cannot retire until the buffer is empty. Propagation of the buffer
 *          head is itself a step of the search, so out-of-order execution of stores versus loads emerges
 *          from the interleaving rather than being asserted by hand.
 *   SC   - the same search with the store buffers switched off: stores take effect immediately, which is
 *          exactly a global order over all memory operations.
 *
 * A second flag (`fifo: false`) lets any buffered store propagate, modelling a relaxed model without
 * store order, which is how the tests show *which* guarantee a FIFO buffer buys.
 *
 * Verified: the classic outcomes are reproduced and hand-explained - store buffering (SB) reaches
 * (r0=0, r1=0) under TSO but not under SC, and an mfence in each thread removes it; message passing (MP)
 * never yields (r0=1, r1=0) under TSO because stores drain in order, while the relaxed flag makes that
 * outcome appear, which is exactly the guarantee a FIFO buffer buys; load buffering (LB) cannot produce
 * (1,1) even under TSO, because load -> store order is preserved and the outcome would need a cyclic
 * visibility order, so TSO and SC agree there (a weaker model than TSO would be needed). In every test
 * the TSO outcome set is a superset of the SC one, the exploration is exhausted and deterministic, and
 * the visited-state counts are reported.
 */
"use strict";

const assert = require("node:assert");

const write = (loc, value) => ({ op: "write", loc, value });
const read = (loc, reg) => ({ op: "read", loc, reg });
const fence = () => ({ op: "fence" });
const litmus = (name, threads, watch) => ({ name, threads, watch });

/** Store buffering: each thread stores one location and then loads the other. */
const storeBuffering = () => litmus("SB", [
  [write("x", 1), read("y", "r0")],
  [write("y", 1), read("x", "r1")],
], [[0, "r0"], [1, "r1"]]);

/** Store buffering with an mfence between the store and the load of each thread. */
const storeBufferingFenced = () => litmus("SB+fence", [
  [write("x", 1), fence(), read("y", "r0")],
  [write("y", 1), fence(), read("x", "r1")],
], [[0, "r0"], [1, "r1"]]);

/** Message passing: one producer stores two locations, the consumer loads them in the same order. */
const messagePassing = () => litmus("MP", [
  [write("x", 1), write("y", 1)],
  [read("y", "r0"), read("x", "r1")],
], [[1, "r0"], [1, "r1"]]);

/** Load buffering: each thread loads one location and then stores the other. */
const loadBuffering = () => litmus("LB", [
  [read("x", "r0"), write("y", 1)],
  [read("y", "r1"), write("x", 1)],
], [[0, "r0"], [1, "r1"]]);

const clone = (state, threads) => ({
  pcs: [...state.pcs],
  buffers: state.buffers.map((buffer) => [...buffer]),
  regs: state.regs.map((regs) => new Map(regs)),
  memory: new Map(state.memory),
  threads,
});

const stateKey = (state) => JSON.stringify({
  pcs: state.pcs,
  buffers: state.buffers,
  regs: state.regs.map((regs) => [...regs].sort()),
  memory: [...state.memory].sort(),
});

const outcomeOf = (program, state) =>
  `(${program.watch.map(([thread, reg]) => state.regs[thread].get(reg) ?? 0).join(",")})`;

/**
 * Exhaustive search over every interleaving of instruction steps and store propagations. `storeBuffers`
 * false gives sequential consistency; `fifo` false lets any buffered store drain, relaxing store order.
 */
function explore(program, options = {}) {
  const { storeBuffers = true, fifo = true, maxStates = 40000 } = options;
  const threads = program.threads.length;
  const start = {
    pcs: program.threads.map(() => 0),
    buffers: program.threads.map(() => []),
    regs: program.threads.map(() => new Map()),
    memory: new Map(),
    threads,
  };
  const seen = new Set([stateKey(start)]);
  const stack = [start];
  const outcomes = new Set();
  let visited = 0;
  let terminal = 0;
  while (stack.length > 0 && visited < maxStates) {
    const state = stack.pop();
    visited += 1;
    const successors = [];
    for (let t = 0; t < threads; t++) {
      const code = program.threads[t];
      if (state.pcs[t] >= code.length) continue;
      const op = code[state.pcs[t]];
      const next = clone(state, threads);
      if (op.op === "fence") {
        if (storeBuffers && next.buffers[t].length > 0) continue; // the fence waits for the buffer to drain
        next.pcs[t] += 1;
      } else if (op.op === "write") {
        if (storeBuffers) next.buffers[t] = [...next.buffers[t], { loc: op.loc, value: op.value }];
        else next.memory = new Map([...next.memory, [op.loc, op.value]]);
        next.pcs[t] += 1;
      } else {
        const buffered = [...next.buffers[t]].reverse().find((entry) => entry.loc === op.loc);
        const value = buffered ? buffered.value : next.memory.get(op.loc) ?? 0;
        next.regs[t] = new Map([...next.regs[t], [op.reg, value]]);
        next.pcs[t] += 1;
      }
      successors.push(next);
    }
    if (storeBuffers) {
      for (let t = 0; t < threads; t++) {
        if (state.buffers[t].length === 0) continue;
        if (fifo) {
          const next = clone(state, threads);
          const [head, ...rest] = next.buffers[t];
          next.buffers[t] = rest;
          next.memory = new Map([...next.memory, [head.loc, head.value]]);
          successors.push(next);
        } else {
          for (let index = 0; index < state.buffers[t].length; index++) {
            const next = clone(state, threads);
            const entry = next.buffers[t][index];
            next.buffers[t] = next.buffers[t].filter((_, i) => i !== index);
            next.memory = new Map([...next.memory, [entry.loc, entry.value]]);
            successors.push(next);
          }
        }
      }
    }
    if (successors.length === 0) {
      terminal += 1;
      outcomes.add(outcomeOf(program, state));
      continue;
    }
    for (const successor of successors) {
      const key = stateKey(successor);
      if (seen.has(key)) continue;
      seen.add(key);
      stack.push(successor);
    }
  }
  return { outcomes: [...outcomes].sort(), visited, terminal, exhausted: stack.length === 0 };
}

const has = (result, outcome) => result.outcomes.includes(outcome);

function main() {
  // (1) Store buffering: TSO allows (0,0), sequential consistency forbids it.
  const sbTso = explore(storeBuffering());
  const sbSc = explore(storeBuffering(), { storeBuffers: false });
  assert.strictEqual(has(sbTso, "(0,0)"), true, "documented: TSO lets both loads see 0 (store buffering)");
  assert.strictEqual(has(sbSc, "(0,0)"), false, "documented: SC cannot, because one store must come first");
  assert.deepStrictEqual(sbSc.outcomes, ["(0,1)", "(1,0)", "(1,1)"], "SC reaches the other three outcomes");
  assert.deepStrictEqual(sbTso.outcomes, ["(0,0)", "(0,1)", "(1,0)", "(1,1)"], "TSO reaches all four");
  assert.ok(sbTso.visited > 20 && sbSc.visited > 5,
    `the search visited ${sbTso.visited} TSO states and ${sbSc.visited} SC states`);

  // (2) A fence in each thread drains the buffer and removes the relaxed outcome.
  const sbFenced = explore(storeBufferingFenced());
  assert.strictEqual(has(sbFenced, "(0,0)"), false, "documented: with mfences the buffers are drained");
  assert.strictEqual(has(sbFenced, "(0,1)"), true, "while the sequential outcomes remain reachable");
  assert.deepStrictEqual(sbFenced.outcomes, sbSc.outcomes, "so a fenced SB behaves like SC here");

  // (3) Message passing: FIFO store buffers preserve store order, so (1,0) is impossible.
  const mpTso = explore(messagePassing());
  assert.strictEqual(has(mpTso, "(1,0)"), false,
    "documented: seeing y := 1 before x := 1 is impossible when stores drain in order");
  assert.deepStrictEqual(mpTso.outcomes, ["(0,0)", "(0,1)", "(1,1)"], "the other three outcomes are reachable");
  const mpRelaxed = explore(messagePassing(), { fifo: false });
  assert.strictEqual(has(mpRelaxed, "(1,0)"), true,
    "and the relaxed flag (any store may drain first) makes exactly that outcome appear");
  assert.deepStrictEqual(mpRelaxed.outcomes, ["(0,0)", "(0,1)", "(1,0)", "(1,1)"], "the relaxed model reaches all four");
  assert.deepStrictEqual(explore(messagePassing(), { storeBuffers: false }).outcomes, mpTso.outcomes,
    "SC agrees with TSO on this test");

  // (4) Load buffering: (1,1) would need each store visible before a load that precedes it, which even
  // TSO forbids - load -> store order is preserved, so TSO and SC agree here (a weaker model is needed).
  const lbTso = explore(loadBuffering());
  const lbSc = explore(loadBuffering(), { storeBuffers: false });
  assert.strictEqual(has(lbTso, "(1,1)"), false, "documented: (1,1) needs a cyclic visibility order");
  assert.deepStrictEqual(lbTso.outcomes, lbSc.outcomes, "so TSO and SC agree on LB");
  assert.deepStrictEqual(lbSc.outcomes, ["(0,0)", "(0,1)", "(1,0)"], "SC reaches the three ordered outcomes");
  assert.deepStrictEqual(explore(loadBuffering(), { fifo: false }).outcomes, lbSc.outcomes,
    "and relaxing store order does not help: the constraint is the load/store order, not the buffer order");

  // (5) The invariant across every litmus test: TSO is never more restrictive than SC.
  const tests = [storeBuffering(), storeBufferingFenced(), messagePassing(), loadBuffering()];
  let compared = 0;
  let tsoOutcomes = 0;
  for (const program of tests) {
    const tso = explore(program);
    const sc = explore(program, { storeBuffers: false });
    assert.ok(sc.outcomes.every((outcome) => tso.outcomes.includes(outcome)),
      `${program.name}: every SC outcome must also be allowed under TSO`);
    assert.ok(tso.outcomes.length >= sc.outcomes.length, `${program.name}: TSO allows at least as much`);
    assert.ok(tso.exhausted && sc.exhausted, `${program.name}: the search space was exhausted`);
    assert.ok(tso.terminal > 0 && sc.terminal > 0, `${program.name}: terminal states were recorded`);
    assert.ok(tso.visited >= sc.visited, `${program.name}: the buffered model has the larger state space`);
    compared += sc.outcomes.length;
    tsoOutcomes += tso.outcomes.length;
  }
  assert.deepStrictEqual([compared, tsoOutcomes], [12, 13], "the outcome sets have the documented sizes");

  // (6) Sanity of the encoding: the watch list decides the printed tuple, and reads default to 0.
  const trivial = explore(litmus("trivial", [[read("q", "r0")], []], [[0, "r0"], [0, "missing"]]));
  assert.deepStrictEqual(trivial.outcomes, ["(0,0)"], "an unread location and a missing register read as 0");
  assert.strictEqual(trivial.visited, 2, "one state for the load and one terminal state");
  const noThreads = explore(litmus("empty", [[], []], []));
  assert.deepStrictEqual([noThreads.outcomes, noThreads.visited], [["()"], 1], "a program with no operations has one outcome");

  // (7) Determinism: the same test explores the same state space and reports the same outcomes.
  for (const program of tests) {
    const first = explore(program);
    const second = explore(program);
    assert.deepStrictEqual([first.outcomes, first.visited, first.terminal], [second.outcomes, second.visited, second.terminal],
      `${program.name}: the exploration is deterministic`);
  }
  const relaxedA = explore(loadBuffering(), { fifo: false });
  const relaxedB = explore(loadBuffering(), { fifo: false });
  assert.deepStrictEqual([relaxedA.outcomes, relaxedA.visited], [relaxedB.outcomes, relaxedB.visited],
    "including the relaxed propagation mode");
}

module.exports = { write, read, fence, litmus, storeBuffering, storeBufferingFenced, messagePassing,
  loadBuffering, explore, stateKey, outcomeOf, has };

if (require.main === module) {
  main();
  console.log("PASS");
}
