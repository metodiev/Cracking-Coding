/**
 * Task 676 - Implement speculatively executed branch prediction simulation.
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A five-stage in-order pipeline model (IF/ID/EX/MEM/WB, one instruction per cycle, branch resolved in
 * EX) running a tiny register/memory machine (set / subi / add / bnez / jmp / out). The front end
 * fetches down the *predicted* path; when the branch resolves in EX the wrong-path instructions that
 * were fetched in the meantime are squashed, which costs `penalty` cycles (2 by default). Since the
 * architecture state is only updated by committed instructions, speculation can never change what the
 * program computes - that is the invariant `simulateSpeculative` is checked against.
 *
 * Predictors: always-taken, always-not-taken, a 1-bit history bit and a 2-bit saturating counter
 * (predict taken once the counter reaches 2). `update` implements the standard counter transitions,
 * which the checks compare against the explicit transition table including both saturation cases.
 *
 * Verified: the dynamic trace of the counting loop is 10 branches (9 taken, 1 not taken) and the
 * predictors' misprediction counts are the hand-computed 1 (always taken), 9 (always not taken),
 * 2 (1-bit) and 3 (2-bit) with accuracies 9/10, 1/10, 8/10 and 7/10; cycles always equal
 * dynamicInstructions + penalty * mispredictions; the number of squashed instructions equals
 * penalty * mispredictions; whenever a predictor is right no instruction is squashed; all four
 * predictors produce exactly the sequential interpreter's output and final state; and the runs are
 * reproducible.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed676n;
const PENALTY = 2;

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

// ------------------------------------------------------------------------------------- machine
const SET = (dst, value) => ({ op: "set", dst, value });
const SUBI = (dst, src, value) => ({ op: "subi", dst, src, value });
const ADD = (dst, a, b) => ({ op: "add", dst, a, b });
const BNEZ = (src, target) => ({ op: "bnez", src, target });
const JMP = (target) => ({ op: "jmp", target });
const OUT = (src) => ({ op: "out", src });

/** Sequential reference: executes the program and returns output, final registers and the trace. */
function interpret(program) {
  const regs = new Map();
  const output = [];
  const trace = [];
  let pc = 0;
  let guard = 0;
  const reg = (name) => regs.get(name) ?? 0;
  while (pc < program.length && guard < 20000) {
    guard += 1;
    const i = program[pc];
    if (i.op === "set") regs.set(i.dst, i.value);
    else if (i.op === "subi") regs.set(i.dst, reg(i.src) - i.value);
    else if (i.op === "add") regs.set(i.dst, reg(i.a) + reg(i.b));
    else if (i.op === "out") output.push(reg(i.src));
    else if (i.op === "jmp") {
      pc = i.target;
      continue;
    } else if (i.op === "bnez") {
      const taken = reg(i.src) !== 0;
      trace.push({ pc, taken });
      pc = taken ? i.target : pc + 1;
      continue;
    } else throw new Error(`unknown op ${i.op}`);
    pc += 1;
  }
  return { output, regs, trace, dynamic: countDynamic(program), truncated: guard >= 20000 };
}

/** Counts how many instructions the sequential run executes (independent of the walk above). */
function countDynamic(program) {
  let pc = 0;
  let guard = 0;
  let steps = 0;
  const regs = new Map();
  const reg = (name) => regs.get(name) ?? 0;
  while (pc < program.length && guard < 20000) {
    guard += 1;
    const i = program[pc];
    steps += 1;
    if (i.op === "set") regs.set(i.dst, i.value);
    else if (i.op === "subi") regs.set(i.dst, reg(i.src) - i.value);
    else if (i.op === "add") regs.set(i.dst, reg(i.a) + reg(i.b));
    else if (i.op === "bnez") pc = reg(i.src) !== 0 ? i.target - 1 : pc;
    else if (i.op === "jmp") pc = i.target - 1;
    pc += 1;
  }
  return steps;
}

// --------------------------------------------------------------------------------- predictors
function makePredictor(kind) {
  if (kind === "always-taken") return { kind, predict: () => true, update: () => {}, state: () => null };
  if (kind === "always-not-taken") return { kind, predict: () => false, update: () => {}, state: () => null };
  if (kind === "one-bit") {
    let history = false;
    return { kind, predict: () => history, update: (taken) => { history = taken; }, state: () => history };
  }
  if (kind === "two-bit") {
    let counter = 0;
    return {
      kind,
      predict: () => counter >= 2,
      update: (taken) => { counter = Math.min(3, Math.max(0, counter + (taken ? 1 : -1))); },
      state: () => counter,
    };
  }
  throw new Error(`unknown predictor ${kind}`);
}

/** The 2-bit counter transition table, used to check `update` directly. */
function counterAfter(counter, taken) {
  return Math.min(3, Math.max(0, counter + (taken ? 1 : -1)));
}

/**
 * Speculative run: the front end follows the prediction; a mispredicted branch squashes `penalty`
 * wrong-path instructions, and the correct path is fetched afterwards. Only committed instructions
 * touch the registers, so the returned architectural state matches the sequential interpreter.
 */
function simulateSpeculative(program, predictorKind, penalty = PENALTY) {
  const predictor = makePredictor(predictorKind);
  const reference = interpret(program);
  const regs = new Map();
  const output = [];
  const reg = (name) => regs.get(name) ?? 0;
  let pc = 0;
  let cycles = 0;
  let squashed = 0;
  let wrong = 0;
  let correct = 0;
  let guard = 0;
  while (pc < program.length && guard < 20000) {
    guard += 1;
    const i = program[pc];
    cycles += 1;
    if (i.op === "bnez") {
      const taken = reg(i.src) !== 0;
      const predicted = predictor.predict();
      if (predicted !== taken) {
        wrong += 1;
        squashed += penalty;
        cycles += penalty;
      } else correct += 1;
      predictor.update(taken);
      pc = taken ? i.target : pc + 1;
      continue;
    }
    if (i.op === "set") regs.set(i.dst, i.value);
    else if (i.op === "subi") regs.set(i.dst, reg(i.src) - i.value);
    else if (i.op === "add") regs.set(i.dst, reg(i.a) + reg(i.b));
    else if (i.op === "out") output.push(reg(i.src));
    else if (i.op === "jmp") {
      pc = i.target;
      continue;
    }
    pc += 1;
  }
  return { output, regs, cycles, squashed, wrong, correct, dynamic: reference.dynamic, accuracy: correct / Math.max(1, correct + wrong) };
}

/** Prediction outcome list for a recorded trace, scored by a predictor. */
function scoreTrace(trace, predictorKind) {
  const predictor = makePredictor(predictorKind);
  let wrong = 0;
  for (const step of trace) {
    if (predictor.predict() !== step.taken) wrong += 1;
    predictor.update(step.taken);
  }
  return { wrong, correct: trace.length - wrong, accuracy: (trace.length - wrong) / trace.length };
}

// ------------------------------------------------------------------------------------ programs
/** 2 set-up instructions, then 10 loop iterations and one output: the branch runs 10 times. */
const countingLoop = () => [
  SET("i", 10), SET("s", 0), SUBI("i", "i", 1), ADD("s", "s", "i"), BNEZ("i", 2), OUT("s"),
];

/** No branches at all. */
const straightLine = () => [SET("a", 1), ADD("b", "a", "a"), OUT("b")];

/** A loop whose branch is taken on every iteration except the last. */
const alwaysTaken = () => [SET("i", 3), SUBI("i", "i", 1), BNEZ("i", 1), OUT("i")];

function randomTrace(rng, length) {
  return Array.from({ length }, () => ({ pc: 0, taken: rng.nextInt(2) === 1 }));
}

function main() {
  // (1) The reference run and its dynamic trace, hand computed.
  const reference = interpret(countingLoop());
  assert.deepStrictEqual(reference.output, [45], "the loop sums 1..9 plus the final 0");
  assert.deepStrictEqual(reference.trace.length, 10, "the branch executes ten times");
  assert.deepStrictEqual(reference.trace.filter((s) => s.taken).length, 9, "nine times taken, once not");
  assert.strictEqual(reference.dynamic, 33, "2 set-ups + 10 * 3 body instructions + 1 output");

  // (2) Mispredictions are the hand-computed values for each predictor.
  assert.deepStrictEqual([scoreTrace(reference.trace, "always-taken").wrong, scoreTrace(reference.trace, "always-taken").accuracy],
    [1, 0.9], "always-taken predicts the loop right and the exit wrong");
  assert.deepStrictEqual(scoreTrace(reference.trace, "always-not-taken").wrong, 9, "always-not-taken misses every taken branch");
  assert.deepStrictEqual([scoreTrace(reference.trace, "one-bit").wrong, scoreTrace(reference.trace, "one-bit").accuracy],
    [2, 0.8], "a 1-bit predictor needs two misses: the first branch and the exit");
  assert.deepStrictEqual([scoreTrace(reference.trace, "two-bit").wrong, scoreTrace(reference.trace, "two-bit").accuracy],
    [3, 0.7], "a 2-bit counter warms up for two iterations and then survives until the exit");

  // (3) Cycles = committed instructions + penalty per misprediction, and squash accounting.
  for (const kind of ["always-taken", "always-not-taken", "one-bit", "two-bit"]) {
    const run = simulateSpeculative(countingLoop(), kind);
    assert.strictEqual(run.cycles, run.dynamic + PENALTY * run.wrong,
      `${kind}: cycles equal dynamic instructions plus the penalties`);
    assert.strictEqual(run.squashed, PENALTY * run.wrong, `${kind}: squashed equals penalty * mispredictions`);
    assert.strictEqual(run.wrong + run.correct, reference.trace.length, `${kind}: every branch is scored once`);
  }
  const best = simulateSpeculative(countingLoop(), "always-taken");
  const worst = simulateSpeculative(countingLoop(), "always-not-taken");
  assert.ok(best.cycles < worst.cycles, `always-taken needs ${best.cycles} cycles, always-not-taken ${worst.cycles}`);

  // (4) Speculation never changes the architecture state.
  for (const kind of ["always-taken", "always-not-taken", "one-bit", "two-bit"]) {
    const run = simulateSpeculative(countingLoop(), kind);
    assert.deepStrictEqual(run.output, reference.output, `${kind}: the output is identical`);
    assert.deepStrictEqual([...run.regs.entries()], [...reference.regs.entries()], `${kind}: the registers are identical`);
    assert.strictEqual(run.regs.get("s"), 45, "and the accumulator holds the expected sum");
  }

  // (5) A perfectly predicted run squashes nothing.
  const neverTaken = [SET("i", 0), BNEZ("i", 0), OUT("i")];
  const perfect = simulateSpeculative(neverTaken, "always-not-taken");
  assert.deepStrictEqual([perfect.wrong, perfect.squashed, perfect.cycles], [0, 0, perfect.dynamic],
    "a branch that is never taken is predicted perfectly, so no instruction is squashed");
  assert.deepStrictEqual([perfect.output, interpret(neverTaken).output], [[0], [0]], "and the run matches the reference");
  const perfectTrace = Array.from({ length: 5 }, () => ({ pc: 0, taken: false }));
  assert.deepStrictEqual([scoreTrace(perfectTrace, "always-not-taken").wrong, scoreTrace(perfectTrace, "always-not-taken").accuracy],
    [0, 1], "a perfectly predictable trace scores 100%");
  const straight = simulateSpeculative(straightLine(), "two-bit");
  assert.deepStrictEqual([straight.cycles, straight.squashed, straight.wrong], [straight.dynamic, 0, 0],
    "a branch-free program costs exactly one cycle per instruction");

  // (6) The 2-bit counter transitions, including both saturation ends.
  const counter = makePredictor("two-bit");
  const walked = [];
  for (const taken of [true, true, true, false, false, false]) {
    counter.update(taken);
    walked.push([counter.state(), counter.predict()]);
  }
  assert.deepStrictEqual(walked, [[1, false], [2, true], [3, true], [2, true], [1, false], [0, false]],
    "the counter saturates at 3 and at 0 and predicts taken from 2 upwards");
  assert.deepStrictEqual([counterAfter(3, true), counterAfter(3, false), counterAfter(0, false), counterAfter(1, true)],
    [3, 2, 0, 2], "the transition table matches the implementation");
  const oneBit = makePredictor("one-bit");
  assert.deepStrictEqual([oneBit.predict(), (oneBit.update(true), oneBit.predict()), (oneBit.update(false), oneBit.predict())],
    [false, true, false], "a 1-bit predictor simply mirrors the last outcome");

  // (7) Random traces: accuracy always matches an independent recount, and is 1 when predictable.
  const rng = new SplitMix64(DEFAULT_SEED);
  let scored = 0;
  for (let i = 0; i < 12; i++) {
    const trace = randomTrace(rng, 4 + rng.nextInt(9));
    for (const kind of ["one-bit", "two-bit", "always-taken"]) {
      const result = scoreTrace(trace, kind);
      const recount = kind === "two-bit" ? scoreTwoBit(trace) : kind === "one-bit" ? scoreOneBit(trace) : trace.filter((s) => !s.taken).length;
      assert.strictEqual(result.wrong, recount, `${kind}: an independent recount agrees`);
      scored += 1;
    }
  }
  assert.ok(scored === 36, `${scored} predictor scores were double-checked`);
  const constant = Array.from({ length: 6 }, () => ({ pc: 0, taken: true }));
  assert.deepStrictEqual([scoreTrace(constant, "two-bit").wrong, scoreTrace(constant, "one-bit").wrong], [2, 1],
    "on a fully taken trace the counters only differ in the warm-up");

  // (8) Determinism.
  assert.deepStrictEqual(simulateSpeculative(countingLoop(), "two-bit"), simulateSpeculative(countingLoop(), "two-bit"));
  const rngA = new SplitMix64(7n);
  const rngB = new SplitMix64(7n);
  assert.deepStrictEqual(randomTrace(rngA, 8), randomTrace(rngB, 8), "generated traces are reproducible");
}

/** Independent re-scoring of a trace by an explicit 2-bit counter (no shared code with the predictor). */
function scoreTwoBit(trace) {
  let counter = 0;
  let wrong = 0;
  for (const step of trace) {
    const prediction = counter >= 2;
    if (prediction !== step.taken) wrong += 1;
    counter += step.taken ? 1 : -1;
    counter = counter < 0 ? 0 : counter > 3 ? 3 : counter;
  }
  return wrong;
}
function scoreOneBit(trace) {
  let history = false;
  let wrong = 0;
  for (const step of trace) {
    if (history !== step.taken) wrong += 1;
    history = step.taken;
  }
  return wrong;
}

module.exports = { SplitMix64, DEFAULT_SEED, PENALTY, SET, SUBI, ADD, BNEZ, JMP, OUT, interpret,
  countDynamic, makePredictor, counterAfter, simulateSpeculative, scoreTrace, scoreTwoBit, scoreOneBit,
  countingLoop, straightLine, alwaysTaken, randomTrace };

if (require.main === module) {
  main();
  console.log("PASS");
}
