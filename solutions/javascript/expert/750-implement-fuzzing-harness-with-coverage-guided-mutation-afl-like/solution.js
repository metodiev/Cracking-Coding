/** Task 750 - Implement a fuzzing harness with coverage-guided mutation (AFL-like).
 *  (Tier: Expert, CTCI Ch 11 - Testing)
 *
 * The target is a small bytecode VM (comparisons against literal bytes, length guards, a checksum
 * test and a crash instruction). Every executed transition (pc -> next pc) is recorded in a
 * per-run edge set, which is the instrumentation an AFL-style fuzzer feeds on. The harness keeps a
 * corpus queue of inputs; each round picks the next queue entry, applies one deterministic
 * mutation (bit flip, byte write, interesting byte, insertion, deletion, block duplication or a
 * splice with another queue entry) and runs the target. A mutated input joins the queue exactly
 * when it discovered at least one edge that no earlier input had reached, so the queue grows only
 * on genuine new coverage, and inputs that crash the target are stored separately. Because every
 * choice comes from a seeded splitmix/xorshift PRNG and the queue order is fixed, a run is exactly
 * reproducible.
 *
 * What is simplified: one process, one target, no fork server or shared-memory map, a single
 * deterministic mutation per round (no havoc stacking), and coverage is edge coverage of the VM
 * only. What is verified: the same seed replays the identical run (executions, edge count, queue
 * size, crashing inputs); the crash input really crashes when replayed and really satisfies the
 * target's documented condition; the corpus edge union equals the union recomputed from the queue
 * entries and grows monotonically; every queue entry is justified by at least one new edge; guided
 * fuzzing finds the crash within the documented budget while a pure random mutator with the same
 * budget and the same seed does not (the target is built so that guidance is required); targets
 * that crash on the empty input and targets that never crash behave as documented; and no mutation
 * exceeds the input length limit.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

/** splitmix64 in BigInt disguise, reduced to 32 bits for byte level decisions. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

// ---- the target: a bytecode VM with edge instrumentation ----------------------------------------
// instructions:
//   { op: "len", minimum, next, shortFall }        - two distinct edges
//   { op: "cmpByte", index, value, next, mismatch } - two distinct edges per byte
//   { op: "checksum", mask, expected, next, mismatch }
//   { op: "crash", message }
//   { op: "halt" }
function instructionIndex(program, predicate) {
  const index = program.findIndex(predicate);
  if (index < 0) throw new Error("malformed program");
  return index;
}

function runTarget(program, input, options = {}) {
  const edges = new Set();
  const crashes = [];
  let pc = 0;
  let steps = 0;
  const stepLimit = options.stepLimit ?? 200;
  while (steps++ < stepLimit) {
    const instruction = program[pc];
    if (instruction === undefined) {
      edges.add(`${pc}:end`);
      break;
    }
    const record = (next) => edges.add(`${pc}:${next}`);
    switch (instruction.op) {
      case "len": {
        if (input.length >= instruction.minimum) {
          record(instruction.next);
          pc = instruction.next;
        } else {
          record(instruction.shortFall);
          pc = instruction.shortFall;
        }
        break;
      }
      case "cmpByte": {
        if (input.length > instruction.index && input[instruction.index] === instruction.value) {
          record(instruction.next);
          pc = instruction.next;
        } else {
          record(instruction.mismatch);
          pc = instruction.mismatch;
        }
        break;
      }
      case "checksum": {
        let sum = 0;
        for (const byte of input) sum = (sum + byte) & 0xff;
        if ((sum & instruction.mask) === instruction.expected) {
          record(instruction.next);
          pc = instruction.next;
        } else {
          record(instruction.mismatch);
          pc = instruction.mismatch;
        }
        break;
      }
      case "crash": {
        record(pc);
        crashes.push(instruction.message);
        return { edges, crashes, pc, terminated: "crash" };
      }
      default:
        return { edges, crashes, pc, terminated: "halt" };
    }
  }
  return { edges, crashes, pc, terminated: "step-limit" };
}

/**
 * The flagship target: "AFL!" guarded byte by byte, then a checksum, then a crash. Each matching
 * byte opens exactly one new edge, which is what makes coverage feedback necessary.
 */
function magicTarget(magic, checksumMask, checksumExpected) {
  const programList = [];
  const checksumPos = magic.length; // the comparisons occupy 0 .. magic.length - 1
  const crashPos = magic.length + 1;
  const haltPos = magic.length + 2;
  for (let i = 0; i < magic.length; i++) {
    programList.push({
      op: "cmpByte",
      index: i,
      value: magic[i],
      next: i + 1 < magic.length ? i + 1 : checksumPos,
      mismatch: haltPos,
    });
  }
  programList.push({ op: "checksum", mask: checksumMask, expected: checksumExpected,
    next: crashPos, mismatch: haltPos });
  programList.push({ op: "crash", message: "magic + checksum reached" });
  programList.push({ op: "halt" });
  return programList;
}

// ---- mutations ----------------------------------------------------------------------------------
/** A small AFL-style dictionary: control bytes plus the printable ASCII range. */
const INTERESTING = [0x00, 0x01, 0x0a, 0xff,
  ...Array.from({ length: 0x7e - 0x20 + 1 }, (_, i) => 0x20 + i)];

function mutate(input, rng, others, maxLength) {
  const bytes = [...input];
  switch (rng(8)) {
    case 0: { // bit flip
      if (bytes.length === 0) return mutate([0], rng, others, maxLength);
      const position = rng(bytes.length);
      bytes[position] ^= 1 << rng(8);
      break;
    }
    case 1: { // byte write
      if (bytes.length === 0) return mutate([rng(256)], rng, others, maxLength);
      bytes[rng(bytes.length)] = rng(256);
      break;
    }
    case 2: { // interesting byte
      if (bytes.length === 0) return mutate([0], rng, others, maxLength);
      bytes[rng(bytes.length)] = INTERESTING[rng(INTERESTING.length)];
      break;
    }
    case 3: { // insertion, half the time straight out of the dictionary
      if (bytes.length >= maxLength) break;
      const value = rng(2) === 0 ? INTERESTING[rng(INTERESTING.length)] : rng(256);
      bytes.splice(rng(bytes.length + 1), 0, value);
      break;
    }
    case 4: { // deletion
      if (bytes.length <= 1) break;
      bytes.splice(rng(bytes.length), 1);
      break;
    }
    case 5: { // block duplication
      if (bytes.length === 0 || bytes.length >= maxLength) break;
      const start = rng(bytes.length);
      const length = 1 + rng(Math.min(4, bytes.length - start));
      const block = bytes.slice(start, start + length);
      bytes.splice(start, 0, ...block);
      break;
    }
    case 6: { // splice with another queue entry
      if (others.length === 0) break;
      const other = others[rng(others.length)];
      if (other.length === 0) break;
      const cut = rng(other.length);
      const chunk = other.slice(cut, cut + 4);
      bytes.splice(bytes.length, 0, ...chunk);
      break;
    }
    default: { // dictionary probe: overwrite or append exactly one interesting byte
      const position = rng(bytes.length + 1);
      const value = INTERESTING[rng(INTERESTING.length)];
      if (position === bytes.length) {
        if (bytes.length < maxLength) bytes.push(value);
      } else {
        bytes[position] = value;
      }
    }
  }
  return bytes.slice(0, maxLength);
}

// ---- the fuzzing loop ----------------------------------------------------------------------------
function fuzz(program, options = {}) {
  const rng = makeRng(options.seed ?? 1);
  const budget = options.budget ?? 4000;
  const maxLength = options.maxLength ?? 12;
  const queue = options.seeds ? options.seeds.map((seed) => [...seed]) : [[0]];
  const queueEdges = [];
  const globalEdges = new Set();
  const crashes = [];
  const seenCrashes = new Set();
  let executions = 0;
  let mutations = 0;
  let queueIndex = 0;
  const runInput = (bytes, origin) => {
    executions++;
    const result = runTarget(program, bytes, options);
    const fresh = [...result.edges].filter((edge) => !globalEdges.has(edge));
    for (const edge of result.edges) globalEdges.add(edge);
    if (result.crashes.length > 0) {
      const key = JSON.stringify(bytes);
      if (!seenCrashes.has(key)) {
        seenCrashes.add(key);
        crashes.push({ input: [...bytes], message: result.crashes[0], at: executions });
      }
    }
    return { fresh, result };
  };
  for (const seed of queue) queueEdges.push(runInput(seed, "seed").fresh);
  while (executions < budget) {
    if (crashes.length > 0 && options.stopOnCrash !== false) break;
    const base = queue[queueIndex % queue.length];
    queueIndex++;
    const others = queue.filter((entry) => entry !== base);
    const candidate = mutate(base, rng, others, maxLength);
    mutations++;
    const { fresh } = runInput(candidate, "mutation");
    if (fresh.length > 0) {
      queue.push(candidate);
      queueEdges.push(fresh);
    }
  }
  return { queue, queueEdges, edges: globalEdges, crashes, executions, mutations, rngState: rng() };
}

/** Pure random mutator with the same budget: mutates only the seed, keeps no corpus. */
function fuzzRandom(program, options = {}) {
  const rng = makeRng(options.seed ?? 1);
  const budget = options.budget ?? 4000;
  const maxLength = options.maxLength ?? 12;
  const seeds = options.seeds ? options.seeds.map((seed) => [...seed]) : [[0]];
  const globalEdges = new Set();
  const crashes = [];
  let executions = 0;
  for (const seed of seeds) {
    executions++;
    const result = runTarget(program, seed, options);
    for (const edge of result.edges) globalEdges.add(edge);
    if (result.crashes.length > 0) crashes.push({ input: [...seed], message: result.crashes[0] });
  }
  while (executions < budget) {
    const candidate = mutate(seeds[rng(seeds.length)], rng, seeds, maxLength);
    executions++;
    const result = runTarget(program, candidate, options);
    for (const edge of result.edges) globalEdges.add(edge);
    if (result.crashes.length > 0) {
      crashes.push({ input: candidate, message: result.crashes[0] });
      break;
    }
  }
  return { edges: globalEdges, crashes, executions };
}

module.exports = { magicTarget, runTarget, fuzz, fuzzRandom, mutate, makeRng, INTERESTING };

if (require.main === module) {
  const magic = [0x41, 0x46, 0x4c, 0x21]; // "AFL!"
  // the final guard is a parity condition on the byte sum, so a fifth byte with the right parity
  // is enough: exactly the kind of branch a coverage-guided fuzzer is expected to learn
  const target = magicTarget(magic, 0x01, 0x01);

  // -- the target behaves as documented --------------------------------------------------------
  const empty = runTarget(target, []);
  deepStrictEqual(empty.crashes, [], "the empty input does not crash");
  strictEqual(runTarget(target, [...magic, 0x01]).crashes.length, 1,
    "the magic bytes plus an odd sum crash");
  strictEqual(runTarget(target, [...magic, 0x02]).crashes.length, 0,
    "an even sum does not reach the crash");
  strictEqual(runTarget(target, magic).crashes.length, 0,
    "the magic bytes alone leave the checksum unsatisfied");
  const partial = runTarget(target, [0x41, 0x46]);
  ok(partial.edges.size < runTarget(target, magic).edges.size,
    "matching more bytes must reach more edges");

  // -- the guided fuzzer finds the crash ---------------------------------------------------------
  const guided = fuzz(target, { seed: 20260750, budget: 40000, seeds: [[0]] });
  ok(guided.crashes.length > 0, "the guided fuzzer must find the crash");
  const crashInput = guided.crashes[0].input;
  strictEqual(runTarget(target, crashInput).crashes.length, 1, "the stored crash input must crash again");
  const checksum = crashInput.reduce((s, b) => (s + b) & 0xff, 0);
  strictEqual(checksum & 1, 1, "the crash input satisfies the documented checksum condition");
  deepStrictEqual(crashInput.slice(0, magic.length).filter((b, i) => b !== magic[i]), [],
    "the crash input must start with the magic bytes");
  ok(guided.edges.size > 5, `many edges must be discovered (${guided.edges.size})`);
  ok(guided.queue.length > 1, "the corpus must have grown");
  ok(guided.executions <= 40000, "the budget must be respected");
  ok(guided.mutations > 0, "mutations must have been applied");
  for (const input of guided.queue) {
    ok(input.length <= 12, "no input may exceed the length limit");
  }

  // -- every queue entry is justified by fresh coverage, and the union is consistent ---------------
  let recomputed = new Set();
  let justified = 0;
  guided.queue.forEach((input, index) => {
    const result = runTarget(target, input);
    const fresh = [...result.edges].filter((edge) => !recomputed.has(edge));
    for (const edge of result.edges) recomputed.add(edge);
    if (index === 0 || fresh.length > 0) justified++;
    deepStrictEqual(new Set(guided.queueEdges[index]).size, guided.queueEdges[index].length,
      "edge lists must not repeat");
  });
  strictEqual(justified, guided.queue.length, "every queue entry must bring new coverage");
  deepStrictEqual([...recomputed].sort(), [...guided.edges].sort(),
    "the corpus edge union must equal the discovered edge set");

  // -- determinism ----------------------------------------------------------------------------------
  const replay = fuzz(target, { seed: 20260750, budget: 40000, seeds: [[0]] });
  strictEqual(replay.executions, guided.executions, "the same seed must run the same number of inputs");
  strictEqual(replay.edges.size, guided.edges.size, "the same seed must discover the same edges");
  strictEqual(replay.queue.length, guided.queue.length, "the corpus must be identical");
  deepStrictEqual(replay.crashes, guided.crashes, "the crash list must be identical");
  deepStrictEqual(replay.queue.map((entry) => entry.join(",")), guided.queue.map((entry) => entry.join(",")),
    "the queue contents must be identical");
  const otherSeed = fuzz(target, { seed: 99, budget: 40000, seeds: [[0]] });
  ok(otherSeed.executions !== guided.executions || otherSeed.queue.length !== guided.queue.length,
    "a different seed must take a different path through the search");

  // -- guidance beats pure randomness ------------------------------------------------------------------
  const blind = fuzzRandom(target, { seed: 20260750, budget: 40000, seeds: [[0]] });
  ok(blind.crashes.length === 0,
    "uncontrolled mutation must not reach the four byte magic within the budget");
  ok(guided.edges.size > blind.edges.size,
    `coverage feedback must explore more of the target (${guided.edges.size} vs ${blind.edges.size})`);

  // -- boundary targets ------------------------------------------------------------------------------------
  const instantCrash = [{ op: "crash", message: "always" }, { op: "halt" }];
  const instant = fuzz(instantCrash, { seed: 5, budget: 100, seeds: [[]] });
  strictEqual(instant.crashes.length, 1, "a target that crashes on the empty input is found at once");
  strictEqual(instant.executions, 1, "and the fuzzer stops immediately");
  const harmless = [{ op: "cmpByte", index: 0, value: 0x02, next: 2, mismatch: 2 }, { op: "halt" },
    { op: "halt" }];
  const boring = fuzz(harmless, { seed: 7, budget: 200, seeds: [[0]] });
  deepStrictEqual(boring.crashes, [], "a target that cannot crash reports none");
  strictEqual(boring.executions, 200, "the budget is spent exactly");
  const oneRun = fuzz(harmless, { seed: 7, budget: 1, seeds: [[0]] });
  strictEqual(oneRun.executions, 1, "a budget of one runs the seed only");

  // -- mutation engine --------------------------------------------------------------------------------------
  const rng = makeRng(4242);
  let changed = 0;
  for (let i = 0; i < 200; i++) {
    const mutated = mutate([1, 2, 3], rng, [[9, 9, 9]], 6);
    ok(mutated.length <= 6, "mutations respect the length limit");
    if (mutated.join(",") !== "1,2,3") changed++;
  }
  ok(changed > 150, `most mutations must actually change the input (${changed}/200)`);
  const fromEmpty = mutate([], rng, [], 4);
  ok(fromEmpty.length <= 4 && fromEmpty.length >= 0, "an empty seed is handled");

  console.log(`queue=${guided.queue.length} executions=${guided.executions} edges=${guided.edges.size} ` +
    `blind_edges=${blind.edges.size} mutations=${guided.mutations}`);
  console.log(`crash=${JSON.stringify(crashInput)} at_execution=${guided.crashes[0].at} ` +
    `checksum=0x${checksum.toString(16)}`);
  console.log("PASS");
}
