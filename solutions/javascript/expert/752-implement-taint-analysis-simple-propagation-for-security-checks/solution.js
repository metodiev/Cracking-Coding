/** Task 752 - Implement taint analysis simple propagation for security checks. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * The analysed program is a tiny three-address IR: basic blocks of instructions (`source`,
 * `copy`, `add`, `sanitize`, `literal`, `call`, `selector`, `countdown`, `branch`, `sink`)
 * wired into a CFG by `successors`. Taint is a set of variables propagated forward: a
 * `source` taints its destination, `copy`/`add`/`call` join the taint of their operands, and
 * every other definition *kills* the destination (a strong update, so an overwrite erases the
 * earlier taint). Merges union the incoming sets, which makes this a may-analysis: a sink
 * argument is reported when it is tainted on *some* path. The transfer function distributes
 * over union, so the chaotic-iteration fix point equals the union over all paths; the code
 * iterates a worklist and caps it so a non-convergent fix point is an error, not a hang.
 * Every report carries a witness chain rebuilt by a backward walk over def-use edges.
 *
 * What is simplified: no inter-procedural, field-sensitive or alias analysis and no implicit
 * flows through `branch` conditions — the toy IR is deliberately coarse.
 * What is verified: (a) soundness — every sink the dynamic taint interpreter taints is also
 * reported statically; (b) no over-reporting — a sanitizer and an overwrite kill the taint
 * statically *and* at run time; (c) on loop-free programs the fix point equals an independent
 * path-enumeration analysis exactly; (d) every witness chain follows real def-use edges
 * starting at a source; (e) sinks in unreachable blocks are never analysed.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

// --- toy IR instructions -------------------------------------------------------------
const source = (dst, label) => ({ op: "source", dst, label });
const copy = (dst, src) => ({ op: "copy", dst, src });
const add = (dst, lhs, rhs) => ({ op: "add", dst, lhs, rhs });
const sanitize = (dst, src) => ({ op: "sanitize", dst, src });
const literal = (dst) => ({ op: "literal", dst });
const call = (dst, callee, args) => ({ op: "call", dst, callee, args });
const selector = (dst) => ({ op: "selector", dst });
const countdown = (dst) => ({ op: "countdown", dst });
const branch = (src) => ({ op: "branch", src });
const sink = (name, args) => ({ op: "sink", name, args });
const KILLS = new Set(["sanitize", "literal", "selector", "countdown"]);
const PROPAGATES = new Set(["source", "copy", "add", "call"]);

/** Def-use operands of an instruction; definitions without operands return []. */
function operandsOf(instruction) {
  const { op } = instruction;
  if (op === "copy" || op === "sanitize" || op === "branch") return [instruction.src];
  if (op === "add") return [instruction.lhs, instruction.rhs];
  if (op === "call") return instruction.args.slice();
  return [];
}

const sameSet = (a, b) => a.size === b.size && [...a].every((value) => b.has(value));

/** One transfer step; mutates `state` and hands every tainted sink argument to `onSink`. */
function applyInstruction(state, instruction, index, onSink) {
  const { op } = instruction;
  if (op === "branch") return;
  if (op === "sink") { for (const arg of instruction.args) if (state.has(arg)) onSink(arg, index); return; }
  if (KILLS.has(op)) { state.delete(instruction.dst); return; }
  if (!PROPAGATES.has(op)) throw new TypeError(`unknown instruction '${op}'`);
  if (op === "source" || operandsOf(instruction).some((operand) => state.has(operand))) state.add(instruction.dst);
  else state.delete(instruction.dst);
}

function reachableBlocks(program) {
  const reachable = new Set();
  const pending = [program.entry];
  while (pending.length > 0) {
    const blockId = pending.pop();
    if (reachable.has(blockId) || !(blockId in program.blocks)) continue;
    reachable.add(blockId);
    pending.push(...program.blocks[blockId].successors);
  }
  return reachable;
}

const predecessorsOf = (program, blockId) => Object.entries(program.blocks)
  .filter(([, block]) => block.successors.includes(blockId)).map(([id]) => id);

/** Definition table variable -> [{block, index, instruction}], reachable blocks only. */
function collectDefiners(program, reachable = reachableBlocks(program)) {
  const definers = new Map();
  for (const blockId of Object.keys(program.blocks).sort()) {
    if (!reachable.has(blockId)) continue;
    program.blocks[blockId].instructions.forEach((instruction, index) => {
      if (instruction.dst === undefined) return;
      definers.set(instruction.dst, [...(definers.get(instruction.dst) ?? []), { block: blockId, index, instruction }]);
    });
  }
  return definers;
}

/** Shortest backward def-use walk from `target` to a variable produced by a source. */
function findChain(definers, target) {
  const pending = [[target]];
  const seen = new Set([target]);
  while (pending.length > 0) {
    const path = pending.shift();
    const definitions = definers.get(path[0]) ?? [];
    if (definitions.some(({ instruction }) => instruction.op === "source")) return path;
    for (const { instruction } of definitions) {
      for (const operand of operandsOf(instruction)) {
        if (!seen.has(operand)) { seen.add(operand); pending.push([operand, ...path]); }
      }
    }
  }
  return null;
}
/** True when the chain is a run of real def-use edges starting at a source definition. */
function validateChain(definers, chain) {
  if (!Array.isArray(chain) || chain.length === 0) return false;
  if (!(definers.get(chain[0]) ?? []).some(({ instruction }) => instruction.op === "source")) return false;
  return chain.slice(1).every((variable, i) => (definers.get(variable) ?? [])
    .some(({ instruction }) => operandsOf(instruction).includes(chain[i])));
}
/** Forward may-taint fix point over the CFG; reports are collected in a final sweep. */
function analyseTaint(program) {
  const reachable = reachableBlocks(program);
  const blockIds = Object.keys(program.blocks).sort();
  const inSets = new Map(blockIds.map((id) => [id, new Set()]));
  const outSets = new Map(blockIds.map((id) => [id, new Set()]));
  const worklist = [program.entry];
  const queued = new Set(worklist);
  const visited = new Set();
  let updates = 0;
  while (worklist.length > 0) {
    const blockId = worklist.shift();
    queued.delete(blockId);
    const incoming = new Set();
    for (const predecessor of predecessorsOf(program, blockId)) {
      if (reachable.has(predecessor)) for (const variable of outSets.get(predecessor)) incoming.add(variable);
    }
    const state = new Set(incoming);
    for (const instruction of program.blocks[blockId].instructions) applyInstruction(state, instruction, 0, () => {});
    const changed = !visited.has(blockId)
      || !sameSet(inSets.get(blockId), incoming) || !sameSet(outSets.get(blockId), state);
    visited.add(blockId);
    inSets.set(blockId, incoming);
    outSets.set(blockId, state);
    if (changed && ++updates > 100000) throw new Error("taint fix point did not converge");
    if (!changed) continue;
    for (const successor of program.blocks[blockId].successors) {
      if (reachable.has(successor) && !queued.has(successor)) { queued.add(successor); worklist.push(successor); }
    }
  }
  const definers = collectDefiners(program, reachable);
  const violations = [];
  for (const blockId of blockIds) {
    if (!reachable.has(blockId)) continue;
    const state = new Set(inSets.get(blockId));
    program.blocks[blockId].instructions.forEach((instruction, index) => applyInstruction(state, instruction, index,
      (arg, sinkIndex) => violations.push({
        block: blockId, index: sinkIndex, sink: instruction.name, arg, chain: findChain(definers, arg) })));
  }
  return { violations, updates, inSets, outSets, reachable };
}

/** Behavioural oracle: concrete interpreter with dynamic taint tracking. */
function runDynamic(program, inputs, maxSteps = 400) {
  const state = new Set();
  const values = new Map(Object.entries(inputs));
  const violations = [];
  let block = program.entry;
  let steps = 0;
  while (block !== undefined) {
    if (steps++ > maxSteps) return { violations, truncated: true };
    const { instructions, successors } = program.blocks[block];
    let next;
    instructions.forEach((instruction, index) => {
      if (instruction.op === "selector" && !values.has(instruction.dst)) throw new RangeError(`no input '${instruction.dst}'`);
      if (instruction.op === "countdown") values.set(instruction.dst, (values.get(instruction.dst) ?? 1) - 1);
      if (instruction.op === "branch") { next = values.get(instruction.src) ? successors[0] : successors[1]; return; }
      applyInstruction(state, instruction, index, (arg) => violations.push(`${block}#${index}:${arg}`));
    });
    block = next === undefined ? successors[0] : next;
  }
  return { violations, truncated: false };
}
/** Independent oracle 1: enumerate whole paths and union the taint seen at each sink. */function enumeratePaths(program, maxPaths = 4000) {
  const reachable = reachableBlocks(program);
  const found = new Set();
  const stack = [{ block: program.entry, state: new Set() }];
  let expanded = 0;
  while (stack.length > 0) {
    const { block, state } = stack.pop();
    if (!reachable.has(block)) continue;
    if (++expanded > maxPaths) return { found, truncated: true };
    const local = new Set(state);
    program.blocks[block].instructions.forEach((instruction, index) => {
      applyInstruction(local, instruction, index, (arg, sinkIndex) => found.add(`${block}#${sinkIndex}:${arg}`));
    });
    for (const successor of program.blocks[block].successors) stack.push({ block: successor, state: new Set(local) });
  }
  return { found, truncated: false };
}
const violationKeys = (violations) => new Set(violations.map((v) => `${v.block}#${v.index}:${v.arg}`));

/** Every 2^k selector assignment, so the dynamic oracle covers every path. */
function allInputCombinations(program) {
  const names = [];
  for (const block of Object.values(program.blocks)) {
    for (const instruction of block.instructions) if (instruction.op === "selector") names.push(instruction.dst);
  }
  return Array.from({ length: 1 << names.length }, (_, mask) => {
    const inputs = {};
    names.forEach((name, bit) => { inputs[name] = (mask >> bit) & 1; });
    return inputs;
  });
}

/** Deterministic xorshift32. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x752752;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
/** Random IR generator; `allowLoops` false keeps every successor strictly forward. */
function randomProgram(rng, blockCount, allowLoops) {
  const ids = Array.from({ length: blockCount }, (_, i) => `b${i}`);
  const blocks = {};
  const variables = [];
  let counter = 0;
  const pick = () => variables[rng(variables.length)];
  for (let i = 0; i < blockCount; i++) {
    const instructions = [];
    for (let k = 0, count = 1 + rng(3); k < count; k++) {
      const choice = variables.length === 0 ? 0 : rng(6);
      const dst = `v${counter++}`;
      if (choice === 0) instructions.push(source(dst, "request"));
      else if (choice === 1) instructions.push(copy(dst, pick()));
      else if (choice === 2 && rng(2) === 0) instructions.push(sanitize(dst, pick()));
      else if (choice === 3) instructions.push(add(dst, pick(), pick()));
      else if (choice === 4) instructions.push(literal(dst));
      else { instructions.push(sink("exec", [pick()])); continue; }
      variables.push(dst);
    }
    const successors = i + 1 < blockCount ? [ids[i + 1]] : [];
    if (allowLoops && i > 0 && rng(3) === 0) successors.push(ids[rng(i)]);
    else if (!allowLoops && rng(3) === 0 && i + 2 < blockCount) successors.push(ids[i + 1 + rng(blockCount - i - 1)]);
    blocks[ids[i]] = { instructions, successors };
  }
  return { entry: ids[0], blocks };
}

module.exports = {
  source, copy, add, sanitize, literal, call, selector, countdown, branch, sink,
  applyInstruction, analyseTaint, runDynamic, enumeratePaths, validateChain,
  findChain, collectDefiners, reachableBlocks, allInputCombinations, makeRng, randomProgram,
};
if (require.main === module) {
  const program = (blocks, entry = "b0") => ({ entry, blocks });
  const block = (instructions, successors = []) => ({ instructions, successors });
  // 1. empty program: nothing flows, nothing is reported
  deepStrictEqual(analyseTaint(program({ b0: block([]) })).violations, []);
  // 2. direct flow, with a witness chain that must survive validation
  const direct = program({ b0: block([source("param", "http.query"), sink("sql", ["param"])]) });
  const directResult = analyseTaint(direct);
  strictEqual(directResult.violations.length, 1, "one tainted sink");
  deepStrictEqual(directResult.violations[0].chain, ["param"]);
  ok(validateChain(collectDefiners(direct), directResult.violations[0].chain), "witness chain must be valid");
  // 3. a sanitizer kills the taint: no false positive, statically or dynamically
  const sanitized = program({ b0: block([source("raw", "http.query"), sanitize("clean", "raw"), sink("sql", ["clean"])]) });
  deepStrictEqual(analyseTaint(sanitized).violations, [], "a sanitized value must not be reported");
  deepStrictEqual(runDynamic(sanitized, {}).violations, [], "and it is clean at run time too");
  // 4. strong update: the overwrite erases the taint of the first sink only
  const overwritten = program({ b0: block([source("x", "http.query"), sink("log", ["x"]), literal("x"), sink("log", ["x"])]) });
  const overwrittenResult = analyseTaint(overwritten);
  strictEqual(overwrittenResult.violations.length, 1, "only the sink before the overwrite is tainted");
  strictEqual(overwrittenResult.violations[0].index, 1, "the reported sink is the second instruction");
  // 5. taint joins through arithmetic and through a non-sanitizing call
  const joined = program({ b0: block([source("a", "header"), literal("b"), add("c", "a", "b"),
    call("d", "escapeUrl", ["c"]), sink("html", ["c", "b"]), sink("shell", ["d"])]) });
  deepStrictEqual([...violationKeys(analyseTaint(joined).violations)].sort(), ["b0#4:c", "b0#5:d"], "propagation kinds");
  // 6. may-analysis across a branch merge, cross-checked both ways
  const merged = program({
    b0: block([selector("flag"), branch("flag")], ["b1", "b2"]),
    b1: block([source("t", "cookie")], ["b3"]),
    b2: block([literal("t")], ["b3"]),
    b3: block([sink("sql", ["t"])]),
  });
  strictEqual(analyseTaint(merged).violations.length, 1, "taint on one branch is enough for a may-analysis");
  deepStrictEqual(runDynamic(merged, { flag: 1 }).violations, ["b3#0:t"], "the tainted branch really reaches the sink");
  deepStrictEqual(runDynamic(merged, { flag: 0 }).violations, [], "the clean branch really does not");
  // 7. loop-carried taint: the fix point converges and stays sound on every path
  const loop = program({
    b0: block([selector("n"), literal("acc")], ["b1"]),
    b1: block([branch("n")], ["b2", "b3"]),
    b2: block([source("input", "socket"), add("acc", "acc", "input"), countdown("n")], ["b1"]),
    b3: block([sink("file", ["acc"])]),
  });
  const loopResult = analyseTaint(loop);
  ok(loopResult.updates > 3 && loopResult.updates <= 8, `the fix point converges in a few updates (${loopResult.updates})`);
  strictEqual(loopResult.violations.length, 1, "loop-carried taint must be reported");
  for (const inputs of allInputCombinations(loop)) {
    const dynamic = runDynamic(loop, inputs);
    ok(!dynamic.truncated, "the counted loop must terminate");
    for (const key of dynamic.violations) ok(violationKeys(loopResult.violations).has(key), "dynamic ⊆ static");
  }
  strictEqual(runDynamic(loop, { n: 0 }).violations.length, 0, "zero iterations leave the accumulator clean");
  strictEqual(runDynamic(loop, { n: 3 }).violations.length, 1, "three iterations taint the accumulator");
  // 8. unreachable blocks are neither analysed nor enumerated
  const unreachable = program({ b0: block([literal("safe")]), b9: block([source("dead", "query"), sink("sql", ["dead"])]) });
  deepStrictEqual(analyseTaint(unreachable).violations, [], "a sink in an unreachable block cannot be reached");
  strictEqual(enumeratePaths(unreachable).found.size, 0, "path enumeration agrees");
  // 9. a source-free program is clean
  const clean = program({ b0: block([literal("a"), copy("b", "a"), sink("sql", ["b"])]) });
  deepStrictEqual(analyseTaint(clean).violations, []);
  strictEqual(findChain(collectDefiners(clean), "b"), null, "no chain into a clean variable");
  // 10. randomized cross-checks against the three oracles
  const rng = makeRng(0x752752);
  let acyclicTrials = 0;
  let dynamicChecks = 0;
  for (let trial = 0; trial < 30; trial++) {
    const acyclic = trial % 5 !== 4;
    const candidate = randomProgram(rng, 3 + rng(4), !acyclic);
    const staticResult = analyseTaint(candidate);
    const staticKeys = violationKeys(staticResult.violations);
    const definers = collectDefiners(candidate, staticResult.reachable);
    for (const violation of staticResult.violations) {
      ok(validateChain(definers, violation.chain), `witness chain of ${violation.block}#${violation.index}`);
    }
    for (const inputs of allInputCombinations(candidate)) {
      const dynamic = runDynamic(candidate, inputs);
      dynamicChecks++;
      for (const key of dynamic.violations) ok(staticKeys.has(key), `dynamic taint ${key} must be reported (trial ${trial})`);
      ok(dynamic.violations.length <= staticResult.violations.length, "dynamic findings never exceed static ones");
    }
    if (acyclic) {
      const paths = enumeratePaths(candidate);
      ok(!paths.truncated, "acyclic programs must enumerate completely");
      deepStrictEqual([...staticKeys].sort(), [...paths.found].sort(), `path enumeration equality in trial ${trial}`);
      acyclicTrials++;
    }
  }
  ok(acyclicTrials >= 20, "most randomized programs must be acyclic");
  // 11. boundaries and malformed input
  deepStrictEqual(analyseTaint(program({ b0: block([source("s", "q"), sink("ping", [])]) })).violations, [], "no args, no leak");
  throws(() => applyInstruction(new Set(), { op: "nope" }, 0, () => {}), TypeError);
  throws(() => runDynamic(merged, {}), RangeError);
  const backedge = program({
    b0: block([branch("c"), source("v", "q"), sink("sql", ["v"])], ["b1", "b1"]),
    b1: block([sink("sql", ["v"])]),
  });
  ok(violationKeys(analyseTaint(backedge).violations).has("b1#0:v"), "a successor-less block still sees the flow");
  deepStrictEqual(runDynamic(backedge, { c: 0 }).violations, ["b0#2:v", "b1#0:v"], "a duplicate successor executes once");

  console.log(`randomized=30 acyclic=${acyclicTrials} dynamicChecks=${dynamicChecks} loopUpdates=${loopResult.updates}`);
  console.log("PASS");
}
