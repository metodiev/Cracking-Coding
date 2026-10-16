/** Task 973 - Implement automated hint generator using solved-solution decomposition.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A canonical solution is stored as a small DAG of logical steps (code block + concept tag +
 * prerequisites). The generator topologically orders the steps and emits three escalating hint
 * levels per step: L1 a concept question, L2 the technique plus a prose outline, L3 a code skeleton
 * whose secret tokens are blanked out. Prose levels must not quote the solution (not one secret
 * occurrence), while L3 must account for every code line - either verbatim or as a blanked hole.
 * Simplified: the concept lexicon is hand-written data rather than an LLM and the skeleton blanks
 * whole tokens instead of rewriting expressions. Verified: DAG ordering, the leak gate, a line-level
 * completeness invariant for the skeleton, level/time boundaries, cycle rejection and determinism.
 */
"use strict";
const assert = require("node:assert");

const SOLUTION = {
  id: "two-sum",
  steps: [
    { id: "s1", tag: "parseInput", dependsOn: [], code: [
      "const nums = input.nums;",
      "const target = input.target;",
    ] },
    { id: "s2", tag: "hashMap", dependsOn: ["s1"], code: ["const seen = new Map();"] },
    { id: "s3", tag: "singlePass", dependsOn: ["s2"], code: [
      "for (let i = 0; i < nums.length; i += 1) {",
      "  const need = target - nums[i];",
      "  if (seen.has(need)) return [seen.get(need), i];",
      "  seen.set(nums[i], i);",
      "}",
    ] },
    { id: "s4", tag: "noSolution", dependsOn: ["s3"], code: ["return [];"] },
  ],
};

/** Concept lexicon: the question, the outline, and the tokens that must never be leaked. */
const LEXICON = {
  parseInput: { artifact: "input contract", goal: "read the array and the target exactly once",
    technique: "input unpacking", outline: "copy the array and the target out of the payload first",
    secrets: ["input.nums", "input.target"] },
  hashMap: { artifact: "lookup structure", goal: "answer membership in constant time",
    technique: "one hash table", outline: "index every visited value by its position as you scan",
    secrets: ["Map", "seen"] },
  singlePass: { artifact: "single scan", goal: "spot the complement while iterating",
    technique: "one-pass hashing", outline: "walk the sequence once and test whether the complement was recorded",
    secrets: ["need", "target - nums[i]", "seen.get", "seen.has", "seen.set"] },
  noSolution: { artifact: "termination case", goal: "return a well-defined empty answer",
    technique: "defensive return", outline: "after the loop, hand back the empty pair instead of undefined",
    secrets: ["return [];"] },
};
const FALLBACK = { artifact: "code block", goal: "finish this step",
  technique: "unnamed technique", outline: "study the block inputs and outputs before writing it",
  secrets: [] };
const LEVELS = [1, 2, 3];
const LEVEL_NAMES = { 1: "nudge", 2: "structure", 3: "skeleton" };

/** Kahn topological sort over the step DAG, alphabetical among the ready steps. */
function orderSteps(steps) {
  const ids = steps.map((step) => step.id);
  const indegree = new Map(ids.map((id) => [id, 0]));
  const dependents = new Map(ids.map((id) => [id, []]));
  for (const step of steps) {
    for (const pre of step.dependsOn) {
      if (!indegree.has(pre)) throw new Error(`${step.id} depends on unknown step ${pre}`);
      indegree.set(step.id, indegree.get(step.id) + 1);
      dependents.get(pre).push(step.id);
    }
  }
  const ready = ids.filter((id) => indegree.get(id) === 0).sort();
  const order = [];
  while (ready.length) {
    const id = ready.shift();
    order.push(id);
    for (const next of dependents.get(id)) {
      indegree.set(next, indegree.get(next) - 1);
      if (indegree.get(next) === 0) { ready.push(next); ready.sort(); }
    }
  }
  if (order.length !== ids.length) throw new Error("step graph has a cycle");
  return order;
}

const entryFor = (tag, lexicon) => lexicon[tag] || FALLBACK;
const secretHits = (text, secrets) => secrets.filter((secret) => text.includes(secret));

/** L3 skeleton: every secret occurrence in the code becomes a `___` hole. */
function skeleton(step, entry) {
  return step.code.map((line) => entry.secrets.reduce((text, secret) => text.split(secret).join("___"), line));
}

/** One hint at one level; level 3 returns the skeleton's lines. */
function hintFor(step, level, lexicon = LEXICON) {
  assert(LEVELS.includes(level), `hint level ${level} must be one of ${LEVELS.join("/")}`);
  const entry = entryFor(step.tag, lexicon);
  if (level === 1) return `Which ${entry.artifact} lets you ${entry.goal}?`;
  if (level === 2) return `${entry.technique}: ${entry.outline}.`;
  return skeleton(step, entry);
}

/** Full hint plan: steps in DAG order, three levels each, with a leak gate on every level. */
function buildHints(solution, lexicon = LEXICON) {
  const byId = new Map(solution.steps.map((step) => [step.id, step]));
  return orderSteps(solution.steps).map((id) => {
    const step = byId.get(id);
    const entry = entryFor(step.tag, lexicon);
    const hints = LEVELS.map((level) => {
      const text = hintFor(step, level, lexicon);
      return { level, name: LEVEL_NAMES[level], text: Array.isArray(text) ? text.join("\n") : text };
    });
    for (const hint of hints) {
      const leaked = secretHits(hint.text, entry.secrets);
      assert.strictEqual(leaked.length, 0, `hint ${id}/L${hint.level} leaks ${leaked.join(",")}`);
    }
    return { stepId: id, tag: step.tag, lines: step.code.length, covered: hints.length === LEVELS.length,
      hints, secretCount: entry.secrets.length };
  });
}

const textOf = (entry, level) => entry.hints.find((hint) => hint.level === level).text;
const countOccurrences = (text, needle) => text.split(needle).length - 1;
const verbatimLines = (hint, code) => code.filter((line) => hint.includes(line)).length;
const blankedLines = (hint) => hint.split("\n").filter((line) => line.includes("___")).length;

function main() {
  const plan = buildHints(SOLUTION);

  // 1. Happy path: every step is covered at all three levels, in topological order.
  assert.deepStrictEqual(plan.map((entry) => entry.stepId), ["s1", "s2", "s3", "s4"], "topological order");
  assert(plan.every((entry) => entry.covered), "every step is covered");
  assert(plan.every((entry) => entry.hints.map((h) => h.level).join() === "1,2,3"), "all three levels emitted");
  assert(plan[0].hints[0].text.startsWith("Which input contract"), `L1 phrasing: ${plan[0].hints[0].text}`);
  assert(plan[2].hints[1].text.startsWith("one-pass hashing:"), "L2 names the technique");
  assert.deepStrictEqual(plan.map((entry) => entry.hints.map((h) => h.name).join("/")),
    ["nudge/structure/skeleton", "nudge/structure/skeleton", "nudge/structure/skeleton", "nudge/structure/skeleton"]);

  // 2. Independent scan: the prose levels contain no secret occurrence of their step.
  for (const entry of plan) {
    const step = SOLUTION.steps.find((s) => s.id === entry.stepId);
    const entryLexicon = entryFor(step.tag, LEXICON);
    for (const level of [1, 2]) {
      assert.strictEqual(secretHits(textOf(entry, level), entryLexicon.secrets).length, 0,
        `${entry.stepId} L${level} must not quote the solution`);
    }
    assert(secretHits(textOf(entry, 3), entryLexicon.secrets).length === 0, `${entry.stepId} L3 blanks its secrets`);
    assert(step.code.some((line) => textOf(entry, 3).includes(line)) || entryLexicon.secrets.length === 0
      || blankedLines(textOf(entry, 3)) > 0, `${entry.stepId} L3 keeps the block shape`);
  }

  // 3. Line-level completeness: every code line is either verbatim or blanked, never dropped.
  for (const entry of plan) {
    const step = SOLUTION.steps.find((s) => s.id === entry.stepId);
    const skeletonText = textOf(entry, 3);
    assert.strictEqual(verbatimLines(skeletonText, step.code) + blankedLines(skeletonText), step.code.length,
      `${entry.stepId} L3 accounts for all ${step.code.length} lines`);
    assert.strictEqual(skeletonText.split("\n").length, step.code.length, `${entry.stepId} line count preserved`);
  }
  const skeletonText = textOf(plan[2], 3);
  assert(skeletonText.includes("for (let i = 0; i < nums.length; i += 1) {"), "control flow stays visible");
  assert(skeletonText.endsWith("}"), "closing braces survive");
  assert.strictEqual(countOccurrences(skeletonText, "___"), 7, "seven secret occurrences are blanked");

  // 4. Independent hole count: occurrences measured on the untouched code must match the holes.
  const occurrences = SOLUTION.steps[2].code.reduce((sum, line) => sum
    + LEXICON.singlePass.secrets.reduce((n, secret) => n + countOccurrences(line, secret), 0), 0);
  assert.strictEqual(countOccurrences(skeletonText, "___"), occurrences, "hole count matches the scan");
  assert(occurrences > LEXICON.singlePass.secrets.length, "secrets repeat inside the block");

  // 5. Boundaries: one-step solutions, an empty solution and an unknown tag.
  const single = buildHints({ id: "one", steps: [SOLUTION.steps[0]] });
  assert.strictEqual(single.length, 1, "single-step solutions work");
  assert.strictEqual(blankedLines(textOf(single[0], 3)), 2, "both lines of the block are blanked");
  assert.deepStrictEqual(buildHints({ id: "empty", steps: [] }), [], "an empty solution yields no hints");
  const unknown = buildHints({ id: "x", steps: [{ id: "s1", tag: "quantum", dependsOn: [], code: ["return 1;"] }] });
  assert(textOf(unknown[0], 1).includes("code block") && textOf(unknown[0], 2).startsWith("unnamed technique"),
    "unknown tags fall back to a generic hint");
  assert.strictEqual(blankedLines(textOf(unknown[0], 3)), 0, "a fallback keeps the code visible");

  // 6. Negative: illegal levels and broken graphs are rejected.
  assert.throws(() => hintFor(SOLUTION.steps[0], 4), /must be one of/, "level 4 is rejected");
  assert.throws(() => hintFor(SOLUTION.steps[0], 0), /must be one of/, "level 0 is rejected");
  assert.throws(() => orderSteps([{ id: "a", tag: "x", dependsOn: ["b"], code: [] },
    { id: "b", tag: "x", dependsOn: ["a"], code: [] }]), /cycle/, "cyclic steps are rejected");
  assert.throws(() => orderSteps([{ id: "a", tag: "x", dependsOn: ["ghost"], code: [] }]),
    /unknown step/, "dangling dependencies are rejected");

  // 7. Oracle: the leak gate is exercised by a deliberately leaky lexicon.
  const leaky = { singlePass: { ...LEXICON.singlePass, outline: "record seen.set(nums[i], i) as you go" } };
  assert.throws(() => buildHints(SOLUTION, { ...LEXICON, ...leaky }), /leaks/, "a leaking outline is rejected");
  const leakyNudge = { hashMap: { ...LEXICON.hashMap, goal: "grow seen into a Map" } };
  assert.throws(() => buildHints(SOLUTION, { ...LEXICON, ...leakyNudge }), /L1 leaks Map/, "L1 is gated too");

  // 8. Determinism: the plan depends on the DAG, not on the input order or the run.
  assert(JSON.stringify(buildHints(SOLUTION)) === JSON.stringify(plan), "hints are reproducible");
  assert(JSON.stringify(buildHints({ ...SOLUTION, steps: [...SOLUTION.steps].reverse() })) === JSON.stringify(plan),
    "reversing the input list changes nothing");
  const plainStep = { id: "s", tag: "noSolution", dependsOn: [], code: SOLUTION.steps[3].code };
  assert(buildHints({ id: "s", steps: [plainStep] })[0].hints[2].text.includes("___"), "single secret line blanked");

  console.log(`hints: ${plan.length} steps x ${LEVELS.length} levels, ` +
    `${plan.reduce((sum, entry) => sum + entry.secretCount, 0)} secrets, ${occurrences} holes in the deepest skeleton, ` +
    `skeleton lines ${plan.reduce((sum, entry) => sum + entry.lines, 0)}`);
  console.log("PASS");
}
module.exports = { SOLUTION, LEXICON, FALLBACK, LEVELS, orderSteps, hintFor, skeleton, buildHints,
  secretHits, countOccurrences };
if (require.main === module) main();
