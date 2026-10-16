/** Task 974 - Implement step-by-step solution explainer generator using AST of canonical solution.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A miniature imperative language (let/assign/if/else/while/return over integer arithmetic and
 * comparisons) is tokenized and parsed into an AST, then executed twice: once by a plain recursive
 * evaluator that only returns the final environment, and once by a tracing interpreter that records
 * one explanation step per executed statement - including one step per loop-condition evaluation -
 * together with a deep copy of the variable state at that moment. Simplified: no functions, strings
 * or arrays, integers only, division truncates toward zero. Verified: both interpreters agree on
 * value and environment (independent oracle), hand-computed step counts, snapshot independence,
 * precedence, the literal/empty/deep-nesting boundaries, syntax and runtime errors, determinism.
 */
"use strict";
const assert = require("node:assert");
const MULTI_CHAR = ["==", "!=", "<=", ">="];
const SINGLE_CHAR = ["(", ")", "{", "}", ";", "=", "<", ">", "+", "-", "*", "/", "%", "!"];
const PRECEDENCE = { "==": 1, "!=": 1, "<": 1, "<=": 1, ">": 1, ">=": 1, "+": 2, "-": 2, "*": 3, "/": 3, "%": 3 };
function tokenize(source) {
  const tokens = [];
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    if (/\s/.test(ch)) { i += 1; continue; }
    if (/[0-9]/.test(ch)) {
      const match = /^[0-9]+/.exec(source.slice(i));
      tokens.push({ type: "num", value: Number(match[0]), pos: i });
      i += match[0].length;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      const match = /^[A-Za-z_][A-Za-z0-9_]*/.exec(source.slice(i));
      tokens.push({ type: "name", value: match[0], pos: i });
      i += match[0].length;
      continue;
    }
    const op = MULTI_CHAR.concat(SINGLE_CHAR).find((candidate) => source.startsWith(candidate, i));
    if (op === undefined) throw new SyntaxError(`unexpected character "${ch}" at ${i}`);
    tokens.push({ type: "punct", value: op, pos: i });
    i += op.length;
  }
  tokens.push({ type: "eof", value: "<eof>", pos: source.length });
  return tokens;
}
function parse(source) {
  const tokens = tokenize(source);
  let pos = 0;
  const peek = () => tokens[pos];
  const next = () => tokens[pos++];
  const expect = (value) => {
    const token = next();
    if (token.value !== value) throw new SyntaxError(`expected "${value}" but found "${token.value}" at ${token.pos}`);
    return token;
  };
  const unary = () => {
    const token = peek();
    if (token.value === "-" || token.value === "!") { next(); return { type: "Unary", op: token.value, arg: unary() }; }
    if (token.value === "(") { next(); const inner = expression(); expect(")"); return inner; }
    next();
    if (token.type === "num") return { type: "Num", value: token.value };
    if (token.type === "name") return { type: "Ident", name: token.value };
    throw new SyntaxError(`unexpected token "${token.value}" at ${token.pos}`);
  };
  const expression = (minPrecedence = 0) => {
    let left = unary();
    while (peek().type === "punct" && (PRECEDENCE[peek().value] || 0) > minPrecedence) {
      const op = next().value;
      left = { type: "Binary", op, left, right: expression(PRECEDENCE[op]) };
    }
    return left;
  };
  const block = () => {
    expect("{");
    const body = [];
    while (peek().value !== "}") body.push(statement());
    expect("}");
    return { type: "Block", body };
  };
  const statement = () => {
    const token = peek();
    if (token.value === "{") return block();
    if (token.type === "name" && token.value === "let") {
      next();
      const name = next();
      if (name.type !== "name") throw new SyntaxError(`expected a variable name at ${name.pos}`);
      expect("="); const value = expression(); expect(";");
      return { type: "Let", name: name.value, value };
    }
    if (token.type === "name" && token.value === "if") {
      next(); expect("("); const test = expression(); expect(")");
      const then = block();
      const alt = peek().type === "name" && peek().value === "else" ? (next(), block()) : null;
      return { type: "If", test, then, alt };
    }
    if (token.type === "name" && token.value === "while") {
      next(); expect("("); const test = expression(); expect(")");
      return { type: "While", test, body: block() };
    }
    if (token.type === "name" && token.value === "return") {
      next(); const value = expression(); expect(";");
      return { type: "Return", value };
    }
    next();
    if (token.type !== "name") throw new SyntaxError(`expected an assignment at ${token.pos}`);
    expect("="); const value = expression(); expect(";");
    return { type: "Assign", name: token.value, value };
  };
  const body = [];
  while (peek().type !== "eof") body.push(statement());
  return { type: "Program", body };
}
const OPERATORS = { "+": (a, b) => a + b, "-": (a, b) => a - b, "*": (a, b) => a * b,
  "/": (a, b) => Math.trunc(a / b), "%": (a, b) => a % b, "==": (a, b) => a === b, "!=": (a, b) => a !== b,
  "<": (a, b) => a < b, "<=": (a, b) => a <= b, ">": (a, b) => a > b, ">=": (a, b) => a >= b };
const applyOperator = (op, a, b) => OPERATORS[op](a, b);
/** Oracle: a plain evaluator with no tracing, so its shape is independent of the explainer. */
function evaluateProgram(ast) {
  const env = new Map();
  const value = (node) => {
    if (node.type === "Num") return node.value;
    if (node.type === "Ident") {
      if (!env.has(node.name)) throw new Error(`unknown variable "${node.name}"`);
      return env.get(node.name);
    }
    if (node.type === "Unary") return node.op === "-" ? -value(node.arg) : !value(node.arg);
    if (node.type === "Binary") return applyOperator(node.op, value(node.left), value(node.right));
    throw new Error(`cannot evaluate ${node.type}`);
  };
  let result;
  const run = (statements) => {
    for (const statement of statements) {
      if (statement.type === "Let" || statement.type === "Assign") env.set(statement.name, value(statement.value));
      else if (statement.type === "If") {
        const branch = value(statement.test) ? statement.then.body : (statement.alt ? statement.alt.body : []);
        if (run(branch)) return true;
      } else if (statement.type === "While") {
        while (value(statement.test)) if (run(statement.body.body)) return true;
      } else if (statement.type === "Return") { result = value(statement.value); return true; }
      else if (statement.type === "Block") { if (run(statement.body)) return true; }
      else throw new Error(`cannot execute ${statement.type}`);
    }
    return false;
  };
  run(ast.body);
  return { result, env: Object.fromEntries(env) };
}
/** Tracing interpreter: one step per executed statement, with a snapshot of the environment. */
function explain(ast, { maxSteps = 5000 } = {}) {
  const env = new Map();
  const steps = [];
  const record = (kind, detail) => {
    if (steps.length >= maxSteps) throw new Error(`step budget of ${maxSteps} exceeded`);
    steps.push({ index: steps.length, kind, detail, state: Object.fromEntries(env) });
  };
  const value = (node) => {
    if (node.type === "Num") return node.value;
    if (node.type === "Ident") {
      if (!env.has(node.name)) throw new Error(`unknown variable "${node.name}"`);
      return env.get(node.name);
    }
    if (node.type === "Unary") return node.op === "-" ? -value(node.arg) : !value(node.arg);
    if (node.type === "Binary") return applyOperator(node.op, value(node.left), value(node.right));
    throw new Error(`cannot evaluate ${node.type}`);
  };
  let result;
  const run = (statements) => {
    for (const statement of statements) {
      if (statement.type === "Let" || statement.type === "Assign") {
        const assigned = value(statement.value);
        env.set(statement.name, assigned);
        record(statement.type === "Let" ? "let" : "assign",
          `${statement.name} ${statement.type === "Let" ? "starts at" : "becomes"} ${assigned}`);
      } else if (statement.type === "If") {
        const taken = value(statement.test);
        record("branch", `condition is ${taken ? "true, so run the then-block" : "false, so run the else-block"}`);
        if (run(taken ? statement.then.body : (statement.alt ? statement.alt.body : []))) return true;
      } else if (statement.type === "While") {
        let iteration = 0;
        while (value(statement.test)) {
          iteration += 1;
          record("loop", `iteration ${iteration} begins`);
          if (run(statement.body.body)) return true;
        }
        record("loop-end", `condition is false after ${iteration} iterations`);
      } else if (statement.type === "Return") {
        result = value(statement.value);
        record("return", `the answer is ${result}`);
        return true;
      } else if (statement.type === "Block") { if (run(statement.body)) return true; }
      else throw new Error(`cannot execute ${statement.type}`);
    }
    return false;
  };
  run(ast.body);
  return { steps, result, env: Object.fromEntries(env), budget: maxSteps };
}
const PROGRAMS = {
  sum: "let total = 0; let i = 1; while (i <= 10) { total = total + i; i = i + 1; } return total;",
  threes: "let n = 15; let hits = 0; while (n > 0) { if (n % 3 == 0) { hits = hits + 1; } n = n - 1; } return hits;",
  branch: "let a = 2; let b = 3; if (a < b) { a = a * b + 1; } else { b = b * 2; } return a + b;",
  literal: "return 42;",
  empty: "",
  runaway: "while (1 < 2) { } return 1;",
};
const countKind = (steps, kind) => steps.filter((step) => step.kind === kind).length;
function main() {
  // 1. Happy path: three programs produce the arithmetic a reader expects.
  const sum = explain(parse(PROGRAMS.sum));
  const threes = explain(parse(PROGRAMS.threes));
  const branch = explain(parse(PROGRAMS.branch));
  assert.strictEqual(sum.result, 55, "1..10 sums to 55");
  assert.strictEqual(threes.result, 5, "the multiples of three in 1..15 are five");
  assert.strictEqual(branch.result, 10, "2*3+1 then plus 3");
  assert(sum.steps.some((step) => step.detail.includes("iteration 10")), "the tenth iteration is narrated");
  assert(sum.steps.at(-1).detail === "the answer is 55", "the last step states the result");
  // 2. Independent oracle: the plain evaluator agrees on value and final environment.
  for (const [name, source] of Object.entries(PROGRAMS)) {
    if (name === "runaway") continue;
    const traced = explain(parse(source));
    const plain = evaluateProgram(parse(source));
    assert.strictEqual(traced.result, plain.result, `${name}: results agree`);
    assert.deepStrictEqual(traced.env, plain.env, `${name}: final environments agree`);
  }
  assert.deepStrictEqual(evaluateProgram(parse("let z = 1 + 2 * 3;")).env, { z: 7 }, "precedence: * before +");
  assert.deepStrictEqual(evaluateProgram(parse("let z = (1 + 2) * 3;")).env, { z: 9 }, "parentheses win");
  assert.deepStrictEqual(evaluateProgram(parse("let z = -7 / 2;")).env, { z: -3 }, "division truncates toward zero");
  assert.deepStrictEqual(evaluateProgram(parse("let z = 17 % 5;")).env, { z: 2 }, "modulo");
  // 3. Hand-computed step counts act as an exact trace oracle.
  assert.strictEqual(sum.steps.length, 34, `sum trace: 2 lets + 11 conditions + 20 assigns + 1 return`);
  assert.strictEqual(countKind(sum.steps, "let"), 2, "two declarations");
  assert.strictEqual(countKind(sum.steps, "assign"), 20, "two assignments per iteration");
  assert.strictEqual(countKind(sum.steps, "loop"), 10, "ten loop iterations");
  assert.strictEqual(countKind(sum.steps, "loop-end"), 1, "one loop exit");
  assert.strictEqual(threes.steps.length, 54, "threes trace: 16 conditions + 15 branches + 20 assigns + 3");
  assert.strictEqual(countKind(threes.steps, "branch"), 15, "one branch decision per iteration");
  assert.strictEqual(countKind(threes.steps, "assign"), 20, "five hits plus fifteen decrements");
  // 4. Snapshot invariants: indices are dense, snapshots are copies, state only ever grows.
  for (const program of [sum, threes, branch]) {
    program.steps.forEach((step, i) => assert.strictEqual(step.index, i, "step indices are dense"));
    const keys = program.steps.map((step) => Object.keys(step.state).length);
    for (let i = 1; i < keys.length; i += 1) assert(keys[i] >= keys[i - 1], "variables are never forgotten");
    program.steps[1].state.injected = 1;
    assert(!program.steps[2].state.injected, "snapshots are independent copies");
    delete program.steps[1].state.injected;
    assert(program.steps[1].state !== program.steps[2].state, "each snapshot is its own object");
  }
  assert.deepStrictEqual(threes.steps.at(-1).state, { n: 0, hits: 5 }, "the final state is visible in the trace");
  // 5. Boundaries: a literal program, an empty program and deep nesting.
  const literal = explain(parse(PROGRAMS.literal));
  assert.strictEqual(literal.steps.length, 1, "a literal program is one step long");
  assert.deepStrictEqual(literal.env, {}, "no variables are bound");
  const empty = explain(parse(PROGRAMS.empty));
  assert.strictEqual(empty.steps.length, 0, "an empty program has no steps");
  assert.strictEqual(empty.result, undefined, "an empty program returns nothing");
  const nestedSource = `return ${"(".repeat(100)}1${")".repeat(100)} + 1;`;
  assert.strictEqual(explain(parse(nestedSource)).result, 2, "100 levels of parentheses still evaluate");
  const longSource = `let acc = 0; ${Array.from({ length: 50 }, () => "acc = acc + 1;").join(" ")} return acc;`;
  assert.strictEqual(evaluateProgram(parse(longSource)).result, 50, "a long straight-line program works");
  // 6. Negative paths: syntax errors carry a position, unknown names and runaway loops are caught.
  assert.throws(() => parse("let = 5;"), SyntaxError, "a missing identifier is a syntax error");
  assert.throws(() => parse("let a = 1"), /expected ";"/, "a missing semicolon is reported with context");
  assert.throws(() => parse("return 1 + ;"), /unexpected token/, "a dangling operator is rejected");
  assert.throws(() => parse("let a = $1;"), /unexpected character/, "unknown characters are rejected");
  assert.throws(() => explain(parse("return missing;")), /unknown variable "missing"/, "unknown names fail");
  assert.throws(() => explain(parse(PROGRAMS.runaway)), /step budget/, "an infinite loop hits the step budget");
  assert.strictEqual(evaluateProgram(parse("if (0) { } return 3;")).result, 3, "a false branch without else falls through");
  // 7. The explainer reads exactly like the execution order (loop bodies then the exit note).
  const ordered = threes.steps.map((step) => step.kind).join(",");
  assert(ordered.startsWith("let,let,loop,branch,assign,assign"), `trace prefix: ${ordered.slice(0, 40)}`);
  assert(ordered.endsWith("loop-end,return"), "the trace ends with the loop exit and the return");
  assert.strictEqual(threes.steps.filter((step) => step.kind === "loop").length, 15, "fifteen iterations narrated");
  // 8. Determinism: parsing and explaining twice is byte-identical.
  assert(JSON.stringify(explain(parse(PROGRAMS.threes))) === JSON.stringify(threes), "the trace is reproducible");
  assert(JSON.stringify(parse(PROGRAMS.sum)) === JSON.stringify(parse(PROGRAMS.sum)), "the AST is reproducible");
  console.log(`explainer: sum ${sum.steps.length} steps, threes ${threes.steps.length} steps, ` +
    `branch ${branch.steps.length} steps, kinds ${[...new Set(sum.steps.map((s) => s.kind))].join("/")}`);
  console.log("PASS");
}
module.exports = { PRECEDENCE, tokenize, parse, applyOperator, evaluateProgram, explain, PROGRAMS };
if (require.main === module) main();
