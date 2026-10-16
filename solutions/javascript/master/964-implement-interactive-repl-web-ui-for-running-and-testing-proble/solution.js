/** Task 964 - Implement interactive REPL/web UI for running and testing problems in-browser (sandboxed).
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * The evaluation half of a browser REPL, in memory: a lexer, a precedence-climbing parser with unary
 * minus, right-associative `^`, calls and a small statement grammar (`let x = e`, `x = e`, `print(e)`,
 * bare expressions); an evaluator over an environment Map; and a sandbox policy enforced during
 * evaluation rather than afterwards -- a step counter kills runaway work, a whitelist keeps the builtin
 * surface tiny, a banned-identifier list rejects host objects at parse time, and allocations obey a
 * memory cap. A failed line never mutates the environment, so the session stays consistent, and
 * replaying the recorded history rebuilds the environment exactly.
 * Simplified: this is not a JavaScript interpreter -- the language is arithmetic over numbers, arrays
 * and whitelisted builtins, with no loops or functions, which is enough to model the sandbox contract
 * (budget, whitelist, banned globals, allocation cap). Verified: closed-form precedence and associativity, environment mutation with rejection of assignment to unknowns, the step budget with an unchanged environment, banned identifiers, the allocation cap, history replay, completion boundaries, deterministic re-runs.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
class SandboxError extends Error {}
const BANNED = new Set(["process", "require", "globalThis", "eval", "Function", "module", "exports", "this", "import"]);
const BUILTINS = {
  abs: ([x]) => Math.abs(x), sqrt: ([x]) => Math.sqrt(x), len: ([x]) => x.length,
  min: (args) => Math.min(...args), max: (args) => Math.max(...args), sum: ([xs]) => xs.reduce((a, b) => a + b, 0),
  range: ([n]) => {
    if (n < 0) throw new SandboxError("range needs a non-negative length");
    if (n > 64) throw new SandboxError("allocation limit exceeded");
    return Array.from({ length: n }, (_, i) => i);
  },
};
const BINARY = { "+": 9, "-": 9, "*": 10, "/": 10, "%": 10 };
const OPS = { "+": (a, b) => a + b, "-": (a, b) => a - b, "*": (a, b) => a * b, "/": (a, b) => a / b, "%": (a, b) => a % b };
/** Sticky lexer: whitespace, numbers, identifiers and single punctuation characters; nothing else. */
function tokenize(source) {
  const text = source.trim();
  if (text === "") throw new SandboxError("empty input");
  const tokens = [];
  const pattern = /\s*(\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*|[^\sA-Za-z0-9_])/y;
  let consumed = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) { tokens.push(match[1]); consumed = pattern.lastIndex; }
  if (consumed !== text.length) throw new SandboxError(`unexpected character at ${consumed}`);
  return tokens;
}
/** Precedence climbing; `-` binds looser than `^`, so -2^2 is -(2^2) = -4. */
function parse(tokens) {
  let index = 0;
  const peek = () => tokens[index];
  const eat = (text) => (peek() === text ? (index += 1, true) : false);
  const fail = (message) => { throw new SandboxError(`parse error at token ${index}: ${message}`); };
  const primary = () => {
    const token = peek();
    if (token === undefined) fail("expression expected");
    if (/^\d/.test(token)) { index += 1; return { kind: "num", value: Number(token) }; }
    if (/^[A-Za-z_]/.test(token)) {
      index += 1;
      if (BANNED.has(token)) fail(`banned identifier ${token}`);
      if (!eat("(")) return { kind: "var", name: token };
      const args = [];
      if (!eat(")")) {
        do { args.push(expression(0)); } while (eat(","));
        if (!eat(")")) fail("expected )");
      }
      return { kind: "call", name: token, args };
    }
    if (eat("(")) { const node = expression(0); if (!eat(")")) fail("expected )"); return node; }
    return fail(`unexpected ${token}`);
  };
  const unary = () => (eat("-") ? { kind: "neg", operand: unary() } : power());
  const power = () => {
    const base = primary();
    return eat("^") ? { kind: "pow", base, exponent: unary() } : base;
  };
  const expression = (min) => {
    let left = unary();
    for (;;) {
      const token = peek();
      const precedence = BINARY[token];
      if (precedence === undefined || precedence < min) return left;
      index += 1;
      left = { kind: "bin", op: token, left, right: expression(precedence + 1) };
    }
  };
  const statement = () => {
    if (eat("let")) {
      const name = peek();
      if (name === undefined || !/^[A-Za-z_]/.test(name)) fail("let needs a name");
      if (BANNED.has(name)) fail(`banned identifier ${name}`);
      index += 1;
      if (!eat("=")) fail("let needs =");
      return { kind: "let", name, value: expression(0) };
    }
    if (peek() === "print" && tokens[index + 1] === "(") { index += 2; const value = expression(0); if (!eat(")")) fail("expected )"); return { kind: "print", value }; }
    const first = peek();
    if (first !== undefined && /^[A-Za-z_]/.test(first) && tokens[index + 1] === "=") {
      if (BANNED.has(first)) fail(`banned identifier ${first}`);
      index += 2;
      return { kind: "assign", name: first, value: expression(0) };
    }
    return { kind: "expr", value: expression(0) };
  };
  const node = statement();
  if (index !== tokens.length) fail(`unexpected ${tokens[index]}`);
  return node;
}
/** Evaluate with the sandbox policy live: every node and call costs a step against the budget. */
function evaluate(node, env, ctx) {
  ctx.steps += 1;
  if (ctx.steps > ctx.budget) throw new SandboxError(`step budget exceeded (${ctx.budget} steps)`);
  switch (node.kind) {
    case "num": return node.value;
    case "var": {
      if (env.has(node.name)) return env.get(node.name);
      throw new SandboxError(`unknown identifier ${node.name}`);
    }
    case "neg": return -evaluate(node.operand, env, ctx);
    case "pow": return evaluate(node.base, env, ctx) ** evaluate(node.exponent, env, ctx);
    case "bin": {
      const left = evaluate(node.left, env, ctx);
      const right = evaluate(node.right, env, ctx);
      if ((node.op === "/" || node.op === "%") && right === 0) throw new SandboxError("division by zero");
      return OPS[node.op](left, right);
    }
    case "call": {
      const fn = ctx.builtins[node.name];
      if (fn === undefined) throw new SandboxError(`unknown function ${node.name}`);
      const args = node.args.map((arg) => evaluate(arg, env, ctx));
      ctx.steps += 1;
      return fn(args);
    }
  }
}
const format = (value) => (Array.isArray(value) ? `[${value.join(", ")}]` : String(value));
/** A REPL session: environment, output log, history, budget and completion index. */
class Session {
  constructor({ budget = 200, builtins = BUILTINS } = {}) {
    this.budget = budget; this.builtins = builtins;
    this.env = new Map(); this.logs = []; this.history = [];
  }  run(line) {
    const entry = { line, ok: true, steps: 0 };
    try {
      const node = parse(tokenize(line));
      const ctx = { steps: 0, budget: this.budget, builtins: this.builtins };
      const value = evaluate(node.value, this.env, ctx); // every statement kind carries its expression
      entry.steps = ctx.steps;
      // Environment mutation happens only after a fully successful evaluation.
      if (node.kind === "let") this.env.set(node.name, value);
      if (node.kind === "assign") {
        if (!this.env.has(node.name)) throw new SandboxError(`unknown identifier ${node.name}`);
        this.env.set(node.name, value);
      }
      entry.value = value;
      entry.display = format(value);
      if (node.kind === "print") this.logs.push(entry.display);
    } catch (error) {
      entry.ok = false;
      entry.error = error.message;
      entry.kind = error instanceof SandboxError ? "sandbox" : "internal";
    }
    this.history.push(entry);
    return entry;
  }
  snapshot() { return [...this.env.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)); }
  /** Replay the transcript into a fresh session: the environment must come back identical. */
  replay(options) {
    const clone = new Session(options || { budget: this.budget, builtins: this.builtins });
    for (const entry of this.history) clone.run(entry.line);
    return clone;
  }
  completion(prefix) {
    const names = [...Object.keys(this.builtins), ...this.env.keys()];
    return [...new Set(names)].filter((name) => name.startsWith(prefix)).sort().slice(0, 8);
  }
}

function main() {
  const rng = makeRng(20240964);
  const session = new Session({ budget: 200 });
  // 1. Precedence, associativity and unary minus against closed form.
  const closedForm = [["2 + 3 * 4", 14], ["(2 + 3) * 4", 20], ["2 ^ 3 ^ 2", 512], ["-2 ^ 2", -4], ["10 - 3 - 2", 5],
    ["10 % 3", 1], ["8 / 4 / 2", 1], ["sum(range(8))", 28]];
  for (const [line, expected] of closedForm) {
    const entry = session.run(line);
    assert(entry.ok && entry.value === expected, `${line} -> ${entry.value} expected ${expected}`);
  }
  // 2. Statements: let, reassignment, expression; the environment is a sorted snapshot.
  assert(session.run("let a = 5").ok && session.run("let b = a * 2 + 1").ok, "let evaluates and stores");
  assert(session.run("b").value === 11 && session.run("a + b").value === 16, "stored variables are readable");
  assert(JSON.stringify(session.snapshot()) === '[["a",5],["b",11]]', `snapshot ${JSON.stringify(session.snapshot())}`);
  const unknownAssign = session.run("c = 1");
  assert(!unknownAssign.ok && unknownAssign.error === "unknown identifier c", "assigning to an unknown name is rejected");
  assert(!session.env.has("c"), "a rejected line leaves no binding behind");
  // 3. Step budget: a 200-term sum exceeds a 100-step budget without touching the environment.
  const tight = new Session({ budget: 100 });
  const long = Array.from({ length: 200 }, () => "1").join(" + ");
  const blown = tight.run(`let big = ${long}`);
  assert(!blown.ok && blown.error.startsWith("step budget exceeded"), `budget error: ${blown.error}`);
  assert(!tight.env.has("big") && tight.snapshot().length === 0, "no partial state survives a sandbox kill");
  assert(tight.run("let ok = 1 + 2").ok && tight.snapshot().length === 1, "a smaller line still fits the same budget");
  // 4. Banned identifiers never reach evaluation and the whitelist blocks unknown functions.
  for (const line of ["process.exit(1)", "eval(1)", "require(1)", "globalThis"]) {
    const entry = session.run(line);
    assert(!entry.ok && entry.error.includes("banned identifier"), `${line} -> ${entry.error}`);
  }
  const missingFn = session.run("shutdown(1)");
  assert(!missingFn.ok && missingFn.error === "unknown function shutdown", "only whitelisted builtins exist");
  // 5. Allocation cap: range obeys the memory limit, and the surviving result matches a closed form.
  const huge = session.run("range(1000000)");
  assert(!huge.ok && huge.error === "allocation limit exceeded", `allocation error: ${huge.error}`);
  assert(session.run("let xs = range(10)").ok && session.run("len(xs)").value === 10, "a legal range still allocates");
  assert(session.run("sum(xs)").value === 45 && session.run("range(-1)").error === "range needs a non-negative length",
    "the allocated list sums to n(n-1)/2 and negative lengths are rejected");
  // 6. Completion: prefix search over builtins and live variables, with the empty-prefix boundary.
  assert(session.completion("ra").join(",") === "range" && session.completion("m").join(",") === "max,min",
    "prefix search over builtins");
  assert(session.completion("a").join(",") === "a,abs", "a variable and a builtin both match");
  assert(session.completion("zzz").length === 0 && session.completion("").length === 8, "unknown prefix is empty, bare prefix caps at 8");
  // 7. Error rendering: parse, runtime and division errors are messages, not crashes.
  assert(session.run("2 +").error.startsWith("parse error at token") && session.run("(1 + 2").error.includes("expected )"),
    "dangling operators and unclosed groups are parse errors");
  assert(session.run("1 / 0").error === "division by zero", "division by zero is a sandbox error");
  assert(session.run("nope + 1").error === "unknown identifier nope" && session.run("").error === "empty input",
    "unknown identifiers and empty lines are reported");
  assert(session.run("3 + 4").value === 7, "the session survives every error");
  assert(session.run("print(2 ^ 5)").display === "32" && session.logs.join(",") === "32", "print captures output");
  // 8. Replay and determinism: the transcript rebuilds the same environment, twice.
  const replay = session.replay();
  assert(JSON.stringify(replay.snapshot()) === JSON.stringify(session.snapshot()), "replay reproduces the environment");
  assert(replay.logs.join(",") === session.logs.join(","), "replay reproduces the output log");
  assert(JSON.stringify(replay.history) === JSON.stringify(session.history), "replay reproduces the transcript");
  const fresh = new Session({ budget: 200 });
  for (const entry of session.history) fresh.run(entry.line);
  assert(JSON.stringify(fresh.snapshot()) === JSON.stringify(session.snapshot()), "a second replay agrees");
  const seeded = Array.from({ length: 5 }, (_, i) => `let v${i} = ${rng.int(50)} * 2`);
  const first = seeded.reduce((target, line) => (target.run(line), target), new Session());
  const second = seeded.reduce((target, line) => (target.run(line), target), new Session());
  assert.deepStrictEqual(first.snapshot(), second.snapshot(), "a seeded script replays identically");
  assert(first.snapshot().length === 5 && first.logs.length === 0, `seeded bindings ${first.snapshot().length}`);
  console.log(`repl: lines=${session.history.length} ok=${session.history.filter((e) => e.ok).length} `
    + `bindings=${session.snapshot().length} logs=${session.logs.length} completions=${session.completion("").length}`);
  console.log("PASS");
}
module.exports = { makeRng, SandboxError, BANNED, BUILTINS, tokenize, parse, evaluate, Session, format };
if (require.main === module) main();
