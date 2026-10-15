/**
 * Task 342 - Expression evaluator with variables and precedence. (Tier: Intermediate, CTCI Ch 8)
 *
 * Shunting-yard parser for + - * /, parentheses, unary minus ('~' in the RPN) and variables from
 * a plain object.  Everything is evaluated in JavaScript numbers (IEEE-754 double), so 100/8 is
 * 12.5 - no integer truncation.  Malformed input throws Error with a message.  The brief's
 * example 3+4*2/(1-5) is 1 (3 + 8/(-4)), not 2.
 */
"use strict";

const BINARY_PRECEDENCE = { "+": 1, "-": 1, "*": 2, "/": 2 };
const UNARY_PRECEDENCE = 4;            // unary minus binds tightest, right-associative

function tokenize(expr) {
  const tokens = [];
  let i = 0;
  while (i < expr.length) {
    const ch = expr[i];
    if (ch === " ") {
      i++;
    } else if (ch >= "0" && ch <= "9") {
      let j = i;
      while (j < expr.length && expr[j] >= "0" && expr[j] <= "9") j++;
      tokens.push({ kind: "num", value: Number(expr.slice(i, j)) });
      i = j;
    } else if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < expr.length && /[A-Za-z0-9_]/.test(expr[j])) j++;
      tokens.push({ kind: "var", name: expr.slice(i, j) });
      i = j;
    } else if ("+-*/()".includes(ch)) {
      tokens.push({ kind: "op", symbol: ch });
      i++;
    } else {
      throw new Error(`unexpected character '${ch}'`);
    }
  }
  return tokens;
}

function toRpn(tokens) {
  const output = [];
  const ops = [];
  let expectOperand = true;
  for (const token of tokens) {
    if (token.kind === "num") {
      if (!expectOperand) throw new Error(`expected an operator before ${token.value}`);
      output.push(token);
      expectOperand = false;
    } else if (token.kind === "var") {
      if (!expectOperand) throw new Error(`expected an operator before ${token.name}`);
      output.push(token);
      expectOperand = false;
    } else if (token.symbol === "(") {
      if (!expectOperand) throw new Error("implicit multiplication is not supported");
      ops.push(token);
    } else if (token.symbol === ")") {
      if (expectOperand) throw new Error("empty or dangling parenthesis group");
      while (ops.length > 0 && ops[ops.length - 1].symbol !== "(") output.push(ops.pop());
      if (ops.length === 0) throw new Error("unbalanced ')'");
      ops.pop();
      expectOperand = false;
    } else if (expectOperand) {
      if (token.symbol === "-") ops.push({ kind: "op", symbol: "~" });
      else if (token.symbol !== "+") throw new Error(`operator '${token.symbol}' has no left operand`);
    } else {
      const precedence = BINARY_PRECEDENCE[token.symbol];
      while (ops.length > 0) {
        const top = ops[ops.length - 1];
        if (top.symbol === "(") break;
        const topPrecedence = top.symbol === "~" ? UNARY_PRECEDENCE : BINARY_PRECEDENCE[top.symbol];
        if (topPrecedence > precedence || (topPrecedence === precedence && top.symbol !== "~")) output.push(ops.pop());
        else break;
      }
      ops.push(token);
      expectOperand = true;
    }
  }
  if (expectOperand && tokens.length > 0) throw new Error("dangling operator at end of expression");
  while (ops.length > 0) {
    const top = ops.pop();
    if (top.symbol === "(") throw new Error("unbalanced '('");
    output.push(top);
  }
  return output;
}

function evaluateRpn(rpn, variables) {
  const stack = [];
  for (const token of rpn) {
    if (token.kind === "num") {
      stack.push(token.value);
    } else if (token.kind === "var") {
      if (!(token.name in variables)) throw new Error(`unknown variable '${token.name}'`);
      stack.push(Number(variables[token.name]));
    } else if (token.symbol === "~") {
      if (stack.length === 0) throw new Error("malformed expression");
      stack.push(-stack.pop());
    } else {
      if (stack.length < 2) throw new Error("malformed expression");
      const b = stack.pop();
      const a = stack.pop();
      if (token.symbol === "+") stack.push(a + b);
      else if (token.symbol === "-") stack.push(a - b);
      else if (token.symbol === "*") stack.push(a * b);
      else {
        if (b === 0) throw new Error("division by zero");
        stack.push(a / b);
      }
    }
  }
  if (stack.length !== 1) throw new Error("malformed expression");
  return stack[0];
}

function evaluate(expr, variables = {}) {
  const tokens = tokenize(expr);
  if (tokens.length === 0) throw new Error("empty expression");
  return evaluateRpn(toRpn(tokens), variables);
}

module.exports = { tokenize, toRpn, evaluate };

if (require.main === module) {
  const assert = require("node:assert");

  // brief examples - the first one is 1, not 2
  assert.strictEqual(evaluate("3+4*2/(1-5)"), 1);
  assert.strictEqual(evaluate("x+y*2", { x: 3, y: 4 }), 11);
  assert.strictEqual(evaluate("(1+2)*(3+4)"), 21);

  // values cross-checked against CPython's expression parser
  assert.strictEqual(evaluate("1+2*3-4/8"), 6.5);
  assert.strictEqual(evaluate("2*(3+4)-5"), 9);
  assert.strictEqual(evaluate("100/8"), 12.5);
  assert.strictEqual(evaluate("-(3+4)*2"), -14);
  assert.strictEqual(evaluate("3*-2"), -6);
  assert.strictEqual(evaluate("2*-3+4"), -2);
  assert.strictEqual(evaluate("10-4-3"), 3, "left associative");
  assert.strictEqual(evaluate("16/4/2"), 2, "(16/4)/2");
  assert.strictEqual(evaluate("2*(1+2*(3-1))"), 10);
  assert.strictEqual(evaluate("((7))"), 7);
  assert.strictEqual(evaluate("8/(2+2)"), 2);
  assert.strictEqual(evaluate("--5"), 5, "chained unary minus");
  assert.strictEqual(evaluate("-(-5)"), 5);
  assert.strictEqual(evaluate("3--2"), 5, "binary minus then unary minus");
  assert.strictEqual(evaluate("2+3*4-6/3"), 12);
  assert.strictEqual(evaluate("((1+2)*(3+4))/3"), 7);
  assert.strictEqual(evaluate("1+((2+3)*4)"), 21);
  assert.strictEqual(evaluate("7/2"), 3.5);
  assert.strictEqual(evaluate("(8-3)*(2+1)/5"), 3);
  assert.strictEqual(evaluate(" 1 + 2 "), 3, "whitespace ignored");
  assert.strictEqual(evaluate("2*-5+20/4"), -5);
  assert.strictEqual(evaluate("x*y+z", { x: 3, y: 4, z: 5 }), 17);
  assert.strictEqual(evaluate("-x*2", { x: 3 }), -6);
  assert.strictEqual(evaluate("+x", { x: 7 }), 7, "unary plus");
  assert.strictEqual(evaluate("price*quantity", { price: 2.5, quantity: 4 }), 10);

  // malformed input: documented Error
  const bad = ["", "1+", "*3", "1**2", "(1+2", "1+2)", "()", "( )", "1 2", "3x", "1(2)", "&", "1&2", "4/0", "0/0", "(1+)"];
  for (const expr of bad) assert.throws(() => evaluate(expr), Error, `expected throw for ${JSON.stringify(expr)}`);
  assert.throws(() => evaluate("x+1"), Error, "unknown variable");
  assert.throws(() => evaluate("sum", { total: 1 }), Error, "unknown variable with a map");
  console.log("PASS");
}
