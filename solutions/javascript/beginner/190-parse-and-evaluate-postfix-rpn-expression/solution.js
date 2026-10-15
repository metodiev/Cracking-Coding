/** Task 190 - Parse and evaluate postfix (RPN) expression. (Tier: Beginner, CTCI Ch 3) */
"use strict";

const OPERATORS = new Set(["+", "-", "*", "/"]);
const INTEGER_TOKEN = /^-?\d+$/;

/** Split a whitespace-separated RPN expression into tokens. */
function tokenizeRpn(expression) {
  const trimmed = expression.trim();
  return trimmed === "" ? [] : trimmed.split(/\s+/);
}

/**
 * Tokens are separated by whitespace: integer operands and the operators + - * /.
 * Evaluation is a stack sweep — push operands, pop two for each operator. '/' is integer
 * division truncating toward zero, and division by zero, unknown tokens, missing operands
 * or leftover values all throw Error.
 */
function evalRpn(expression) {
  const stack = [];
  for (const token of tokenizeRpn(expression)) {
    if (OPERATORS.has(token)) {
      if (stack.length < 2) throw new Error(`not enough operands for ${token}`);
      const right = stack.pop();
      const left = stack.pop();
      if (token === "+") stack.push(left + right);
      else if (token === "-") stack.push(left - right);
      else if (token === "*") stack.push(left * right);
      else {
        if (right === 0) throw new Error("division by zero");
        const quotient = Math.trunc(left / right);
        stack.push(quotient === 0 ? 0 : quotient); // Math.trunc can produce -0
      }
    } else {
      if (!INTEGER_TOKEN.test(token)) throw new Error(`invalid token: ${token}`);
      stack.push(Number(token));
    }
  }
  if (stack.length !== 1) throw new Error(`expression leaves ${stack.length} values on the stack`);
  return stack[0];
}

module.exports = { tokenizeRpn, evalRpn };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(tokenizeRpn("5 1 2 + 4 * + 3 -"), ["5", "1", "2", "+", "4", "*", "+", "3", "-"]);
  assert.deepStrictEqual(tokenizeRpn(""), []);
  assert.deepStrictEqual(tokenizeRpn("  7   2 / "), ["7", "2", "/"]);

  assert.strictEqual(evalRpn("3 4 +"), 7);
  assert.strictEqual(evalRpn("5 1 2 + 4 * + 3 -"), 14);
  assert.strictEqual(evalRpn("7 2 /"), 3);
  assert.strictEqual(evalRpn("42"), 42);
  assert.strictEqual(evalRpn("  42  "), 42);
  assert.strictEqual(evalRpn("2 3 4 * +"), 14);
  assert.strictEqual(evalRpn("1 2 + 3 *"), 9);
  assert.strictEqual(evalRpn("10 3 - 4 2 / *"), 14);
  assert.strictEqual(evalRpn("5 1 - 1 -"), 3);
  assert.strictEqual(evalRpn("0 5 /"), 0);

  // integer division truncates toward zero
  assert.strictEqual(evalRpn("-7 2 /"), -3);
  assert.strictEqual(evalRpn("7 -2 /"), -3);
  assert.strictEqual(evalRpn("-7 -2 /"), 3);
  assert.strictEqual(evalRpn("-1 3 /"), 0);

  for (const bad of ["1 0 /", "0 0 /", "3 4", "1 2 + 3", "+", "", "   ", "abc", "1 2 &", "1.5 2 +", "+5 1 +", "3 -"]) {
    assert.throws(() => evalRpn(bad), Error, `rejects ${bad}`);
  }
  console.log("PASS");
}
