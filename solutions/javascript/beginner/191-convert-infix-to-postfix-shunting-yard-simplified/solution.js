/** Task 191 - Convert infix to postfix (shunting-yard simplified). (Tier: Beginner, CTCI Ch 3) */
"use strict";

const PRECEDENCE = { "+": 1, "-": 1, "*": 2, "/": 2 };

function isOperator(token) {
  return Object.prototype.hasOwnProperty.call(PRECEDENCE, token);
}

function infixToPostfix(expr) {
  const trimmed = expr.trim();
  const tokens = trimmed === "" ? [] : trimmed.split(/\s+/);
  const output = [];
  const stack = [];
  for (const token of tokens) {
    if (/^\d+$/.test(token)) {
      output.push(token);
    } else if (isOperator(token)) {
      while (stack.length > 0 && isOperator(stack[stack.length - 1])
             && PRECEDENCE[stack[stack.length - 1]] >= PRECEDENCE[token]) {
        output.push(stack.pop());
      }
      stack.push(token);
    } else if (token === "(") {
      stack.push(token);
    } else if (token === ")") {
      while (stack.length > 0 && stack[stack.length - 1] !== "(") output.push(stack.pop());
      if (stack.length === 0) throw new Error("unmatched ')' in expression");
      stack.pop();
    } else {
      throw new Error(`unknown token: ${token}`);
    }
  }
  while (stack.length > 0) {
    if (stack[stack.length - 1] === "(") throw new Error("unmatched '(' in expression");
    output.push(stack.pop());
  }
  return output.join(" ");
}

module.exports = { infixToPostfix };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(infixToPostfix("3 + 4 * 2"), "3 4 2 * +");
  assert.strictEqual(infixToPostfix("( 1 + 2 ) * 3"), "1 2 + 3 *");
  assert.strictEqual(infixToPostfix("1 + 2 - 3"), "1 2 + 3 -");
  assert.strictEqual(infixToPostfix("8 / 4 / 2"), "8 4 / 2 /");
  assert.strictEqual(infixToPostfix("2 * 3 + 4 * 5"), "2 3 * 4 5 * +");
  assert.strictEqual(infixToPostfix("( 1 + 2 ) * ( 3 - 4 )"), "1 2 + 3 4 - *");
  assert.strictEqual(infixToPostfix("( ( 12 ) )"), "12");
  assert.strictEqual(infixToPostfix(""), "");
  assert.throws(() => infixToPostfix("1 + 2 )"), /unmatched/);
  assert.throws(() => infixToPostfix("( 1 + 2"), /unmatched/);
  assert.throws(() => infixToPostfix("( ( 1 ) + 2 ) )"), /unmatched/);
  assert.throws(() => infixToPostfix("1 + a"), /unknown token/);
  assert.throws(() => infixToPostfix("1 ^ 2"), /unknown token/);
  console.log("PASS");
}
