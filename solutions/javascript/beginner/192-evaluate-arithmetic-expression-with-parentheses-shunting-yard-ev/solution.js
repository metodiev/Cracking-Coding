/** Task 192 - Evaluate arithmetic expression with +,-,*,/, parentheses (shunting-yard + eval). (Tier: Beginner, CTCI Ch 3) */
"use strict";

const PRECEDENCE = { "+": 1, "-": 1, "*": 2, "/": 2 };

function isOperator(token) {
  return Object.prototype.hasOwnProperty.call(PRECEDENCE, token);
}

function tokenize(expr) {
  const trimmed = expr.trim();
  const tokens = trimmed === "" ? [] : trimmed.split(/\s+/);
  for (const token of tokens) {
    if (!/^\d+$/.test(token) && !isOperator(token) && token !== "(" && token !== ")") {
      throw new Error(`unknown token: ${token}`);
    }
  }
  return tokens;
}

function infixToPostfix(tokens) {
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
    }
  }
  while (stack.length > 0) {
    if (stack[stack.length - 1] === "(") throw new Error("unmatched '(' in expression");
    output.push(stack.pop());
  }
  return output;
}

function apply(op, left, right) {
  switch (op) {
    case "+": return left + right;
    case "-": return left - right;
    case "*": return left * right;
    default:
      if (right === 0) throw new Error("division by zero");
      return Math.trunc(left / right);                 // truncate toward zero
  }
}

function evaluatePostfix(tokens) {
  const stack = [];
  for (const token of tokens) {
    if (isOperator(token)) {
      if (stack.length < 2) throw new Error("malformed expression: operator without operands");
      const right = stack.pop();
      const left = stack.pop();
      stack.push(apply(token, left, right));
    } else {
      stack.push(Number.parseInt(token, 10));
    }
  }
  if (stack.length !== 1) throw new Error("malformed expression: stray operands");
  return stack[0];
}

function evaluate(expr) {
  return evaluatePostfix(infixToPostfix(tokenize(expr)));
}

module.exports = { evaluate, infixToPostfix, evaluatePostfix, tokenize };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(evaluate("3 + 4 * 2"), 11);
  assert.strictEqual(evaluate("( 1 + 2 ) * 3"), 9);
  assert.strictEqual(evaluate("10 / 3"), 3);
  assert.strictEqual(evaluate("2 * ( 3 + 4 ) - 5"), 9);
  assert.strictEqual(evaluate("( 0 - 7 ) / 2"), -3);
  assert.strictEqual(evaluate("100 / 7 / 2"), 7);
  assert.strictEqual(evaluate("7"), 7);
  assert.strictEqual(evaluate("( ( 2 + 3 ) * ( 4 - 1 ) )"), 15);
  assert.throws(() => evaluate("10 / 0"), /division by zero/);
  assert.throws(() => evaluate("( 1 + 2"), /unmatched/);
  assert.throws(() => evaluate("1 + 2 )"), /unmatched/);
  assert.throws(() => evaluate("( 1 + 2 ) )"), /unmatched/);
  assert.throws(() => evaluate("1 +"), /malformed/);
  assert.throws(() => evaluate(""), /malformed/);
  assert.throws(() => evaluate("3 @ 4"), /unknown token/);
  console.log("PASS");
}
