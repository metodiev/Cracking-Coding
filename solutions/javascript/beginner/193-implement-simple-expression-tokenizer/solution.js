/** Task 193 - Implement simple expression tokenizer. (Tier: Beginner, CTCI Ch 3) */
"use strict";

const OPERATORS = "+-*/";

class Token {
  constructor(type, value) {
    this.type = type;
    this.value = value;
  }
}

function tokenize(expr) {
  const tokens = [];
  let index = 0;
  while (index < expr.length) {
    const ch = expr[index];
    if (/\s/.test(ch)) {
      index++;
    } else if (/[0-9]/.test(ch)) {
      let end = index;
      while (end < expr.length && /[0-9]/.test(expr[end])) end++;
      tokens.push(new Token("INT", Number.parseInt(expr.slice(index, end), 10)));
      index = end;
    } else if (OPERATORS.includes(ch)) {
      tokens.push(new Token("OP", ch));
      index++;
    } else if (ch === "(") {
      tokens.push(new Token("LPAREN", ch));
      index++;
    } else if (ch === ")") {
      tokens.push(new Token("RPAREN", ch));
      index++;
    } else {
      throw new Error(`unknown character '${ch}' at index ${index}`);
    }
  }
  return tokens;
}

module.exports = { tokenize, Token };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(tokenize("12 + (3*4)"), [
    new Token("INT", 12), new Token("OP", "+"), new Token("LPAREN", "("),
    new Token("INT", 3), new Token("OP", "*"), new Token("INT", 4), new Token("RPAREN", ")"),
  ]);
  assert.deepStrictEqual(tokenize("1+2"), [new Token("INT", 1), new Token("OP", "+"), new Token("INT", 2)]);
  assert.deepStrictEqual(tokenize("  7  "), [new Token("INT", 7)]);
  assert.deepStrictEqual(tokenize(""), []);
  assert.deepStrictEqual(tokenize(" \t\n "), []);
  assert.deepStrictEqual(tokenize("10/2-3"), [
    new Token("INT", 10), new Token("OP", "/"), new Token("INT", 2),
    new Token("OP", "-"), new Token("INT", 3),
  ]);
  assert.throws(() => tokenize("12.5"), /unknown character '\.' at index 2/);
  assert.throws(() => tokenize("a+b"), /unknown character 'a' at index 0/);
  assert.throws(() => tokenize("1 $ 2"), /unknown character '\$' at index 2/);
  assert.throws(() => tokenize("#"), /unknown character '#' at index 0/);
  console.log("PASS");
}
