/** Task 288 - Generate all balanced parentheses strings. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * Open-first DFS: at each step add "(" if some pair is still unopened, then ")" if the
 * closing count is below the opening count. This emits strings in lexicographic order
 * ("(" < ")"), so results are deterministic. Counts must equal the Catalan numbers and
 * every string is validated with an independent balance checker.
 */
function generateParentheses(n) {
  if (n < 0) return [];
  const results = [];

  const build = (current, opens, closes) => {
    if (current.length === 2 * n) {
      results.push(current);
      return;
    }
    if (opens < n) build(`${current}(`, opens + 1, closes);
    if (closes < opens) build(`${current})`, opens, closes + 1);
  };

  build("", 0, 0);
  return results;
}

/** Independent checker: only parentheses, no negative prefix, ends at depth 0. */
function isBalanced(text) {
  let depth = 0;
  for (const char of text) {
    if (char === "(") {
      depth++;
    } else if (char === ")") {
      depth--;
      if (depth < 0) return false;
    } else {
      return false;
    }
  }
  return depth === 0;
}

/** C(0)=1, C(n+1) = C(n) * 2 * (2n + 1) / (n + 2). */
function catalan(n) {
  let value = 1;
  for (let i = 0; i < n; i++) value = (value * 2 * (2 * i + 1)) / (i + 2);
  return value;
}

/** Enumerates all 2^(2n) parenthesis strings and keeps the balanced ones. */
function bruteForceBalanced(n) {
  const results = [];
  for (let mask = 0; mask < 1 << (2 * n); mask++) {
    let text = "";
    for (let bit = 0; bit < 2 * n; bit++) text += (mask >> bit) & 1 ? "(" : ")";
    if (isBalanced(text)) results.push(text);
  }
  results.sort();
  return results;
}

module.exports = { generateParentheses, isBalanced, catalan, bruteForceBalanced };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(generateParentheses(0), [""]);
  assert.deepStrictEqual(generateParentheses(1), ["()"]);
  assert.deepStrictEqual(generateParentheses(2), ["(())", "()()"]);
  assert.deepStrictEqual(generateParentheses(3), ["((()))", "(()())", "(())()", "()(())", "()()()"]);
  assert.deepStrictEqual(generateParentheses(-1), []);

  for (let n = 0; n <= 10; n++) {
    const strings = generateParentheses(n);
    assert.strictEqual(strings.length, catalan(n), `catalan ${n}`);
    for (let i = 1; i < strings.length; i++) {
      assert.ok(strings[i - 1] < strings[i], `lexicographic ${n}`);
    }
    for (const text of strings) {
      assert.strictEqual(text.length, 2 * n, `length ${n}`);
      assert.ok(isBalanced(text), `balanced ${n} ${text}`);
      const opens = [...text].filter((c) => c === "(").length;
      assert.strictEqual(opens, n, `pairs ${n}`);
      assert.strictEqual(text.length - opens, n, `pairs close ${n}`);
    }
    assert.deepStrictEqual(generateParentheses(n), strings, `deterministic ${n}`);
  }

  for (let n = 0; n <= 7; n++) {
    assert.deepStrictEqual(generateParentheses(n), bruteForceBalanced(n), `oracle ${n}`);
  }

  assert.strictEqual(catalan(0), 1);
  assert.strictEqual(catalan(1), 1);
  assert.strictEqual(catalan(2), 2);
  assert.strictEqual(catalan(3), 5);
  assert.strictEqual(catalan(4), 14);
  assert.strictEqual(catalan(5), 42);
  assert.strictEqual(catalan(8), 1430);
  assert.strictEqual(catalan(10), 16796);

  console.log("PASS");
}
