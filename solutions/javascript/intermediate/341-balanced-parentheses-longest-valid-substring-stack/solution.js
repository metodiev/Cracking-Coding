/**
 * Task 341 - Longest valid parentheses substring (stack of indices). (Tier: Intermediate, CTCI Ch 8)
 *
 * Two independent implementations - a stack of indices and the DP recurrence - cross-checked
 * against a brute-force oracle on hand cases and LCG-generated strings.
 */
"use strict";

function longestValidSubstringStack(s) {
  const stack = [-1];                  // base: position of the last unmatched ')'
  let best = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "(") {
      stack.push(i);
    } else {
      stack.pop();
      if (stack.length === 0) stack.push(i);          // this ')' becomes the new base
      else best = Math.max(best, i - stack[stack.length - 1]);
    }
  }
  return best;
}

function longestValidSubstringDp(s) {
  const dp = new Array(s.length).fill(0);
  let best = 0;
  for (let i = 0; i < s.length; i++) {
    if (s[i] !== ")") continue;
    if (i >= 1 && s[i - 1] === "(") {
      dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
    } else if (i >= 1 && dp[i - 1] > 0) {
      const j = i - dp[i - 1] - 1;
      if (j >= 0 && s[j] === "(") dp[i] = dp[i - 1] + 2 + (j >= 1 ? dp[j - 1] : 0);
    }
    best = Math.max(best, dp[i]);
  }
  return best;
}

function longestValidSubstringBrute(s) {
  let best = 0;
  for (let i = 0; i < s.length; i++) {
    let depth = 0;
    for (let j = i; j < s.length; j++) {
      depth += s[j] === "(" ? 1 : -1;
      if (depth < 0) break;
      if (depth === 0) best = Math.max(best, j - i + 1);
    }
  }
  return best;
}

function lcgStrings(count, maxLen, seedIn = 34142) {
  let seed = seedIn;
  const strings = [];
  for (let k = 0; k < count; k++) {
    seed = (seed * 75 + 74) % 65537;
    const length = seed % (maxLen + 1);
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += "()"[seed % 2];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { longestValidSubstringStack, longestValidSubstringDp, longestValidSubstringBrute, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(longestValidSubstringStack("(()"), 2);
  assert.strictEqual(longestValidSubstringStack(")()())"), 4);
  assert.strictEqual(longestValidSubstringStack(""), 0);
  assert.strictEqual(longestValidSubstringStack("()(())"), 6);
  assert.strictEqual(longestValidSubstringStack("()(()"), 2);
  assert.strictEqual(longestValidSubstringStack("((())"), 4);
  assert.strictEqual(longestValidSubstringStack("())"), 2);
  assert.strictEqual(longestValidSubstringStack(")("), 0);
  assert.strictEqual(longestValidSubstringStack("((("), 0);
  assert.strictEqual(longestValidSubstringStack("()"), 2);
  assert.strictEqual(longestValidSubstringStack("()()"), 4);
  assert.strictEqual(longestValidSubstringStack(")()("), 2);

  const hand = ["(()", ")()())", "", "()(())", "()(()", "((())", "())", ")(", "(((", "()", "()()", ")()("];
  for (const s of hand) {
    assert.strictEqual(longestValidSubstringDp(s), longestValidSubstringStack(s), `dp vs stack ${s}`);
    assert.strictEqual(longestValidSubstringBrute(s), longestValidSubstringStack(s), `brute vs stack ${s}`);
  }

  for (const s of lcgStrings(150, 24)) {
    const expected = longestValidSubstringBrute(s);
    assert.strictEqual(longestValidSubstringStack(s), expected, `stack ${s}`);
    assert.strictEqual(longestValidSubstringDp(s), expected, `dp ${s}`);
  }
  console.log("PASS");
}
