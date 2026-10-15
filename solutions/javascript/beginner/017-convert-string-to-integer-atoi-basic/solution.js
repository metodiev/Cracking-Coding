/** Task 017 - Convert string to integer (atoi) basic. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function atoi(s) {
  if (s.length === 0) throw new Error("empty input");
  let i = 0, sign = 1;
  if (s[0] === "+" || s[0] === "-") {
    sign = s[0] === "-" ? -1 : 1;
    i = 1;
  }
  if (i === s.length) throw new Error("no digits");
  let value = 0;
  for (; i < s.length; i++) {
    const code = s.charCodeAt(i) - 48;
    if (code < 0 || code > 9) throw new Error("invalid char: " + s[i]);
    value = value * 10 + code;
  }
  return sign * value;
}

module.exports = { atoi };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(atoi("42"), 42);
  assert.strictEqual(atoi("-123"), -123);
  assert.strictEqual(atoi("+7"), 7);
  console.log("PASS");
}
