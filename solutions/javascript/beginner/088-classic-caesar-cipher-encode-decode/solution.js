/** Task 088 - Classic Caesar cipher encode/decode. (Tier: Beginner, CTCI Ch 1)
 *
 * Shifts lowercase a-z with wrap-around; every shift is reduced mod 26 so large
 * and negative shifts work. Non-lowercase characters (including uppercase
 * letters, digits, spaces, and punctuation) are left unchanged, so
 * caesarDecode(caesarEncode(text, k), k) === text.
 */
"use strict";

function caesarEncode(text, shift) {
  const normalized = ((shift % 26) + 26) % 26;
  let out = "";
  for (const ch of text) {
    if (ch >= "a" && ch <= "z") {
      const code = ((ch.charCodeAt(0) - 97 + normalized) % 26) + 97;
      out += String.fromCharCode(code);
    } else {
      out += ch;
    }
  }
  return out;
}

function caesarDecode(text, shift) {
  return caesarEncode(text, -shift);
}

module.exports = { caesarEncode, caesarDecode };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(caesarEncode("abc", 3), "def");
  assert.strictEqual(caesarEncode("xyz", 3), "abc");
  assert.strictEqual(caesarEncode("hello, world!", 3), "khoor, zruog!");
  assert.strictEqual(caesarEncode("abc", -3), "xyz");
  assert.strictEqual(caesarEncode("abc", 29), "def");
  assert.strictEqual(caesarEncode("abc", 26), "abc");
  assert.strictEqual(caesarEncode("abc", -29), "xyz");
  assert.strictEqual(caesarEncode("", 5), "");

  assert.strictEqual(caesarDecode("khoor", 3), "hello");
  assert.strictEqual(caesarDecode("def", 3), "abc");
  assert.strictEqual(caesarDecode(caesarEncode("attack at dawn!", 7), 7), "attack at dawn!");
  assert.strictEqual(caesarDecode(caesarEncode("the quick brown fox 123", 13), 13), "the quick brown fox 123");

  // non-letters are unchanged; uppercase is not lowercase, so it is left alone
  assert.strictEqual(caesarEncode("a1 B2 c!", 1), "b1 B2 d!");
  assert.strictEqual(caesarEncode("Hello", 1), "Hfmmp");
  console.log("PASS");
}
