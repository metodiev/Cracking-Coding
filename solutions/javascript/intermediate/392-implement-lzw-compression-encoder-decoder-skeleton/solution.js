/** Task 392 - Implement LZW compression (encoder/decoder skeleton). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function lzwEncode(text, alphabet) {
  const dictionary = new Map();
  for (let i = 0; i < alphabet.length; i++) dictionary.set(alphabet[i], i);
  let nextCode = dictionary.size;
  const codes = [];
  let current = "";
  for (const ch of text) {
    const candidate = current + ch;
    if (dictionary.has(candidate)) {
      current = candidate;
    } else {
      if (current === "") throw new Error("symbol outside the alphabet: " + ch);
      codes.push(dictionary.get(current));
      dictionary.set(candidate, nextCode++);
      current = ch;
      if (!dictionary.has(ch)) throw new Error("symbol outside the alphabet: " + ch);
    }
  }
  if (current !== "") codes.push(dictionary.get(current));
  return codes;
}

function lzwDecode(codes, alphabet) {
  const dictionary = new Map();
  for (let i = 0; i < alphabet.length; i++) dictionary.set(i, alphabet[i]);
  let nextCode = dictionary.size;
  if (codes.length === 0) return "";
  const first = codes[0];
  if (!dictionary.has(first)) throw new Error("invalid first code");
  let previous = dictionary.get(first);
  let out = previous;
  for (let k = 1; k < codes.length; k++) {
    const code = codes[k];
    let entry;
    if (dictionary.has(code)) entry = dictionary.get(code);
    else if (code === nextCode) entry = previous + previous[0];   // the cSc rule
    else throw new Error("invalid code: " + code);
    out += entry;
    dictionary.set(nextCode, previous + entry[0]);
    nextCode++;
    previous = entry;
  }
  return out;
}

function lcgStrings(alphabet, length, count, seed = 5150) {
  const strings = [];
  for (let k = 0; k < count; k++) {
    let s = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      s += alphabet[seed % alphabet.length];
    }
    strings.push(s);
  }
  return strings;
}

module.exports = { lzwEncode, lzwDecode, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(lzwEncode("abababa", "ab"), [0, 1, 2, 4]);
  assert.deepStrictEqual(lzwEncode("aaaa", "ab"), [0, 2, 0]);
  assert.deepStrictEqual(lzwEncode("abcabcabcabc", "abc"), [0, 1, 2, 3, 5, 4, 6]);
  assert.deepStrictEqual(lzwEncode("", "ab"), []);
  assert.deepStrictEqual(lzwEncode("a", "ab"), [0]);

  assert.strictEqual(lzwDecode([0, 1, 2, 4], "ab"), "abababa");
  assert.strictEqual(lzwDecode([0, 2, 0], "ab"), "aaaa");
  assert.strictEqual(lzwDecode([0, 1, 2, 3, 5, 4, 6], "abc"), "abcabcabcabc");
  assert.strictEqual(lzwDecode([], "ab"), "");
  assert.strictEqual(lzwDecode([0], "ab"), "a");

  for (const s of ["abababa", "aaaa", "", "a", ...lcgStrings("ab", 30, 40)]) {
    assert.strictEqual(lzwDecode(lzwEncode(s, "ab"), "ab"), s, s);
  }
  for (const s of lcgStrings("abcd", 40, 40)) {
    assert.strictEqual(lzwDecode(lzwEncode(s, "abcd"), "abcd"), s, s);
  }

  const repetitive = "ab".repeat(20);
  assert.ok(lzwEncode(repetitive, "ab").length < repetitive.length);

  const csc = lzwEncode("abababab", "ab");
  assert.strictEqual(lzwDecode(csc, "ab"), "abababab");
  console.log("PASS");
}
