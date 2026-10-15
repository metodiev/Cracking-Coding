/** Task 388 - Implement string compression (run-length encoding + decode). (Tier: Intermediate, CTCI Ch 1) */
"use strict";

function isLetter(ch) {
  return (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z");
}

function isDigitChar(ch) {
  return ch >= "0" && ch <= "9";
}

/** Runs of length 1 stay bare; runs of length >= 2 become character + decimal count. */
function rleEncode(text) {
  let out = "";
  let i = 0;
  while (i < text.length) {
    let j = i;
    while (j < text.length && text[j] === text[i]) j++;
    const run = j - i;
    out += run >= 2 ? text[i] + String(run) : text[i];
    i = j;
  }
  return out;
}

/** Rejects non-canonical counts 0 and 1, and non-letter characters. */
function rleDecode(encoded) {
  let out = "";
  let i = 0;
  while (i < encoded.length) {
    const ch = encoded[i];
    if (!isLetter(ch)) throw new Error(`expected a letter at position ${i}`);
    i++;
    let count = 0;
    let hasDigits = false;
    while (i < encoded.length && isDigitChar(encoded[i])) {
      count = count * 10 + (encoded.charCodeAt(i) - 48);
      hasDigits = true;
      i++;
    }
    if (!hasDigits) {
      out += ch;
      continue;
    }
    if (count < 2) throw new Error(`count ${count} is not canonical`);
    out += ch.repeat(count);
  }
  return out;
}

/** First character of every maximal run, in order (the letters of the encoding). */
function runCharacters(text) {
  let out = "";
  for (let i = 0; i < text.length; i++) {
    if (i === 0 || text[i] !== text[i - 1]) out += text[i];
  }
  return out;
}

module.exports = { isLetter, isDigitChar, rleEncode, rleDecode, runCharacters };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(rleEncode("aaabbc"), "a3b2c");
  assert.strictEqual(rleDecode("a3b2c"), "aaabbc");

  // empty string, single characters, no runs
  assert.strictEqual(rleEncode(""), "");
  assert.strictEqual(rleDecode(""), "");
  assert.strictEqual(rleEncode("a"), "a");
  assert.strictEqual(rleDecode("a"), "a");
  assert.strictEqual(rleEncode("abcd"), "abcd");
  assert.strictEqual(rleDecode("abcd"), "abcd");

  // runs of exactly two keep their length; runs of three or more shrink
  assert.strictEqual(rleEncode("aab"), "a2b");
  assert.strictEqual(rleEncode("abb"), "ab2");
  assert.strictEqual(rleEncode("aabbaa"), "a2b2a2");
  assert.strictEqual(rleDecode("a2b2a2"), "aabbaa");
  assert.strictEqual(rleEncode("aaabccccd"), "a3bc4d");
  assert.strictEqual(rleDecode("a3bc4d"), "aaabccccd");
  assert.strictEqual(rleDecode("z2"), "zz");

  // multi-digit counts (the 9 -> 10 boundary)
  assert.strictEqual(rleEncode("a".repeat(9) + "b"), "a9b");
  assert.strictEqual(rleEncode("a".repeat(10)), "a10");
  assert.strictEqual(rleDecode("a10"), "a".repeat(10));
  assert.strictEqual(rleEncode("a".repeat(12)), "a12");
  assert.strictEqual(rleDecode("a12"), "a".repeat(12));
  assert.strictEqual(rleDecode("a99"), "a".repeat(99));

  // canonical outputs round trip through decode then encode
  for (const encoded of ["a3b2c", "a2b2a2", "a9b", "a10", "a12", "a3bc4d", "abcd", "a", "a99"]) {
    assert.strictEqual(rleEncode(rleDecode(encoded)), encoded, `canonical ${encoded}`);
  }

  // malformed input is rejected (documented)
  for (const bad of ["3", "a0", "a1", "ab1"]) {
    assert.throws(() => rleDecode(bad), `malformed rejected ${bad}`);
  }

  // LCG strings: round trips, canonical idempotence, and the letters of the encoding
  let seed = 5150;
  for (let trial = 0; trial < 200; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const length = seed % 60;
    seed = (seed * 75 + 74) % 65537;
    const alphabetSize = 2 + seed % 4;
    let text = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      text += String.fromCharCode("a".charCodeAt(0) + (seed % alphabetSize));
    }
    const encoded = rleEncode(text);
    assert.ok(encoded.length <= text.length, `encoding never grows ${trial}`);
    assert.strictEqual([...encoded].filter(isLetter).join(""), runCharacters(text),
      `run letters ${trial}`);
    assert.strictEqual(rleDecode(encoded), text, `round trip ${trial}`);
    assert.strictEqual(rleEncode(rleDecode(encoded)), encoded, `canonical idempotence ${trial}`);
  }

  console.log("PASS");
}
