/** Task 391 - Implement adaptive Huffman (outline, conceptual). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function buildTree(freq, alphabet) {
  let seq = 0;
  const work = [];
  for (const ch of alphabet) work.push({ weight: freq.get(ch), seq: seq++, ch, left: null, right: null });
  while (work.length > 1) {
    work.sort((a, b) => (a.weight - b.weight) || (a.seq - b.seq));
    const x = work.shift();
    const y = work.shift();
    work.push({ weight: x.weight + y.weight, seq: seq++, ch: null, left: x, right: y });
  }
  return work[0];
}

function collectCodes(root, prefix = "", out = new Map()) {
  if (root.ch !== null) {
    out.set(root.ch, prefix);
    return out;
  }
  collectCodes(root.left, prefix + "0", out);
  collectCodes(root.right, prefix + "1", out);
  return out;
}

function adaptiveEncode(text, alphabet) {
  const freq = new Map();
  for (const ch of alphabet) freq.set(ch, 1);
  const bits = [];
  for (const ch of text) {
    if (!freq.has(ch)) throw new Error("symbol outside the alphabet: " + ch);
    const codes = collectCodes(buildTree(freq, alphabet));
    bits.push(codes.get(ch));
    freq.set(ch, freq.get(ch) + 1);
  }
  return bits.join("");
}

function adaptiveDecode(bits, alphabet) {
  const freq = new Map();
  for (const ch of alphabet) freq.set(ch, 1);
  const out = [];
  let i = 0;
  while (i < bits.length) {
    let node = buildTree(freq, alphabet);
    while (node.ch === null) {
      node = bits[i] === "0" ? node.left : node.right;
      i++;
    }
    out.push(node.ch);
    freq.set(node.ch, freq.get(node.ch) + 1);
  }
  return out.join("");
}

function lcgStrings(alphabet, length, count, seed = 6060) {
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

module.exports = { buildTree, collectCodes, adaptiveEncode, adaptiveDecode, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(adaptiveEncode("a", "abc"), "10");
  assert.strictEqual(adaptiveEncode("aaabbc", "abc"), "1001000110");
  assert.strictEqual(adaptiveEncode("aaa", "abc"), "1001");
  assert.strictEqual(adaptiveEncode("", "abc"), "");
  assert.strictEqual(adaptiveDecode("1001000110", "abc"), "aaabbc");

  for (const s of ["a", "abc", "aaabbc", "abababab", "cccbaaa", ...lcgStrings("abc", 24, 40)]) {
    assert.strictEqual(adaptiveDecode(adaptiveEncode(s, "abc"), "abc"), s, s);
  }

  const codes = collectCodes(buildTree(new Map([["a", 41], ["b", 2], ["c", 2]]), "abc"));
  assert.strictEqual(codes.get("a"), "1");
  assert.strictEqual(codes.get("b").length, 2);
  assert.strictEqual(codes.get("c").length, 2);

  assert.throws(() => adaptiveEncode("d", "abc"), /outside the alphabet/);
  console.log("PASS");
}
