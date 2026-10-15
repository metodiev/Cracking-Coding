/** Task 393 - Implement a basic DEFLATE/Brotli idea demo (LZ77 + Huffman, simplified). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function lz77Tokens(text, window) {
  const n = text.length;
  const tokens = [];
  let i = 0;
  while (i < n) {
    let bestLen = 0;
    let bestDist = 0;
    for (let dist = 1; dist <= Math.min(window, i); dist++) {
      let length = 0;
      while (i + length < n && text[i + length] === text[i - dist + length] && length < 255) length++;
      if (length > bestLen) {
        bestLen = length;
        bestDist = dist;
      }
    }
    if (bestLen >= 3) {
      tokens.push([1, bestLen, bestDist]);
      i += bestLen;
    } else {
      tokens.push([0, text.charCodeAt(i), 0]);
      i++;
    }
  }
  return tokens;
}

function lz77Decompress(tokens) {
  const out = [];
  for (const [kind, a, b] of tokens) {
    if (kind === 0) out.push(a);
    else for (let k = 0; k < a; k++) out.push(out[out.length - b]);
  }
  return String.fromCharCode(...out);
}

function tokensToBytes(tokens) {
  const stream = [];
  for (const [kind, a, b] of tokens) {
    stream.push(kind, a);
    if (kind === 1) stream.push(b);
  }
  return stream;
}

function bytesToTokens(stream) {
  const tokens = [];
  let i = 0;
  while (i < stream.length) {
    if (stream[i] === 0) {
      tokens.push([0, stream[i + 1], 0]);
      i += 2;
    } else {
      tokens.push([1, stream[i + 1], stream[i + 2]]);
      i += 3;
    }
  }
  return tokens;
}

function buildHuffman(freq) {
  let seq = 0;
  const work = [];
  for (const byteValue of [...freq.keys()].sort((a, b) => a - b)) {
    work.push({ weight: freq.get(byteValue), seq: seq++, byteValue, left: null, right: null });
  }
  while (work.length > 1) {
    work.sort((a, b) => (a.weight - b.weight) || (a.seq - b.seq));
    const x = work.shift();
    const y = work.shift();
    work.push({ weight: x.weight + y.weight, seq: seq++, byteValue: null, left: x, right: y });
  }
  return work[0];
}

function collectCodes(root, prefix = "", out = new Map()) {
  if (root.byteValue !== null) {
    out.set(root.byteValue, prefix);
    return out;
  }
  collectCodes(root.left, prefix + "0", out);
  collectCodes(root.right, prefix + "1", out);
  return out;
}

function huffmanEncode(data) {
  if (data.length === 0) return { bits: "", codes: new Map() };
  const freq = new Map();
  for (const byteValue of data) freq.set(byteValue, (freq.get(byteValue) || 0) + 1);
  const codes = collectCodes(buildHuffman(freq));
  let bits = "";
  for (const byteValue of data) bits += codes.get(byteValue);
  return { bits, codes };
}

function huffmanDecode(bits, codes) {
  const reverse = new Map();
  for (const [byteValue, code] of codes) reverse.set(code, byteValue);
  const out = [];
  let current = "";
  for (const bit of bits) {
    current += bit;
    if (reverse.has(current)) {
      out.push(reverse.get(current));
      current = "";
    }
  }
  if (current !== "") throw new Error("truncated code stream");
  return out;
}

function isPrefixFree(codes) {
  const table = [...codes.values()];
  for (let i = 0; i < table.length; i++) {
    for (let j = 0; j < table.length; j++) {
      if (i !== j && table[j].startsWith(table[i])) return false;
    }
  }
  return true;
}

function compressText(text, window = 8) {
  return huffmanEncode(tokensToBytes(lz77Tokens(text, window)));
}

function decompressText(bits, codes) {
  return lz77Decompress(bytesToTokens(huffmanDecode(bits, codes)));
}

function lcgStrings(alphabet, length, count, seed = 3131) {
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

const key = (tokens) => tokens.map((t) => t.join(",")).join(";");

module.exports = { lz77Tokens, lz77Decompress, tokensToBytes, bytesToTokens, huffmanEncode, huffmanDecode, isPrefixFree, compressText, decompressText, lcgStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(key(lz77Tokens("abcabcabc", 8)), "0,97,0;0,98,0;0,99,0;1,6,3");
  assert.strictEqual(key(lz77Tokens("aaaaaaaa", 8)), "0,97,0;1,7,1");
  assert.strictEqual(key(lz77Tokens("abc".repeat(10), 8)), "0,97,0;0,98,0;0,99,0;1,27,3");
  assert.deepStrictEqual(lz77Tokens("", 8), []);
  assert.strictEqual(key(lz77Tokens("ab", 8)), "0,97,0;0,98,0");

  assert.strictEqual(lz77Decompress([[0, 97, 0], [0, 98, 0], [0, 99, 0], [1, 6, 3]]), "abcabcabc");
  assert.strictEqual(lz77Decompress([[0, 97, 0], [1, 7, 1]]), "aaaaaaaa");

  assert.deepStrictEqual(tokensToBytes([[0, 97, 0], [1, 6, 3]]), [0, 97, 1, 6, 3]);
  assert.strictEqual(key(bytesToTokens([0, 97, 1, 6, 3])), "0,97,0;1,6,3");

  const samples = ["abcabcabc", "aaaaaaaa", "abc".repeat(10), "", "ab", "aabaabaa", ...lcgStrings("abc", 18, 30)];
  for (const s of samples) {
    assert.strictEqual(lz77Decompress(lz77Tokens(s, 8)), s, s);
    assert.strictEqual(key(bytesToTokens(tokensToBytes(lz77Tokens(s, 8)))), key(lz77Tokens(s, 8)), s);
  }
  for (const s of samples) {
    const { bits, codes } = compressText(s, 8);
    assert.ok(isPrefixFree(codes), s);
    const weighted = tokensToBytes(lz77Tokens(s, 8)).reduce((acc, byteValue) => acc + codes.get(byteValue).length, 0);
    assert.strictEqual(bits.length, weighted, s);
    assert.strictEqual(decompressText(bits, codes), s, s);
  }
  for (const text of ["abcabcabc", "aaaaaaaa", "abc".repeat(10)]) {
    const { bits } = compressText(text, 8);
    assert.ok(bits.length < 8 * text.length, text);
  }
  assert.strictEqual(key(lz77Tokens("abcabcabc", 2)), "0,97,0;0,98,0;0,99,0;0,97,0;0,98,0;0,99,0;0,97,0;0,98,0;0,99,0");
  console.log("PASS");
}
