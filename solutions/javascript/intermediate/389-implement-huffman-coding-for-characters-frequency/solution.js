/** Task 389 - Implement Huffman coding for characters frequency. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function characterFrequencies(text) {
  const frequencies = new Map();
  for (const ch of text) frequencies.set(ch, (frequencies.get(ch) || 0) + 1);
  return frequencies;
}

function less(a, b) {
  return a.freq !== b.freq ? a.freq < b.freq : a.id < b.id;
}

function heapPush(heap, node) {
  heap.push(node);
  let i = heap.length - 1;
  while (i > 0) {
    const parent = (i - 1) >> 1;
    if (less(heap[parent], heap[i])) break;
    [heap[parent], heap[i]] = [heap[i], heap[parent]];
    i = parent;
  }
}

function heapPop(heap) {
  const top = heap[0];
  const last = heap.pop();
  if (heap.length) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      const left = 2 * i + 1;
      const right = 2 * i + 2;
      let smallest = i;
      if (left < heap.length && less(heap[left], heap[smallest])) smallest = left;
      if (right < heap.length && less(heap[right], heap[smallest])) smallest = right;
      if (smallest === i) break;
      [heap[smallest], heap[i]] = [heap[i], heap[smallest]];
      i = smallest;
    }
  }
  return top;
}

/**
 * Greedy min-heap merge with the documented deterministic tie-break:
 * nodes ordered by (frequency, creation id); first popped node becomes the left
 * child (bit '0'), the second the right child (bit '1').
 */
function huffmanCodes(frequencies) {
  const codes = new Map();
  if (!frequencies.size) return codes;
  const chars = [...frequencies.keys()].sort();
  const heap = [];
  let id = 0;
  for (const ch of chars) heapPush(heap, { freq: frequencies.get(ch), id: id++, ch, leaf: true });
  let nextId = id;
  while (heap.length > 1) {
    const left = heapPop(heap);
    const right = heapPop(heap);
    heapPush(heap, { freq: left.freq + right.freq, id: nextId++, ch: null, leaf: false, left, right });
  }
  const walk = (node, prefix) => {
    if (node.leaf) {
      codes.set(node.ch, prefix.length ? prefix : "0");
      return;
    }
    walk(node.left, prefix + "0");
    walk(node.right, prefix + "1");
  };
  walk(heap[0], "");
  return codes;
}

function weightedLength(codes, frequencies) {
  let total = 0;
  for (const [ch, code] of codes) total += frequencies.get(ch) * code.length;
  return total;
}

function huffmanEncode(text, codes) {
  let bits = "";
  for (const ch of text) bits += codes.get(ch);
  return bits;
}

function huffmanDecode(bits, codes) {
  const reverse = new Map();
  for (const [ch, code] of codes) reverse.set(code, ch);
  let out = "";
  let current = "";
  for (const bit of bits) {
    current += bit;
    if (reverse.has(current)) {
      out += reverse.get(current);
      current = "";
    }
  }
  if (current.length) throw new Error("bits end with an incomplete code");
  return out;
}

function isPrefixFree(codes) {
  for (const code of codes.values()) {
    for (const other of codes.values()) {
      if (code !== other && other.startsWith(code)) return false;
    }
  }
  return true;
}

/** sum 2^(-len) == 1 for a complete binary tree; the single-symbol table "0" is a lone leaf. */
function kraftSumOk(codes) {
  if (!codes.size) return true;
  if (codes.size === 1) {
    for (const code of codes.values()) return code === "0";
  }
  let longest = 0;
  for (const code of codes.values()) longest = Math.max(longest, code.length);
  let sum = 0;
  for (const code of codes.values()) sum += 2 ** (longest - code.length);
  return sum === 2 ** longest;
}

function ceilLog2(value) {
  let bits = 0;
  while (2 ** bits < value) bits++;
  return bits;
}

module.exports = { characterFrequencies, huffmanCodes, weightedLength, huffmanEncode, huffmanDecode,
  isPrefixFree, kraftSumOk, ceilLog2 };

if (require.main === module) {
  const assert = require("node:assert");

  // hand-verified small example: a:2, b:3, c:4 -> {c: "0", a: "10", b: "11"}
  const text = "aabbbcccc";
  const freq = characterFrequencies(text);
  assert.strictEqual(freq.get("a"), 2);
  assert.strictEqual(freq.get("b"), 3);
  assert.strictEqual(freq.get("c"), 4);
  const codes = huffmanCodes(freq);
  assert.strictEqual(codes.size, 3);
  assert.strictEqual(codes.get("a"), "10");
  assert.strictEqual(codes.get("b"), "11");
  assert.strictEqual(codes.get("c"), "0");
  const weighted = weightedLength(codes, freq);
  assert.strictEqual(weighted, 14);
  assert.strictEqual(weighted, 2 * 2 + 3 * 2 + 4 * 1);
  const bits = huffmanEncode(text, codes);
  assert.strictEqual(bits, "10101111110000");
  assert.strictEqual(bits.length, weighted);
  assert.strictEqual(huffmanDecode(bits, codes), text);

  // single distinct character has the single code "0"
  const freq1 = characterFrequencies("aaaa");
  const codes1 = huffmanCodes(freq1);
  assert.strictEqual(codes1.size, 1);
  assert.strictEqual(codes1.get("a"), "0");
  assert.strictEqual(weightedLength(codes1, freq1), 4);
  const bits1 = huffmanEncode("aaaa", codes1);
  assert.strictEqual(bits1, "0000");
  assert.strictEqual(huffmanDecode(bits1, codes1), "aaaa");

  // empty text
  const emptyCodes = huffmanCodes(new Map());
  assert.strictEqual(emptyCodes.size, 0);
  assert.strictEqual(huffmanEncode("", emptyCodes), "");
  assert.strictEqual(huffmanDecode("", emptyCodes), "");

  // second hand-verified structure, built straight from frequencies
  const bigFreq = new Map([["a", 35], ["b", 17], ["c", 17], ["d", 16], ["e", 15]]);
  const bigCodes = huffmanCodes(bigFreq);
  assert.strictEqual(bigCodes.size, 5);
  assert.strictEqual(bigCodes.get("a"), "0");
  assert.strictEqual(bigCodes.get("e"), "100");
  assert.strictEqual(bigCodes.get("d"), "101");
  assert.strictEqual(bigCodes.get("b"), "110");
  assert.strictEqual(bigCodes.get("c"), "111");
  assert.strictEqual(weightedLength(bigCodes, bigFreq), 230);
  assert.strictEqual(weightedLength(bigCodes, bigFreq), 35 * 1 + (17 + 17 + 16 + 15) * 3);

  // prefix-free property and the Kraft equality (a complete binary tree)
  for (const table of [codes, codes1, bigCodes]) {
    assert.ok(isPrefixFree(table), "prefix free");
    assert.ok(kraftSumOk(table), "kraft");
  }

  // decoding rejects a dangling prefix (documented error)
  assert.throws(() => huffmanDecode("1", codes));

  // LCG strings: round trips, bit length equals the weighted path length
  let seed = 2718;
  for (let trial = 0; trial < 120; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const length = seed % 80;
    seed = (seed * 75 + 74) % 65537;
    const alphabetSize = 1 + seed % 6;
    let sample = "";
    for (let i = 0; i < length; i++) {
      seed = (seed * 75 + 74) % 65537;
      sample += String.fromCharCode(97 + (seed % alphabetSize));
    }
    const sampleFreq = characterFrequencies(sample);
    const sampleCodes = huffmanCodes(sampleFreq);
    const sampleBits = huffmanEncode(sample, sampleCodes);
    const sampleWeight = weightedLength(sampleCodes, sampleFreq);
    assert.strictEqual(huffmanDecode(sampleBits, sampleCodes), sample, `LCG round trip ${trial}`);
    assert.strictEqual(sampleBits.length, sampleWeight, `bit length ${trial}`);
    assert.ok(isPrefixFree(sampleCodes) && kraftSumOk(sampleCodes), `LCG prefix ${trial}`);
    if (length > 0) {
      const distinct = new Set(sample).size;
      assert.ok(sampleWeight <= length * Math.max(1, ceilLog2(distinct)),
        `fixed-length bound ${trial}`);
    }
  }

  console.log("PASS");
}
