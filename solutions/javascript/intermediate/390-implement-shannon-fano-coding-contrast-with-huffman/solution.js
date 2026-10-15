/** Task 390 - Implement Shannon-Fano coding (contrast with Huffman). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
"use strict";

function characterFrequencies(text) {
  const frequencies = new Map();
  for (const ch of text) frequencies.set(ch, (frequencies.get(ch) || 0) + 1);
  return frequencies;
}

/**
 * Top-down split: frequency-descending order (ties by character), cut where the halves'
 * total-frequency difference is minimal (earliest split on ties), '0' left, '1' right.
 */
function shannonFanoCodes(frequencies) {
  const codes = new Map();
  if (!frequencies.size) return codes;
  const items = [...frequencies.entries()]
    .sort((x, y) => y[1] - x[1] || (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0));
  const rec = (part, prefix) => {
    if (part.length === 1) {
      codes.set(part[0][0], prefix.length ? prefix : "0");
      return;
    }
    const total = part.reduce((sum, entry) => sum + entry[1], 0);
    let bestSplit = 1;
    let bestDiff = null;
    let leftSum = 0;
    for (let k = 1; k < part.length; k++) {
      leftSum += part[k - 1][1];
      const diff = Math.abs(total - 2 * leftSum);
      if (bestDiff === null || diff < bestDiff) {
        bestSplit = k;
        bestDiff = diff;
      }
    }
    rec(part.slice(0, bestSplit), prefix + "0");
    rec(part.slice(bestSplit), prefix + "1");
  };
  rec(items, "");
  return codes;
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

/** Local Huffman re-implementation (same tie-break as task 389). */
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

function encode(text, codes) {
  let bits = "";
  for (const ch of text) bits += codes.get(ch);
  return bits;
}

function decode(bits, codes) {
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

module.exports = { characterFrequencies, shannonFanoCodes, huffmanCodes, weightedLength, encode,
  decode, isPrefixFree, kraftSumOk };

if (require.main === module) {
  const assert = require("node:assert");

  // tie example: aabbbcccc -> both codecs cost 14 bits
  const tieFreq = new Map([["a", 2], ["b", 3], ["c", 4]]);
  const sf = shannonFanoCodes(tieFreq);
  const huff = huffmanCodes(tieFreq);
  assert.strictEqual(sf.size, 3);
  assert.strictEqual(sf.get("c"), "0");
  assert.strictEqual(sf.get("b"), "10");
  assert.strictEqual(sf.get("a"), "11");
  assert.strictEqual(huff.get("c"), "0");
  assert.strictEqual(huff.get("a"), "10");
  assert.strictEqual(huff.get("b"), "11");
  assert.strictEqual(weightedLength(sf, tieFreq), 14);
  assert.strictEqual(weightedLength(huff, tieFreq), 14);
  assert.strictEqual(weightedLength(sf, tieFreq), weightedLength(huff, tieFreq)); // documented tie
  for (const codes of [sf, huff]) {
    assert.ok(isPrefixFree(codes) && kraftSumOk(codes));
    assert.strictEqual(decode(encode("aabbbcccc", codes), codes), "aabbbcccc");
  }

  // strict example: Huffman 230 < Shannon-Fano 231
  const bigFreq = new Map([["a", 35], ["b", 17], ["c", 17], ["d", 16], ["e", 15]]);
  const sfBig = shannonFanoCodes(bigFreq);
  const huffBig = huffmanCodes(bigFreq);
  assert.strictEqual(sfBig.size, 5);
  assert.strictEqual(sfBig.get("a"), "00");
  assert.strictEqual(sfBig.get("b"), "01");
  assert.strictEqual(sfBig.get("c"), "10");
  assert.strictEqual(sfBig.get("d"), "110");
  assert.strictEqual(sfBig.get("e"), "111");
  assert.strictEqual(huffBig.get("a"), "0");
  assert.strictEqual(huffBig.get("e"), "100");
  assert.strictEqual(huffBig.get("d"), "101");
  assert.strictEqual(huffBig.get("b"), "110");
  assert.strictEqual(huffBig.get("c"), "111");
  assert.strictEqual(weightedLength(sfBig, bigFreq), 231);
  assert.strictEqual(weightedLength(huffBig, bigFreq), 230);
  assert.ok(weightedLength(huffBig, bigFreq) < weightedLength(sfBig, bigFreq));
  assert.strictEqual(sfBig.get("a").length, 2);
  assert.strictEqual(huffBig.get("a").length, 1);
  for (const codes of [sfBig, huffBig]) {
    assert.ok(isPrefixFree(codes) && kraftSumOk(codes));
  }

  // edge cases
  assert.strictEqual(shannonFanoCodes(new Map()).size, 0);
  assert.strictEqual(huffmanCodes(new Map()).size, 0);
  const single = new Map([["a", 4]]);
  assert.strictEqual(shannonFanoCodes(single).get("a"), "0");
  assert.strictEqual(huffmanCodes(single).get("a"), "0");
  assert.strictEqual(decode("", new Map()), "");
  assert.strictEqual(encode("", new Map()), "");

  // LCG strings: both codecs round trip, prefix-free, and Huffman <= Shannon-Fano
  let seed = 8642;
  let tieCount = 0;
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
    const freq = characterFrequencies(sample);
    const sfCodes = shannonFanoCodes(freq);
    const huffCodes = huffmanCodes(freq);
    for (const codes of [sfCodes, huffCodes]) {
      assert.ok(isPrefixFree(codes) && kraftSumOk(codes), `LCG prefix ${trial}`);
      const bits = encode(sample, codes);
      assert.strictEqual(decode(bits, codes), sample, `LCG round trip ${trial}`);
      assert.strictEqual(bits.length, weightedLength(codes, freq), `LCG weight ${trial}`);
    }
    const sfWeight = weightedLength(sfCodes, freq);
    const huffWeight = weightedLength(huffCodes, freq);
    assert.ok(huffWeight <= sfWeight, `huffman <= shannon-fano ${trial}`);
    if (huffWeight === sfWeight) tieCount++;
  }
  assert.ok(tieCount > 0, "ties are observed among the samples");

  console.log("PASS");
}
