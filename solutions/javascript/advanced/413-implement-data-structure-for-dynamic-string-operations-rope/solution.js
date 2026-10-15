/** Task 413 - Implement a data structure for dynamic string operations (rope). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A rope stores a string as a sequence of chunks (leaves of a binary tree). The real
 * structure is a balanced tree of chunks with O(log n) split/concat; this implementation
 * keeps the chunk sequence FLAT and balanced instead, a documented simulation with the
 * same interface:
 *   * chunks hold at most CHUNK = 64 characters and are rebalanced so their count stays
 *     within ~2x of the ideal ceil(n / 64),
 *   * index/charAt/slice locate by walking the chunk lengths (O(#chunks + CHUNK)),
 *   * insert/delete/split/concat/reverse splice chunks and rebalance (O(n + CHUNK));
 *     a tree-based rope would perform the same operations by re-linking subtrees.
 * All operations are functional: they return NEW ropes, leaving the receiver untouched.
 */
"use strict";

const CHUNK = 64;

function normalize(chunks) {
  const whole = chunks.filter((c) => c.length > 0).join("");
  const n = whole.length;
  if (n === 0) return [];
  const target = Math.max(1, Math.ceil(n / CHUNK));
  const size = Math.ceil(n / target);
  const out = [];
  for (let i = 0; i < n; i += size) out.push(whole.slice(i, Math.min(n, i + size)));
  return out;
}

function maybeRebalance(chunks) {
  const kept = chunks.filter((c) => c.length > 0);
  const n = kept.reduce((acc, c) => acc + c.length, 0);
  const target = n > 0 ? Math.max(1, Math.ceil(n / CHUNK)) : 1;
  if (kept.length > 2 * target + 2) return normalize(kept);
  return kept;
}

class Rope {
  constructor(text = "") {
    this.chunks = normalize(text.length > 0 ? [text] : []);
  }

  static of(chunks) {
    const r = new Rope("");
    r.chunks = maybeRebalance(chunks);
    return r;
  }

  get length() {
    return this.chunks.reduce((acc, c) => acc + c.length, 0);
  }

  to_string() {
    return this.chunks.join("");
  }

  /** [chunk index, offset] for pos; offset may equal the chunk length at a boundary. */
  locate(pos) {
    for (let idx = 0; idx < this.chunks.length; idx++) {
      const len = this.chunks[idx].length;
      if (pos <= len) return [idx, pos];
      pos -= len;
    }
    return [this.chunks.length, 0];
  }

  charAt(pos) {
    if (pos < 0 || pos >= this.length) throw new RangeError("rope index out of range");
    for (const c of this.chunks) {
      if (pos < c.length) return c[pos];
      pos -= c.length;
    }
    throw new RangeError("rope index out of range");
  }

  slice(start, end) {
    if (start < 0 || end > this.length || start > end) throw new RangeError("rope slice out of range");
    let out = "";
    let need = end - start;
    let [idx, off] = this.locate(start);
    while (need > 0) {
      const c = this.chunks[idx];
      const take = Math.min(need, c.length - off);
      out += c.slice(off, off + take);
      need -= take;
      idx++;
      off = 0;
    }
    return out;
  }

  insert(pos, text) {
    if (pos < 0 || pos > this.length) throw new RangeError("rope insert out of range");
    const [idx, off] = this.locate(pos);
    const out = this.chunks.slice(0, idx);
    if (off > 0) out.push(this.chunks[idx].slice(0, off));
    if (text.length > 0) out.push(...normalize([text]));
    if (idx < this.chunks.length && off < this.chunks[idx].length) {
      out.push(this.chunks[idx].slice(off));
    }
    if (idx < this.chunks.length) out.push(...this.chunks.slice(idx + 1));
    return Rope.of(out);
  }

  delete(start, end) {
    if (start < 0 || end > this.length || start > end) throw new RangeError("rope delete out of range");
    if (start === end) return Rope.of(this.chunks.slice());
    const [li, lo] = this.locate(start);
    const [ri, ro] = this.locate(end);
    const out = this.chunks.slice(0, li);
    if (lo > 0) out.push(this.chunks[li].slice(0, lo));
    if (ro < this.chunks[ri].length) out.push(this.chunks[ri].slice(ro));
    out.push(...this.chunks.slice(ri + 1));
    return Rope.of(out);
  }

  split(pos) {
    if (pos < 0 || pos > this.length) throw new RangeError("rope split out of range");
    const [idx, off] = this.locate(pos);
    const left = this.chunks.slice(0, idx);
    if (off > 0) left.push(this.chunks[idx].slice(0, off));
    const right = this.chunks.slice(idx + 1);
    if (idx < this.chunks.length && off < this.chunks[idx].length) {
      right.unshift(this.chunks[idx].slice(off));
    }
    return [Rope.of(left), Rope.of(right)];
  }

  concat(other) {
    return Rope.of(this.chunks.concat(other.chunks));
  }

  reverse() {
    return Rope.of(this.chunks.slice().reverse().map((c) => c.split("").reverse().join("")));
  }
}

class Lcg {
  constructor(state) {
    this.state = state;
  }

  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state;
  }
}

module.exports = { Rope, Lcg, CHUNK, normalize, maybeRebalance };

if (require.main === module) {
  const assert = require("node:assert");

  const r = new Rope("hello");
  assert.strictEqual(r.length, 5);
  assert.strictEqual(r.charAt(0), "h");
  assert.strictEqual(r.charAt(4), "o");
  assert.strictEqual(r.charAt(2), "l");
  assert.strictEqual(r.to_string(), "hello");
  assert.strictEqual(r.slice(1, 4), "ell");
  assert.strictEqual(new Rope("abef").insert(2, "cd").to_string(), "abcdef");
  assert.strictEqual(new Rope("hello").insert(0, "xyz").to_string(), "xyzhello");
  // "xyzhello" (8 chars) minus [3, 6) 'h','e','l' leaves "xyzlo"
  assert.strictEqual(new Rope("hello").insert(0, "xyz").delete(3, 6).to_string(), "xyzlo");
  const [left, right] = new Rope("xyzlo").split(2);
  assert.strictEqual(left.to_string(), "xy");
  assert.strictEqual(right.to_string(), "zlo");
  assert.strictEqual(left.concat(right).to_string(), "xyzlo");
  assert.strictEqual(new Rope("xyzlo").reverse().to_string(), "olzyx");
  assert.strictEqual(new Rope("").length, 0);
  assert.strictEqual(new Rope("").to_string(), "");
  assert.strictEqual(new Rope("").insert(0, "abc").to_string(), "abc");
  assert.strictEqual(new Rope("abc").delete(1, 2).to_string(), "ac");
  assert.strictEqual(new Rope("abc").delete(0, 0).to_string(), "abc");
  assert.strictEqual(new Rope("abc").split(0)[1].to_string(), "abc");
  assert.strictEqual(new Rope("abc").split(3)[0].to_string(), "abc");
  // functional semantics: the receiver never changes
  const original = new Rope("keepme");
  original.insert(0, "X");
  original.delete(0, 3);
  original.reverse();
  assert.strictEqual(original.to_string(), "keepme");

  // cross-chunk operations on a 200-character rope
  let big = "";
  for (let i = 0; i < 200; i++) big += String.fromCharCode("a".charCodeAt(0) + (i % 26));
  const rb = new Rope(big);
  assert.strictEqual(rb.length, 200);
  assert.strictEqual(rb.to_string(), big);
  assert.strictEqual(rb.charAt(0), big[0]);
  assert.strictEqual(rb.charAt(199), big[199]);
  assert.strictEqual(rb.charAt(63), big[63]);
  assert.strictEqual(rb.charAt(64), big[64]);
  assert.strictEqual(rb.slice(0, 70), big.slice(0, 70));
  assert.strictEqual(rb.slice(60, 150), big.slice(60, 150));
  const oracle = big.slice(0, 130) + "XYZ" + big.slice(130);
  const ins = rb.insert(130, "XYZ");
  assert.strictEqual(ins.to_string(), oracle);
  assert.strictEqual(ins.length, 203);
  const del = ins.delete(125, 135);
  assert.strictEqual(del.to_string(), oracle.slice(0, 125) + oracle.slice(135));
  const [bigLeft, bigRight] = new Rope(big).split(64);
  assert.strictEqual(bigLeft.length, 64);
  assert.strictEqual(bigRight.length, 136);
  assert.strictEqual(bigLeft.to_string(), big.slice(0, 64));
  assert.strictEqual(bigRight.to_string(), big.slice(64));
  assert.strictEqual(bigLeft.concat(bigRight).to_string(), big);
  assert.strictEqual(new Rope(big).reverse().to_string(), big.split("").reverse().join(""));

  // chunk length invariant after structural operations
  const probe = new Rope(big)
    .insert(0, "z".repeat(40))
    .delete(10, 90)
    .reverse()
    .concat(new Rope("q".repeat(300)));
  for (const c of probe.chunks) assert.ok(c.length <= CHUNK, "chunk cap");
  assert.strictEqual(probe.length, 200 + 40 - 80 + 300);

  // LCG-driven fuzz against a plain-string oracle; every operation must agree,
  // and no op script can desynchronise the rope from the oracle string.
  const gen = new Lcg(31337);
  const alphabet = "abcdef";
  let rope = new Rope("");
  let oracleStr = "";
  for (let step = 0; step < 320; step++) {
    const kind = gen.next() % 5;
    if (kind === 0 || kind === 1) {
      const pos = gen.next() % (oracleStr.length + 1);
      const ln = 1 + (gen.next() % 12);
      let text = "";
      for (let k = 0; k < ln; k++) text += alphabet[gen.next() % alphabet.length];
      rope = rope.insert(pos, text);
      oracleStr = oracleStr.slice(0, pos) + text + oracleStr.slice(pos);
    } else if (kind === 2) {
      if (oracleStr.length > 0) {
        let a = gen.next() % (oracleStr.length + 1);
        let b = gen.next() % (oracleStr.length + 1);
        if (a > b) [a, b] = [b, a];
        rope = rope.delete(a, b);
        oracleStr = oracleStr.slice(0, a) + oracleStr.slice(b);
      }
    } else if (kind === 3) {
      rope = rope.reverse();
      oracleStr = oracleStr.split("").reverse().join("");
    } else {
      let a = gen.next() % (oracleStr.length + 1);
      let b = gen.next() % (oracleStr.length + 1);
      if (a > b) [a, b] = [b, a];
      assert.strictEqual(rope.slice(a, b), oracleStr.slice(a, b), `fuzz slice step ${step}`);
      if (oracleStr.length > 0) {
        const pos = gen.next() % oracleStr.length;
        assert.strictEqual(rope.charAt(pos), oracleStr[pos], `fuzz char step ${step}`);
      }
      const mid = a;
      const [sp, sq] = rope.split(mid);
      assert.strictEqual(sp.to_string(), oracleStr.slice(0, mid), `fuzz split left ${step}`);
      assert.strictEqual(sq.to_string(), oracleStr.slice(mid), `fuzz split right ${step}`);
      rope = sp.concat(sq);
    }
    assert.strictEqual(rope.to_string(), oracleStr, `fuzz step ${step}`);
    assert.strictEqual(rope.length, oracleStr.length, `fuzz length step ${step}`);
  }

  // final sweep: index and slice for every position of the fuzz result
  for (let i = 0; i < oracleStr.length; i++) {
    assert.strictEqual(rope.charAt(i), oracleStr[i], `sweep ${i}`);
  }
  assert.strictEqual(rope.slice(0, oracleStr.length), oracleStr);

  console.log("PASS");
}
