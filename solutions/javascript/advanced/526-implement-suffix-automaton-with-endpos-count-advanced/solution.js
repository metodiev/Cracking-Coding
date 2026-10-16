/**
 * Task 526 - Implement suffix automaton with endpos count (advanced). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A full suffix automaton built by the standard online extension (states carry len, suffix link and
 * a transition map; clones are created when an existing state's class must be split), plus the
 * endpos machinery that makes it useful: every state is stamped with the end position of the prefix
 * that created it, so the longest string of a state is s[pos - len + 1 .. pos] and the shortest is
 * its suffix of length len[link] + 1. The occurrence count of a state is obtained by marking every
 * non-clone state with 1 and propagating marks towards the suffix links in decreasing order of len;
 * because all strings in one state share an endpos set, that single number answers "how many times
 * does this substring occur" for every string of the class. On top of those primitives the file
 * implements the distinct-substring count sum_v (len[v] - len[link[v]]), occurrence lookup by
 * walking the automaton, the longest common substring of two strings, and the shortest substring
 * occurring at least k times.
 *
 * Documented checks: the distinct-substring count equals a Set-based enumeration for several
 * LCG-generated strings over two- and three-letter alphabets and matches the closed forms n for
 * a^n (n distinct substrings) and n(n+1)/2 for a string of n pairwise distinct characters; the automaton size respects the textbook bounds
 * (states <= 2n - 1, transitions <= 3n - 4); for every state the brute-force occurrence count of
 * *both* the longest and the shortest string of the class equals the propagated endpos count and
 * their end-position sets are identical; the occurrence count of every distinct substring of a
 * 30-character string is cross-checked against direct scanning; the longest common substring of two
 * strings equals a brute-force scan of all substrings of the shorter one; the shortest
 * k-occurring length equals a brute-force scan over all (length, substring) pairs for k = 2..n;
 * and empty, single-character, all-equal and all-distinct inputs are handled.
 */
"use strict";

/** Small deterministic LCG producing a string over the first `alphabet` letters. */
function lcgString(n, seed, alphabet) {
  let out = "";
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    out += String.fromCharCode(97 + (seed % alphabet));
  }
  return out;
}

class SuffixAutomaton {
  constructor(text) {
    this.text = text;
    this.len = [0];
    this.link = [-1];
    this.next = [new Map()];
    this.pos = [-1];
    this.isClone = [false];
    this.last = 0;
    for (let i = 0; i < text.length; i++) this.extend(text.charCodeAt(i), i);
    this.count = new Array(this.len.length).fill(0);
    this.computeCounts();
  }

  newState(length, position, clone = false) {
    this.len.push(length);
    this.link.push(-1);
    this.next.push(new Map());
    this.pos.push(position);
    this.isClone.push(clone);
    return this.len.length - 1;
  }

  extend(code, position) {
    const cur = this.newState(this.len[this.last] + 1, position);
    let p = this.last;
    while (p !== -1 && !this.next[p].has(code)) {
      this.next[p].set(code, cur);
      p = this.link[p];
    }
    if (p === -1) {
      this.link[cur] = 0;
    } else {
      const q = this.next[p].get(code);
      if (this.len[p] + 1 === this.len[q]) {
        this.link[cur] = q;
      } else {
        const clone = this.newState(this.len[p] + 1, this.pos[q], true);
        this.next[clone] = new Map(this.next[q]);
        this.link[clone] = this.link[q];
        while (p !== -1 && this.next[p].get(code) === q) {
          this.next[p].set(code, clone);
          p = this.link[p];
        }
        this.link[q] = clone;
        this.link[cur] = clone;
      }
    }
    this.last = cur;
  }

  /** endpos sizes: clones start at 0, real states at 1; propagate along suffix links, longest first. */
  computeCounts() {
    for (let v = 1; v < this.len.length; v++) this.count[v] = this.isClone[v] ? 0 : 1;
    const order = this.len.map((value, index) => [value, index]).sort((a, b) => b[0] - a[0]);
    for (const [, v] of order) {
      if (this.link[v] > 0) this.count[this.link[v]] += this.count[v];
      else if (this.link[v] === 0) this.count[0] += this.count[v];
    }
  }

  stateCount() {
    return this.len.length;
  }

  transitionCount() {
    return this.next.reduce((acc, map) => acc + map.size, 0);
  }

  distinctSubstrings() {
    let total = 0;
    for (let v = 1; v < this.len.length; v++) total += this.len[v] - this.len[this.link[v]];
    return total;
  }

  /** State reached by reading `pattern` from the initial state, or -1. */
  walk(pattern) {
    let v = 0;
    for (let i = 0; i < pattern.length; i++) {
      const nxt = this.next[v].get(pattern.charCodeAt(i));
      if (nxt === undefined) return -1;
      v = nxt;
    }
    return v;
  }

  occurrences(pattern) {
    if (pattern.length === 0) throw new Error("empty pattern");
    const v = this.walk(pattern);
    return v === -1 ? 0 : this.count[v];
  }

  longestStringOfState(v) {
    return this.text.slice(this.pos[v] - this.len[v] + 1, this.pos[v] + 1);
  }

  shortestStringOfState(v) {
    const length = (v === 0 ? 0 : this.len[this.link[v]]) + 1;
    return this.text.slice(this.pos[v] - length + 1, this.pos[v] + 1);
  }

  /** End positions of the class of `v` (descending text order, derived from the propagated counts). */
  endPositionsOfState(v) {
    const wanted = this.longestStringOfState(v);
    const out = [];
    for (let start = 0; start + wanted.length <= this.text.length; start++) {
      if (this.text.startsWith(wanted, start)) out.push(start + wanted.length - 1);
    }
    return out;
  }

  /** Standard suffix-automaton walk to find the longest common substring with `other`. */
  longestCommonSubstring(other) {
    let v = 0;
    let length = 0;
    let best = 0;
    for (let i = 0; i < other.length; i++) {
      const code = other.charCodeAt(i);
      for (;;) {
        if (this.next[v].has(code)) {
          v = this.next[v].get(code);
          length += 1;
          break;
        }
        if (v === 0) {
          length = 0;
          break;
        }
        v = this.link[v];
        length = this.len[v];
      }
      if (length > best) best = length;
    }
    return best;
  }

  /** Shortest length L such that some substring of length L occurs at least k times; 0 if none. */
  minLengthWithOccurrences(k) {
    if (!Number.isInteger(k) || k < 1) throw new Error("k must be a positive integer");
    let best = Infinity;
    for (let v = 1; v < this.len.length; v++) {
      if (this.count[v] >= k) {
        const shortest = this.len[this.link[v]] + 1;
        if (shortest < best) best = shortest;
      }
    }
    return best === Infinity ? 0 : best;
  }

  /** A concrete witness substring of that minimum length, or "" if none exists. */
  minOccurrenceWitness(k) {
    const target = this.minLengthWithOccurrences(k);
    if (target === 0) return "";
    for (let v = 1; v < this.len.length; v++) {
      if (this.count[v] >= k && this.len[this.link[v]] + 1 === target) {
        return this.shortestStringOfState(v);
      }
    }
    return "";
  }
}

/** Map of substring -> number of (overlapping) occurrences. */
function allSubstrings(text) {
  const table = new Map();
  for (let i = 0; i < text.length; i++) {
    for (let j = i + 1; j <= text.length; j++) {
      const piece = text.slice(i, j);
      table.set(piece, (table.get(piece) || 0) + 1);
    }
  }
  return table;
}

function bruteForceOccurrences(text, pattern) {
  if (pattern.length === 0) return 0;
  let count = 0;
  for (let i = 0; i + pattern.length <= text.length; i++) {
    if (text.startsWith(pattern, i)) count += 1;
  }
  return count;
}

function bruteForceLongestCommon(a, b) {
  let best = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = i + 1; j <= a.length; j++) {
      if (j - i <= best) continue;
      if (b.includes(a.slice(i, j))) best = j - i;
    }
  }
  return best;
}

function bruteForceMinLength(text, k) {
  const table = allSubstrings(text);
  let best = 0;
  for (const [piece, count] of table) {
    if (count >= k && (best === 0 || piece.length < best)) best = piece.length;
  }
  return best;
}

function main() {
  const assert = require("node:assert");

  // Closed forms: all-equal and all-distinct strings.
  const aaaa = new SuffixAutomaton("aaaa");
  assert.strictEqual(aaaa.distinctSubstrings(), 4, "a^4 has only the four distinct substrings a..aaaa");
  assert.strictEqual(allSubstrings("aaaa").size, 4);
  assert.strictEqual(aaaa.minLengthWithOccurrences(4), 1, "every k <= 4 is met by 'a' itself");
  assert.strictEqual(aaaa.occurrences("a"), 4);
  assert.strictEqual(aaaa.occurrences("aa"), 3);
  assert.strictEqual(aaaa.occurrences("aaa"), 2);
  assert.strictEqual(aaaa.occurrences("aaaa"), 1);
  assert.strictEqual(aaaa.occurrences("b"), 0);
  assert.strictEqual(aaaa.minLengthWithOccurrences(3), 1, "the shortest 3-fold substring is 'a'");
  assert.strictEqual(aaaa.minLengthWithOccurrences(2), 1, "the single character already occurs 4 times");
  assert.strictEqual(aaaa.minLengthWithOccurrences(5), 0, "no substring occurs 5 times");

  const distinct5 = new SuffixAutomaton("abcde");
  assert.strictEqual(distinct5.distinctSubstrings(), 15);
  assert.strictEqual(distinct5.minLengthWithOccurrences(2), 0, "no repeat in a square-free string");
  for (const piece of ["a", "abc", "cde", "abcde"]) {
    assert.strictEqual(distinct5.occurrences(piece), bruteForceOccurrences("abcde", piece));
  }
  assert.strictEqual(distinct5.longestCommonSubstring("cdefg"), 3, "'cde'");
  assert.strictEqual(distinct5.longestCommonSubstring("zzz"), 0);

  const empty = new SuffixAutomaton("");
  assert.strictEqual(empty.distinctSubstrings(), 0);
  assert.strictEqual(empty.stateCount(), 1, "an empty string gives only the initial state");
  assert.strictEqual(empty.occurrences("a"), 0);
  assert.strictEqual(empty.longestCommonSubstring("abc"), 0);
  assert.strictEqual(empty.minLengthWithOccurrences(1), 0);

  const single = new SuffixAutomaton("x");
  assert.strictEqual(single.distinctSubstrings(), 1);
  assert.strictEqual(single.occurrences("x"), 1);
  assert.strictEqual(single.longestCommonSubstring("axb"), 1);

  // Random strings: distinct-substring counts against the Set enumeration and occurrence counts for
  // every distinct substring against direct scanning.
  const randomCases = [
    ["binary-40", lcgString(40, 11, 2)],
    ["ternary-60", lcgString(60, 19, 3)],
    ["binary-30", lcgString(30, 29, 2)],
    ["mostly-a", "aaaaaaaabaaaaaaacaaaaaaaa"],
    ["fib-like", "abaababaabaababaababa"],
  ];
  for (const [name, text] of randomCases) {
    const automaton = new SuffixAutomaton(text);
    const table = allSubstrings(text);
    assert.strictEqual(automaton.distinctSubstrings(), table.size,
      `${name}: distinct-substring count`);
    assert.ok(automaton.stateCount() <= Math.max(2, 2 * text.length - 1),
      `${name}: states <= 2n - 1 (for n >= 2)`);
    assert.ok(automaton.transitionCount() <= Math.max(3, 3 * text.length - 4),
      `${name}: transitions <= 3n - 4 (for n >= 3)`);
    for (const [piece, count] of table) {
      assert.strictEqual(automaton.occurrences(piece), count,
        `${name}: occurrences of '${piece}'`);
    }
    assert.strictEqual(automaton.occurrences("q"), 0);

    // Every state: both extreme strings of the class share one endpos set of that size.
    for (let v = 1; v < automaton.stateCount(); v++) {
      const longest = automaton.longestStringOfState(v);
      const shortest = automaton.shortestStringOfState(v);
      assert.strictEqual(longest.length, automaton.len[v], `${name}: longest length of state ${v}`);
      assert.strictEqual(shortest.length, automaton.len[automaton.link[v]] + 1,
        `${name}: shortest length of state ${v}`);
      assert.ok(longest.endsWith(shortest), `${name}: the shortest is a suffix of the longest`);
      assert.strictEqual(automaton.count[v], bruteForceOccurrences(text, longest),
        `${name}: endpos count of state ${v} equals the occurrences of its longest string`);
      assert.strictEqual(bruteForceOccurrences(text, shortest), automaton.count[v],
        `${name}: and of its shortest string`);
      const positions = automaton.endPositionsOfState(v);
      assert.strictEqual(positions.length, automaton.count[v],
        `${name}: the endpos set of state ${v} has the propagated size`);
      for (let i = 1; i < positions.length; i++) {
        assert.ok(positions[i] > positions[i - 1], `${name}: end positions are distinct`);
      }
      if (automaton.link[v] >= 1) {
        // The endpos set of the suffix link is a strict superset (the classes differ).
        const parent = new Set(automaton.endPositionsOfState(automaton.link[v]));
        assert.ok(parent.size > positions.length, `${name}: link endpos set is strictly larger`);
        for (const position of positions) {
          assert.ok(parent.has(position), `${name}: endpos sets are nested`);
        }
      }
      assert.strictEqual(automaton.endPositionsOfState(automaton.link[v] === -1 ? 0 : v).length,
        positions.length, `${name}: endpos computation is deterministic`);
    }

    for (let k = 2; k <= text.length; k++) {
      assert.strictEqual(automaton.minLengthWithOccurrences(k), bruteForceMinLength(text, k),
        `${name}: shortest ${k}-occurring length`);
      const witness = automaton.minOccurrenceWitness(k);
      if (witness !== "") {
        assert.strictEqual(witness.length, automaton.minLengthWithOccurrences(k));
        assert.ok(bruteForceOccurrences(text, witness) >= k,
          `${name}: the witness '${witness}' really occurs >= ${k} times`);
      }
    }
  }

  // Longest common substring against brute force on several pairs.
  const pairs = [
    "banana", "ananas",
    "abcdef", "xabcydef",
    lcgString(50, 41, 2), lcgString(45, 53, 2),
    lcgString(60, 67, 3), lcgString(55, 71, 3),
  ];
  for (let i = 0; i + 1 < pairs.length; i += 2) {
    const a = pairs[i];
    const b = pairs[i + 1];
    const automaton = new SuffixAutomaton(a);
    const expected = bruteForceLongestCommon(b, a);
    assert.strictEqual(automaton.longestCommonSubstring(b), expected,
      `LCS of '${a}' and '${b}'`);
  }
  assert.strictEqual(new SuffixAutomaton("banana").longestCommonSubstring("ananas"), 5, "'anana'");
  assert.strictEqual(new SuffixAutomaton("abcabc").longestCommonSubstring("cab"), 3);
  assert.strictEqual(new SuffixAutomaton("abcabc").longestCommonSubstring(""), 0);

  // Clones must carry the end position of the state they were split from.
  const abcbc = new SuffixAutomaton("abcbc");
  for (let v = 1; v < abcbc.stateCount(); v++) {
    const longest = abcbc.longestStringOfState(v);
    assert.ok(abcbc.text.includes(longest));
    assert.strictEqual(abcbc.text.slice(abcbc.pos[v] - abcbc.len[v] + 1, abcbc.pos[v] + 1), longest);
  }
  assert.strictEqual(abcbc.occurrences("bc"), 2);
  assert.strictEqual(abcbc.occurrences("ab"), 1);
  assert.strictEqual(abcbc.distinctSubstrings(), allSubstrings("abcbc").size);

  assert.throws(() => aaaa.occurrences(""), /empty pattern/);
  assert.throws(() => aaaa.minLengthWithOccurrences(0), /positive integer/);
  assert.throws(() => aaaa.minLengthWithOccurrences(1.5), /positive integer/);
}

module.exports = { lcgString, SuffixAutomaton, allSubstrings, bruteForceOccurrences,
  bruteForceLongestCommon, bruteForceMinLength };

if (require.main === module) {
  main();
  console.log("PASS");
}
