/** Task 289 - Restore IP addresses from a digit string. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * Parts are extended left to right, trying segment lengths 1, 2, 3 in that order, so the
 * emitted addresses are deterministic. A segment is valid when it is 1-3 digits, at most
 * 255, and has no leading zero unless it is exactly "0". Cross-checked against an oracle
 * that enumerates every cut position triple.
 */
function isValidSegment(segment) {
  if (!segment || segment.length > 3) return false;
  if (segment[0] === "0") return segment === "0";
  return Number(segment) <= 255;
}

function restoreIpAddresses(s) {
  if (s.length < 4 || s.length > 12) return [];
  const results = [];
  const parts = [];

  const backtrack = (index) => {
    if (parts.length === 4) {
      if (index === s.length) results.push(parts.join("."));
      return;
    }
    for (const length of [1, 2, 3]) {
      if (index + length > s.length) break;
      const segment = s.slice(index, index + length);
      if (!isValidSegment(segment)) continue;
      parts.push(segment);
      backtrack(index + length);
      parts.pop();
    }
  };

  backtrack(0);
  return results;
}

/** Oracle: try every triple of cut positions and validate the four segments. */
function bruteForceIpAddresses(s) {
  const results = [];
  const n = s.length;
  for (let i = 1; i < Math.min(4, n); i++) {
    for (let j = i + 1; j < Math.min(i + 4, n); j++) {
      for (let k = j + 1; k < Math.min(j + 4, n); k++) {
        const parts = [s.slice(0, i), s.slice(i, j), s.slice(j, k), s.slice(k)];
        if (parts.every(isValidSegment)) results.push(parts.join("."));
      }
    }
  }
  results.sort();
  return results;
}

function isValidIp(text, s) {
  const parts = text.split(".");
  return parts.length === 4 && parts.every(isValidSegment) && parts.join("") === s;
}

const rng = {
  state: 13579,
  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state;
  },
};

module.exports = { restoreIpAddresses, isValidSegment, bruteForceIpAddresses, isValidIp };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(restoreIpAddresses("25525511135"), ["255.255.11.135", "255.255.111.35"]);
  assert.deepStrictEqual(restoreIpAddresses("0000"), ["0.0.0.0"]);
  assert.deepStrictEqual(restoreIpAddresses("1111"), ["1.1.1.1"]);
  assert.deepStrictEqual(restoreIpAddresses("010010"), ["0.10.0.10", "0.100.1.0"]);
  assert.deepStrictEqual(restoreIpAddresses("101023"),
    ["1.0.10.23", "1.0.102.3", "10.1.0.23", "10.10.2.3", "101.0.2.3"]);
  assert.deepStrictEqual(restoreIpAddresses("255255255255"), ["255.255.255.255"]);

  assert.deepStrictEqual(restoreIpAddresses("111"), []);            // too short for four parts
  assert.deepStrictEqual(restoreIpAddresses(""), []);
  assert.deepStrictEqual(restoreIpAddresses("1111111111111"), []);  // 13 digits: too long
  assert.deepStrictEqual(restoreIpAddresses("256256256256"), []);   // every segment exceeds 255
  assert.deepStrictEqual(restoreIpAddresses("111111111111111111111"), []);

  for (let trial = 0; trial < 200; trial++) {
    const length = 1 + (rng.next() % 14);
    let s = "";
    for (let i = 0; i < length; i++) s += rng.next() % 10;
    const results = restoreIpAddresses(s);
    assert.deepStrictEqual([...results].sort(), bruteForceIpAddresses(s), s);
    for (const text of results) assert.ok(isValidIp(text, s), `${s} ${text}`);
    assert.strictEqual(results.length, new Set(results).size, s);
  }

  console.log("PASS");
}
