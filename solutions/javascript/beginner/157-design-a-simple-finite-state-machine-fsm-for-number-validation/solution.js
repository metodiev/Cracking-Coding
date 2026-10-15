/** Task 157 - Design a simple FSM for number validation. (Tier: Beginner, CTCI Ch 7) */
"use strict";

const START = "START";
const SIGN = "SIGN";
const INT = "INT";
const DOT = "DOT";
const FRAC = "FRAC";

/** A valid number must end in one of these states. */
const ACCEPTING = new Set([INT, FRAC]);

/**
 * Data-driven transition table: state -> { character class -> next state }.
 * Exponent notation (e.g. "1e5") is intentionally unsupported and rejected.
 */
const TRANSITIONS = {
  [START]: { digit: INT, sign: SIGN },
  [SIGN]: { digit: INT },
  [INT]: { digit: INT, dot: DOT },
  [DOT]: { digit: FRAC },
  [FRAC]: { digit: FRAC },
};

/** Character class the FSM consumes; anything else (spaces, letters) is rejected. */
function charClass(ch) {
  if (ch >= "0" && ch <= "9") return "digit";
  if (ch === "+" || ch === "-") return "sign";
  if (ch === ".") return "dot";
  return null;
}

/** True iff `s` is a signed decimal number accepted by the FSM above. */
function accepts(s) {
  let state = START;
  for (const ch of s) {
    const cls = charClass(ch);
    if (cls === null) return false;
    const row = TRANSITIONS[state];
    const next = row ? row[cls] : undefined;
    if (!next) return false;
    state = next;
  }
  return ACCEPTING.has(state);
}

module.exports = { START, SIGN, INT, DOT, FRAC, ACCEPTING, TRANSITIONS, accepts };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(accepts("123"), true);
  assert.strictEqual(accepts("+1"), true);
  assert.strictEqual(accepts("-1.5"), true);
  assert.strictEqual(accepts("0"), true);
  assert.strictEqual(accepts("-0.0"), true);
  assert.strictEqual(accepts("007"), true);            // leading zeros are fine
  assert.strictEqual(accepts("1."), false);            // DOT is not accepting
  assert.strictEqual(accepts(".5"), false);            // needs a leading digit
  assert.strictEqual(accepts(""), false);
  assert.strictEqual(accepts("+"), false);
  assert.strictEqual(accepts("-"), false);
  assert.strictEqual(accepts("."), false);
  assert.strictEqual(accepts("1.2.3"), false);
  assert.strictEqual(accepts("12a"), false);
  assert.strictEqual(accepts(" 12"), false);           // whitespace invalid
  assert.strictEqual(accepts("12 "), false);
  assert.strictEqual(accepts("+-1"), false);
  assert.strictEqual(accepts("1e5"), false);           // exponents unsupported

  // the transition table is explicit and data-driven
  assert.ok(!("dot" in TRANSITIONS[START]), "no leading dot transition");
  assert.strictEqual(TRANSITIONS[INT].dot, DOT);
  assert.strictEqual(TRANSITIONS[DOT].digit, FRAC);
  assert.deepStrictEqual([...ACCEPTING].sort(), [FRAC, INT].sort());
  console.log("PASS");
}
