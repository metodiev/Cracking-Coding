/** Task 003 - Count vowels and consonants in a string. (Tier: Beginner, CTCI Ch 1) */
"use strict";

function countVowelsAndConsonants(s) {
  let vowels = 0, consonants = 0;
  for (const ch of s.toLowerCase()) {
    if (!/[a-z]/.test(ch)) continue;
    if ("aeiou".includes(ch)) vowels++; else consonants++;
  }
  return [vowels, consonants];
}

module.exports = { countVowelsAndConsonants };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(countVowelsAndConsonants("hello"), [2, 3]);
  assert.deepStrictEqual(countVowelsAndConsonants(""), [0, 0]);
  assert.deepStrictEqual(countVowelsAndConsonants("AEIOU"), [5, 0]);
  console.log("PASS");
}
