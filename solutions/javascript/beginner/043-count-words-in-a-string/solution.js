/** Task 043 - Count words in a string. (Tier: Beginner, CTCI Ch 1) */
function countWords(s) {
  const trimmed = s.trim();
  return trimmed === "" ? 0 : trimmed.split(/\s+/).length;
}

module.exports = { countWords };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(countWords("the sky is blue"), 4);
  assert.strictEqual(countWords("   "), 0);
  assert.strictEqual(countWords("one"), 1);
  console.log("PASS");
}
