/** Task 045 - Replace spaces with %20 (URLify). (Tier: Beginner, CTCI Ch 1) */
function urlify(chars, length) {
  let write = chars.length - 1;
  for (let read = length - 1; read >= 0; read--) {
    if (chars[read] === " ") {
      chars[write] = "0";
      chars[write - 1] = "2";
      chars[write - 2] = "%";
      write -= 3;
    } else {
      chars[write--] = chars[read];
    }
  }
  return chars;
}

module.exports = { urlify };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(urlify(("a b c" + " ".repeat(4)).split(""), 5).join(""), "a%20b%20c");
  assert.strictEqual(urlify("abc".split(""), 3).join(""), "abc");
  console.log("PASS");
}
