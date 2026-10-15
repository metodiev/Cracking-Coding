/** Task 042 - Check balanced parentheses with three bracket types. (Tier: Beginner, CTCI Ch 3) */
function isBalanced(s) {
  const pairs = { ")": "(", "]": "[", "}": "{" };
  const stack = [];
  for (const ch of s) {
    if ("([{".includes(ch)) stack.push(ch);
    else if (pairs[ch]) {
      if (stack.pop() !== pairs[ch]) return false;
    }
  }
  return stack.length === 0;
}

module.exports = { isBalanced };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isBalanced("{[()]}"), true);
  assert.strictEqual(isBalanced("()[]{}"), true);
  assert.strictEqual(isBalanced("{[(])}"), false);
  assert.strictEqual(isBalanced(")"), false);
  console.log("PASS");
}
