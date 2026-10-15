/** Task 235 - Serialize and deserialize N-ary tree. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

class NAryNode {
  constructor(val, children = []) {
    this.val = val;
    this.children = children;
  }
}

/**
 * Wire format (identical across the Python, Java, JavaScript and C# solutions):
 * a recursive preorder where every node is written as "value[childCount]" followed
 * by the serialized children, e.g. root 1 with children [3, 2, 4] where 3 has
 * children [5, 6] serializes to "1[3]3[2]5[0]6[0]2[0]4[0]"; empty tree -> "".
 */
function serialize(root) {
  if (root === null) return "";
  let out = root.val + "[" + root.children.length + "]";
  for (const child of root.children) out += serialize(child);
  return out;
}

function deserialize(data) {
  if (data.length === 0) return null;
  let pos = 0;
  const readNode = () => {
    const open = data.indexOf("[", pos);
    const val = parseInt(data.slice(pos, open), 10);
    const close = data.indexOf("]", open);
    const childCount = parseInt(data.slice(open + 1, close), 10);
    pos = close + 1;
    const node = new NAryNode(val);
    for (let i = 0; i < childCount; i++) node.children.push(readNode());
    return node;
  };
  return readNode();
}

function depth(node) {
  if (node === null) return 0;
  let best = 0;
  for (const child of node.children) best = Math.max(best, depth(child));
  return 1 + best;
}

function treesEqual(a, b) {
  if (a === null || b === null) return a === b;
  if (a.val !== b.val || a.children.length !== b.children.length) return false;
  return a.children.every((child, i) => treesEqual(child, b.children[i]));
}

module.exports = { NAryNode, serialize, deserialize, depth, treesEqual };

if (require.main === module) {
  const assert = require("node:assert");

  // exact wire format for the documented example
  const example = new NAryNode(1);
  const three = new NAryNode(3, [new NAryNode(5), new NAryNode(6)]);
  example.children = [three, new NAryNode(2), new NAryNode(4)];
  assert.strictEqual(serialize(example), "1[3]3[2]5[0]6[0]2[0]4[0]");
  assert.ok(treesEqual(deserialize("1[3]3[2]5[0]6[0]2[0]4[0]"), example));
  assert.strictEqual(serialize(deserialize(serialize(example))), serialize(example));

  // empty tree
  assert.strictEqual(serialize(null), "");
  assert.strictEqual(deserialize(""), null);

  // single node
  assert.strictEqual(serialize(new NAryNode(42)), "42[0]");
  assert.ok(treesEqual(deserialize("42[0]"), new NAryNode(42)));

  // deep chain of 50 nodes
  let chain = new NAryNode(50);
  for (let v = 49; v >= 1; v--) chain = new NAryNode(v, [chain]);
  assert.strictEqual(depth(chain), 50);
  assert.ok(serialize(chain).startsWith("1[1]2[1]3[1]"));
  assert.ok(treesEqual(deserialize(serialize(chain)), chain));
  assert.strictEqual(depth(deserialize(serialize(chain))), 50);

  // node with many children
  const many = new NAryNode(1, []);
  for (let v = 0; v < 10; v++) many.children.push(new NAryNode(v));
  let expected = "1[10]";
  for (let v = 0; v < 10; v++) expected += v + "[0]";
  assert.strictEqual(serialize(many), expected);
  const restoredMany = deserialize(serialize(many));
  assert.ok(treesEqual(restoredMany, many));
  assert.strictEqual(restoredMany.children.length, 10);
  assert.deepStrictEqual(restoredMany.children.map((c) => c.val), [...Array(10).keys()]);

  // negative values are handled by the scanner
  assert.strictEqual(serialize(new NAryNode(-5, [new NAryNode(-7)])), "-5[1]-7[0]");
  assert.ok(treesEqual(deserialize("-5[1]-7[0]"), new NAryNode(-5, [new NAryNode(-7)])));

  console.log("PASS");
}
