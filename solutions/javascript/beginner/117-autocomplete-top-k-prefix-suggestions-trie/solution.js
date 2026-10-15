/** Task 117 - Autocomplete top-k prefix suggestions (trie). (Tier: Beginner, CTCI Ch 4) */
"use strict";

class AcNode {
  constructor() {
    this.kids = new Map();
    this.weight = 0;
    this.best = 0;
  }
}

class Autocomplete {
  constructor() {
    this.root = new AcNode();
  }

  add(word, weight = 1) {
    let node = this.root;
    for (const ch of word) {
      if (!node.kids.has(ch)) node.kids.set(ch, new AcNode());
      node = node.kids.get(ch);
    }
    node.weight += weight;
    this.recompute(this.root);
  }

  recompute(node) {
    let best = node.weight;
    for (const kid of node.kids.values()) best = Math.max(best, this.recompute(kid));
    node.best = best;
    return best;
  }

  suggest(prefix, k) {
    let node = this.root;
    for (const ch of prefix) {
      node = node.kids.get(ch);
      if (!node) return [];
    }
    const found = [];
    this.collect(node, prefix.split(""), found);
    found.sort((a, b) => (b[1] - a[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    return found.slice(0, k).map(([word]) => word);
  }

  collect(node, path, out) {
    if (node.weight > 0) out.push([path.join(""), node.weight]);
    for (const [ch, kid] of node.kids) {
      path.push(ch);
      this.collect(kid, path, out);
      path.pop();
    }
  }
}

module.exports = { Autocomplete };

if (require.main === module) {
  const assert = require("node:assert");
  const ac = new Autocomplete();
  ac.add("car", 5);
  ac.add("cart", 3);
  ac.add("carbon", 2);
  ac.add("care", 1);
  ac.add("dog", 9);
  assert.deepStrictEqual(ac.suggest("car", 2), ["car", "cart"]);
  assert.deepStrictEqual(ac.suggest("car", 10), ["car", "cart", "carbon", "care"]);
  assert.deepStrictEqual(ac.suggest("do", 5), ["dog"]);
  assert.deepStrictEqual(ac.suggest("z", 3), []);
  assert.deepStrictEqual(ac.suggest("", 1), ["dog"]);
  console.log("PASS");
}
