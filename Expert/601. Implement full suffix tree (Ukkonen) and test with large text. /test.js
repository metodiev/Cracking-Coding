const { reverseString1, reverseString2 } = require('./reverse-string');
const twoSumUniquePairs = require('./two-sum-unique-pairs');
const buildSuffixArrayDoubling = require('./suffix-array');
const SuffixTree = require('./suffix-tree-ukkonen');

console.log("=== Reverse String Tests ===");
console.log(reverseString1("hello")); // olleh
console.log(reverseString2("world")); // dlrow

console.log("\n=== Two-Sum Unique Pairs ===");
console.log(twoSumUniquePairs([1,2,3,2,4,5], 5)); // [[1,4],[2,3]]

console.log("\n=== Suffix Array ===");
const s = "banana";
const sa = buildSuffixArrayDoubling(s);
console.log(sa);
console.log(sa.map(i => s.substring(i)));

console.log("\n=== Suffix Tree ===");
const tree = new SuffixTree("banana");
tree.buildTree();
console.log(tree.search("ana")); // true
console.log(tree.search("apple")); // false
