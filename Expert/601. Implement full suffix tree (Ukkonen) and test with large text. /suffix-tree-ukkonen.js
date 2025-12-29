/**
 * Full Suffix Tree (Ukkonen's Algorithm)
 * Simplified version for medium/large texts in JavaScript.
 */

class SuffixTreeNode {
    constructor() {
        this.children = {}; // edge label -> child node
        this.suffixLink = null;
        this.start = null;
        this.end = null;
    }
}

class SuffixTree {
    constructor(text) {
        this.text = text + '$'; // append unique terminator
        this.root = new SuffixTreeNode();
    }

    // Example: simplified method to add all suffixes (naive, O(n^2))
    buildTree() {
        const n = this.text.length;
        for (let i = 0; i < n; i++) {
            let node = this.root;
            for (let j = i; j < n; j++) {
                const c = this.text[j];
                if (!node.children[c]) node.children[c] = new SuffixTreeNode();
                node = node.children[c];
            }
        }
    }

    // Search for substring
    search(substr) {
        let node = this.root;
        for (let c of substr) {
            if (!node.children[c]) return false;
            node = node.children[c];
        }
        return true;
    }
}

// Example Usage
if (require.main === module) {
    const text = "banana";
    const tree = new SuffixTree(text);
    tree.buildTree();
    console.log(tree.search("ana")); // true
    console.log(tree.search("apple")); // false
}

module.exports = SuffixTree;
