/** Task 477 - Implement B+ tree leaf-range query and insert basics. (Tier: Advanced, CTCI Ch 14 — Databases) */
import java.util.*;

class BNode {
    final boolean isLeaf;
    final List<Integer> keys = new ArrayList<>();
    final List<BNode> children = new ArrayList<>(); // internal nodes only
    BNode next;                                     // leaf nodes only

    BNode(boolean isLeaf) {
        this.isLeaf = isLeaf;
    }
}

class Split {
    final int sep;
    final BNode right;

    Split(int sep, BNode right) {
        this.sep = sep;
        this.right = right;
    }
}

class BPlusTree {
    final int maxKeys;
    BNode root = new BNode(true);
    int size = 0;

    BPlusTree(int maxKeys) {
        this.maxKeys = maxKeys;
    }

    BNode leafFor(int key) {
        BNode node = root;
        while (!node.isLeaf) {
            int idx = 0;
            while (idx < node.keys.size() && node.keys.get(idx) <= key) idx++;
            node = node.children.get(idx);
        }
        return node;
    }

    boolean contains(int key) {
        return leafFor(key).keys.contains(key);
    }

    List<Integer> allKeys() {
        BNode leaf = leafFor(Integer.MIN_VALUE);
        List<Integer> out = new ArrayList<>();
        while (leaf != null) {
            out.addAll(leaf.keys);
            leaf = leaf.next;
        }
        return out;
    }

    List<Integer> range(int lo, int hi) {
        List<Integer> out = new ArrayList<>();
        if (lo > hi) return out;
        BNode leaf = leafFor(lo);
        while (leaf != null) {
            for (int key : leaf.keys) {
                if (key > hi) return out;
                if (key >= lo) out.add(key);
            }
            leaf = leaf.next;
        }
        return out;
    }

    boolean insert(int key) {
        if (contains(key)) return false;
        Split res = insertInto(root, key);
        if (res != null) {
            BNode newRoot = new BNode(false);
            newRoot.keys.add(res.sep);
            newRoot.children.add(root);
            newRoot.children.add(res.right);
            root = newRoot;
        }
        size++;
        return true;
    }

    private Split insertInto(BNode node, int key) {
        if (node.isLeaf) {
            int pos = 0;
            while (pos < node.keys.size() && node.keys.get(pos) < key) pos++;
            node.keys.add(pos, key);
            if (node.keys.size() <= maxKeys) return null;
            int mid = node.keys.size() / 2;
            BNode right = new BNode(true);
            right.keys.addAll(new ArrayList<>(node.keys.subList(mid, node.keys.size())));
            while (node.keys.size() > mid) node.keys.remove(node.keys.size() - 1);
            right.next = node.next;
            node.next = right;
            return new Split(right.keys.get(0), right);
        }
        int idx = 0;
        while (idx < node.keys.size() && node.keys.get(idx) <= key) idx++;
        Split res = insertInto(node.children.get(idx), key);
        if (res == null) return null;
        node.keys.add(idx, res.sep);
        node.children.add(idx + 1, res.right);
        if (node.keys.size() <= maxKeys) return null;
        int mid = node.children.size() / 2;
        int promoted = node.keys.get(mid - 1);
        BNode rightNode = new BNode(false);
        rightNode.children.addAll(new ArrayList<>(node.children.subList(mid, node.children.size())));
        rightNode.keys.addAll(new ArrayList<>(node.keys.subList(mid, node.keys.size())));
        while (node.children.size() > mid) node.children.remove(node.children.size() - 1);
        while (node.keys.size() > mid - 1) node.keys.remove(node.keys.size() - 1);
        return new Split(promoted, rightNode);
    }
}

public class Solution {
    static int lcg(int seed) {
        return (seed * 75 + 74) % 65537;
    }

    static int minKey(BNode node) {
        while (!node.isLeaf) node = node.children.get(0);
        return node.keys.get(0);
    }

    static List<BNode> collectLeaves(BNode node) {
        List<BNode> out = new ArrayList<>();
        if (node.isLeaf) {
            out.add(node);
            return out;
        }
        for (BNode child : node.children) out.addAll(collectLeaves(child));
        return out;
    }

    static List<BNode> chainLeaves(BPlusTree tree) {
        BNode leaf = tree.leafFor(Integer.MIN_VALUE);
        List<BNode> out = new ArrayList<>();
        while (leaf != null) {
            out.add(leaf);
            leaf = leaf.next;
        }
        return out;
    }

    static void checkInvariants(BPlusTree tree) {
        List<Integer> depths = new ArrayList<>();
        walkInvariants(tree, tree.root, 0, depths);
        Set<Integer> uniqueDepths = new HashSet<>(depths);
        check(uniqueDepths.size() == 1, "all leaves at equal depth");
        List<BNode> byChain = chainLeaves(tree);
        List<BNode> byDfs = collectLeaves(tree.root);
        check(byChain.size() == byDfs.size(), "leaf chain covers every leaf");
        for (int i = 0; i < byChain.size(); i++) {
            check(byChain.get(i).keys.equals(byDfs.get(i).keys), "leaf chain order matches tree order");
        }
    }

    static void walkInvariants(BPlusTree tree, BNode node, int depth, List<Integer> depths) {
        List<Integer> sorted = new ArrayList<>(node.keys);
        Collections.sort(sorted);
        check(node.keys.equals(sorted), "keys sorted");
        check(node.keys.size() <= tree.maxKeys, "node capacity");
        if (node.isLeaf) {
            depths.add(depth);
            return;
        }
        check(node.children.size() == node.keys.size() + 1, "children = keys + 1");
        for (int i = 0; i < node.keys.size(); i++) {
            check(minKey(node.children.get(i + 1)) == node.keys.get(i), "separator equals right subtree minimum");
        }
        for (BNode child : node.children) walkInvariants(tree, child, depth + 1, depths);
    }

    static int leafCount(BPlusTree tree) {
        return collectLeaves(tree.root).size();
    }

    static int height(BPlusTree tree) {
        int depth = 1;
        BNode node = tree.root;
        while (!node.isLeaf) {
            node = node.children.get(0);
            depth++;
        }
        return depth;
    }

    public static void main(String[] args) {
        // Ascending inserts force a deterministic split cascade (hand-traced).
        BPlusTree tree = new BPlusTree(3);
        List<Integer> brute = new ArrayList<>();
        int[][] probes = {{1, 12}, {4, 9}, {5, 5}, {3, 3}, {10, 20}, {0, 0}, {12, 12}, {2, 11}};
        for (int key = 1; key <= 12; key++) {
            check(tree.insert(key), "insert " + key);
            brute.add(key);
            checkInvariants(tree);
            for (int[] probe : probes) {
                check(tree.range(probe[0], probe[1]).equals(filter(brute, probe[0], probe[1])), "range " + probe[0] + ".." + probe[1]);
            }
        }
        check(tree.root.keys.equals(Arrays.asList(5)), "root separator hand-traced to [5]");
        check(leafCount(tree) == 6, "six leaves");
        check(height(tree) == 3, "three levels");
        check(tree.allKeys().equals(brute), "all keys in order");
        check(!tree.insert(7), "duplicates are rejected");
        check(tree.allKeys().equals(brute) && tree.size == 12, "duplicate left the tree unchanged");

        // Descending inserts exercise the left-splitting mirror image.
        BPlusTree tree2 = new BPlusTree(3);
        List<Integer> brute2 = new ArrayList<>();
        for (int key = 12; key >= 1; key--) {
            check(tree2.insert(key), "insert " + key);
            brute2.add(key);
            checkInvariants(tree2);
            List<Integer> sorted = new ArrayList<>(brute2);
            Collections.sort(sorted);
            check(tree2.range(1, 12).equals(sorted), "descending range");
        }
        List<Integer> oneToTwelve = new ArrayList<>();
        for (int i = 1; i <= 12; i++) oneToTwelve.add(i);
        check(tree2.allKeys().equals(oneToTwelve), "descending keys complete");

        // Randomized inserts and range scans against the sorted-list oracle.
        BPlusTree tree3 = new BPlusTree(4);
        Set<Integer> inserted = new TreeSet<>();
        int seed = 4242;
        for (int i = 0; i < 40; i++) {
            seed = lcg(seed);
            int key = seed % 64;
            if (inserted.contains(key)) {
                check(!tree3.insert(key), "duplicate rejected");
            } else {
                check(tree3.insert(key), "insert " + key);
                inserted.add(key);
            }
            checkInvariants(tree3);
        }
        List<Integer> keysSorted = new ArrayList<>(inserted);
        check(keysSorted.size() > 25, "random phase inserted enough keys");
        check(tree3.allKeys().equals(keysSorted), "all keys");
        for (int i = 0; i < 80; i++) {
            seed = lcg(seed);
            int lo = seed % 70;
            seed = lcg(seed);
            int hi = lo + seed % (70 - lo);
            check(tree3.range(lo, hi).equals(filter(keysSorted, lo, hi)), "random range " + lo + ".." + hi);
        }
        check(tree3.range(65, 63).isEmpty(), "reversed range is empty");
        check(tree3.range(200, 300).isEmpty(), "beyond max is empty");
        List<Integer> atZero = tree3.range(0, 0);
        check(atZero.equals(inserted.contains(0) ? Arrays.asList(0) : Collections.emptyList()), "singleton range");
        System.out.println("PASS");
    }

    static List<Integer> filter(List<Integer> values, int lo, int hi) {
        List<Integer> out = new ArrayList<>();
        for (int v : values) if (v >= lo && v <= hi) out.add(v);
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
