/** Task 437 - Implement B-tree node insert/delete basics. (Tier: Advanced, CTCI Ch 14)
 *
 * A B-tree of order m = 4 (minimum degree t = 2): every node holds at most 2t-1 = 3 keys and, when
 * internal, 2..4 children; non-root nodes keep at least t-1 = 1 key, and all leaves sit at the same
 * depth. Multiset semantics are obtained by storing collision-free composite keys
 * key = (value << 20) | serial: each insertion gets a fresh serial, so the textbook unique-key
 * insert/delete apply unchanged while the sorted traversal still yields a sorted multiset of values.
 * Insert uses the preemptive split rule - a full node (3 keys) is split around its median before
 * descending, the median rises into the parent and the upper half becomes a new right sibling.
 * Delete follows CLRS: a key found in an internal node is replaced by its in-order predecessor or
 * successor; before descending into a child that holds only t-1 keys the child is repaired by
 * borrowing a key from a sibling (rotation through the parent) or by merging with a sibling - the
 * documented merge/borrow counter is asserted to be positive after the LCG workload. The sorted
 * traversal is compared against a sorted multiset oracle on LCG operation scripts, and structure
 * invariants (key order, occupancy, uniform leaf depth) are re-validated throughout.
 */
import java.util.*;

class BTreeNode {
    final List<Integer> keys = new ArrayList<>();
    final List<BTreeNode> children = new ArrayList<>();
}

class BTree {
    static final int T = 2;
    static final int SHIFT = 20;

    BTreeNode root = new BTreeNode();
    int size;
    int nextSerial;
    int splits;
    int merges;
    int borrows;

    static int lowerBound(List<Integer> keys, int key) {
        int lo = 0;
        int hi = keys.size();
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (keys.get(mid) < key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    void insert(int value) {
        int key = (value << SHIFT) | nextSerial;
        nextSerial++;
        if (root.keys.size() == 2 * T - 1) {
            BTreeNode newRoot = new BTreeNode();
            newRoot.children.add(root);
            splitChild(newRoot, 0);
            root = newRoot;
        }
        insertNonfull(root, key);
        size++;
    }

    private void splitChild(BTreeNode parent, int index) {
        BTreeNode node = parent.children.get(index);
        BTreeNode right = new BTreeNode();
        int median = node.keys.get(T - 1);
        right.keys.addAll(node.keys.subList(T, node.keys.size()));
        node.keys.subList(T - 1, node.keys.size()).clear();
        if (!node.children.isEmpty()) {
            right.children.addAll(node.children.subList(T, node.children.size()));
            node.children.subList(T, node.children.size()).clear();
        }
        parent.keys.add(index, median);
        parent.children.add(index + 1, right);
        splits++;
    }

    private void insertNonfull(BTreeNode node, int key) {
        int index = lowerBound(node.keys, key);
        if (!node.children.isEmpty()) {
            if (node.children.get(index).keys.size() == 2 * T - 1) {
                splitChild(node, index);
                if (key > node.keys.get(index)) index++;
            }
            insertNonfull(node.children.get(index), key);
        } else {
            node.keys.add(index, key);
        }
    }

    private Integer lowerBoundKey(BTreeNode node, int base) {
        int index = lowerBound(node.keys, base);
        if (!node.children.isEmpty()) {
            Integer found = lowerBoundKey(node.children.get(index), base);
            if (found != null) return found;
            if (index < node.keys.size()) return node.keys.get(index);
            return null;
        }
        return index < node.keys.size() ? node.keys.get(index) : null;
    }

    boolean contains(int value) {
        Integer key = lowerBoundKey(root, value << SHIFT);
        return key != null && (key >> SHIFT) == value;
    }

    boolean delete(int value) {
        Integer key = lowerBoundKey(root, value << SHIFT);
        if (key == null || (key >> SHIFT) != value) return false;
        delete(root, key);
        if (root.keys.isEmpty() && !root.children.isEmpty()) {
            BTreeNode child = root.children.get(0);
            root.keys.clear();
            root.keys.addAll(child.keys);
            root.children.clear();
            root.children.addAll(child.children);
        }
        size--;
        return true;
    }

    private void delete(BTreeNode node, int key) {
        int index = lowerBound(node.keys, key);
        if (index < node.keys.size() && node.keys.get(index) == key) {
            if (node.children.isEmpty()) {
                node.keys.remove(index);
            } else {
                deleteInternal(node, index);
            }
            return;
        }
        if (node.children.isEmpty()) return;
        boolean flag = index == node.keys.size();
        if (node.children.get(index).keys.size() == T - 1) {
            fill(node, index);
            if (flag && index > node.keys.size()) index--;
        }
        delete(node.children.get(index), key);
    }

    private void deleteInternal(BTreeNode node, int index) {
        int key = node.keys.get(index);
        BTreeNode left = node.children.get(index);
        BTreeNode right = node.children.get(index + 1);
        if (left.keys.size() > T - 1) {
            int predecessor = maxKey(left);
            node.keys.set(index, predecessor);
            delete(left, predecessor);
        } else if (right.keys.size() > T - 1) {
            int successor = minKey(right);
            node.keys.set(index, successor);
            delete(right, successor);
        } else {
            merge(node, index);
            delete(node.children.get(index), key);
        }
    }

    private void fill(BTreeNode node, int index) {
        if (index > 0 && node.children.get(index - 1).keys.size() > T - 1) {
            borrowFromLeft(node, index);
        } else if (index < node.keys.size() && node.children.get(index + 1).keys.size() > T - 1) {
            borrowFromRight(node, index);
        } else if (index < node.keys.size()) {
            merge(node, index);
        } else {
            merge(node, index - 1);
        }
    }

    private void borrowFromLeft(BTreeNode node, int index) {
        BTreeNode child = node.children.get(index);
        BTreeNode left = node.children.get(index - 1);
        child.keys.add(0, node.keys.get(index - 1));
        node.keys.set(index - 1, left.keys.remove(left.keys.size() - 1));
        if (!left.children.isEmpty()) {
            child.children.add(0, left.children.remove(left.children.size() - 1));
        }
        borrows++;
    }

    private void borrowFromRight(BTreeNode node, int index) {
        BTreeNode child = node.children.get(index);
        BTreeNode right = node.children.get(index + 1);
        child.keys.add(node.keys.get(index));
        node.keys.set(index, right.keys.remove(0));
        if (!right.children.isEmpty()) {
            child.children.add(right.children.remove(0));
        }
        borrows++;
    }

    private void merge(BTreeNode node, int index) {
        BTreeNode left = node.children.get(index);
        BTreeNode right = node.children.get(index + 1);
        left.keys.add(node.keys.remove(index));
        left.keys.addAll(right.keys);
        if (!left.children.isEmpty()) {
            left.children.addAll(right.children);
        }
        node.children.remove(index + 1);
        merges++;
    }

    private int minKey(BTreeNode node) {
        while (!node.children.isEmpty()) node = node.children.get(0);
        return node.keys.get(0);
    }

    private int maxKey(BTreeNode node) {
        while (!node.children.isEmpty()) node = node.children.get(node.children.size() - 1);
        return node.keys.get(node.keys.size() - 1);
    }

    List<Integer> inorderValues() {
        List<Integer> out = new ArrayList<>();
        inorder(root, out);
        return out;
    }

    private void inorder(BTreeNode node, List<Integer> out) {
        for (int i = 0; i < node.keys.size(); i++) {
            if (!node.children.isEmpty()) inorder(node.children.get(i), out);
            out.add(node.keys.get(i) >> SHIFT);
        }
        if (!node.children.isEmpty()) inorder(node.children.get(node.keys.size()), out);
    }

    boolean validate() {
        return validate(root, true, 0, -1) >= 0;
    }

    private int validate(BTreeNode node, boolean isRoot, int depth, int leafDepth) {
        if (node.keys.size() > 2 * T - 1) return -1;
        for (int i = 1; i < node.keys.size(); i++) {
            if (node.keys.get(i - 1) >= node.keys.get(i)) return -1;
        }
        if (!node.children.isEmpty()) {
            if (node.children.size() != node.keys.size() + 1) return -1;
            if (node.children.size() < (isRoot ? 2 : T)) return -1;
            for (BTreeNode child : node.children) {
                leafDepth = validate(child, false, depth + 1, leafDepth);
                if (leafDepth < 0) return -1;
            }
            return leafDepth;
        }
        if (!isRoot && node.keys.size() < T - 1) return -1;
        if (leafDepth != -1 && leafDepth != depth) return -1;
        return depth;
    }
}

public class Solution {
    static void sortedInsert(List<Integer> oracle, int value) {
        oracle.add(BTree.lowerBound(oracle, value), value);
    }

    public static void main(String[] args) {
        BTree tree = new BTree();
        check(tree.validate(), "empty validate");
        check(tree.inorderValues().isEmpty(), "empty order");
        check(!tree.delete(5), "delete from empty");

        for (int value : new int[]{10, 20, 30, 40}) tree.insert(value);
        check(tree.splits >= 1, "root split " + tree.splits);
        check(tree.inorderValues().toString().equals("[10, 20, 30, 40]"), "hand order");
        check(tree.validate(), "hand validate");
        check(tree.contains(30) && !tree.contains(35), "hand contains");
        check(tree.size == 4, "hand size");

        tree.insert(10);
        tree.insert(10);
        check(tree.inorderValues().toString().equals("[10, 10, 10, 20, 30, 40]"), "duplicates");
        check(tree.delete(10), "delete duplicate");
        check(tree.inorderValues().toString().equals("[10, 10, 20, 30, 40]"), "after one delete");
        check(tree.delete(10) && tree.delete(10), "delete remaining duplicates");
        check(!tree.contains(10), "no 10 left");
        check(!tree.delete(10), "delete absent 10");
        check(tree.inorderValues().toString().equals("[20, 30, 40]"), "after duplicate drain");

        for (int value : new int[]{35, 25, 15, 5, 45, 50, 55, 60}) tree.insert(value);
        check(tree.inorderValues().toString().equals("[5, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]"), "second hand order");
        check(tree.validate(), "second hand validate");
        for (int value : new int[]{30, 5, 60, 20, 45, 15, 55, 25, 35, 50, 40}) {
            check(tree.delete(value), "scripted hand delete");
        }
        check(tree.inorderValues().isEmpty(), "hand drained");
        check(tree.size == 0, "hand drained size");
        check(tree.validate(), "hand drained validate");

        tree = new BTree();
        List<Integer> oracle = new ArrayList<>();
        int seed = 4370;
        for (int step = 0; step < 300; step++) {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 4;
            seed = (seed * 75 + 74) % 65537;
            if (op <= 1) {
                int value = seed % 30;
                tree.insert(value);
                sortedInsert(oracle, value);
            } else if (op == 2 && !oracle.isEmpty()) {
                int value = oracle.remove(seed % oracle.size());
                check(tree.delete(value), "random delete step " + step);
                boolean stillPresent = oracle.contains(value);
                check(tree.delete(value) == stillPresent, "random second delete step " + step);
                if (stillPresent) oracle.remove(Integer.valueOf(value));
            } else {
                int value = 100 + seed % 5;
                check(!tree.delete(value), "random absent step " + step);
            }
            check(tree.size == oracle.size(), "random size step " + step);
            if (step % 20 == 0 || step == 299) {
                check(tree.inorderValues().equals(oracle), "random order step " + step);
                check(tree.validate(), "random validate step " + step);
            }
        }
        check(!oracle.isEmpty(), "workload left values");
        check(tree.inorderValues().equals(oracle), "final order");
        check(tree.validate(), "final validate");
        check(tree.splits > 0 && tree.merges > 0, "counters " + tree.splits + "/" + tree.merges + "/" + tree.borrows);

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
