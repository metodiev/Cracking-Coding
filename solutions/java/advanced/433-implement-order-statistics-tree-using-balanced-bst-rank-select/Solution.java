/** Task 433 - Implement order statistics tree using balanced BST (rank/select). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The balanced BST is a treap: a BST ordered by key plus a min-heap ordered by priority, kept
 * balanced by rotations on insert and by merging the two children on delete. Each node caches the
 * size of its subtree and a multiplicity count, so it acts as a multiset: rank(x) returns the
 * number of stored elements strictly below x and select(k) returns the k-th smallest (1-based,
 * counting multiplicities). Priorities are drawn from the shared LCG stream, which makes the tree
 * shape - and therefore every assertion - fully deterministic. Assertions compare insert/delete/
 * rank/select/count against a sorted-list oracle on LCG operation scripts and re-check the
 * heap/BST/size invariants with a structural validator.
 */
import java.util.*;

class TreapNode {
    int key;
    int priority;
    int count;
    int subtreeSize;
    TreapNode left;
    TreapNode right;

    TreapNode(int key, int priority) {
        this.key = key;
        this.priority = priority;
        this.count = 1;
        this.subtreeSize = 1;
    }
}

class OrderStatisticsTree {
    private TreapNode root;
    private int total;
    private boolean removed;

    int size() {
        return total;
    }

    private static int sizeOf(TreapNode node) {
        return node == null ? 0 : node.subtreeSize;
    }

    private static void update(TreapNode node) {
        node.subtreeSize = sizeOf(node.left) + node.count + sizeOf(node.right);
    }

    private TreapNode rotateRight(TreapNode node) {
        TreapNode child = node.left;
        node.left = child.right;
        child.right = node;
        update(node);
        update(child);
        return child;
    }

    private TreapNode rotateLeft(TreapNode node) {
        TreapNode child = node.right;
        node.right = child.left;
        child.left = node;
        update(node);
        update(child);
        return child;
    }

    void insert(int key, int priority) {
        root = insert(root, key, priority);
        total++;
    }

    private TreapNode insert(TreapNode node, int key, int priority) {
        if (node == null) return new TreapNode(key, priority);
        if (key == node.key) {
            node.count++;
            node.subtreeSize++;
            return node;
        }
        if (key < node.key) {
            node.left = insert(node.left, key, priority);
            if (node.left.priority < node.priority) return rotateRight(node);
        } else {
            node.right = insert(node.right, key, priority);
            if (node.right.priority < node.priority) return rotateLeft(node);
        }
        update(node);
        return node;
    }

    boolean delete(int key) {
        removed = false;
        root = delete(root, key);
        if (removed) total--;
        return removed;
    }

    private TreapNode delete(TreapNode node, int key) {
        if (node == null) return null;
        if (key == node.key) {
            removed = true;
            if (node.count > 1) {
                node.count--;
                node.subtreeSize--;
                return node;
            }
            return merge(node.left, node.right);
        }
        if (key < node.key) node.left = delete(node.left, key);
        else node.right = delete(node.right, key);
        update(node);
        return node;
    }

    private TreapNode merge(TreapNode left, TreapNode right) {
        if (left == null) return right;
        if (right == null) return left;
        if (left.priority <= right.priority) {
            left.right = merge(left.right, right);
            update(left);
            return left;
        }
        right.left = merge(left, right.left);
        update(right);
        return right;
    }

    int rank(int key) {
        return rank(root, key);
    }

    private int rank(TreapNode node, int key) {
        if (node == null) return 0;
        if (key <= node.key) return rank(node.left, key);
        return sizeOf(node.left) + node.count + rank(node.right, key);
    }

    int select(int k) {
        if (k < 1 || k > total) throw new IndexOutOfBoundsException("select index out of range");
        return select(root, k);
    }

    private int select(TreapNode node, int k) {
        int leftSize = sizeOf(node.left);
        if (k <= leftSize) return select(node.left, k);
        if (k <= leftSize + node.count) return node.key;
        return select(node.right, k - leftSize - node.count);
    }

    int countOf(int key) {
        TreapNode node = root;
        while (node != null) {
            if (key == node.key) return node.count;
            node = key < node.key ? node.left : node.right;
        }
        return 0;
    }

    List<Integer> inorder() {
        List<Integer> out = new ArrayList<>();
        inorder(root, out);
        return out;
    }

    private void inorder(TreapNode node, List<Integer> out) {
        if (node == null) return;
        inorder(node.left, out);
        for (int i = 0; i < node.count; i++) out.add(node.key);
        inorder(node.right, out);
    }

    boolean validate() {
        int size = validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
        return size == total;
    }

    private int validate(TreapNode node, long low, long high) {
        if (node == null) return 0;
        if (!(low < node.key && node.key < high)) return -1;
        int leftSize = validate(node.left, low, node.key);
        int rightSize = validate(node.right, node.key, high);
        if (leftSize < 0 || rightSize < 0) return -1;
        if (node.subtreeSize != leftSize + node.count + rightSize) return -1;
        if (node.left != null && node.left.priority < node.priority) return -1;
        if (node.right != null && node.right.priority < node.priority) return -1;
        return node.subtreeSize;
    }
}

public class Solution {
    static void sortedInsert(List<Integer> sortedValues, int value) {
        int index = Collections.binarySearch(sortedValues, value);
        if (index < 0) index = -(index + 1);
        sortedValues.add(index, value);
    }

    public static void main(String[] args) {
        OrderStatisticsTree tree = new OrderStatisticsTree();
        check(tree.size() == 0, "empty size");
        int[][] seeded = {{5, 100}, {1, 90}, {9, 80}, {3, 70}, {3, 110}};
        for (int[] pair : seeded) tree.insert(pair[0], pair[1]);
        check(tree.inorder().toString().equals("[1, 3, 3, 5, 9]"), "hand inorder");
        check(tree.size() == 5, "hand size");
        check(tree.validate(), "hand validate");
        check(tree.select(1) == 1 && tree.select(2) == 3 && tree.select(3) == 3, "hand select low");
        check(tree.select(4) == 5 && tree.select(5) == 9, "hand select high");
        check(tree.rank(1) == 0 && tree.rank(3) == 1, "hand rank low");
        check(tree.rank(4) == 3 && tree.rank(9) == 4, "hand rank high");
        check(tree.countOf(3) == 2 && tree.countOf(4) == 0, "hand counts");
        check(tree.delete(3), "hand delete");
        check(tree.inorder().toString().equals("[1, 3, 5, 9]"), "inorder after delete");
        check(tree.select(3) == 5, "select after delete");
        check(!tree.delete(100), "absent delete");
        check(tree.validate(), "validate after delete");

        tree = new OrderStatisticsTree();
        List<Integer> oracle = new ArrayList<>();
        int seed = 4331;
        for (int step = 0; step < 400; step++) {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 5;
            if (op <= 1) {
                seed = (seed * 75 + 74) % 65537;
                int key = seed % 60;
                seed = (seed * 75 + 74) % 65537;
                tree.insert(key, seed);
                sortedInsert(oracle, key);
            } else if (op == 2) {
                seed = (seed * 75 + 74) % 65537;
                int key = seed % 60;
                boolean expected = oracle.contains(key);
                if (expected) oracle.remove(Integer.valueOf(key));
                check(tree.delete(key) == expected, "random delete step " + step);
            } else if (op == 3) {
                seed = (seed * 75 + 74) % 65537;
                int key = seed % 60;
                int below = 0;
                int count = 0;
                for (int value : oracle) {
                    if (value < key) below++;
                    if (value == key) count++;
                }
                check(tree.rank(key) == below, "random rank step " + step);
                check(tree.countOf(key) == count, "random count step " + step);
            } else if (!oracle.isEmpty()) {
                seed = (seed * 75 + 74) % 65537;
                int k = 1 + seed % oracle.size();
                check(tree.select(k) == oracle.get(k - 1), "random select step " + step);
            }
            check(tree.size() == oracle.size(), "random size step " + step);
            if (step % 25 == 0) {
                check(tree.inorder().equals(oracle), "random inorder step " + step);
                check(tree.validate(), "random validate step " + step);
            }
        }
        check(!oracle.isEmpty(), "workload left values");
        check(tree.inorder().equals(oracle), "final inorder");
        check(tree.validate(), "final validate");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
