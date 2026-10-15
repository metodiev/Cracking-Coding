/** Task 434 - Implement scapegoat tree basics (rebuild heuristic). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A scapegoat tree is an unbalanced BST that stays near-balanced by whole-subtree rebuilds instead
 * of rotations. Documented constants: balance factor alpha = 0.75 (a node may not have a child
 * subtree larger than alpha * its own size) and the delete trigger size < alpha * max_size, where
 * max_size is the peak size since the last full rebuild. Insert walks down, then walks back up
 * updating cached sizes and picks the deepest ancestor whose child exceeds alpha * its size - the
 * scapegoat - and rebuilds its subtree into a perfectly balanced half-split. Delete is physical:
 * the key is replaced by its in-order successor and sizes are repaired on the way up; falling below
 * alpha * max_size rebuilds the whole tree. Keys are unique (duplicate inserts are no-ops), matching
 * the sorted-list oracle used on LCG operation scripts. Rebuilds are counted: the adversarial
 * ascending insert 1..200 triggers at least one rebuild (a final assert documents the count), and
 * every state is checked for BST order, cached sizes and O(log n) height.
 */
import java.util.*;

class ScapegoatNode {
    int key;
    ScapegoatNode left;
    ScapegoatNode right;
    int subtreeSize;

    ScapegoatNode(int key) {
        this.key = key;
        this.subtreeSize = 1;
    }
}

class ScapegoatTree {
    private final double alpha;
    private ScapegoatNode root;
    private int total;
    private int maxSize;
    int rebuilds;

    ScapegoatTree() {
        this(0.75);
    }

    ScapegoatTree(double alpha) {
        this.alpha = alpha;
    }

    int size() {
        return total;
    }

    boolean contains(int key) {
        ScapegoatNode node = root;
        while (node != null) {
            if (key == node.key) return true;
            node = key < node.key ? node.left : node.right;
        }
        return false;
    }

    boolean insert(int key) {
        if (root == null) {
            root = new ScapegoatNode(key);
            total = 1;
            maxSize = 1;
            return true;
        }
        List<ScapegoatNode> path = new ArrayList<>();
        ScapegoatNode node = root;
        while (true) {
            path.add(node);
            if (key == node.key) return false;
            if (key < node.key) {
                if (node.left == null) {
                    node.left = new ScapegoatNode(key);
                    break;
                }
                node = node.left;
            } else {
                if (node.right == null) {
                    node.right = new ScapegoatNode(key);
                    break;
                }
                node = node.right;
            }
        }
        int scapegoatIndex = -1;
        for (int i = path.size() - 1; i >= 0; i--) {
            ScapegoatNode current = path.get(i);
            current.subtreeSize++;
            if (scapegoatIndex == -1) {
                int left = current.left == null ? 0 : current.left.subtreeSize;
                int right = current.right == null ? 0 : current.right.subtreeSize;
                if (left > alpha * current.subtreeSize || right > alpha * current.subtreeSize) {
                    scapegoatIndex = i;
                }
            }
        }
        total++;
        maxSize = Math.max(maxSize, total);
        if (scapegoatIndex >= 0) rebuildAt(path, scapegoatIndex);
        return true;
    }

    boolean delete(int key) {
        if (!contains(key)) return false;
        root = delete(root, key);
        total--;
        if (total < alpha * maxSize) {
            List<Integer> keys = collect(root);
            root = build(keys, 0, keys.size() - 1);
            maxSize = total;
            rebuilds++;
        }
        return true;
    }

    private ScapegoatNode delete(ScapegoatNode node, int key) {
        if (key < node.key) node.left = delete(node.left, key);
        else if (key > node.key) node.right = delete(node.right, key);
        else return deleteNode(node);
        update(node);
        return node;
    }

    private ScapegoatNode deleteNode(ScapegoatNode node) {
        if (node.left == null) return node.right;
        if (node.right == null) return node.left;
        ScapegoatNode successor = node.right;
        while (successor.left != null) successor = successor.left;
        node.key = successor.key;
        node.right = delete(node.right, successor.key);
        update(node);
        return node;
    }

    List<Integer> inorder() {
        List<Integer> out = new ArrayList<>();
        Deque<ScapegoatNode> stack = new ArrayDeque<>();
        ScapegoatNode node = root;
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            out.add(node.key);
            node = node.right;
        }
        return out;
    }

    int height() {
        return height(root);
    }

    private int height(ScapegoatNode node) {
        if (node == null) return 0;
        return 1 + Math.max(height(node.left), height(node.right));
    }

    boolean validate() {
        return validate(root, Long.MIN_VALUE, Long.MAX_VALUE) == total;
    }

    private int validate(ScapegoatNode node, long low, long high) {
        if (node == null) return 0;
        if (!(low < node.key && node.key < high)) return -1;
        int leftSize = validate(node.left, low, node.key);
        int rightSize = validate(node.right, node.key, high);
        if (leftSize < 0 || rightSize < 0) return -1;
        if (node.subtreeSize != 1 + leftSize + rightSize) return -1;
        return node.subtreeSize;
    }

    private List<Integer> collect(ScapegoatNode node) {
        List<Integer> out = new ArrayList<>();
        Deque<ScapegoatNode> stack = new ArrayDeque<>();
        while (!stack.isEmpty() || node != null) {
            while (node != null) {
                stack.push(node);
                node = node.left;
            }
            node = stack.pop();
            out.add(node.key);
            node = node.right;
        }
        return out;
    }

    private void update(ScapegoatNode node) {
        node.subtreeSize = 1 + (node.left == null ? 0 : node.left.subtreeSize)
                + (node.right == null ? 0 : node.right.subtreeSize);
    }

    private void rebuildAt(List<ScapegoatNode> path, int index) {
        ScapegoatNode node = path.get(index);
        ScapegoatNode rebuilt = build(collect(node), 0, node.subtreeSize - 1);
        if (index == 0) {
            root = rebuilt;
        } else {
            ScapegoatNode parent = path.get(index - 1);
            if (parent.left == node) parent.left = rebuilt;
            else parent.right = rebuilt;
        }
        rebuilds++;
    }

    private ScapegoatNode build(List<Integer> keys, int lo, int hi) {
        if (lo > hi) return null;
        int mid = (lo + hi) / 2;
        ScapegoatNode node = new ScapegoatNode(keys.get(mid));
        node.left = build(keys, lo, mid - 1);
        node.right = build(keys, mid + 1, hi);
        update(node);
        return node;
    }
}

public class Solution {
    static long heightBound(int size) {
        return 3L * (long) Math.ceil(Math.log(size + 1) / Math.log(2)) + 3;
    }

    public static void main(String[] args) {
        ScapegoatTree tree = new ScapegoatTree();
        check(tree.size() == 0, "empty size");
        check(tree.inorder().isEmpty(), "empty order");
        check(tree.validate(), "empty validate");
        check(!tree.delete(0), "delete from empty");

        for (int key : new int[]{10, 5, 15, 3}) check(tree.insert(key), "hand insert");
        check(!tree.insert(10), "duplicate insert");
        check(tree.inorder().toString().equals("[3, 5, 10, 15]"), "hand order");
        check(tree.contains(15) && !tree.contains(4), "contains");
        check(tree.validate(), "hand validate");
        check(tree.delete(5), "hand delete");
        check(!tree.delete(5), "repeat delete");
        check(tree.inorder().toString().equals("[3, 10, 15]"), "order after delete");
        check(tree.validate(), "validate after delete");

        tree = new ScapegoatTree();
        for (int key = 1; key <= 200; key++) tree.insert(key);
        List<Integer> ascending = new ArrayList<>();
        for (int key = 1; key <= 200; key++) ascending.add(key);
        check(tree.inorder().equals(ascending), "ascending order");
        check(tree.rebuilds >= 1, "ascending rebuilds " + tree.rebuilds);
        check(tree.height() <= heightBound(tree.size()), "ascending height " + tree.height());
        for (int key = 2; key <= 200; key += 2) check(tree.delete(key), "ascending delete");
        List<Integer> odds = new ArrayList<>();
        for (int key = 1; key <= 200; key += 2) odds.add(key);
        check(tree.inorder().equals(odds), "odds order");
        check(tree.validate(), "odds validate");
        for (int key = 1; key <= 200; key += 2) check(tree.delete(key), "drain delete");
        check(tree.size() == 0 && tree.inorder().isEmpty(), "drained");
        check(tree.validate(), "drained validate");

        tree = new ScapegoatTree();
        List<Integer> oracle = new ArrayList<>();
        int seed = 4340;
        for (int step = 0; step < 400; step++) {
            seed = (seed * 75 + 74) % 65537;
            int op = seed % 3;
            seed = (seed * 75 + 74) % 65537;
            int key = seed % 100;
            if (op == 0) {
                boolean got = tree.insert(key);
                boolean expected = !oracle.contains(key);
                if (expected) {
                    int index = Collections.binarySearch(oracle, key);
                    oracle.add(index < 0 ? -(index + 1) : index, key);
                }
                check(got == expected, "random insert step " + step);
            } else {
                boolean got = tree.delete(key);
                boolean expected = oracle.contains(key);
                if (expected) oracle.remove(Integer.valueOf(key));
                check(got == expected, "random delete step " + step);
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
        check(tree.height() <= heightBound(tree.size()), "final height " + tree.height());

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
