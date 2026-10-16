/** Task 612 - Implement advanced persistent data structures with path copying optimization. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Path copying means an update never mutates a node: it rebuilds the nodes on the search path only and
 * shares every untouched subtree with the previous version, so both versions stay valid and an update
 * costs O(log n) fresh nodes instead of the O(n) of copying the whole structure. Two structures are
 * built that way here:
 *   - PersistentStack, a cons list with push/pop that returns a new head and shares the tail;
 *   - PersistentAvl, an immutable AVL tree whose insert/delete rebalance with rotations that also copy
 *     their path (a single rotation allocates exactly two nodes) and which returns the *same* root
 *     when the operation is a no-op (a duplicate insert or a delete of a missing key), the cheap way to
 *     recognise "this version is unchanged".
 * Every allocation goes through the AvlNode constructor, so the cost of an operation is measured, not
 * estimated.
 *
 * Verified: the AVL invariants (ordering, balance factor, height) hold for every version; each version
 * holds exactly the keys of its TreeSet oracle snapshot, including all older versions after later
 * updates, which is what persistence means; random search hits and misses agree with the oracle; an
 * update allocates at most c * log2(n) nodes (checked per operation) and the whole version chain
 * allocates far fewer nodes than one full copy per version; no-op updates return the identical root;
 * empty structures and delete-to-empty and re-insert are covered.
 */
import java.util.*;

class PersistentStack {
    static final class Cell {
        final int value;
        final Cell next;
        final int depth;                       // number of elements below this cell, so size is O(1)
        Cell(int value, Cell next) {
            this.value = value;
            this.next = next;
            this.depth = next == null ? 1 : next.depth + 1;
        }
    }

    static Cell empty() { return null; }

    static Cell push(Cell top, int value) { return new Cell(value, top); }

    static Cell pop(Cell top) {
        if (top == null) throw new IllegalStateException("pop of an empty stack");
        return top.next;
    }

    static int size(Cell top) { return top == null ? 0 : top.depth; }

    static int peek(Cell top) {
        if (top == null) throw new IllegalStateException("peek of an empty stack");
        return top.value;
    }

    static int[] toArray(Cell top) {
        int[] out = new int[size(top)];
        int i = out.length;
        for (Cell c = top; c != null; c = c.next) out[--i] = c.value;
        return out;
    }
}

class PersistentAvl {
    static long allocated = 0;                 // every node of every version is counted here

    static final class Node {
        final int key;
        final Node left, right;
        final int height;
        final int size;                        // O(1) size, kept in the node like the height

        Node(int key, Node left, Node right) {
            this.key = key;
            this.left = left;
            this.right = right;
            this.height = 1 + Math.max(height(left), height(right));
            this.size = 1 + size(left) + size(right);
            allocated++;
        }
    }

    static int height(Node n) { return n == null ? 0 : n.height; }

    static int size(Node n) { return n == null ? 0 : n.size; }

    /** The 4-ary tree shape is a strict AVL tree; this is the textbook bound, used in the checks. */
    static int heightBound(int size) {
        int bound = 0;
        long fib = 1, prev = 0;
        while (fib - 1 < size) {
            long next = fib + prev + 1;
            prev = fib;
            fib = next;
            bound++;
        }
        return bound;
    }

    static int balance(Node n) { return height(n.left) - height(n.right); }

    static Node insert(Node t, int key) {
        if (contains(t, key)) return t;          // identity shortcut: an unchanged version is returned
        return insertNew(t, key);
    }

    private static Node insertNew(Node t, int key) {
        if (t == null) return new Node(key, null, null);
        if (key < t.key) return rebalance(new Node(t.key, insertNew(t.left, key), t.right));
        return rebalance(new Node(t.key, t.left, insertNew(t.right, key)));
    }

    static Node delete(Node t, int key) {
        if (t == null || !contains(t, key)) return t;                   // no-op: the very same version
        return deleteFound(t, key);
    }

    private static Node deleteFound(Node t, int key) {
        if (key < t.key) return rebalance(new Node(t.key, deleteFound(t.left, key), t.right));
        if (key > t.key) return rebalance(new Node(t.key, t.left, deleteFound(t.right, key)));
        if (t.left == null) return t.right;
        if (t.right == null) return t.left;
        Node successor = minimum(t.right);
        return rebalance(new Node(successor.key, t.left, deleteFound(t.right, successor.key)));
    }

    static Node minimum(Node t) {
        while (t.left != null) t = t.left;
        return t;
    }

    static boolean contains(Node t, int key) {
        while (t != null) {
            if (key == t.key) return true;
            t = key < t.key ? t.left : t.right;
        }
        return false;
    }

    /** Rotations copy the two nodes they touch, so a rotation costs exactly two fresh nodes. */
    static Node rebalance(Node t) {
        int b = balance(t);
        if (b > 1) {
            Node l = t.left;
            if (balance(l) >= 0) return rotateRight(t);
            Node fixed = new Node(t.key, rotateLeft(l), t.right);
            return rotateRight(fixed);
        }
        if (b < -1) {
            Node r = t.right;
            if (balance(r) <= 0) return rotateLeft(t);
            Node fixed = new Node(t.key, t.left, rotateRight(r));
            return rotateLeft(fixed);
        }
        return t;
    }

    static Node rotateRight(Node t) {
        Node l = t.left;
        Node inner = new Node(t.key, l.right, t.right);
        return new Node(l.key, l.left, inner);
    }

    static Node rotateLeft(Node t) {
        Node r = t.right;
        Node inner = new Node(t.key, t.left, r.left);
        return new Node(r.key, inner, r.right);
    }

    static int[] toArray(Node t) {
        int[] out = new int[size(t)];
        fill(t, out, 0);
        return out;
    }

    private static int fill(Node n, int[] out, int at) {
        if (n == null) return at;
        at = fill(n.left, out, at);
        out[at++] = n.key;
        return fill(n.right, out, at);
    }

    /** Structural check of the AVL invariant over a whole version. */
    static boolean isValid(Node n) {
        if (n == null) return true;
        if (Math.abs(balance(n)) > 1) return false;
        if (n.height != 1 + Math.max(height(n.left), height(n.right))) return false;
        if (n.size != 1 + size(n.left) + size(n.right)) return false;
        if (n.left != null && n.left.key >= n.key) return false;
        if (n.right != null && n.right.key <= n.key) return false;
        return isValid(n.left) && isValid(n.right);
    }
}

public class Solution {
    private static long state = 0x13198A2E03707344L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static void checkStack() {
        PersistentStack.Cell empty = PersistentStack.empty();
        check(PersistentStack.size(empty) == 0, "an empty stack has size 0");
        PersistentStack.Cell[] versions = new PersistentStack.Cell[64];
        versions[0] = empty;
        for (int i = 1; i < versions.length; i++) {
            versions[i] = PersistentStack.push(versions[i - 1], i);
            check(PersistentStack.size(versions[i]) == i, "size of stack version " + i);
            check(PersistentStack.peek(versions[i]) == i, "top of stack version " + i);
        }
        for (int i = 0; i < versions.length; i++) {
            int[] expected = new int[i];
            for (int j = 0; j < i; j++) expected[j] = j + 1;
            check(Arrays.equals(PersistentStack.toArray(versions[i]), expected),
                    "stack version " + i + " keeps its own contents");
            check(PersistentStack.size(versions[i]) == i, "older stack versions stay untouched");
        }
        PersistentStack.Cell shorter = PersistentStack.pop(versions[10]);
        check(PersistentStack.size(shorter) == 9 && PersistentStack.peek(shorter) == 9, "pop shares the tail");
        check(PersistentStack.size(versions[10]) == 10, "the popped version is unchanged");
        boolean threw = false;
        try { PersistentStack.pop(empty); } catch (IllegalStateException e) { threw = true; }
        check(threw, "pop of an empty stack rejected");
    }

    static void checkPersistentAvl() {
        List<PersistentAvl.Node> versions = new ArrayList<>();
        List<Set<Integer>> snapshots = new ArrayList<>();
        PersistentAvl.Node root = null;
        Set<Integer> model = new TreeSet<>();
        versions.add(root);
        snapshots.add(new TreeSet<>(model));
        long worstCost = 0;
        long fullCopyCost = 0;

        for (int step = 0; step < 240; step++) {
            long before = PersistentAvl.allocated;
            boolean grow = model.size() < 60 && (nextLong() >>> 33) % 100 < 60;
            int key = (int) ((nextLong() >>> 33) % 90);
            if (grow) {
                PersistentAvl.Node sameRootIfPresent = PersistentAvl.contains(root, key) ? root : null;
                root = PersistentAvl.insert(root, key);
                if (sameRootIfPresent != null) check(root == sameRootIfPresent, "duplicate insert is a no-op");
                model.add(key);
            } else if (!model.isEmpty()) {
                boolean present = model.remove(key);
                PersistentAvl.Node before2 = root;
                root = PersistentAvl.delete(root, key);
                if (!present) check(root == before2, "deleting a missing key is a no-op");
            } else {
                continue;
            }
            long cost = PersistentAvl.allocated - before;
            worstCost = Math.max(worstCost, cost);
            fullCopyCost += model.size();
            long height = PersistentAvl.height(root);
            check(cost <= 3 * (height + 2), "path copying cost " + cost + " at height " + height);
            check(PersistentAvl.isValid(root), "the AVL invariant after step " + step);
            check(PersistentAvl.height(root) <= PersistentAvl.heightBound(model.size()) + 1,
                    "height stays logarithmic at step " + step);
            check(PersistentAvl.size(root) == model.size(), "size after step " + step);
            int[] array = PersistentAvl.toArray(root);
            int[] expected = new int[model.size()];
            int i = 0;
            for (int value : model) expected[i++] = value;
            check(Arrays.equals(array, expected), "in-order traversal is sorted, step " + step);
            versions.add(root);
            snapshots.add(new TreeSet<>(model));
        }

        check(model.size() > 5, "the random walk kept a non-trivial tree");
        // every historical version must still hold exactly the keys of its own snapshot
        for (int v = 0; v < versions.size(); v++) {
            PersistentAvl.Node version = versions.get(v);
            Set<Integer> snapshot = snapshots.get(v);
            check(PersistentAvl.size(version) == snapshot.size(), "version " + v + " size");
            int[] array = PersistentAvl.toArray(version);
            int[] expected = new int[snapshot.size()];
            int i = 0;
            for (int value : snapshot) expected[i++] = value;
            check(Arrays.equals(array, expected), "version " + v + " contents survive later updates");
            for (int probe = 0; probe < 90; probe++) {
                check(PersistentAvl.contains(version, probe) == snapshot.contains(probe),
                        "version " + v + " search for " + probe);
            }
            check(PersistentAvl.isValid(version), "version " + v + " is a valid AVL");
        }

        // deleting everything, then re-inserting, must behave like a fresh tree
        PersistentAvl.Node shrinking = versions.get(versions.size() - 1);
        Set<Integer> keys = new TreeSet<>(model);
        for (int key : keys) shrinking = PersistentAvl.delete(shrinking, key);
        check(shrinking == null, "deleting every key leaves the empty tree");
        check(PersistentAvl.delete(null, 5) == null && !PersistentAvl.contains(null, 5), "empty tree edges");
        check(PersistentAvl.insert(null, 5).size == 1, "insert into the empty tree");

        long total = PersistentAvl.allocated;
        System.out.println("persistent AVL: " + versions.size() + " versions, " + total
                + " nodes allocated, " + fullCopyCost + " nodes if each version were copied whole,"
                + " worst path-copying cost " + worstCost);
        check(total < fullCopyCost, "path copying beats copying whole versions");
    }

    public static void main(String[] args) {
        checkStack();
        checkPersistentAvl();

        int[] fixed = {5, 3, 8, 1, 4, 7, 9, 2, 6};
        PersistentAvl.Node tree = null;
        for (int key : fixed) tree = PersistentAvl.insert(tree, key);
        check(PersistentAvl.height(tree) == 4, "the classic shape has height 4");
        check(Arrays.equals(PersistentAvl.toArray(tree), new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9}), "sorted");
        PersistentAvl.Node without = PersistentAvl.delete(tree, 5);
        check(PersistentAvl.size(without) == 8 && PersistentAvl.size(tree) == 9, "delete keeps the old version");
        check(Arrays.equals(PersistentAvl.toArray(without), new int[]{1, 2, 3, 4, 6, 7, 8, 9}), "delete result");
        check(PersistentAvl.isValid(tree) && PersistentAvl.isValid(without), "both versions are AVLs");
        PersistentAvl.Node again = PersistentAvl.insert(without, 5);
        check(Arrays.equals(PersistentAvl.toArray(again), new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                "re-inserting the deleted key restores the set");
        check(PersistentAvl.contains(again, 9) && !PersistentAvl.contains(again, 42), "search hit and miss");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
