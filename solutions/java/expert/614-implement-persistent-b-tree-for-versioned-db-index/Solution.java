/** Task 614 - Implement persistent B-tree for versioned DB index. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * A versioned index keeps one B-tree root per committed version. Every put/delete is copy-on-write:
 * the nodes on the root-to-leaf path are rebuilt (O(t * log_t n) fresh nodes, where t is the minimum
 * degree), every untouched subtree is shared with the previous version, and the returned root is
 * appended as a new version. Old roots stay readable, which is exactly a snapshot index for MVCC.
 * Insert follows CLRS: split a full root first, then split any full child on the way down. Delete is
 * the full CLRS deletion with preemptive borrowing and merging so that no node ever holds fewer than
 * t-1 keys, which is what makes the "all leaves at the same depth" invariant survive deletions.
 * A put of an existing key or a delete of a missing key returns the identical root (and allocates
 * nothing), so "this version did not change" is decidable by reference.
 *
 * Verified on a chain of versions: the in-order keys of every version equal its TreeSet snapshot, all
 * versions stay readable after later updates, the structural invariants (key counts per node, sorted
 * keys, separator keys, equal leaf depth) hold for every version, range scans match the oracle, the
 * number of nodes allocated per put/delete is bounded, and the empty tree, single key, delete-everything
 * and re-insert-after-delete cases are covered.
 */
import java.util.*;

class BTree {
    static final int T = 3;                        // minimum degree: nodes hold T-1 .. 2T-1 keys
    static long allocated = 0;

    static final class Node {
        final boolean leaf;
        final int[] keys = new int[2 * T - 1];
        final Node[] children = new Node[2 * T];
        int count;

        Node(boolean leaf) {
            this.leaf = leaf;
            allocated++;
        }
    }

    static Node emptyLeaf() { return new Node(true); }

    /** First slot whose key is >= key (or count when all keys are smaller). */
    static int slot(Node n, int key) {
        int i = 0;
        while (i < n.count && n.keys[i] < key) i++;
        return i;
    }

    static boolean contains(Node n, int key) {
        while (n != null) {
            int i = slot(n, key);
            if (i < n.count && n.keys[i] == key) return true;
            if (n.leaf) return false;
            n = n.children[i];
        }
        return false;
    }

    static int minKey(Node n) {
        while (!n.leaf) n = n.children[0];
        return n.keys[0];
    }

    static int maxKey(Node n) {
        while (!n.leaf) n = n.children[n.count];
        return n.keys[n.count - 1];
    }

    static Node copyOf(Node n) {
        Node copy = new Node(n.leaf);
        copy.count = n.count;
        System.arraycopy(n.keys, 0, copy.keys, 0, n.count);
        if (!n.leaf) System.arraycopy(n.children, 0, copy.children, 0, n.count + 1);
        return copy;
    }

    /** Inserts key at slot `at`; the new child slot `at` is `child`, or the old pointer when null. */
    static Node insertKeyChild(Node n, int at, int key, Node child) {
        Node copy = new Node(n.leaf);
        for (int i = 0; i < at; i++) {
            copy.keys[i] = n.keys[i];
            if (!n.leaf) copy.children[i] = n.children[i];
        }
        copy.keys[at] = key;
        if (!n.leaf) copy.children[at] = child != null ? child : n.children[at];
        for (int i = at; i < n.count; i++) {
            copy.keys[i + 1] = n.keys[i];
            if (!n.leaf) copy.children[i + 1] = n.children[i];
        }
        if (!n.leaf) copy.children[n.count + 1] = n.children[n.count];
        copy.count = n.count + 1;
        return copy;
    }

    /** Removes key at slot i (and, for internal nodes, the child right after it). */
    static Node removeKeyChild(Node n, int i) {
        Node copy = new Node(n.leaf);
        for (int k = 0; k < i; k++) {
            copy.keys[k] = n.keys[k];
            if (!n.leaf) copy.children[k] = n.children[k];
        }
        for (int k = i + 1; k < n.count; k++) {
            copy.keys[k - 1] = n.keys[k];
            if (!n.leaf) copy.children[k - 1] = n.children[k];
        }
        if (!n.leaf) copy.children[n.count - 1] = n.children[n.count];
        copy.count = n.count - 1;
        return copy;
    }

    static Node replaceChild(Node n, int i, Node child) {
        Node copy = copyOf(n);
        copy.children[i] = child;
        return copy;
    }

    static Node replaceKey(Node n, int i, int key) {
        Node copy = copyOf(n);
        copy.keys[i] = key;
        return copy;
    }

    /** Child at slot i is full: promote its median and return the rebuilt parent. */
    static Node splitChild(Node n, int i) {
        Node full = n.children[i];
        Node left = new Node(full.leaf);
        Node right = new Node(full.leaf);
        int median = full.keys[T - 1];
        left.count = T - 1;
        right.count = T - 1;
        for (int k = 0; k < T - 1; k++) {
            left.keys[k] = full.keys[k];
            right.keys[k] = full.keys[k + T];
        }
        if (!full.leaf) {
            for (int k = 0; k < T; k++) left.children[k] = full.children[k];
            for (int k = 0; k < T; k++) right.children[k] = full.children[k + T];
        }
        Node mid = insertKeyChild(n, i, median, null);
        mid = replaceChild(mid, i, left);
        return replaceChild(mid, i + 1, right);
    }

    static Node mergeWithKey(Node left, int key, Node right) {
        Node merged = new Node(left.leaf);
        for (int k = 0; k < left.count; k++) merged.keys[k] = left.keys[k];
        merged.keys[left.count] = key;
        for (int k = 0; k < right.count; k++) merged.keys[left.count + 1 + k] = right.keys[k];
        if (!left.leaf) {
            for (int k = 0; k <= left.count; k++) merged.children[k] = left.children[k];
            for (int k = 0; k <= right.count; k++) merged.children[left.count + 1 + k] = right.children[k];
        }
        merged.count = left.count + right.count + 1;
        return merged;
    }

    /** Merges child i, key i and child i+1 into slot i of a rebuilt parent. */
    static Node mergeChildren(Node n, int i) {
        Node merged = mergeWithKey(n.children[i], n.keys[i], n.children[i + 1]);
        Node shrunk = removeKeyChild(n, i);
        return replaceChild(shrunk, i, merged);
    }

    static final class Descent {
        final Node parent;
        final int index;
        Descent(Node parent, int index) {
            this.parent = parent;
            this.index = index;
        }
    }

    static Descent borrowFromLeft(Node n, int i) {
        Node left = n.children[i - 1];
        Node child = n.children[i];
        int moved = left.keys[left.count - 1];
        Node newLeft = copyOf(left);
        newLeft.count--;
        Node newChild = insertKeyChild(child, 0, n.keys[i - 1], left.leaf ? null : left.children[left.count]);
        Node parent = replaceKey(n, i - 1, moved);
        parent = replaceChild(parent, i - 1, newLeft);
        return new Descent(replaceChild(parent, i, newChild), i);
    }

    /** The right sibling lends its first key: the parent's key i moves down as the child's last key. */
    static Descent borrowFromRight(Node n, int i) {
        Node right = n.children[i + 1];
        Node child = n.children[i];
        int moved = right.keys[0];
        Node movedChild = right.leaf ? null : right.children[0];
        Node newRight = removeKeyChild(right, 0);
        Node parent = replaceKey(n, i, moved);
        parent = replaceChild(parent, i + 1, newRight);
        Node newChild = insertKeyChild(child, child.count, n.keys[i], null);
        if (!child.leaf) newChild = replaceChild(newChild, child.count + 1, movedChild);
        return new Descent(replaceChild(parent, i, newChild), i);
    }

    /** Makes the child at slot i hold at least T keys, either by borrowing or by merging. */
    static Descent fill(Node n, int i) {
        if (i > 0 && n.children[i - 1].count >= T) return borrowFromLeft(n, i);
        if (i < n.count && n.children[i + 1].count >= T) return borrowFromRight(n, i);
        if (i < n.count) return new Descent(mergeChildren(n, i), i);
        return new Descent(mergeChildren(n, i - 1), i - 1);
    }

    static Node insert(Node root, int key) {
        if (root == null) return insertKeyChild(emptyLeaf(), 0, key, null);
        if (contains(root, key)) return root;
        if (root.count == 2 * T - 1) {
            Node newRoot = new Node(false);
            newRoot.children[0] = root;
            root = splitChild(newRoot, 0);
        }
        return insertNonFull(root, key);
    }

    private static Node insertNonFull(Node n, int key) {
        int i = slot(n, key);
        if (n.leaf) return insertKeyChild(n, i, key, null);
        if (n.children[i].count == 2 * T - 1) {
            Node fixed = splitChild(n, i);
            i = key < fixed.keys[i] ? i : i + 1;
            return replaceChild(fixed, i, insertNonFull(fixed.children[i], key));
        }
        return replaceChild(n, i, insertNonFull(n.children[i], key));
    }

    static Node delete(Node root, int key) {
        if (root == null || !contains(root, key)) return root;
        Node result = deleteRec(root, key);
        if (result.count == 0) return result.leaf ? null : result.children[0];   // empty leaf, or the root shrank
        return result;
    }

    private static Node deleteRec(Node n, int key) {
        int i = slot(n, key);
        boolean found = i < n.count && n.keys[i] == key;
        if (n.leaf) return found ? removeKeyChild(n, i) : n;
        if (found) {
            Node left = n.children[i], right = n.children[i + 1];
            if (left.count >= T) {
                int predecessor = maxKey(left);
                Node newLeft = deleteRec(left, predecessor);
                return replaceChild(replaceKey(n, i, predecessor), i, newLeft);
            }
            if (right.count >= T) {
                int successor = minKey(right);
                Node newRight = deleteRec(right, successor);
                return replaceChild(replaceKey(n, i, successor), i + 1, newRight);
            }
            Node merged = mergeWithKey(left, n.keys[i], right);
            Node shrunk = removeKeyChild(n, i);
            return deleteRec(replaceChild(shrunk, i, merged), key);
        }
        if (n.children[i].count == T - 1) {
            Descent filled = fill(n, i);
            Node target = filled.parent.children[filled.index];
            return replaceChild(filled.parent, filled.index, deleteRec(target, key));
        }
        return replaceChild(n, i, deleteRec(n.children[i], key));
    }

    /** In-order keys of a version. */
    static int[] toArray(Node n) {
        List<Integer> out = new ArrayList<>();
        collect(n, out);
        int[] array = new int[out.size()];
        for (int i = 0; i < array.length; i++) array[i] = out.get(i);
        return array;
    }

    private static void collect(Node n, List<Integer> out) {
        if (n == null) return;
        for (int i = 0; i < n.count; i++) {
            if (!n.leaf) collect(n.children[i], out);
            out.add(n.keys[i]);
        }
        if (!n.leaf) collect(n.children[n.count], out);
    }

    static List<Integer> range(Node root, int lo, int hi) {
        List<Integer> out = new ArrayList<>();
        rangeRec(root, lo, hi, out);
        return out;
    }

    private static void rangeRec(Node n, int lo, int hi, List<Integer> out) {
        if (n == null) return;
        for (int i = 0; i < n.count; i++) {
            if (!n.leaf && n.keys[i] >= lo) rangeRec(n.children[i], lo, hi, out);
            if (n.keys[i] >= lo && n.keys[i] <= hi) out.add(n.keys[i]);
        }
        if (!n.leaf && (n.count == 0 || n.keys[n.count - 1] <= hi)) {
            rangeRec(n.children[n.count], lo, hi, out);
        }
    }

    /** Leaf depth of a version, or -1 when the tree violates a B-tree invariant. */
    static int validate(Node n, boolean isRoot) {
        for (int i = 0; i + 1 < n.count; i++) {
            if (n.keys[i] >= n.keys[i + 1]) return -1;
        }
        if (n.count > 2 * T - 1) return -1;
        if (n.count == 0) return (isRoot && n.leaf) ? 0 : -1;
        if (!isRoot && n.count < T - 1) return -1;
        if (n.leaf) return 0;
        int depth = -1;
        for (int k = 0; k <= n.count; k++) {
            Node child = n.children[k];
            if (child == null) return -1;
            if (k > 0 && minKey(child) <= n.keys[k - 1]) return -1;
            if (k < n.count && maxKey(child) >= n.keys[k]) return -1;
            int childDepth = validate(child, false);
            if (childDepth < 0) return -1;
            if (depth < 0) depth = childDepth;
            else if (depth != childDepth) return -1;
        }
        return depth + 1;
    }
}

class VersionedIndex {
    private final List<BTree.Node> roots = new ArrayList<>();

    VersionedIndex() { roots.add(null); }

    int versionCount() { return roots.size(); }

    BTree.Node rootOf(int version) {
        if (version < 0 || version >= roots.size()) throw new IllegalArgumentException("version " + version);
        return roots.get(version);
    }

    int put(int key) {
        roots.add(BTree.insert(roots.get(roots.size() - 1), key));
        return roots.size() - 1;
    }

    int remove(int key) {
        roots.add(BTree.delete(roots.get(roots.size() - 1), key));
        return roots.size() - 1;
    }

    boolean get(int version, int key) { return BTree.contains(rootOf(version), key); }

    List<Integer> range(int version, int lo, int hi) { return BTree.range(rootOf(version), lo, hi); }
}

public class Solution {
    private static long state = 0x082EFA98EC4E6C89L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int[] snapshotArray(Set<Integer> keys) {
        int[] out = new int[keys.size()];
        int i = 0;
        for (int key : keys) out[i++] = key;
        return out;
    }

    static void verifyVersions(VersionedIndex index, List<Set<Integer>> snapshots) {
        for (int v = 0; v < index.versionCount(); v++) {
            BTree.Node root = index.rootOf(v);
            Set<Integer> snapshot = snapshots.get(v);
            if (root == null) {
                check(snapshot.isEmpty(), "version " + v + " is empty");
                continue;
            }
            check(BTree.validate(root, true) >= 0, "version " + v + " satisfies the B-tree invariants");
            check(Arrays.equals(BTree.toArray(root), snapshotArray(snapshot)), "version " + v + " contents");
            for (int probe = 0; probe < 40; probe++) {
                int key = (int) (probe * 7 - 20);
                check(index.get(v, key) == snapshot.contains(key), "version " + v + " lookup of " + key);
            }
            List<Integer> scanned = index.range(v, 0, 25);
            List<Integer> expected = new ArrayList<>();
            for (int key : new TreeSet<>(snapshot)) if (key >= 0 && key <= 25) expected.add(key);
            check(scanned.equals(expected), "version " + v + " range scan");
        }
    }

    public static void main(String[] args) {
        VersionedIndex index = new VersionedIndex();
        List<Set<Integer>> snapshots = new ArrayList<>();
        snapshots.add(new TreeSet<>());
        check(index.versionCount() == 1 && index.rootOf(0) == null, "the initial version is the empty tree");
        check(!index.get(0, 5) && index.range(0, 0, 9).isEmpty(), "lookups on the empty version");

        int first = index.put(10);
        check(index.get(first, 10) && !index.get(0, 10), "the first put is a new version");
        snapshots.add(new TreeSet<>(Arrays.asList(10)));               // version 1
        int duplicate = index.put(10);
        check(index.rootOf(duplicate) == index.rootOf(first), "a duplicate put shares the root");
        snapshots.add(new TreeSet<>(Arrays.asList(10)));               // version 2
        check(index.versionCount() == 3, "the duplicate put still committed a version");

        long worstCost = 0;
        long height = 0;
        for (int step = 0; step < 260; step++) {
            long before = BTree.allocated;
            BTree.Node previous = index.rootOf(index.versionCount() - 1);
            Set<Integer> model = new TreeSet<>(snapshots.get(snapshots.size() - 1));
            boolean grow = model.size() < 70 && (nextLong() >>> 33) % 100 < 60;
            int key = (int) ((nextLong() >>> 33) % 120);
            int version;
            if (grow) {
                version = index.put(key);
                model.add(key);
            } else if (!model.isEmpty()) {
                model.remove(key);
                version = index.remove(key);
            } else {
                continue;
            }
            long cost = BTree.allocated - before;
            worstCost = Math.max(worstCost, cost);
            BTree.Node root = index.rootOf(version);
            Set<Integer> committed = new TreeSet<>(model);
            if (!committed.isEmpty()) {
                int depth = BTree.validate(root, true);
                check(depth >= 0, "invariants after step " + step);
                height = Math.max(height, depth);
            }
            if (cost > 0) {
                check(cost <= 12L * BTree.T * (height + 2), "copy-on-write cost " + cost + " at step " + step);
            } else {
                check(root == previous, "an unchanged version keeps the identical root");
            }
            snapshots.add(committed);
        }
        check(BTree.allocated > 100, "the walk really rebuilt nodes");
        verifyVersions(index, snapshots);

        System.out.println("versioned index: " + index.versionCount() + " versions, " + BTree.allocated
                + " nodes allocated, worst update cost " + worstCost + ", leaf depth " + height);

        // explicit persistence checks on a small chain
        VersionedIndex small = new VersionedIndex();
        int v1 = small.put(5);
        int v2 = small.put(3);
        int v3 = small.remove(5);
        check(small.get(v1, 5) && small.get(v2, 5) && !small.get(v3, 5), "versions see their own keys");
        check(small.get(v3, 3) && small.get(v1, 3) == false, "delete only affects the new version");
        check(small.remove(99) >= 0, "deleting a missing key is still a version");
        check(!small.get(small.versionCount() - 1, 99), "the missing key stays missing");
        int v4 = small.put(5);
        check(small.get(v4, 5) && small.get(v4, 3), "re-inserting after a delete works");
        check(Arrays.equals(BTree.toArray(small.rootOf(v4)), new int[]{3, 5}), "sorted contents");

        // emptying a bigger tree one key at a time exercises every merge path
        BTree.Node tree = null;
        TreeSet<Integer> keys = new TreeSet<>();
        for (int i = 1; i <= 40; i++) {
            tree = BTree.insert(tree, i * 3);
            keys.add(i * 3);
        }
        check(BTree.validate(tree, true) >= 0, "a 40-key tree is valid");
        for (int key = 1; key <= 40 * 3; key++) {
            if (!keys.contains(key)) continue;
            tree = BTree.delete(tree, key);
            keys.remove(key);
            check(Arrays.equals(BTree.toArray(tree), snapshotArray(keys)), "delete of " + key);
            if (!keys.isEmpty()) check(BTree.validate(tree, true) >= 0, "valid after deleting " + key);
        }
        check(tree == null, "deleting every key collapses the tree");
        check(BTree.toArray(null).length == 0, "the empty tree has no keys");
        check(!BTree.contains(null, 5) && BTree.delete(null, 5) == null, "empty-tree edges");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
