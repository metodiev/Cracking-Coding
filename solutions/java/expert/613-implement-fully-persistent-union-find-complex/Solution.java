/** Task 613 - Implement fully persistent union-find (complex). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A *fully* persistent union-find supports an update on any old version, not just the newest one, so
 * each version is an immutable structure and the versions form a tree. Two things make that work here:
 *   - a persistent array over a fixed power-of-two range stored as a complete binary tree; a write
 *     rebuilds the O(log n) nodes on the path to the leaf and shares all other subtrees, so a version
 *     costs O(log n) fresh nodes and reading an entry is O(log n);
 *   - union by size without path compression. Compression is impossible in the persistent setting
 *     because it would mutate versions that must stay readable; union by size alone bounds the depth
 *     by log2(n), which keeps find() cheap and the node count per union at O(log n).
 * A union whose endpoints already share a root is a no-op and returns the identical version, so the
 * caller can detect "nothing changed" by reference.
 *
 * Verified: for every version a brute-force replay of the union history in a plain (mutable) union-find
 * oracle gives exactly the same roots, connectivity and component sizes; all older versions survive
 * later updates and branch updates from the same version stay independent; the depth of every parent
 * chain is at most ceil(log2 n); a union allocates at most 2 * (depth + 1) nodes; unions of connected
 * elements allocate nothing and return the same object; n = 1 and non-power-of-two sizes and
 * out-of-range indices are covered.
 */
import java.util.*;

class PersistentArray {
    static long allocated = 0;

    /** A leaf carries a value, an internal node carries two subtrees; both are immutable. */
    static final class Node {
        final int value;
        final Node left, right;

        private Node(int value, Node left, Node right) {
            this.value = value;
            this.left = left;
            this.right = right;
        }

        static Node leaf(int value) {
            allocated++;
            return new Node(value, null, null);
        }

        static Node internal(Node left, Node right) {
            allocated++;
            return new Node(0, left, right);
        }

        boolean isLeaf() { return left == null && right == null; }
    }

    /** depth = ceil(log2(capacity)); the value at an index is reached by following its bits. */
    static int depthFor(int capacity) {
        int depth = 0;
        while ((1 << depth) < capacity) depth++;
        return depth;
    }

    static Node build(int[] values, int from, int level) {
        if (level == 0) return Node.leaf(values[from]);
        int half = 1 << (level - 1);
        return Node.internal(build(values, from, level - 1), build(values, from + half, level - 1));
    }

    static int get(Node root, int index, int depth) {
        Node n = root;
        for (int level = depth - 1; level >= 0; level--) {
            n = ((index >>> level) & 1) == 0 ? n.left : n.right;
        }
        return n.value;
    }

    static Node set(Node root, int index, int value, int depth) {
        if (depth == 0) return Node.leaf(value);
        if (((index >>> (depth - 1)) & 1) == 0) {
            return Node.internal(set(root.left, index, value, depth - 1), root.right);
        }
        return Node.internal(root.left, set(root.right, index, value, depth - 1));
    }
}

class PersistentDsu {
    final int n;
    final int capacity;
    final int depth;
    private final PersistentArray.Node parent;
    private final PersistentArray.Node size;

    private PersistentDsu(int n, int capacity, int depth, PersistentArray.Node parent,
                          PersistentArray.Node size) {
        this.n = n;
        this.capacity = capacity;
        this.depth = depth;
        this.parent = parent;
        this.size = size;
    }

    static PersistentDsu create(int n) {
        if (n < 1) throw new IllegalArgumentException("n must be positive");
        int capacity = 1;
        while (capacity < n) capacity <<= 1;
        int depth = PersistentArray.depthFor(capacity);
        int[] parents = new int[capacity];
        int[] sizes = new int[capacity];
        for (int i = 0; i < capacity; i++) {
            parents[i] = i;
            sizes[i] = 1;
        }
        return new PersistentDsu(n, capacity, depth,
                PersistentArray.build(parents, 0, depth), PersistentArray.build(sizes, 0, depth));
    }

    private void require(int x) {
        if (x < 0 || x >= n) throw new IndexOutOfBoundsException("element " + x + " is out of range");
    }

    int parentOf(int x) { return PersistentArray.get(parent, x, depth); }

    int sizeOf(int x) { return PersistentArray.get(size, x, depth); }

    /** No path compression: this must not touch earlier versions. */
    int find(int x) {
        require(x);
        int root = x;
        while (parentOf(root) != root) root = parentOf(root);
        return root;
    }

    boolean connected(int a, int b) {
        require(a);
        require(b);
        return find(a) == find(b);
    }

    int componentSize(int x) { return sizeOf(find(x)); }

    /** Returns this very version when the endpoints are already connected. */
    PersistentDsu union(int a, int b) {
        require(a);
        require(b);
        int ra = find(a), rb = find(b);
        if (ra == rb) return this;
        if (sizeOf(ra) < sizeOf(rb)) {
            int swap = ra;
            ra = rb;
            rb = swap;
        }
        PersistentArray.Node newParent =
                PersistentArray.set(parent, rb, ra, depth);                  // rb hangs under ra
        PersistentArray.Node newSize =
                PersistentArray.set(size, ra, sizeOf(ra) + sizeOf(rb), depth);   // the root carries the total
        return new PersistentDsu(n, capacity, depth, newParent, newSize);
    }
}

public class Solution {
    private static long state = 0x0A4093822299F31DL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int[] bruteRoots(int n, List<int[]> history) {
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        for (int[] op : history) {
            int a = op[0], b = op[1];
            while (parent[a] != a) a = parent[a];
            while (parent[b] != b) b = parent[b];
            if (a != b) parent[a] = b;
        }
        int[] roots = new int[n];
        for (int i = 0; i < n; i++) {
            int r = i;
            while (parent[r] != r) r = parent[r];
            roots[i] = r;
        }
        return roots;
    }

    static int[] bruteSizes(int n, List<int[]> history) {
        int[] roots = bruteRoots(n, history);
        int[] counts = new int[n];
        for (int i = 0; i < n; i++) counts[roots[i]]++;
        return counts;
    }

    static int chainDepth(PersistentDsu dsu, int x) {
        int depth = 0;
        while (dsu.parentOf(x) != x) {
            x = dsu.parentOf(x);
            depth++;
        }
        return depth;
    }

    static void verifyVersion(PersistentDsu dsu, List<int[]> history, String label) {
        int n = dsu.n;
        int[] roots = bruteRoots(n, history);
        int[] sizes = bruteSizes(n, history);
        // root labels may differ between the oracle and the implementation, so compare the partition
        // itself: the smallest element of the component is a canonical label for both.
        int[] oracleCanon = new int[n];
        int[] dsuCanon = new int[n];
        for (int x = 0; x < n; x++) {
            oracleCanon[x] = Integer.MAX_VALUE;
            int root = roots[x];
            for (int y = 0; y < n; y++) if (roots[y] == root) oracleCanon[x] = Math.min(oracleCanon[x], y);
            dsuCanon[x] = Integer.MAX_VALUE;
            int mine = dsu.find(x);
            for (int y = 0; y < n; y++) if (dsu.find(y) == mine) dsuCanon[x] = Math.min(dsuCanon[x], y);
        }
        for (int x = 0; x < n; x++) {
            check(oracleCanon[x] == dsuCanon[x],
                    label + ": component of " + x + " is " + dsuCanon[x] + " want " + oracleCanon[x]);
            check(dsu.componentSize(x) == sizes[roots[x]], label + ": component size of " + x);
        }
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                check(dsu.connected(a, b) == (roots[a] == roots[b]), label + ": connected " + a + " " + b);
            }
        }
        int bound = 0;
        while ((1 << bound) < n) bound++;
        for (int x = 0; x < n; x++) {
            check(chainDepth(dsu, x) <= bound, label + ": chain from " + x + " is short");
        }
    }

    static void randomWalk(int n, int steps) {
        PersistentDsu base = PersistentDsu.create(n);
        List<PersistentDsu> versions = new ArrayList<>();
        List<List<int[]>> histories = new ArrayList<>();
        versions.add(base);
        histories.add(new ArrayList<>());
        long worstCost = 0;
        long wholeCopyCost = 0;
        for (int step = 0; step < steps; step++) {
            int pick = (int) ((nextLong() >>> 33) % versions.size());
            PersistentDsu from = versions.get(pick);
            int a = (int) ((nextLong() >>> 33) % n);
            int b = (int) ((nextLong() >>> 33) % n);
            long before = PersistentArray.allocated;
            PersistentDsu next = from.union(a, b);
            long cost = PersistentArray.allocated - before;
            if (next == from) {
                check(cost == 0, "a union of connected elements allocates nothing");
            } else {
                worstCost = Math.max(worstCost, cost);
                check(cost <= 2 * (next.depth + 1), "union cost " + cost + " at depth " + next.depth);
                versions.add(next);
                List<int[]> history = new ArrayList<>(histories.get(pick));
                history.add(new int[]{a, b});
                histories.add(history);
            }
            wholeCopyCost += 2L * n;
        }
        check(versions.size() > 2, "the walk created versions");
        for (int v = 0; v < versions.size(); v++) {
            verifyVersion(versions.get(v), histories.get(v), "n=" + n + " version " + v);
        }
        if (n >= 32) {
            System.out.println("n=" + n + ": " + versions.size() + " versions, "
                    + PersistentArray.allocated + " nodes allocated, whole-array copies would need "
                    + wholeCopyCost + ", worst union cost " + worstCost);
        }
    }

    public static void main(String[] args) {
        PersistentDsu one = PersistentDsu.create(1);
        check(one.find(0) == 0 && one.connected(0, 0), "a single element is its own root");
        check(one.union(0, 0) == one, "union of the only element is a no-op");
        check(one.componentSize(0) == 1, "the component of one element has size 1");

        PersistentDsu small = PersistentDsu.create(4);
        PersistentDsu merged = small.union(0, 1).union(2, 3);
        check(merged.connected(0, 1) && merged.connected(2, 3) && !merged.connected(1, 2), "two pairs");
        check(!small.connected(0, 1), "the base version is untouched");
        PersistentDsu bridge = merged.union(1, 2);
        check(bridge.connected(0, 3) && bridge.componentSize(0) == 4, "the bridge joins everything");
        check(!merged.connected(0, 3), "the previous version stays a snapshot");
        check(bridge.union(3, 0) == bridge, "a redundant union returns the same version");

        // fully persistent: two different updates on the same old version stay independent
        PersistentDsu base = PersistentDsu.create(6);
        PersistentDsu left = base.union(0, 1).union(1, 2);
        PersistentDsu right = base.union(3, 4).union(4, 5);
        check(left.connected(0, 2) && !left.connected(0, 3), "the left branch");
        check(right.connected(3, 5) && !right.connected(3, 0), "the right branch");
        check(!base.connected(0, 1) && !base.connected(3, 4), "the common ancestor is unchanged");
        PersistentDsu both = left.union(2, 3).union(3, 4).union(4, 5);
        check(both.connected(0, 5) && both.componentSize(0) == 6, "branches can be merged later");
        check(left.componentSize(0) == 3 && right.componentSize(3) == 3, "both branches keep their sizes");

        // member sizes that are not powers of two, and the oracle replay on every version
        randomWalk(3, 20);
        randomWalk(8, 60);
        randomWalk(10, 60);
        randomWalk(31, 80);
        randomWalk(64, 120);

        boolean threw = false;
        try { small.find(9); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "find out of range rejected");
        threw = false;
        try { small.union(-1, 2); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "union out of range rejected");
        threw = false;
        try { PersistentDsu.create(0); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "a zero-element universe is rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
