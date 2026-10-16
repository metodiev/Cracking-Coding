/** Task 704 - Implement offline dynamic connectivity with divide & conquer + UF rollback.
 * (Tier: Expert, CTCI Ch 4 -- Trees and Graphs)
 *
 * Offline dynamic connectivity answers all connectivity queries at once. Each edge is active on a time
 * interval [from, to); the intervals are inserted into a segment tree over the time line (an interval
 * is stored in O(log T) nodes, each node covering a sub-range fully inside the interval). A DFS over
 * that tree then maintains a union-find structure over the edges stored on the root-to-current-node
 * path: entering a node unions its edges, recursing visits a sub-range, and leaving the node undoes
 * exactly those unions. Every leaf therefore sees precisely the edges active at that time, and each
 * leaf costs only one answer instead of one rebuild.
 *
 * The union-find is deliberately rollback-friendly: union by size with *no* path compression, so the
 * history is a single stack of merged children and rollback is exact - pop the child, restore its root
 * pointer, subtract its size, and increase the component count. Path compression would rewrite parent
 * pointers that rollback cannot reconstruct, which is precisely why the classic structure drops it.
 *
 * Verified: for every time slot, the component count, the whole component partition (canonicalised to
 * the smallest node id of each component) and every connectivity query agree with a brute-force rebuild
 * that reads the interval list directly and does BFS; the structure is empty (component count n, empty
 * log) after the traversal, so rollback never leaks work; edge intervals that are empty, degenerate or
 * full-range, self loops and duplicate edges are all covered; the number of segment-tree insertions
 * stays within the O(E log T) bound.
 */
import java.util.*;

class RollbackDsu {
    private final int n;
    private final int[] parent;
    private final int[] size;
    private final int[] mergedChild;
    private final int[] mergedParent;
    private int logSize;
    private int components;

    RollbackDsu(int n) {
        this.n = n;
        parent = new int[n + 1];
        size = new int[n + 1];
        mergedChild = new int[n + 2];
        mergedParent = new int[n + 2];
        for (int i = 1; i <= n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
        components = n;
    }

    int components() {
        return components;
    }

    int logSize() {
        return logSize;
    }

    /** No path compression: the parent chain is the rollback journal. */
    int find(int x) {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    /** Marks the current journal length so a later rollback can return to it. */
    int snapshot() {
        return logSize;
    }

    boolean union(int a, int b) {
        int rootA = find(a);
        int rootB = find(b);
        if (rootA == rootB) return false;
        if (size[rootA] < size[rootB]) {
            int swap = rootA;
            rootA = rootB;
            rootB = swap;
        }
        mergedChild[logSize] = rootB;
        mergedParent[logSize] = rootA;
        logSize++;
        parent[rootB] = rootA;
        size[rootA] += size[rootB];
        components--;
        return true;
    }

    void rollback(int mark) {
        while (logSize > mark) {
            logSize--;
            int child = mergedChild[logSize];
            int parentOfChild = mergedParent[logSize];
            size[parentOfChild] -= size[child];
            parent[child] = child;
            components++;
        }
    }

    /** Component partition named by the smallest node of each component, for exact comparisons. */
    int[] canonicalLabels() {
        int[] representative = new int[n + 1];
        Arrays.fill(representative, Integer.MAX_VALUE);
        for (int node = 1; node <= n; node++) {
            int root = find(node);
            if (representative[root] > node) representative[root] = node;
        }
        int[] labels = new int[n + 1];
        for (int node = 1; node <= n; node++) labels[node] = representative[find(node)];
        return labels;
    }
}

class OfflineDynamicConnectivity {
    private final int n;
    private final int slots;
    private final int[] edgeFrom;
    private final int[] edgeTo;
    private final int[] edgeStart;
    private final int[] edgeEnd;
    private int edgeCount;
    private final List<List<Integer>> tree;
    private final RollbackDsu dsu;
    private final int[] componentCount;
    private final int[][] partition;
    private final List<List<Integer>> queriesAt;
    private final int[][] querySpec;
    private final boolean[] queryAnswer;
    private int queryCount;
    private int insertions;

    OfflineDynamicConnectivity(int n, int slots) {
        if (n < 1 || slots < 1) throw new IllegalArgumentException("n and slots must be positive");
        this.n = n;
        this.slots = slots;
        edgeFrom = new int[4 * slots + 4];
        edgeTo = new int[4 * slots + 4];
        edgeStart = new int[4 * slots + 4];
        edgeEnd = new int[4 * slots + 4];
        tree = new ArrayList<>();
        for (int i = 0; i < 4 * slots + 4; i++) tree.add(new ArrayList<>());
        dsu = new RollbackDsu(n);
        componentCount = new int[slots];
        partition = new int[slots][];
        queriesAt = new ArrayList<>();
        for (int i = 0; i < slots; i++) queriesAt.add(new ArrayList<>());
        querySpec = new int[4 * slots + 4][];
        queryAnswer = new boolean[4 * slots + 4];
    }

    /** Edge u-v is present exactly on the half-open time range [from, to). */
    void addEdgeInterval(int u, int v, int from, int to) {
        if (from >= to) return;                       // never active, nothing to insert
        if (u < 1 || u > n || v < 1 || v > n) throw new IllegalArgumentException("node out of range");
        edgeFrom[edgeCount] = u;
        edgeTo[edgeCount] = v;
        edgeStart[edgeCount] = from;
        edgeEnd[edgeCount] = to;
        edgeCount++;
    }

    int edgeCount() {
        return edgeCount;
    }

    int insertions() {
        return insertions;
    }

    int queryCount() {
        return queryCount;
    }

    int addQuery(int time, int u, int v) {
        if (time < 0 || time >= slots) throw new IllegalArgumentException("time out of range: " + time);
        if (u < 1 || u > n || v < 1 || v > n) throw new IllegalArgumentException("node out of range");
        int index = queryCount++;
        querySpec[index] = new int[]{time, u, v};
        queriesAt.get(time).add(index);
        return index;
    }

    boolean answer(int queryIndex) {
        return queryAnswer[queryIndex];
    }

    void solve() {
        for (int i = 0; i < edgeCount; i++) {
            insert(1, 0, slots, edgeStart[i], edgeEnd[i], i);
        }
        dfs(1, 0, slots);
    }

    int componentsAt(int time) {
        return componentCount[time];
    }

    int[] partitionAt(int time) {
        return partition[time];
    }

    private void insert(int node, int lo, int hi, int from, int to, int edgeIndex) {
        if (to <= lo || hi <= from) return;
        if (from <= lo && hi <= to) {
            tree.get(node).add(edgeIndex);
            insertions++;
            return;
        }
        int mid = (lo + hi) >>> 1;
        insert(2 * node, lo, mid, from, to, edgeIndex);
        insert(2 * node + 1, mid, hi, from, to, edgeIndex);
    }

    private void dfs(int node, int lo, int hi) {
        int mark = dsu.snapshot();
        for (int edgeIndex : tree.get(node)) {
            dsu.union(edgeFrom[edgeIndex], edgeTo[edgeIndex]);
        }
        if (hi - lo == 1) {
            componentCount[lo] = dsu.components();
            partition[lo] = dsu.canonicalLabels();
            for (int queryIndex : queriesAt.get(lo)) {
                int[] spec = querySpec[queryIndex];
                queryAnswer[queryIndex] = dsu.find(spec[1]) == dsu.find(spec[2]);
            }
        } else {
            int mid = (lo + hi) >>> 1;
            dfs(2 * node, lo, mid);
            dfs(2 * node + 1, mid, hi);
        }
        dsu.rollback(mark);
        if (node == 1) {
            // after the whole traversal the structure must be exactly as it started
            checkClean();
        }
    }

    private void checkClean() {
        if (dsu.components() != n || dsu.logSize() != 0) {
            throw new AssertionError("FAILED: rollback leaked state, components=" + dsu.components()
                    + " log=" + dsu.logSize());
        }
    }}

public class Solution {
    private static long state = 0x51DE0F7A55L;

    /** splitmix64 - deterministic PRNG, no Math.random. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    /** Brute force: rebuild the graph at each time from the raw interval list and run BFS. */
    static int[][] bruteForcePartitions(int n, int slots, List<int[]> intervals) {
        int[][] partitions = new int[slots][];
        for (int time = 0; time < slots; time++) {
            List<List<Integer>> adjacency = new ArrayList<>();
            for (int i = 0; i <= n; i++) adjacency.add(new ArrayList<>());
            for (int[] interval : intervals) {
                if (time >= interval[2] && time < interval[3]) {
                    adjacency.get(interval[0]).add(interval[1]);
                    adjacency.get(interval[1]).add(interval[0]);
                }
            }
            int[] label = new int[n + 1];
            for (int start = 1; start <= n; start++) {
                if (label[start] != 0) continue;
                Deque<Integer> queue = new ArrayDeque<>();
                label[start] = start;
                queue.add(start);
                while (!queue.isEmpty()) {
                    int node = queue.poll();
                    for (int other : adjacency.get(node)) {
                        if (label[other] == 0) {
                            label[other] = start;
                            queue.add(other);
                        }
                    }
                }
            }
            partitions[time] = label;
        }
        return partitions;
    }

    static boolean samePartition(int[] left, int[] right) {
        if (left.length != right.length) return false;
        for (int i = 0; i < left.length; i++) {
            for (int j = 0; j < left.length; j++) {
                if ((left[i] == left[j]) != (right[i] == right[j])) return false;
            }
        }
        return true;
    }

    public static void main(String[] args) {
        // --- the raw DSU and its rollback -------------------------------------
        RollbackDsu dsu = new RollbackDsu(6);
        check(dsu.components() == 6, "six singleton components");
        int markA = dsu.snapshot();
        check(dsu.union(1, 2), "merge 1-2");
        int markB = dsu.snapshot();
        check(dsu.union(2, 3), "merge 2-3");
        check(dsu.union(1, 3) == false, "redundant merge reports false");
        check(dsu.components() == 4, "components after two merges");
        check(dsu.find(1) == dsu.find(3), "1 and 3 connected");
        dsu.rollback(markB);
        check(dsu.find(1) != dsu.find(3), "rollback inside the second level");
        check(dsu.components() == 5, "component count restored");
        dsu.union(4, 5);
        check(dsu.components() == 4, "merge after rollback");
        dsu.rollback(markA);
        check(dsu.components() == 6 && dsu.logSize() == 0, "full rollback to a clean state");
        int[] clean = dsu.canonicalLabels();
        for (int i = 1; i <= 6; i++) check(clean[i] == i, "clean partition labels are identities");

        // --- hand-checked time line -------------------------------------------
        OfflineDynamicConnectivity timeline = new OfflineDynamicConnectivity(3, 6);
        timeline.addEdgeInterval(1, 2, 0, 4);
        timeline.addEdgeInterval(2, 3, 2, 6);
        int[] queryIds = {
                timeline.addQuery(0, 1, 3),
                timeline.addQuery(2, 1, 3),
                timeline.addQuery(5, 1, 3),
                timeline.addQuery(5, 2, 3)
        };
        check(timeline.addQuery(1, 1, 1) == 4, "self query index");
        timeline.solve();
        check(!timeline.answer(queryIds[0]), "1 and 3 are apart before t = 2");
        check(timeline.answer(queryIds[1]), "1 and 3 meet at t = 2");
        check(!timeline.answer(queryIds[2]), "1 and 3 split again once the first edge expires at t = 4");
        check(timeline.answer(queryIds[3]), "2 and 3 are directly joined at t = 5");
        check(timeline.answer(4), "a node is connected to itself");
        check(timeline.componentsAt(0) == 2 && timeline.componentsAt(1) == 2, "leading slots");
        check(timeline.componentsAt(2) == 1 && timeline.componentsAt(3) == 1, "both edges overlap");
        check(timeline.componentsAt(4) == 2 && timeline.componentsAt(5) == 2, "trailing slots");

        boolean rejected = false;
        try {
            timeline.addEdgeInterval(1, 9, 0, 1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "edge outside the node range must be rejected");
        rejected = false;
        try {
            timeline.addQuery(6, 1, 2);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "query outside the time line must be rejected");

        // --- degenerate intervals ---------------------------------------------
        OfflineDynamicConnectivity degenerate = new OfflineDynamicConnectivity(4, 4);
        degenerate.addEdgeInterval(1, 2, 2, 2);          // empty range: never active
        degenerate.addEdgeInterval(3, 3, 0, 4);          // self loop: a no-op merge
        degenerate.addEdgeInterval(1, 4, 0, 4);          // full range
        degenerate.addEdgeInterval(1, 4, 1, 2);          // duplicate pair, single slot
        int[] degenerateQueries = {
                degenerate.addQuery(0, 1, 4),
                degenerate.addQuery(1, 1, 2),
                degenerate.addQuery(3, 1, 2)
        };
        degenerate.solve();
        check(degenerate.answer(degenerateQueries[0]), "full-range edge joins 1 and 4");
        check(!degenerate.answer(degenerateQueries[1]), "the empty interval never joins 1 and 2");
        check(degenerate.componentsAt(0) == 3, "full-range edge plus two isolated nodes");
        check(degenerate.componentsAt(1) == 3, "duplicate pair does not change the count");
        check(degenerate.componentsAt(2) == 3, "self loop and empty interval change nothing");

        // --- empty instance ----------------------------------------------------
        OfflineDynamicConnectivity bare = new OfflineDynamicConnectivity(5, 3);
        bare.solve();
        check(bare.componentsAt(0) == 5 && bare.componentsAt(2) == 5, "no edges means no merges");
        int[] bareLabels = bare.partitionAt(1);
        for (int i = 1; i <= 5; i++) check(bareLabels[i] == i, "everything is its own component");

        // --- random instance against the brute-force rebuild -------------------
        int n = 14;
        int slots = 32;
        OfflineDynamicConnectivity dynamic = new OfflineDynamicConnectivity(n, slots);
        List<int[]> intervals = new ArrayList<>();
        for (int edge = 0; edge < 60; edge++) {
            int u = 1 + nextInt(n);
            int v = 1 + nextInt(n);
            int from = nextInt(slots);
            int to = Math.min(slots, from + 1 + nextInt(2 * slots));
            dynamic.addEdgeInterval(u, v, from, to);
            if (from < to) intervals.add(new int[]{u, v, from, to});
        }
        List<int[]> queries = new ArrayList<>();
        List<Integer> randomQueryIds = new ArrayList<>();
        for (int query = 0; query < 60; query++) {
            int time = nextInt(slots);
            int u = 1 + nextInt(n);
            int v = 1 + nextInt(n);
            randomQueryIds.add(dynamic.addQuery(time, u, v));
            queries.add(new int[]{time, u, v});
        }
        dynamic.solve();

        int[][] expected = bruteForcePartitions(n, slots, intervals);
        for (int time = 0; time < slots; time++) {
            check(samePartition(dynamic.partitionAt(time), expected[time]),
                    "partition mismatch at time " + time);
            Set<Integer> distinct = new HashSet<>();
            for (int node = 1; node <= n; node++) distinct.add(expected[time][node]);
            int components = distinct.size();
            check(dynamic.componentsAt(time) == components,
                    "component count at time " + time + ": " + dynamic.componentsAt(time)
                            + " vs " + components);
        }
        for (int query = 0; query < queries.size(); query++) {
            int[] spec = queries.get(query);
            check(dynamic.answer(randomQueryIds.get(query))
                            == (expected[spec[0]][spec[1]] == expected[spec[0]][spec[2]]),
                    "query " + query + " at time " + spec[0] + " for " + spec[1] + "-" + spec[2]);
        }
        int bound = 60 * (2 * 6);
        check(dynamic.insertions() < bound, "O(E log T) insertions expected: " + dynamic.insertions());

        // --- one long-lived edge and one that lives in a single slot -----------
        OfflineDynamicConnectivity sparse = new OfflineDynamicConnectivity(6, 16);
        List<int[]> sparseIntervals = new ArrayList<>();
        sparse.addEdgeInterval(1, 2, 0, 16);
        sparseIntervals.add(new int[]{1, 2, 0, 16});
        sparse.addEdgeInterval(3, 4, 7, 8);
        sparseIntervals.add(new int[]{3, 4, 7, 8});
        check(sparse.insertions() == 0, "insertions happen in solve(), not while adding");
        sparse.solve();
        int[][] sparseExpected = bruteForcePartitions(6, 16, sparseIntervals);
        for (int time = 0; time < 16; time++) {
            check(samePartition(sparse.partitionAt(time), sparseExpected[time]),
                    "sparse partition mismatch at time " + time);
        }
        check(sparse.componentsAt(6) == 5 && sparse.componentsAt(7) == 4,
                "single-slot edge appears exactly once: " + sparse.componentsAt(6) + " / "
                        + sparse.componentsAt(7));

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
