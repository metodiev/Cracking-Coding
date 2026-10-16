/** Task 535 - Implement dynamic programming on DAG with topological order caching. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * Dynamic programming over a weighted DAG in which the recurrence runs backwards along a
 * cached topological order: Kahn's algorithm produces one order per graph and that order is
 * then reused for every query against the same graph (the self-test counts how often an order
 * is materialised and asserts it happens exactly once per graph). Two interchangeable solvers
 * compute, for every vertex v, the weight of the heaviest path that *starts* at v:
 *
 *   - iterative: sweep the cached order in reverse, best[v] = max(0, max over edges v->u of
 *     w(v,u) + best[u]) — the cache is the array best, filled by a single pass;
 *   - memoised DFS: the same recurrence evaluated on demand with a done/memo pair, counting
 *     cache hits and misses, so a repeated query on an already-solved state costs nothing.
 *
 * A second DP counts maximal paths (paths that stop at a sink) modulo 1e9+7, which is exact on
 * the small instances used here. Documented checks: for LCG-generated DAGs the iterative values
 * equal the memoised values, both equal an exhaustive DFS enumeration of every maximal path
 * (heaviest path and path count solved by brute force), every edge points forward in the cached
 * order, the memo table solves each state exactly once and uses n + m calls in total while the
 * uncached recursion revisits states (the test asserts caching strictly wins), Kahn's algorithm
 * reports the back-edge graph as non-topological, and the degenerate cases follow the
 * documented conventions best[sink] = 0 and paths[sink] = 1 (single vertex, isolated vertices,
 * zero-weight edges).
 */
import java.util.*;

class Dag {
    final int n;
    final List<List<int[]>> adj;   // adj[u] = {v, w} pairs
    final int edgeCount;

    Dag(int n, int[][] edges) {
        this.n = n;
        this.adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] e : edges) adj.get(e[0]).add(new int[]{e[1], e[2]});
        this.edgeCount = edges.length;
    }
}

public class Solution {
    static final long MOD = 1_000_000_007L;
    static int orderBuilds = 0;

    /** Kahn's algorithm; returns the full order, or a shorter prefix when a cycle exists. */
    static int[] topoOrder(Dag dag) {
        orderBuilds++;
        int[] indeg = new int[dag.n];
        for (int u = 0; u < dag.n; u++) {
            for (int[] e : dag.adj.get(u)) indeg[e[0]]++;
        }
        Deque<Integer> queue = new ArrayDeque<>();
        for (int v = 0; v < dag.n; v++) {
            if (indeg[v] == 0) queue.add(v);
        }
        int[] order = new int[dag.n];
        int size = 0;
        while (!queue.isEmpty()) {
            int u = queue.poll();
            order[size++] = u;
            for (int[] e : dag.adj.get(u)) {
                if (--indeg[e[0]] == 0) queue.add(e[0]);
            }
        }
        return size == dag.n ? order : Arrays.copyOf(order, size);
    }

    static boolean acyclic(int[] order, int n) {
        return order.length == n;
    }

    /** Cached array DP: heaviest path starting at each vertex, one reverse sweep. */
    static long[] bestStart(Dag dag, int[] order) {
        long[] best = new long[dag.n];
        for (int i = order.length - 1; i >= 0; i--) {
            int v = order[i];
            long value = 0;
            for (int[] e : dag.adj.get(v)) value = Math.max(value, e[1] + best[e[0]]);
            best[v] = value;
        }
        return best;
    }

    /** Cached array DP: number of maximal paths starting at each vertex (mod 1e9+7). */
    static long[] countPaths(Dag dag, int[] order) {
        long[] paths = new long[dag.n];
        for (int i = order.length - 1; i >= 0; i--) {
            int v = order[i];
            if (dag.adj.get(v).isEmpty()) {
                paths[v] = 1;
            } else {
                long total = 0;
                for (int[] e : dag.adj.get(v)) total = (total + paths[e[0]]) % MOD;
                paths[v] = total;
            }
        }
        return paths;
    }

    /** Memoised DFS sharing one cache across all queries against a single graph. */
    static class Memo {
        final Dag dag;
        final long[] memo;
        final boolean[] done;
        long hits;
        long misses;

        Memo(Dag dag) {
            this.dag = dag;
            this.memo = new long[dag.n];
            this.done = new boolean[dag.n];
        }

        long heaviest(int v) {
            if (done[v]) {
                hits++;
                return memo[v];
            }
            misses++;
            long value = 0;
            for (int[] e : dag.adj.get(v)) value = Math.max(value, e[1] + heaviest(e[0]));
            done[v] = true;
            memo[v] = value;
            return value;
        }
    }

    /** Same recurrence with no cache: counts how many states a naive re-evaluation touches. */
    static long heaviestUncached(Dag dag, int v, long[] visits) {
        visits[0]++;
        long value = 0;
        for (int[] e : dag.adj.get(v)) value = Math.max(value, e[1] + heaviestUncached(dag, e[0], visits));
        return value;
    }

    /** Exhaustive DFS over maximal paths: heaviest weight and path count. */
    static void enumerate(Dag dag, int v, long weight, long[] best, long[] paths) {
        if (dag.adj.get(v).isEmpty()) {
            best[0] = Math.max(best[0], weight);
            paths[0]++;
            return;
        }
        for (int[] e : dag.adj.get(v)) enumerate(dag, e[0], weight + e[1], best, paths);
    }

    static int[][] lcgDag(int n, int count, int seed) {
        List<int[]> edges = new ArrayList<>();
        int guard = 0;
        while (edges.size() < count && guard < count * 20) {
            guard++;
            seed = (seed * 75 + 74) % 65537;
            int u = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int v = seed % n;
            if (u >= v) continue;                       // keeps the graph acyclic
            seed = (seed * 75 + 74) % 65537;
            edges.add(new int[]{u, v, seed % 9 + 1});
        }
        return edges.toArray(new int[0][]);
    }

    public static void main(String[] args) {
        Dag diamond = new Dag(5, new int[][]{{0, 1, 5}, {0, 2, 3}, {1, 3, 4}, {2, 3, 4}, {3, 4, 2}});
        int[] order = topoOrder(diamond);
        check(acyclic(order, 5), "diamond is a DAG");
        check(Arrays.equals(bestStart(diamond, order), new long[]{11, 6, 6, 2, 0}),
                "hand-checked heaviest paths from every vertex");
        check(Arrays.equals(countPaths(diamond, order), new long[]{2, 1, 1, 1, 1}),
                "hand-checked maximal path counts");

        Memo memo = new Memo(diamond);
        long[] memoValues = new long[5];
        for (int v = 0; v < 5; v++) memoValues[v] = memo.heaviest(v);
        check(Arrays.equals(memoValues, bestStart(diamond, order)), "memo equals array DP");
        check(memo.misses == 5, "each of the 5 states computed exactly once (" + memo.misses + ")");
        check(memo.hits + memo.misses == 5 + diamond.edgeCount,
                "n + m calls in total: one per vertex plus one per edge");

        long[] visits = new long[1];
        long[] naiveValues = new long[5];
        for (int v = 0; v < 5; v++) naiveValues[v] = heaviestUncached(diamond, v, visits);
        check(Arrays.equals(naiveValues, memoValues), "uncached recursion agrees");
        check(visits[0] > memo.hits + memo.misses,
                "caching evaluates fewer states than re-evaluating per query");

        orderBuilds = 0;
        Dag big = new Dag(9, lcgDag(9, 18, 7));
        int[] bigOrder = topoOrder(big);
        long[] bigBest = bestStart(big, bigOrder);
        long[] again = bestStart(big, bigOrder);
        check(orderBuilds == 1, "the topological order is materialised once and reused");
        check(Arrays.equals(bigBest, again), "replaying the cached order is deterministic");
        Memo bigMemo = new Memo(big);
        for (int v = 0; v < big.n; v++) {
            check(bigMemo.heaviest(v) == bigBest[v], "memoised value for vertex " + v);
        }

        int[] seeds = {3, 11, 19, 27, 35, 43, 51, 59, 67, 75};
        for (int s = 0; s < seeds.length; s++) {
            int n = 4 + s % 5;
            Dag dag = new Dag(n, lcgDag(n, 3 + s % 7, seeds[s]));
            int[] ord = topoOrder(dag);
            check(acyclic(ord, n), "generated graph " + s + " is acyclic");
            Map<Integer, Integer> position = new HashMap<>();
            for (int i = 0; i < ord.length; i++) position.put(ord[i], i);
            for (int u = 0; u < n; u++) {
                for (int[] e : dag.adj.get(u)) {
                    check(position.get(u) < position.get(e[0]), "every edge points forward");
                }
            }
            long[] dp = bestStart(dag, ord);
            long[] counts = countPaths(dag, ord);
            Memo m = new Memo(dag);
            for (int v = 0; v < n; v++) {
                long[] best = {0};
                long[] paths = {0};
                enumerate(dag, v, 0, best, paths);
                check(dp[v] == best[0], "brute force heaviest path from " + v + " agrees");
                check(counts[v] == paths[0] % MOD, "brute force path count from " + v + " agrees");
                check(m.heaviest(v) == dp[v], "memoised heaviest from " + v + " agrees");
            }
            check(m.misses == n, "graph " + s + ": every state solved exactly once");
        }

        Dag cyclic = new Dag(3, new int[][]{{0, 1, 1}, {1, 2, 1}, {2, 0, 1}});
        check(!acyclic(topoOrder(cyclic), 3), "Kahn's algorithm detects the back edge");

        Dag isolated = new Dag(5, new int[0][]);
        int[] isolatedOrder = topoOrder(isolated);
        check(Arrays.equals(bestStart(isolated, isolatedOrder), new long[]{0, 0, 0, 0, 0}),
                "no edges: every vertex is a sink of weight 0");
        check(Arrays.equals(countPaths(isolated, isolatedOrder), new long[]{1, 1, 1, 1, 1}),
                "no edges: one trivial maximal path per vertex");

        Dag single = new Dag(1, new int[0][]);
        check(bestStart(single, topoOrder(single))[0] == 0, "single vertex has weight 0");
        check(countPaths(single, topoOrder(single))[0] == 1, "single vertex has one path");

        Dag zeroWeight = new Dag(3, new int[][]{{0, 1, 0}, {1, 2, 0}, {0, 2, 5}});
        check(Arrays.equals(bestStart(zeroWeight, topoOrder(zeroWeight)), new long[]{5, 0, 0}),
                "zero-weight edges are legal");
        check(Arrays.equals(countPaths(zeroWeight, topoOrder(zeroWeight)), new long[]{2, 1, 1}),
                "two maximal paths from the source");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
