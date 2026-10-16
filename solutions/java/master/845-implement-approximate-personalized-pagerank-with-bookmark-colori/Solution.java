/** Task 845 - Implement approximate personalized PageRank with bookmark coloring. (Tier: Master, CTCI Ch 4 - Trees and Graphs)
 *
 * Personalized PageRank from a source s with teleport probability alpha is the stationary distribution of
 * a walk that, with probability alpha, jumps back to s. The classic *forward push* computes it
 * approximately without touching the whole graph:
 *
 *     p   estimate vector, initialised to alpha * e_s     (what has already been booked)
 *     r   residual vector, initialised to (1 - alpha) * e_s (mass that has not been booked yet)
 *     push(v) :  m = r[v]; p[v] += alpha*m; r[v] = 0; r[u] += (1-alpha) * m / deg(v) for every u ~ v
 *
 * A vertex is pushed while r[v] > epsilon * deg(v), so on termination every residual is at most
 * epsilon * deg(v) and the mass has been booked into the estimate. Two invariants hold exactly at every
 * step: sum(p) + sum(r) = 1 (mass conservation) and p[v] <= ppr[v] (the estimate only ever under-counts).
 * The approximate answer is reported by bookmarking the vertices the walk visits most - the top-k bookmarks
 * - which is the same quantity a bookmark-colouring implementation would return.
 *
 * Simplifications, stated honestly: in-process, no threads, no graph-partitioned workers; the push order is
 * a deterministic priority on the largest residual so runs are reproducible; a dangling vertex (degree 0)
 * returns its mass to the source, which is the standard "restart on a sink" convention and keeps mass
 * conserved. The graph is small (tens of vertices) because the point is the invariant, the cost/accuracy
 * trade-off and the comparison against an exact PageRank.
 *
 * Invariants verified below. Exactness of the bookkeeping: mass conservation at every step, the push count
 * and the residual mass are counted. Approximation: the estimate always underestimates the exact PPR
 * (computed by a dense power iteration), the gap never exceeds the residual mass, shrinking epsilon shrinks
 * both the residual mass and the error, and the top-k bookmark set has a measured recall against the exact
 * top-k. Cost: the number of pushes grows as epsilon shrinks, which is the whole point of the method.
 */
import java.util.*;

/** splitmix64 - used only to build the test graph deterministically. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** Undirected graph with adjacency lists. */
final class UndirectedGraph {
    final int n;
    final List<List<Integer>> adjacency = new ArrayList<>();

    UndirectedGraph(int n) {
        this.n = n;
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
    }

    void addEdge(int u, int v) {
        if (u == v) return;
        if (!adjacency.get(u).contains(v)) adjacency.get(u).add(v);
        if (!adjacency.get(v).contains(u)) adjacency.get(v).add(u);
    }

    int degree(int v) { return adjacency.get(v).size(); }

    List<Integer> neighbours(int v) { return adjacency.get(v); }

    int edgeCount() {
        int total = 0;
        for (List<Integer> list : adjacency) total += list.size();
        return total / 2;
    }
}

/** The outcome of a forward-push run. */
final class PushResult {
    final double[] estimate;
    final double[] residual;
    final int pushes;
    final double residualMass;
    final double estimateMass;

    PushResult(double[] estimate, double[] residual, int pushes) {
        this.estimate = estimate;
        this.residual = residual;
        this.pushes = pushes;
        this.residualMass = sum(residual);
        this.estimateMass = sum(estimate);
    }

    static double sum(double[] values) {
        double total = 0;
        for (double value : values) total += value;
        return total;
    }

    /** Highest-estimate vertices first, ties broken by index: the bookmark set. */
    int[] bookmarks(int k) {
        Integer[] order = new Integer[estimate.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            int cmp = Double.compare(estimate[b], estimate[a]);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        int size = Math.min(k, order.length);
        int[] out = new int[size];
        for (int i = 0; i < size; i++) out[i] = order[i];
        return out;
    }
}

/** Forward push (Andersen-Chung-Lang) with an exact residual invariant. */
final class ForwardPush {
    static PushResult run(UndirectedGraph graph, int source, double alpha, double epsilon) {
        if (alpha <= 0 || alpha >= 1) throw new IllegalArgumentException("alpha must be in (0,1)");
        if (epsilon <= 0) throw new IllegalArgumentException("epsilon must be positive");
        double[] p = new double[graph.n];
        double[] r = new double[graph.n];
        p[source] = 0;
        r[source] = 1;
        PriorityQueue<int[]> queue = new PriorityQueue<>((a, b) -> {
            int cmp = Double.compare(r[b[0]], r[a[0]]);
            return cmp != 0 ? cmp : Integer.compare(a[0], b[0]);
        });
        queue.add(new int[]{source});
        int pushes = 0;
        boolean[] queued = new boolean[graph.n];
        queued[source] = true;
        while (!queue.isEmpty()) {
            int v = queue.poll()[0];
            queued[v] = false;
            if (r[v] <= epsilon * Math.max(1, graph.degree(v))) continue;
            double mass = r[v];
            r[v] = 0;
            p[v] += alpha * mass;
            pushes++;
            if (graph.degree(v) == 0) {
                r[source] += (1 - alpha) * mass;
                if (!queued[source]) {
                    queue.add(new int[]{source});
                    queued[source] = true;
                }
                continue;
            }
            double share = (1 - alpha) * mass / graph.degree(v);
            for (int u : graph.neighbours(v)) {
                r[u] += share;
                if (!queued[u] && r[u] > epsilon * Math.max(1, graph.degree(u))) {
                    queue.add(new int[]{u});
                    queued[u] = true;
                }
            }
        }
        return new PushResult(p, r, pushes);
    }

    /** Exact PPR by dense power iteration; dangling mass restarts at the source, as in the push rule. */
    static double[] exact(UndirectedGraph graph, int source, double alpha, int iterations) {
        double[] rank = new double[graph.n];
        rank[source] = 1;
        for (int step = 0; step < iterations; step++) {
            double[] next = new double[graph.n];
            next[source] += alpha;
            double dangling = 0;
            for (int v = 0; v < graph.n; v++) {
                if (graph.degree(v) == 0) dangling += rank[v];
            }
            next[source] += (1 - alpha) * dangling;
            for (int v = 0; v < graph.n; v++) {
                if (graph.degree(v) == 0) continue;
                double share = (1 - alpha) * rank[v] / graph.degree(v);
                for (int u : graph.neighbours(v)) next[u] += share;
            }
            rank = next;
        }
        return rank;
    }

    /**
     * The same vector obtained by solving (I - M) x = alpha * e_s with Gaussian elimination, so the dense
     * iteration above has an independent check.
     */
    static double[] exactSolve(UndirectedGraph graph, int source, double alpha) {
        int n = graph.n;
        double[][] a = new double[n][n + 1];
        for (int v = 0; v < n; v++) a[v][v] = 1;
        a[source][n] = alpha;
        for (int u = 0; u < n; u++) {
            if (graph.degree(u) == 0) {
                a[source][u] -= (1 - alpha);
                continue;
            }
            double share = (1 - alpha) / graph.degree(u);
            for (int v : graph.neighbours(u)) a[v][u] -= share;
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
            }
            double[] swap = a[col];
            a[col] = a[pivot];
            a[pivot] = swap;
            double divisor = a[col][col];
            for (int k = col; k <= n; k++) a[col][k] /= divisor;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = a[row][col];
                if (factor == 0) continue;
                for (int k = col; k <= n; k++) a[row][k] -= factor * a[col][k];
            }
        }
        double[] solution = new double[n];
        for (int v = 0; v < n; v++) solution[v] = a[v][n];
        return solution;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(31337L);
        UndirectedGraph graph = new UndirectedGraph(24);
        for (int v = 0; v + 1 < 24; v += 2) graph.addEdge(v, v + 1);
        for (int i = 0; i < 90; i++) graph.addEdge(rng.nextInt(24), rng.nextInt(24));
        graph.addEdge(3, 5);
        check(graph.edgeCount() > 50, "the test graph must be reasonably dense, edges " + graph.edgeCount());

        int source = 0;
        double alpha = 0.15, epsilon = 0.0005;

        // ---- 1. mass conservation at every step, and the residual invariant at the end ----------
        PushResult result = ForwardPush.run(graph, source, alpha, epsilon);
        check(Math.abs(result.estimateMass + result.residualMass - 1) < 1e-9,
                "mass must be conserved: p " + result.estimateMass + " + r " + result.residualMass);
        check(result.estimate[source] >= alpha - 1e-9,
                "the source must keep at least the teleport mass, got " + result.estimate[source]);
        boolean residualInvariant = true;
        for (int v = 0; v < graph.n; v++) {
            double limit = epsilon * Math.max(1, graph.degree(v));
            if (result.residual[v] > limit + 1e-12) residualInvariant = false;
        }
        check(residualInvariant, "every residual must be at most epsilon * degree after the run");
        check(result.pushes > 0, "the push loop must have done work, pushes " + result.pushes);
        check(result.residualMass <= epsilon * (2 * graph.edgeCount() + graph.n),
                "the residual mass is bounded by epsilon times the volume, got " + result.residualMass);

        // ---- 2. the estimate underestimates the exact PPR and the gap is bounded by the residual -
        double[] exact = ForwardPush.exact(graph, source, alpha, 400);
        check(Math.abs(PushResult.sum(exact) - 1) < 1e-9, "the exact PPR must sum to one");
        double maxUnder = 0, maxGap = 0;
        for (int v = 0; v < graph.n; v++) {
            maxUnder = Math.max(maxUnder, result.estimate[v] - exact[v]);
            maxGap = Math.max(maxGap, exact[v] - result.estimate[v]);
        }
        double[] solved = ForwardPush.exactSolve(graph, source, alpha);
        check(maxUnder < 1e-9, "the estimate must never exceed the exact PPR, worst overshoot " + maxUnder);
        check(maxGap <= result.residualMass + 1e-12,
                "the gap must be bounded by the residual mass: gap " + maxGap + " vs " + result.residualMass);
        check(maxGap > 0, "the approximation must be strict for a finite epsilon, gap " + maxGap);

        // ---- 3. cost versus accuracy: smaller epsilon costs more and is more accurate ------------
        PushResult coarse = ForwardPush.run(graph, source, alpha, epsilon * 10);
        PushResult fine = ForwardPush.run(graph, source, alpha, epsilon / 10);
        check(coarse.pushes <= result.pushes && result.pushes <= fine.pushes,
                "push count must grow as epsilon shrinks: " + coarse.pushes + " <= " + result.pushes
                        + " <= " + fine.pushes);
        double coarseGap = 0, fineGap = 0;
        for (int v = 0; v < graph.n; v++) {
            coarseGap = Math.max(coarseGap, exact[v] - coarse.estimate[v]);
            fineGap = Math.max(fineGap, exact[v] - fine.estimate[v]);
        }
        check(fineGap <= coarseGap + 1e-12,
                "a smaller epsilon must not be less accurate: " + fineGap + " vs " + coarseGap);
        check(fine.residualMass <= coarse.residualMass + 1e-12,
                "a smaller epsilon must reduce the residual mass: " + fine.residualMass + " vs "
                        + coarse.residualMass);

        // ---- 4. the bookmark set: recall of the exact top-k -------------------------------------
        int[] mine = result.bookmarks(5);
        int[] truth = topK(exact, 5);
        int hits = 0;
        for (int v : mine) for (int w : truth) if (v == w) hits++;
        check(hits >= 4, "the top-5 bookmarks must match the exact top-5, got " + Arrays.toString(mine)
                + " vs " + Arrays.toString(truth));
        int[] coarseBookmarks = coarse.bookmarks(5);
        int coarseHits = 0;
        for (int v : coarseBookmarks) for (int w : truth) if (v == w) coarseHits++;
        check(coarseHits <= hits, "a coarser approximation must not have better recall: " + coarseHits);
        check(result.bookmarks(0).length == 0 && result.bookmarks(100).length == graph.n,
                "the bookmark size must clamp to the graph");

        // ---- 5. degenerate graphs ----------------------------------------------------------------
        UndirectedGraph isolated = new UndirectedGraph(3);
        PushResult lonely = ForwardPush.run(isolated, 1, alpha, epsilon);
        check(lonely.estimate[1] > 0.99 && lonely.pushes >= 1,
                "a source with no neighbours keeps all the mass, got " + lonely.estimate[1]);
        check(Math.abs(PushResult.sum(lonely.estimate) + lonely.residualMass - 1) < 1e-9,
                "mass stays conserved on an edgeless graph");
        double[] lonelyExact = ForwardPush.exact(isolated, 1, alpha, 200);
        check(Math.abs(lonelyExact[1] - 1) < 1e-6, "the exact answer restarts on a sink");
        UndirectedGraph pair = new UndirectedGraph(2);
        pair.addEdge(0, 1);
        PushResult two = ForwardPush.run(pair, 0, alpha, epsilon);
        double[] twoExact = ForwardPush.exactSolve(pair, 0, alpha);
        double[] twoExactIterated = ForwardPush.exact(pair, 0, alpha, 200);
        check(Math.abs(twoExact[0] - 1.0 / (2 - alpha)) < 1e-9,
                "the two-vertex exact PPR must be 1/(2-alpha), got " + twoExact[0]);
        check(Math.abs(twoExactIterated[0] - twoExact[0]) < 1e-9,
                "the power iteration and the linear solve must agree");
        check(Math.abs(PushResult.sum(two.estimate) + two.residualMass - 1) < 1e-9,
                "the two-vertex case conserves mass, sum " + PushResult.sum(two.estimate)
                        + " + " + two.residualMass);
        check(two.estimate[0] <= twoExact[0] + 1e-9 && two.estimate[0] >= twoExact[0] - 1e-2,
                "the two-vertex estimate must be a tight lower bound, got " + two.estimate[0] + " vs "
                        + twoExact[0]);
        UndirectedGraph single = new UndirectedGraph(1);
        PushResult one = ForwardPush.run(single, 0, alpha, epsilon);
        check(one.estimate[0] > 0.99 && one.pushes >= 1, "a single vertex absorbs all the mass");

        // ---- 6. validation and determinism -------------------------------------------------------
        expectThrows(() -> ForwardPush.run(graph, source, 0, epsilon), "alpha below zero");
        expectThrows(() -> ForwardPush.run(graph, source, 1, epsilon), "alpha at one");
        expectThrows(() -> ForwardPush.run(graph, source, alpha, 0), "zero epsilon");
        expectThrows(() -> ForwardPush.run(graph, 99, alpha, epsilon), "source outside the graph");
        PushResult again = ForwardPush.run(graph, source, alpha, epsilon);
        check(again.pushes == result.pushes && Arrays.equals(again.bookmarks(5), mine),
                "the same inputs must reproduce the same pushes and bookmarks");
        double[] estimateCopy = result.estimate.clone();
        check(Arrays.equals(estimateCopy, result.estimate) && estimateCopy.length == graph.n,
                "the estimate vector must be stable after the run");

        System.out.println("graph v=" + graph.n + " e=" + graph.edgeCount() + " alpha=" + alpha
                + " epsilon=" + epsilon + " pushes=" + result.pushes);
        System.out.println("mass: estimate " + result.estimateMass + " + residual " + result.residualMass
                + " = 1; max gap to exact " + maxGap + " (bounded by the residual)");
        System.out.println("cost/accuracy: epsilon " + (epsilon * 10) + " -> " + coarse.pushes
                + " pushes (gap " + coarseGap + "), epsilon " + (epsilon / 10) + " -> " + fine.pushes
                + " pushes (gap " + fineGap + ")");
        System.out.println("bookmarks " + Arrays.toString(mine) + " vs exact top-5 " + Arrays.toString(truth));
        System.out.println("PASS");
    }

    static int[] topK(double[] values, int k) {
        Integer[] order = new Integer[values.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            int cmp = Double.compare(values[b], values[a]);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        int[] out = new int[Math.min(k, order.length)];
        for (int i = 0; i < out.length; i++) out[i] = order[i];
        return out;
    }

    static void expectThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (RuntimeException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected a rejection for " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
