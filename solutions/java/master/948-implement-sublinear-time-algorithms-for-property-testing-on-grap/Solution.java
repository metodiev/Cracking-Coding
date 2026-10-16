/** Task 948 - Implement sublinear time algorithms for property testing on graphs (e.g., bipartiteness).
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A bipartiteness tester in the Goldreich-Ron style: sample s = Theta(1/eps) start vertices, run a bounded
 *  local BFS from each (at most b vertices touched) and reject as soon as an odd cycle shows up. A bipartite
 *  graph can never produce such a witness, and a graph that is eps-far from bipartite is rejected with
 *  probability at least 2/3. Every adjacency probe goes through a counter, and the tester must stay inside
 *  s*(1 + 3b) probes - a constant, independent of n, while the exact 2-colouring checker next to it needs
 *  Theta(n + m) probes. The exact checker also bounds "eps-far" and validates that witnesses correspond to
 *  real odd cycles rather than to the tester's own opinion.
 *  Simplified: the bounded BFS explores at most b vertices instead of a random walk of length O(1/eps), no
 *  edge counting inside the explored subgraph, and only BFS-tree odd cycles are reported as witnesses.
 *  Verified: bipartite graphs are never rejected, witnesses are genuine odd cycles, the eps-far graph rejects
 *  in at least two thirds of the runs, a single violating edge is usually missed, probes stay constant in n
 *  and under budget, and empty, isolated and self-looped graphs behave.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final int SAMPLE_SIZE = 16;            // Theta(1/eps) start vertices
    static final int BUDGET = 24;                 // vertices the bounded BFS may touch per start

    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        int nextInt(int bound) {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return (int) Long.remainderUnsigned(z ^ (z >>> 31), bound);
        }
    }

    /** A graph whose every adjacency probe is counted, exactly like a real sublinear tester would. */
    static final class CounterGraph {
        final int n;
        final int[][] neighbours;
        final int[][] edges;                       // self-loops included, counted once each
        long probes;

        CounterGraph(int n, int[][] neighbours, int[][] edges) {
            this.n = n;
            this.neighbours = neighbours;
            this.edges = edges;
        }

        int degree(int v) {
            probes++;
            return neighbours[v].length;
        }

        int neighbour(int v, int index) {
            probes++;
            return neighbours[v][index];
        }

        int edgeCount() { return edges.length; }
    }

    static CounterGraph of(int n, int[][] edges) {
        List<List<Integer>> lists = new ArrayList<>();
        for (int i = 0; i < n; i++) lists.add(new ArrayList<>());
        for (int[] edge : edges) {
            lists.get(edge[0]).add(edge[1]);
            if (edge[0] != edge[1]) lists.get(edge[1]).add(edge[0]);
        }
        int[][] neighbours = new int[n][];
        for (int v = 0; v < n; v++) {
            neighbours[v] = new int[lists.get(v).size()];
            for (int i = 0; i < neighbours[v].length; i++) neighbours[v][i] = lists.get(v).get(i);
        }
        return new CounterGraph(n, neighbours, edges);
    }

    /** Bounded BFS from `start`; returns an odd cycle as vertex ids, or null when none was seen. */
    static int[] witness(CounterGraph graph, int start, int budget) {
        int limit = Math.min(graph.n, budget);
        int[] colour = new int[graph.n], parent = new int[graph.n], depth = new int[graph.n];
        int[] queue = new int[graph.n];
        Arrays.fill(parent, -1);
        int head = 0, tail = 0, explored = 0;
        colour[start] = 1;
        queue[tail++] = start;
        explored++;
        while (head < tail) {
            int v = queue[head++];
            int degree = graph.degree(v);
            for (int i = 0; i < degree; i++) {
                int w = graph.neighbour(v, i);
                if (colour[w] == 0) {
                    if (explored >= limit) continue;
                    colour[w] = -colour[v];
                    parent[w] = v;
                    depth[w] = depth[v] + 1;
                    queue[tail++] = w;
                    explored++;
                } else if (colour[w] == colour[v]) {
                    List<Integer> left = new ArrayList<>(), right = new ArrayList<>();
                    int a = v, b = w;                              // equal colours: the two paths have equal parity
                    while (depth[a] > depth[b]) { left.add(a); a = parent[a]; }
                    while (depth[b] > depth[a]) { right.add(b); b = parent[b]; }
                    while (a != b) { left.add(a); right.add(b); a = parent[a]; b = parent[b]; }
                    left.add(a);                                   // the lowest common ancestor
                    List<Integer> cycle = new ArrayList<>(left);
                    for (int k = right.size() - 1; k >= 0; k--) cycle.add(right.get(k));
                    cycle.add(v);                                  // close with the conflicting edge
                    int[] result = new int[cycle.size()];
                    for (int k = 0; k < result.length; k++) result[k] = cycle.get(k);
                    return result;
                }
            }
        }
        return null;
    }

    /** The tester: one bounded BFS per sampled start, a witness rejects immediately. */
    static boolean accepts(CounterGraph graph, long seed) {
        if (graph.n == 0) return true;
        Rng rng = new Rng(seed * 0x9E3779B97F4A7C15L + 1);
        for (int attempt = 0; attempt < SAMPLE_SIZE; attempt++) {
            if (witness(graph, rng.nextInt(graph.n), BUDGET) != null) return false;
        }
        return true;
    }

    /** Exact checker: a full 2-colouring plus the number of edges that violate it. */
    static int violatingEdges(CounterGraph graph) {
        int[] colour = new int[graph.n];
        ArrayDeque<Integer> queue = new ArrayDeque<>();
        for (int v = 0; v < graph.n; v++) {
            if (colour[v] != 0) continue;
            colour[v] = 1;
            queue.add(v);
            while (!queue.isEmpty()) {
                int u = queue.poll();
                for (int i = 0; i < graph.neighbours[u].length; i++) {
                    int w = graph.neighbours[u][i];
                    if (colour[w] == 0) {
                        colour[w] = -colour[u];
                        queue.add(w);
                    }
                }
            }
        }
        int violations = 0;
        for (int[] edge : graph.edges) if (colour[edge[0]] == colour[edge[1]]) violations++;
        return violations;
    }

    static boolean isBipartite(CounterGraph graph) { return violatingEdges(graph) == 0; }

    static boolean genuineOddCycle(CounterGraph graph, int[] cycle) {
        if (cycle == null || (cycle.length - 1) % 2 == 0) return false;
        for (int i = 0; i + 1 < cycle.length; i++) {
            boolean found = false;
            for (int w : graph.neighbours[cycle[i]]) if (w == cycle[i + 1]) found = true;
            if (!found) return false;
        }
        return true;
    }

    static int[][] cycleEdges(int n) {
        int[][] edges = new int[n][];
        for (int i = 0; i < n; i++) edges[i] = new int[] {i, (i + 1) % n};
        return edges;
    }

    static int[][] pathEdges(int n) {
        int[][] edges = new int[n - 1][];
        for (int i = 0; i + 1 < n; i++) edges[i] = new int[] {i, i + 1};
        return edges;
    }

    static int[][] starEdges(int n) {
        int[][] edges = new int[n - 1][];
        for (int i = 1; i < n; i++) edges[i - 1] = new int[] {0, i};
        return edges;
    }

    static int[][] completeEdges(int n) {
        List<int[]> edges = new ArrayList<>();
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) edges.add(new int[] {i, j});
        return edges.toArray(new int[0][]);
    }

    /** An even cycle plus chords from vertex 0: each chord creates an odd cycle through the cycle itself. */
    static CounterGraph farFromBipartite(int n, int chords) {
        List<int[]> edges = new ArrayList<>(Arrays.asList(cycleEdges(n)));
        for (int k = 1; k <= chords; k++) edges.add(new int[] {0, 2 * k});
        return of(n, edges.toArray(new int[0][]));
    }

    static double rejectionRate(CounterGraph graph, int runs) {
        int rejected = 0;
        for (long seed = 1; seed <= runs; seed++) if (!accepts(graph, seed)) rejected++;
        return (double) rejected / runs;
    }

    static long probeCount(CounterGraph graph, long seed) {
        graph.probes = 0;
        accepts(graph, seed);
        return graph.probes;
    }

    public static void main(String[] args) {
        CounterGraph path = of(20, pathEdges(20));
        CounterGraph star = of(11, starEdges(11));
        CounterGraph isolated = of(40, new int[0][]);
        CounterGraph evenCycle = of(200, cycleEdges(200));
        int falsePositives = 0;
        for (long seed = 1; seed <= 50; seed++) {
            if (!accepts(path, seed) || !accepts(star, seed) || !accepts(isolated, seed)
                    || !accepts(evenCycle, seed)) falsePositives++;
        }
        check(falsePositives == 0 && isBipartite(path) && isBipartite(star) && isBipartite(isolated)
                        && isBipartite(evenCycle),
                "bipartite graphs must never be rejected over 50 seeds: " + falsePositives + " false positives");

        CounterGraph triangle = of(3, cycleEdges(3));
        CounterGraph fiveCycle = of(5, cycleEdges(5));
        CounterGraph clique = of(6, completeEdges(6));
        int witnessFailures = 0;
        for (int start = 0; start < 5; start++) {
            if (!genuineOddCycle(fiveCycle, witness(fiveCycle, start, BUDGET))) witnessFailures++;
            if (start < 3 && !genuineOddCycle(triangle, witness(triangle, start, BUDGET))) witnessFailures++;
        }
        check(witnessFailures == 0 && violatingEdges(triangle) >= 1 && !isBipartite(triangle)
                        && violatingEdges(clique) > 0 && !accepts(triangle, 1) && !accepts(fiveCycle, 1)
                        && !accepts(clique, 1),
                "every witness must be a genuine odd cycle and odd cycles must be rejected: " + witnessFailures
                        + " failures, triangle violations " + violatingEdges(triangle));

        CounterGraph far = farFromBipartite(80, 12);
        int farViolations = violatingEdges(far);
        double farRate = rejectionRate(far, 200);
        check(farViolations > 0 && farRate >= 2.0 / 3.0,
                "a graph many edges away from bipartite must be rejected: rate " + farRate + " with "
                        + farViolations + " violating edges");

        CounterGraph marginal = farFromBipartite(2000, 1);
        double marginalRate = rejectionRate(marginal, 200);
        check(violatingEdges(marginal) >= 1 && marginalRate >= 0 && marginalRate < 0.5,
                "a single chord in a large graph is a negligible distance and is usually missed: " + marginalRate);

        CounterGraph big = of(2000, cycleEdges(2000));
        long testerProbes = probeCount(big, 11);
        long exactProbes = 2L * big.edgeCount() + big.n;
        check(testerProbes <= (long) SAMPLE_SIZE * (1 + 3L * BUDGET) && testerProbes * 2 < exactProbes,
                "the tester must stay inside its probe budget: " + testerProbes + " probes, budget "
                        + (SAMPLE_SIZE * (1 + 3L * BUDGET)) + ", exact " + exactProbes);

        long smallProbes = probeCount(of(200, cycleEdges(200)), 11);
        check(largeProbesWithin(testerProbes, smallProbes) && testerProbes > 0,
                "the probe count must not grow with the graph: " + smallProbes + " probes at n = 200 vs "
                        + testerProbes + " at n = 2000");

        int soundnessFailures = 0, witnessesSeen = 0;
        for (int start = 0; start < 60; start++) {
            int[] found = witness(far, start, BUDGET);
            if (found == null) continue;
            witnessesSeen++;
            if (!genuineOddCycle(far, found)) soundnessFailures++;
        }
        check(soundnessFailures == 0 && witnessesSeen > 0 && isBipartite(far) == (farViolations == 0)
                        && isBipartite(evenCycle) && !isBipartite(far),
                "witnesses must survive an independent odd-cycle check: " + soundnessFailures + " of "
                        + witnessesSeen);

        CounterGraph selfLoop = of(4, new int[][] {{0, 0}, {1, 2}, {2, 3}});
        CounterGraph empty = of(0, new int[0][]);
        check(!accepts(selfLoop, 3) && violatingEdges(selfLoop) == 1 && accepts(empty, 3)
                        && violatingEdges(empty) == 0 && !isBipartite(selfLoop) && isBipartite(empty),
                "a self-loop must be caught and an empty graph accepted: " + violatingEdges(selfLoop));

        CounterGraph replay = farFromBipartite(80, 12);
        boolean firstVerdict = accepts(replay, 42);
        long firstProbes = probeCount(replay, 42);
        check(accepts(far, 42) == firstVerdict && probeCount(far, 42) == firstProbes
                        && rejectionRate(far, 200) == farRate,
                "the tester must be reproducible, probes included: " + firstProbes);

        check(farRate >= 2.0 / 3.0 && falsePositives == 0 && witnessFailures == 0 && soundnessFailures == 0
                        && testerProbes * 2 < exactProbes && marginalRate < 0.5,
                "the whole battery must hold together: " + farRate + " / " + marginalRate);

        System.out.println("bipartiteness tester: sample " + SAMPLE_SIZE + ", budget " + BUDGET + ", probes "
                + testerProbes + " on " + big.n + " vertices (" + exactProbes + " for the exact checker)");
        System.out.println("rejection rate " + String.format("%.3f", farRate) + " on a 12-chord graph and "
                + String.format("%.3f", marginalRate) + " on the marginal one, " + witnessesSeen
                + " witnesses all genuine");
        System.out.println("PASS");
    }

    static boolean largeProbesWithin(long large, long small) {
        return large <= small * 3 / 2 + 4;
    }
}
