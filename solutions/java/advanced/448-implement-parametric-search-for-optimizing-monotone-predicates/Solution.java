/** Task 448 - Implement parametric search for optimizing monotone predicates. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

interface IntPred {
    boolean test(int value);
}

class WeightedGraph {
    final int n;
    final int[][] edges;

    WeightedGraph(int n, int[][] edges) {
        this.n = n;
        this.edges = edges;
    }
}

class Parametric {
    static Integer parametricMin(int[] candidates, IntPred predicate) {
        int lo = 0, hi = candidates.length;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (predicate.test(candidates[mid])) hi = mid;
            else lo = mid + 1;
        }
        return lo < candidates.length ? Integer.valueOf(candidates[lo]) : null;
    }

    static Integer linearScan(int[] candidates, IntPred predicate) {
        for (int candidate : candidates) if (predicate.test(candidate)) return candidate;
        return null;
    }

    static boolean reachableAll(int n, int[][] edges, int d, int source) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] e : edges) {
            if (e[2] <= d) {
                adjacency.get(e[0]).add(e[1]);
                adjacency.get(e[1]).add(e[0]);
            }
        }
        boolean[] seen = new boolean[n];
        seen[source] = true;
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(source);
        while (!stack.isEmpty()) {
            int x = stack.pop();
            for (int y : adjacency.get(x)) {
                if (!seen[y]) {
                    seen[y] = true;
                    stack.push(y);
                }
            }
        }
        for (boolean flag : seen) if (!flag) return false;
        return true;
    }

    static boolean componentsLeq(int n, int[][] edges, int d, int k) {
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        int components = n;
        for (int[] e : edges) {
            if (e[2] > d) continue;
            int ru = find(parent, e[0]);
            int rv = find(parent, e[1]);
            if (ru != rv) {
                parent[ru] = rv;
                components--;
            }
        }
        return components <= k;
    }

    private static int find(int[] parent, int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    static void assertMonotone(int[] candidates, IntPred predicate) {
        boolean seenTrue = false;
        for (int candidate : candidates) {
            boolean value = predicate.test(candidate);
            if (value) seenTrue = true;
            else if (seenTrue) throw new AssertionError("predicate not monotone at " + candidate);
        }
    }
}

public class Solution {
    private static int seed;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static List<WeightedGraph> lcgGraphs(int count, int maxN, int baseSeed) {
        seed = baseSeed;
        List<WeightedGraph> graphs = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            int n = 3 + seed % (maxN - 2);
            List<int[]> edges = new ArrayList<>();
            for (int node = 1; node < n; node++) {
                int parent = nextSeed() % node;
                int weight = 1 + nextSeed() % 30;
                edges.add(new int[]{parent, node, weight});
            }
            for (int extra = 0; extra < n; extra++) {
                int u = nextSeed() % n;
                int v = nextSeed() % n;
                int weight = 1 + nextSeed() % 30;
                if (u != v) edges.add(new int[]{u, v, weight});
            }
            graphs.add(new WeightedGraph(n, edges.toArray(new int[0][])));
        }
        return graphs;
    }

    private static int[] candidatesOf(int[][] edges) {
        TreeSet<Integer> distinct = new TreeSet<>();
        for (int[] e : edges) distinct.add(e[2]);
        int[] candidates = new int[distinct.size()];
        int idx = 0;
        for (int value : distinct) candidates[idx++] = value;
        return candidates;
    }

    public static void main(String[] args) {
        int[][] path = {{0, 1, 4}, {1, 2, 1}, {2, 3, 7}};
        int[] candidates = {1, 4, 7};
        IntPred reach = d -> Parametric.reachableAll(4, path, d, 0);
        IntPred comps2 = d -> Parametric.componentsLeq(4, path, d, 2);
        IntPred comps1 = d -> Parametric.componentsLeq(4, path, d, 1);
        check(Parametric.parametricMin(candidates, reach) == 7, "bottleneck path");
        check(Parametric.parametricMin(candidates, comps2) == 4, "two components");
        check(Parametric.parametricMin(candidates, comps1) == 7, "single component");
        Parametric.assertMonotone(candidates, reach);
        Parametric.assertMonotone(candidates, comps2);

        int[][] edgeCase = {{0, 1, 5}};
        check(Parametric.parametricMin(new int[]{5}, d -> Parametric.reachableAll(2, edgeCase, d, 0)) == 5,
                "single edge");
        check(Parametric.parametricMin(new int[]{1, 2, 3}, d -> false) == null, "never true");
        check(Parametric.parametricMin(new int[]{1, 2, 3}, d -> true) == 1, "always true");

        int checked = 0;
        for (WeightedGraph graph : lcgGraphs(20, 12, 448)) {
            int[] graphCandidates = candidatesOf(graph.edges);
            for (int source : new int[]{0, graph.n - 1}) {
                IntPred predicate = d -> Parametric.reachableAll(graph.n, graph.edges, d, source);
                Parametric.assertMonotone(graphCandidates, predicate);
                check(Objects.equals(Parametric.parametricMin(graphCandidates, predicate),
                        Parametric.linearScan(graphCandidates, predicate)), "reachable source " + source);
                checked++;
            }
            for (int k : new int[]{1, 2, 4}) {
                IntPred predicate = d -> Parametric.componentsLeq(graph.n, graph.edges, d, k);
                Parametric.assertMonotone(graphCandidates, predicate);
                check(Objects.equals(Parametric.parametricMin(graphCandidates, predicate),
                        Parametric.linearScan(graphCandidates, predicate)), "components <= " + k);
                checked++;
            }
        }
        check(checked == 100, "coverage count");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
