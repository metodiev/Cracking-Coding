/** Task 453 - Implement circulation with lower bounds (flow transform). (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static final int INF = 1_000_000_000;

    /** Edmonds-Karp on an explicit arc list; each arc keeps its own reverse residual. */
    static class MaxFlow {
        final int n;
        final List<Integer> to = new ArrayList<>();
        final List<Integer> initial = new ArrayList<>();
        final List<Integer> cap = new ArrayList<>();
        final List<List<Integer>> head;

        MaxFlow(int n) {
            this.n = n;
            head = new ArrayList<>();
            for (int i = 0; i < n; i++) head.add(new ArrayList<>());
        }

        int addEdge(int u, int v, int capacity) {
            int idx = to.size();
            to.add(v);
            initial.add(capacity);
            cap.add(capacity);
            head.get(u).add(idx);
            to.add(u);
            initial.add(0);
            cap.add(0);
            head.get(v).add(idx + 1);
            return idx;
        }

        int used(int idx) {
            return initial.get(idx) - cap.get(idx);
        }

        int maxFlow(int s, int t) {
            int flow = 0;
            while (true) {
                int[] parentEdge = new int[n];
                Arrays.fill(parentEdge, -1);
                parentEdge[s] = -2;
                ArrayDeque<Integer> q = new ArrayDeque<>();
                q.add(s);
                while (!q.isEmpty()) {
                    int u = q.poll();
                    for (int idx : head.get(u)) {
                        int v = to.get(idx);
                        if (parentEdge[v] == -1 && cap.get(idx) > 0) {
                            parentEdge[v] = idx;
                            q.add(v);
                        }
                    }
                }
                if (parentEdge[t] == -1) return flow;
                int aug = INF;
                for (int v = t; v != s; v = to.get(parentEdge[v] ^ 1)) aug = Math.min(aug, cap.get(parentEdge[v]));
                for (int v = t; v != s; v = to.get(parentEdge[v] ^ 1)) {
                    int idx = parentEdge[v];
                    cap.set(idx, cap.get(idx) - aug);
                    cap.set(idx ^ 1, cap.get(idx ^ 1) + aug);
                }
                flow += aug;
            }
        }
    }

    /** Returns the edge flows for edges = {u, v, lower, upper}, or null when infeasible. */
    static int[] circulation(int n, int[][] edges) {
        int ss = n, tt = n + 1;
        MaxFlow network = new MaxFlow(n + 2);
        int[] balance = new int[n];
        int[] arcIds = new int[edges.length];
        for (int i = 0; i < edges.length; i++) {
            int u = edges[i][0], v = edges[i][1], lower = edges[i][2], upper = edges[i][3];
            balance[v] += lower;
            balance[u] -= lower;
            arcIds[i] = network.addEdge(u, v, upper - lower);
        }
        int required = 0;
        for (int v = 0; v < n; v++) {
            if (balance[v] > 0) {
                network.addEdge(ss, v, balance[v]);
                required += balance[v];
            } else if (balance[v] < 0) {
                network.addEdge(v, tt, -balance[v]);
            }
        }
        if (network.maxFlow(ss, tt) != required) return null;
        int[] flows = new int[edges.length];
        for (int i = 0; i < edges.length; i++) flows[i] = edges[i][2] + network.used(arcIds[i]);
        return flows;
    }

    /** s-t flow of exactly `requiredValue` via the t->s [F, F] circulation transform. */
    static int[] circulationWithRequiredValue(int n, int[][] edges, int s, int t, int requiredValue) {
        int[][] extended = new int[edges.length + 1][];
        System.arraycopy(edges, 0, extended, 0, edges.length);
        extended[edges.length] = new int[]{t, s, requiredValue, requiredValue};
        return circulation(n, extended);
    }

    /** Enumerate every integer flow assignment in [lower, upper] and test conservation. */
    static boolean bruteFeasible(int n, int[][] edges) {
        int m = edges.length;
        int[] values = new int[m];
        while (true) {
            int[] balance = new int[n];
            for (int i = 0; i < m; i++) {
                int f = edges[i][2] + values[i];
                balance[edges[i][0]] -= f;
                balance[edges[i][1]] += f;
            }
            boolean ok = true;
            for (int b : balance) ok &= b == 0;
            if (ok) return true;
            int j = 0;
            while (j < m) {
                if (values[j] + 1 < edges[j][3] - edges[j][2] + 1) {
                    values[j]++;
                    break;
                }
                values[j] = 0;
                j++;
            }
            if (j == m) return false;
        }
    }

    static void checkFlowValid(int n, int[][] edges, int[] flows) {
        int[] balance = new int[n];
        for (int i = 0; i < edges.length; i++) {
            int lower = edges[i][2], upper = edges[i][3];
            check(lower <= flows[i] && flows[i] <= upper, "flow within bounds");
            balance[edges[i][0]] -= flows[i];
            balance[edges[i][1]] += flows[i];
        }
        for (int b : balance) check(b == 0, "conservation");
    }

    static class Instance {
        final int n;
        final int[][] edges;

        Instance(int n, int[][] edges) {
            this.n = n;
            this.edges = edges;
        }
    }

    static List<Instance> lcgInstances(int count, int seed) {
        List<Instance> instances = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 3 + seed % 2;
            seed = (seed * 75 + 74) % 65537;
            int m = 3 + seed % 3;
            List<int[]> pool = new ArrayList<>();
            for (int u = 0; u < n; u++)
                for (int v = 0; v < n; v++)
                    if (u != v) pool.add(new int[]{u, v});
            List<int[]> edges = new ArrayList<>();
            for (int k = 0; k < Math.min(m, pool.size()); k++) {
                seed = (seed * 75 + 74) % 65537;
                int idx = seed % pool.size();
                int[] uv = pool.get(idx);
                pool.set(idx, pool.get(pool.size() - 1));
                pool.remove(pool.size() - 1);
                seed = (seed * 75 + 74) % 65537;
                int lower = seed % 3;
                seed = (seed * 75 + 74) % 65537;
                int upper = lower + seed % 3;
                edges.add(new int[]{uv[0], uv[1], lower, upper});
            }
            instances.add(new Instance(n, edges.toArray(new int[0][])));
        }
        return instances;
    }

    public static void main(String[] args) {
        int[][] triangle = {{0, 1, 1, 3}, {1, 2, 1, 3}, {2, 0, 1, 3}};
        int[] flows = circulation(3, triangle);
        check(flows != null && Arrays.equals(flows, new int[]{1, 1, 1}), "triangle flows");
        checkFlowValid(3, triangle, flows);
        check(bruteFeasible(3, triangle), "triangle brute");

        int[][] tightOut = {{0, 1, 2, 3}, {1, 2, 0, 1}, {2, 0, 1, 3}};
        check(circulation(3, tightOut) == null, "tight out infeasible");
        check(!bruteFeasible(3, tightOut), "tight out brute");

        int[][] twoWay = {{0, 1, 3, 5}, {1, 0, 0, 5}};
        flows = circulation(2, twoWay);
        check(flows != null && Arrays.equals(flows, new int[]{3, 3}), "two way flows");
        checkFlowValid(2, twoWay, flows);
        check(bruteFeasible(2, twoWay), "two way brute");

        int[][] tooMuchIn = {{0, 1, 4, 5}, {1, 0, 0, 2}};
        check(circulation(2, tooMuchIn) == null, "too much in infeasible");
        check(!bruteFeasible(2, tooMuchIn), "too much in brute");

        int[][] disjoint = {{0, 1, 1, 2}, {1, 0, 1, 2}, {2, 3, 0, 1}, {3, 2, 0, 1}};
        flows = circulation(4, disjoint);
        check(flows != null, "disjoint feasible");
        checkFlowValid(4, disjoint, flows);
        check(bruteFeasible(4, disjoint), "disjoint brute");

        int[][] deadEnd = {{0, 1, 1, 1}};
        check(circulation(2, deadEnd) == null, "dead end infeasible");
        check(!bruteFeasible(2, deadEnd), "dead end brute");

        int[][] withIsolated = {{0, 1, 1, 1}, {1, 0, 1, 1}};
        flows = circulation(3, withIsolated);
        check(flows != null && Arrays.equals(flows, new int[]{1, 1}), "isolated flows");
        checkFlowValid(3, withIsolated, flows);
        check(bruteFeasible(3, withIsolated), "isolated brute");

        int[][] stEdges = {{0, 1, 0, 2}, {1, 3, 0, 2}, {0, 2, 0, 1}, {2, 3, 0, 1}};
        flows = circulationWithRequiredValue(4, stEdges, 0, 3, 3);
        check(flows != null && flows[flows.length - 1] == 3, "s-t value 3 feasible");
        int[][] extended = {{0, 1, 0, 2}, {1, 3, 0, 2}, {0, 2, 0, 1}, {2, 3, 0, 1}, {3, 0, 3, 3}};
        checkFlowValid(4, extended, flows);
        check(circulationWithRequiredValue(4, stEdges, 0, 3, 4) == null, "s-t value 4 infeasible");

        int feasibleCount = 0, infeasibleCount = 0;
        for (int seed : new int[]{4531, 4532, 4533}) {
            for (Instance inst : lcgInstances(8, seed)) {
                flows = circulation(inst.n, inst.edges);
                boolean expected = bruteFeasible(inst.n, inst.edges);
                check(flows != null == expected, "LCG feasibility");
                if (flows != null) {
                    checkFlowValid(inst.n, inst.edges, flows);
                    feasibleCount++;
                } else {
                    infeasibleCount++;
                }
            }
        }
        check(feasibleCount >= 1 && infeasibleCount >= 1, "both outcomes present");
        check(feasibleCount + infeasibleCount == 24, "instance count");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
