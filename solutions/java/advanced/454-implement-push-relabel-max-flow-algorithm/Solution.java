/** Task 454 - Implement push-relabel max-flow algorithm. (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static final int INF = 1_000_000_000;

    /** Arc-list flow network: to/cap arrays, reverse arc index is idx ^ 1. */
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

        int flowOn(int idx) {
            return initial.get(idx) - cap.get(idx);
        }

        /** FIFO preflow-push; returns {max flow, push count, relabel count}. */
        int[] pushRelabel(int s, int t) {
            int n = this.n;
            int[] height = new int[n];
            int[] excess = new int[n];
            height[s] = n;
            boolean[] inQueue = new boolean[n];
            ArrayDeque<Integer> queue = new ArrayDeque<>();
            for (int idx : head.get(s)) {
                int amount = cap.get(idx);
                if (amount > 0) {
                    int v = to.get(idx);
                    cap.set(idx, cap.get(idx) - amount);
                    cap.set(idx ^ 1, cap.get(idx ^ 1) + amount);
                    excess[v] += amount;
                    excess[s] -= amount;
                }
            }
            int pushes = 0, relabels = 0;
            for (int v = 0; v < n; v++) {
                if (v != s && v != t && excess[v] > 0) {
                    queue.add(v);
                    inQueue[v] = true;
                }
            }
            while (!queue.isEmpty()) {
                int u = queue.poll();
                inQueue[u] = false;
                while (excess[u] > 0) {
                    boolean pushed = false;
                    for (int idx : head.get(u)) {
                        int v = to.get(idx);
                        if (cap.get(idx) > 0 && height[u] == height[v] + 1) {
                            int amount = Math.min(excess[u], cap.get(idx));
                            cap.set(idx, cap.get(idx) - amount);
                            cap.set(idx ^ 1, cap.get(idx ^ 1) + amount);
                            excess[u] -= amount;
                            excess[v] += amount;
                            pushes++;
                            if (v != s && v != t && !inQueue[v] && excess[v] > 0) {
                                queue.add(v);
                                inQueue[v] = true;
                            }
                            pushed = true;
                            break;
                        }
                    }
                    if (!pushed) {
                        int best = INF;
                        for (int idx : head.get(u)) {
                            if (cap.get(idx) > 0 && height[to.get(idx)] < best) best = height[to.get(idx)];
                        }
                        height[u] = best + 1;
                        relabels++;
                    }
                }
            }
            return new int[]{excess[t], pushes, relabels};
        }

        int edmondsKarp(int s, int t) {
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

    static List<int[]> edgesOf(int[]... raw) {
        return new ArrayList<>(Arrays.asList(raw));
    }

    static class Built {
        final MaxFlow network;
        final int[] ids;

        Built(MaxFlow network, int[] ids) {
            this.network = network;
            this.ids = ids;
        }
    }

    static Built buildNetwork(int n, List<int[]> edges) {
        MaxFlow network = new MaxFlow(n);
        int[] ids = new int[edges.size()];
        for (int i = 0; i < edges.size(); i++) {
            int[] e = edges.get(i);
            ids[i] = network.addEdge(e[0], e[1], e[2]);
        }
        return new Built(network, ids);
    }

    static void checkFlowValid(int n, List<int[]> edges, Built built, int s, int t, int value) {
        int[] balance = new int[n];
        for (int i = 0; i < edges.size(); i++) {
            int[] e = edges.get(i);
            int f = built.network.flowOn(built.ids[i]);
            check(f >= 0 && f <= e[2], "capacity bound");
            balance[e[0]] -= f;
            balance[e[1]] += f;
        }
        check(balance[t] == value, "sink balance");
        check(balance[s] == -value, "source balance");
        for (int node = 0; node < n; node++) {
            if (node != s && node != t) check(balance[node] == 0, "conservation at " + node);
        }
    }

    static class Instance {
        final int n;
        final List<int[]> edges;

        Instance(int n, List<int[]> edges) {
            this.n = n;
            this.edges = edges;
        }
    }

    static List<Instance> lcgFlowGraphs(int count, int seed) {
        List<Instance> graphs = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 4 + seed % 5;
            List<int[]> edges = new ArrayList<>();
            for (int node = 0; node < n - 1; node++) {
                seed = (seed * 75 + 74) % 65537;
                edges.add(new int[]{node, node + 1, 1 + seed % 20});
            }
            for (int k = 0; k < n + 3; k++) {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                seed = (seed * 75 + 74) % 65537;
                if (u != v) edges.add(new int[]{u, v, 1 + seed % 20});
            }
            graphs.add(new Instance(n, edges));
        }
        return graphs;
    }

    public static void main(String[] args) {
        List<int[]> clrs = edgesOf(new int[]{0, 1, 16}, new int[]{0, 2, 13}, new int[]{1, 2, 10},
                new int[]{2, 1, 4}, new int[]{1, 3, 12}, new int[]{3, 2, 9}, new int[]{2, 4, 14},
                new int[]{4, 3, 7}, new int[]{3, 5, 20}, new int[]{4, 5, 4});
        Built built = buildNetwork(6, clrs);
        int[] result = built.network.pushRelabel(0, 5);
        check(result[0] == 23, "CLRS 26.1 value");
        check(result[1] > 0 && result[2] > 0, "CLRS pushes and relabels");
        checkFlowValid(6, clrs, built, 0, 5, 23);
        check(buildNetwork(6, clrs).network.edmondsKarp(0, 5) == 23, "CLRS oracle");

        List<int[]> single = edgesOf(new int[]{0, 1, 5});
        built = buildNetwork(2, single);
        check(built.network.pushRelabel(0, 1)[0] == 5, "single edge");
        checkFlowValid(2, single, built, 0, 1, 5);

        List<int[]> parallel = edgesOf(new int[]{0, 1, 3}, new int[]{0, 1, 4}, new int[]{1, 2, 2});
        check(buildNetwork(3, parallel).network.pushRelabel(0, 2)[0] == 2, "parallel edges");
        check(buildNetwork(3, parallel).network.edmondsKarp(0, 2) == 2, "parallel oracle");

        List<int[]> noPath = edgesOf(new int[]{0, 1, 9}, new int[]{2, 3, 9});
        check(buildNetwork(4, noPath).network.pushRelabel(0, 3)[0] == 0, "no path");
        List<int[]> zeroCap = edgesOf(new int[]{0, 1, 0}, new int[]{1, 2, 5});
        check(buildNetwork(3, zeroCap).network.pushRelabel(0, 2)[0] == 0, "zero capacity");

        int checked = 0, totalPushes = 0, totalRelabels = 0;
        for (int seed : new int[]{4541, 4542, 4543, 4544}) {
            for (Instance inst : lcgFlowGraphs(8, seed)) {
                built = buildNetwork(inst.n, inst.edges);
                result = built.network.pushRelabel(0, inst.n - 1);
                totalPushes += result[1];
                totalRelabels += result[2];
                checkFlowValid(inst.n, inst.edges, built, 0, inst.n - 1, result[0]);
                int expected = buildNetwork(inst.n, inst.edges).network.edmondsKarp(0, inst.n - 1);
                check(result[0] == expected, "LCG value " + result[0] + " vs " + expected);
                checked++;
            }
        }
        check(checked == 32, "graph count");
        check(totalPushes > 0 && totalRelabels > 0, "algorithm exercised");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
