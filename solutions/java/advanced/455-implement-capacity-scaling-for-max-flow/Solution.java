/** Task 455 - Implement capacity scaling for max flow. (Tier: Advanced, CTCI Ch 4) */
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

        int augment(int[] parentEdge, int s, int t) {
            int aug = INF;
            for (int v = t; v != s; v = to.get(parentEdge[v] ^ 1)) aug = Math.min(aug, cap.get(parentEdge[v]));
            for (int v = t; v != s; v = to.get(parentEdge[v] ^ 1)) {
                int idx = parentEdge[v];
                cap.set(idx, cap.get(idx) - aug);
                cap.set(idx ^ 1, cap.get(idx ^ 1) + aug);
            }
            return aug;
        }

        int[] findPath(int s, int t, int minCapacity) {
            int[] parentEdge = new int[n];
            Arrays.fill(parentEdge, -1);
            parentEdge[s] = -2;
            ArrayDeque<Integer> q = new ArrayDeque<>();
            q.add(s);
            while (!q.isEmpty()) {
                int u = q.poll();
                for (int idx : head.get(u)) {
                    int v = to.get(idx);
                    if (parentEdge[v] == -1 && cap.get(idx) >= minCapacity) {
                        parentEdge[v] = idx;
                        q.add(v);
                    }
                }
            }
            return parentEdge[t] != -1 ? parentEdge : null;
        }

        int edmondsKarp(int s, int t) {
            int flow = 0;
            while (true) {
                int[] parentEdge = findPath(s, t, 1);
                if (parentEdge == null) return flow;
                flow += augment(parentEdge, s, t);
            }
        }

        static class Scaling {
            final int value;
            final int[] deltas;
            final int[] perPhase;
            final int[] amounts;

            Scaling(int value, int[] deltas, int[] perPhase, int[] amounts) {
                this.value = value;
                this.deltas = deltas;
                this.perPhase = perPhase;
                this.amounts = amounts;
            }
        }

        Scaling capacityScaling(int s, int t) {
            int maxCapacity = 0;
            for (int idx = 0; idx < to.size(); idx += 2) maxCapacity = Math.max(maxCapacity, initial.get(idx));
            int delta = 0;
            if (maxCapacity >= 1) {
                delta = 1;
                while (delta * 2 <= maxCapacity) delta *= 2;
            }
            List<Integer> deltas = new ArrayList<>();
            List<Integer> perPhase = new ArrayList<>();
            List<Integer> amounts = new ArrayList<>();
            int flow = 0;
            while (delta >= 1) {
                deltas.add(delta);
                int count = 0;
                while (true) {
                    int[] parentEdge = findPath(s, t, delta);
                    if (parentEdge == null) break;
                    int amount = augment(parentEdge, s, t);
                    check(amount >= delta, "augmentation at least delta");
                    amounts.add(amount);
                    flow += amount;
                    count++;
                }
                perPhase.add(count);
                delta /= 2;
            }
            return new Scaling(flow, toIntArray(deltas), toIntArray(perPhase), toIntArray(amounts));
        }
    }

    static int[] toIntArray(List<Integer> values) {
        int[] result = new int[values.size()];
        for (int i = 0; i < result.length; i++) result[i] = values.get(i);
        return result;
    }

    static MaxFlow buildNetwork(int n, int[][] edges) {
        MaxFlow network = new MaxFlow(n);
        for (int[] e : edges) network.addEdge(e[0], e[1], e[2]);
        return network;
    }

    static class Instance {
        final int n;
        final int[][] edges;

        Instance(int n, int[][] edges) {
            this.n = n;
            this.edges = edges;
        }
    }

    static List<Instance> lcgFlowGraphs(int count, int seed) {
        List<Instance> graphs = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 4 + seed % 4;
            List<int[]> edges = new ArrayList<>();
            for (int node = 0; node < n - 1; node++) {
                seed = (seed * 75 + 74) % 65537;
                edges.add(new int[]{node, node + 1, 1 + seed % 500});
            }
            for (int k = 0; k < n + 3; k++) {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                seed = (seed * 75 + 74) % 65537;
                if (u != v) edges.add(new int[]{u, v, 1 + seed % 500});
            }
            graphs.add(new Instance(n, edges.toArray(new int[0][])));
        }
        return graphs;
    }

    static int phaseCountFor(int maxCapacity) {
        int delta = 1;
        int count = 0;
        while (delta <= maxCapacity) {
            count++;
            delta *= 2;
        }
        return count;
    }

    public static void main(String[] args) {
        MaxFlow.Scaling scaling = buildNetwork(2, new int[][]{{0, 1, 6}}).capacityScaling(0, 1);
        check(scaling.value == 6, "single value");
        check(Arrays.equals(scaling.deltas, new int[]{4, 2, 1}), "single deltas");
        check(Arrays.equals(scaling.perPhase, new int[]{1, 0, 0}), "single phases");
        check(Arrays.equals(scaling.amounts, new int[]{6}), "single amounts");

        scaling = buildNetwork(2, new int[][]{{0, 1, 3}, {0, 1, 4}}).capacityScaling(0, 1);
        check(scaling.value == 7, "parallel value");
        check(Arrays.equals(scaling.deltas, new int[]{4, 2, 1}), "parallel deltas");
        check(Arrays.equals(scaling.perPhase, new int[]{1, 1, 0}), "parallel phases");
        check(Arrays.equals(scaling.amounts, new int[]{4, 3}), "parallel amounts");

        int[][] chainEdges = {{0, 1, 1000}, {1, 2, 999}};
        scaling = buildNetwork(3, chainEdges).capacityScaling(0, 2);
        check(scaling.value == 999, "chain value");
        check(scaling.deltas.length == 10 && scaling.deltas[0] == 512 && scaling.deltas[9] == 1, "chain deltas");
        int chainSum = 0;
        for (int a : scaling.amounts) chainSum += a;
        check(chainSum == scaling.value, "chain amounts sum");
        check(buildNetwork(3, chainEdges).edmondsKarp(0, 2) == 999, "chain oracle");

        scaling = buildNetwork(2, new int[][]{{0, 1, 1}}).capacityScaling(0, 1);
        check(scaling.value == 1, "tiny value");
        check(Arrays.equals(scaling.deltas, new int[]{1}) && Arrays.equals(scaling.amounts, new int[]{1}), "tiny phases");

        scaling = buildNetwork(2, new int[][]{}).capacityScaling(0, 1);
        check(scaling.value == 0 && scaling.deltas.length == 0 && scaling.amounts.length == 0, "empty network");

        int[][] clrs = {{0, 1, 16}, {0, 2, 13}, {1, 2, 10}, {2, 1, 4}, {1, 3, 12}, {3, 2, 9},
                {2, 4, 14}, {4, 3, 7}, {3, 5, 20}, {4, 5, 4}};
        scaling = buildNetwork(6, clrs).capacityScaling(0, 5);
        check(scaling.value == 23, "CLRS value");
        check(Arrays.equals(scaling.deltas, new int[]{16, 8, 4, 2, 1}), "CLRS deltas");
        int clrsSum = 0;
        for (int a : scaling.amounts) clrsSum += a;
        check(clrsSum == scaling.value, "CLRS amounts sum");
        check(buildNetwork(6, clrs).edmondsKarp(0, 5) == 23, "CLRS oracle");

        int checked = 0, latePhaseWork = 0, totalAugmentations = 0;
        for (int seed : new int[]{4551, 4552, 4553}) {
            for (Instance inst : lcgFlowGraphs(8, seed)) {
                scaling = buildNetwork(inst.n, inst.edges).capacityScaling(0, inst.n - 1);
                int maxCapacity = 0;
                for (int[] e : inst.edges) maxCapacity = Math.max(maxCapacity, e[2]);
                check(scaling.deltas.length == phaseCountFor(maxCapacity), "phase count");
                check(scaling.deltas[0] * 2 > maxCapacity && scaling.deltas[0] <= maxCapacity, "top delta");
                int sum = 0;
                for (int a : scaling.amounts) sum += a;
                check(sum == scaling.value, "amount sum");
                int phaseSum = 0;
                for (int c : scaling.perPhase) phaseSum += c;
                check(phaseSum == scaling.amounts.length, "phase total");
                totalAugmentations += scaling.amounts.length;
                boolean late = false;
                for (int i = 1; i < scaling.perPhase.length; i++) late |= scaling.perPhase[i] > 0;
                if (late) latePhaseWork++;
                int expected = buildNetwork(inst.n, inst.edges).edmondsKarp(0, inst.n - 1);
                check(scaling.value == expected, "LCG value " + scaling.value + " vs " + expected);
                checked++;
            }
        }
        check(checked == 24, "graph count");
        check(totalAugmentations > 0, "augmentations happened");
        check(latePhaseWork >= 1, "later phases did work");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
