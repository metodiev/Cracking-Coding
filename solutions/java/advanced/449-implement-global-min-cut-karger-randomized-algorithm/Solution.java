/** Task 449 - Implement global min-cut (Karger randomized algorithm). (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

class Lcg {
    private int state;

    Lcg(int seed) {
        state = seed % 65537;
    }

    int next() {
        state = (state * 75 + 74) % 65537;
        return state;
    }

    int below(int bound) {
        return next() % bound;
    }
}

class CutResult {
    final int value;
    final int[] sideA;
    final int[] sideB;

    CutResult(int value, int[] sideA, int[] sideB) {
        this.value = value;
        this.sideA = sideA;
        this.sideB = sideB;
    }
}

class Karger {
    static CutResult minCut(int n, int[][] edges, int seed) {
        Lcg rng = new Lcg(seed);
        int[] label = new int[n];
        for (int i = 0; i < n; i++) label[i] = i;
        List<int[]> remaining = new ArrayList<>();
        for (int[] e : edges) remaining.add(new int[]{e[0], e[1]});
        int components = n;
        while (components > 2) {
            int[] pick = remaining.get(rng.below(remaining.size()));
            int a = label[pick[0]];
            int b = label[pick[1]];
            for (int i = 0; i < n; i++) if (label[i] == b) label[i] = a;
            List<int[]> next = new ArrayList<>();
            for (int[] e : remaining) if (label[e[0]] != label[e[1]]) next.add(e);
            remaining = next;
            components--;
        }
        int first = label[0];
        List<Integer> sideA = new ArrayList<>();
        List<Integer> sideB = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (label[i] == first) sideA.add(i);
            else sideB.add(i);
        }
        int[] arrayA = new int[sideA.size()];
        int[] arrayB = new int[sideB.size()];
        for (int i = 0; i < arrayA.length; i++) arrayA[i] = sideA.get(i);
        for (int i = 0; i < arrayB.length; i++) arrayB[i] = sideB.get(i);
        return new CutResult(remaining.size(), arrayA, arrayB);
    }
}

public class Solution {
    static int bruteMinCut(int n, int[][] edges) {
        int best = Integer.MAX_VALUE;
        for (int mask = 1; mask < (1 << (n - 1)); mask++) {
            int cut = 0;
            for (int[] e : edges) {
                int bu = e[0] > 0 ? (mask >> (e[0] - 1)) & 1 : 0;
                int bv = e[1] > 0 ? (mask >> (e[1] - 1)) & 1 : 0;
                if (bu != bv) cut++;
            }
            best = Math.min(best, cut);
        }
        return best;
    }

    static void checkWitness(int n, int[][] edges, int value, int[] sideA, int[] sideB) {
        boolean[] onA = new boolean[n];
        for (int v : sideA) onA[v] = true;
        for (int v : sideB) if (onA[v]) throw new AssertionError("FAILED: sides overlap");
        if (sideA.length == 0 || sideB.length == 0) throw new AssertionError("FAILED: empty side");
        if (sideA.length + sideB.length != n) throw new AssertionError("FAILED: not a partition");
        int crossing = 0;
        for (int[] e : edges) if (onA[e[0]] != onA[e[1]]) crossing++;
        if (crossing != value) throw new AssertionError("FAILED: witness crossing " + crossing + " != " + value);
    }

    static int[] kargerBestOf(int n, int[][] edges, int baseSeed, int trials, int expected) {
        int best = Integer.MAX_VALUE;
        int successes = 0;
        for (int t = 0; t < trials; t++) {
            int seed = (baseSeed * 31 + t * 17 + 1) % 65537;
            CutResult result = Karger.minCut(n, edges, seed);
            checkWitness(n, edges, result.value, result.sideA, result.sideB);
            if (result.value < expected) {
                throw new AssertionError("FAILED: cut below minimum " + result.value + " < " + expected);
            }
            if (result.value == expected) successes++;
            best = Math.min(best, result.value);
        }
        return new int[]{best, successes};
    }

    private static int seed;

    private static List<Object[]> lcgGraphs(int count, int maxN, int baseSeed) {
        seed = baseSeed;
        List<Object[]> graphs = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            int n = 3 + seed % (maxN - 2);
            List<int[]> edges = new ArrayList<>();
            for (int node = 1; node < n; node++) {
                seed = (seed * 75 + 74) % 65537;
                int parent = seed % node;
                edges.add(new int[]{node, parent});
            }
            int extras = seed % (n + 2);
            for (int e = 0; e < extras; e++) {
                seed = (seed * 75 + 74) % 65537;
                int u = seed % n;
                seed = (seed * 75 + 74) % 65537;
                int v = seed % n;
                if (u != v) edges.add(new int[]{u, v});
            }
            graphs.add(new Object[]{n, edges.toArray(new int[0][])});
        }
        return graphs;
    }

    public static void main(String[] args) {
        int[][] triangle = {{0, 1}, {1, 2}, {0, 2}};
        check(bruteMinCut(3, triangle) == 2, "triangle brute");
        for (int seedValue : new int[]{1, 2, 3, 65536}) {
            CutResult result = Karger.minCut(3, triangle, seedValue);
            checkWitness(3, triangle, result.value, result.sideA, result.sideB);
            check(result.value == 2, "triangle seed " + seedValue);
        }

        List<int[]> k4List = new ArrayList<>();
        for (int u = 0; u < 4; u++) {
            for (int v = u + 1; v < 4; v++) k4List.add(new int[]{u, v});
        }
        int[][] k4 = k4List.toArray(new int[0][]);
        check(bruteMinCut(4, k4) == 3, "K4 brute");
        int[] k4Stats = kargerBestOf(4, k4, 42, 60, 3);
        check(k4Stats[0] == 3, "K4 best");
        check(k4Stats[1] > 0, "K4 successes");

        int totalTrials = 0;
        for (Object[] graph : lcgGraphs(3, 6, 449)) {
            int n = (Integer) graph[0];
            int[][] edges = (int[][]) graph[1];
            int expected = bruteMinCut(n, edges);
            for (int baseSeed : new int[]{1, 42, 1234, 65536}) {
                int[] stats = kargerBestOf(n, edges, baseSeed, 400, expected);
                check(stats[0] == expected, "LCG best seed " + baseSeed);
                check(stats[1] > 0, "LCG successes seed " + baseSeed);
                totalTrials += 400;
            }
        }
        check(totalTrials == 4800, "trial coverage");

        List<int[]> k6List = new ArrayList<>();
        for (int u = 0; u < 6; u++) {
            for (int v = u + 1; v < 6; v++) k6List.add(new int[]{u, v});
        }
        int[][] k6 = k6List.toArray(new int[0][]);
        check(bruteMinCut(6, k6) == 5, "K6 brute");
        int[] k6Stats = kargerBestOf(6, k6, 1234, 400, 5);
        check(k6Stats[0] == 5, "K6 best");
        check(k6Stats[1] > 0, "K6 successes");

        int[][] parallel = {{0, 1}, {1, 0}, {0, 1}};
        check(bruteMinCut(2, parallel) == 3, "parallel brute");
        int[] parallelStats = kargerBestOf(2, parallel, 7, 25, 3);
        check(parallelStats[0] == 3 && parallelStats[1] == 25, "parallel trials");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
