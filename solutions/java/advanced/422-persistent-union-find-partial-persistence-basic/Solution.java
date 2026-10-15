/** Task 422 - Persistent union-find (partial persistence) basic. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class PersistentUnionFind {
    final int n;
    final List<List<int[]>> parentHistory = new ArrayList<>();   // (time, new parent)
    final List<List<int[]>> sizeHistory = new ArrayList<>();     // (time, new root size)

    PersistentUnionFind(int n) {
        this.n = n;
        for (int i = 0; i < n; i++) {
            parentHistory.add(new ArrayList<>());
            sizeHistory.add(new ArrayList<>());
        }
    }

    private static Integer valueAt(List<int[]> history, int time) {
        int lo = 0, hi = history.size();
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (history.get(mid)[0] <= time) lo = mid + 1; else hi = mid;
        }
        return lo == 0 ? null : history.get(lo - 1)[1];
    }

    int find(int node, int time) {
        while (true) {
            Integer parent = valueAt(parentHistory.get(node), time);
            if (parent == null) return node;
            node = parent;
        }
    }

    int size(int node, int time) {
        Integer recorded = valueAt(sizeHistory.get(node), time);
        return recorded == null ? 1 : recorded;
    }

    boolean union(int a, int b, int time) {
        int rootA = find(a, time - 1);
        int rootB = find(b, time - 1);
        if (rootA == rootB) return false;
        int sizeA = size(rootA, time - 1);
        int sizeB = size(rootB, time - 1);
        if (sizeA < sizeB) {
            int tmp = rootA; rootA = rootB; rootB = tmp;
            tmp = sizeA; sizeA = sizeB; sizeB = tmp;
        }
        parentHistory.get(rootB).add(new int[]{time, rootA});
        sizeHistory.get(rootA).add(new int[]{time, sizeA + sizeB});
        return true;
    }

    int componentCount(int time) {
        Set<Integer> roots = new HashSet<>();
        for (int node = 0; node < n; node++) roots.add(find(node, time));
        return roots.size();
    }
}

public class Solution {
    private static int rootOf(int[] parent, int node) {
        while (parent[node] != node) node = parent[node];
        return node;
    }

    static int[] replay(int n, int[][] operations, int time) {
        int[] parent = new int[n];
        int[] size = new int[n];
        for (int i = 0; i < n; i++) { parent[i] = i; size[i] = 1; }
        for (int step = 1; step <= time; step++) {
            int rootA = rootOf(parent, operations[step - 1][0]);
            int rootB = rootOf(parent, operations[step - 1][1]);
            if (rootA != rootB) {
                if (size[rootA] < size[rootB]) { int tmp = rootA; rootA = rootB; rootB = tmp; }
                parent[rootB] = rootA;
                size[rootA] += size[rootB];
            }
        }
        return parent;
    }

    private static int[] oracleRoots(int n, int[][] operations, int time) {
        int[] parent = replay(n, operations, time);
        int[] roots = new int[n];
        for (int node = 0; node < n; node++) roots[node] = rootOf(parent, node);
        return roots;
    }

    public static void main(String[] args) {
        int[][] operations = {{0, 1}, {2, 3}, {1, 3}, {0, 2}};
        PersistentUnionFind uf = new PersistentUnionFind(4);
        boolean[] merges = new boolean[4];
        for (int time = 0; time < 4; time++) merges[time] = uf.union(operations[time][0], operations[time][1], time + 1);
        check(Arrays.equals(merges, new boolean[]{true, true, true, false}), "merge flags");

        check(uf.componentCount(0) == 4, "time 0 count");
        check(uf.componentCount(1) == 3, "time 1 count");
        check(uf.componentCount(2) == 2, "time 2 count");
        check(uf.componentCount(3) == 1, "time 3 count");
        check(uf.componentCount(4) == 1, "time 4 count");

        check(uf.find(1, 1) == 0 && uf.find(0, 1) == 0, "first union representative");
        check(uf.find(3, 1) == 3, "second union not yet applied");
        check(uf.find(3, 2) == 2, "after second union");
        check(uf.find(3, 3) == 0, "after third union");
        check(uf.find(3, 4) == 0, "after fourth union");
        check(uf.find(2, 2) == 2, "own root early");
        check(uf.find(1, 0) == 1 && uf.find(2, 0) == 2, "pre-union representatives");

        for (int time = 0; time <= 4; time++) {
            int[] expected = oracleRoots(4, operations, time);
            Set<Integer> distinct = new HashSet<>();
            for (int root : expected) distinct.add(root);
            check(uf.componentCount(time) == distinct.size(), "oracle count at " + time);
            for (int node = 0; node < 4; node++) {
                check(uf.find(node, time) == expected[node], "oracle representative " + time + "/" + node);
            }
        }

        for (int seed : new int[]{422, 31415, 271828}) {
            int n = 2 + seed % 9;
            PersistentUnionFind structure = new PersistentUnionFind(n);
            int[][] ops = new int[40][];
            int local = seed;
            for (int i = 0; i < 40; i++) {
                local = (local * 75 + 74) % 65537;
                int a = local % n;
                local = (local * 75 + 74) % 65537;
                ops[i] = new int[]{a, local % n};
            }
            int applied = 0;
            for (int time = 1; time <= 40; time++) {
                if (structure.union(ops[time - 1][0], ops[time - 1][1], time)) applied++;
                check(structure.componentCount(time) == n - applied, "lcg count");
            }
            for (int step = 0; step < 60; step++) {
                local = (local * 75 + 74) % 65537;
                int time = local % 41;
                int[] expected = oracleRoots(n, ops, time);
                Set<Integer> distinct = new HashSet<>();
                for (int root : expected) distinct.add(root);
                check(structure.componentCount(time) == distinct.size(), "lcg oracle count");
                for (int node = 0; node < n; node++) {
                    check(structure.find(node, time) == expected[node], "lcg oracle representative");
                }
            }
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
