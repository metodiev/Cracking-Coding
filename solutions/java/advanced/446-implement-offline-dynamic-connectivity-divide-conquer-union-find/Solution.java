/** Task 446 - Implement offline dynamic connectivity (divide & conquer + union find). (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

class RollbackDSU {
    private final int[] parent;
    private final int[] size;
    private final int[] stackA;
    private final int[] stackB;
    private int top;

    RollbackDSU(int n) {
        parent = new int[n];
        size = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
            size[i] = 1;
        }
        stackA = new int[n];
        stackB = new int[n];
    }

    int find(int x) {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    boolean union(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return false;
        if (size[ra] > size[rb]) {
            int swap = ra;
            ra = rb;
            rb = swap;
        }
        parent[ra] = rb;
        size[rb] += size[ra];
        stackA[top] = ra;
        stackB[top] = rb;
        top++;
        return true;
    }

    boolean same(int a, int b) {
        return find(a) == find(b);
    }

    int snapshot() {
        return top;
    }

    void rollback(int snapshot) {
        while (top > snapshot) {
            top--;
            int ra = stackA[top];
            int rb = stackB[top];
            parent[ra] = ra;
            size[rb] -= size[ra];
        }
    }
}

class OfflineConnectivity {
    private final int n;
    private final int[][] ops;
    private final RollbackDSU dsu;
    private final Boolean[] answers;

    OfflineConnectivity(int n, int[][] ops) {
        this.n = n;
        this.ops = ops;
        this.dsu = new RollbackDSU(n);
        this.answers = new Boolean[ops.length];
    }

    Boolean[] run() {
        int q = ops.length;
        if (q == 0) return answers;
        Map<Long, Integer> counts = new HashMap<>();
        Map<Long, int[]> open = new HashMap<>();
        List<int[]> intervals = new ArrayList<>();
        for (int t = 0; t < q; t++) {
            int kind = ops[t][0], u = ops[t][1], v = ops[t][2];
            long key = (long) Math.min(u, v) * 1000 + Math.max(u, v);
            if (kind == 0) {
                int c = counts.getOrDefault(key, 0) + 1;
                counts.put(key, c);
                if (c == 1) open.put(key, new int[]{Math.min(u, v), Math.max(u, v), t});
            } else if (kind == 1) {
                int c = counts.getOrDefault(key, 0);
                if (c > 0) {
                    counts.put(key, c - 1);
                    if (c - 1 == 0) {
                        int[] info = open.remove(key);
                        if (info[2] <= t - 1) intervals.add(new int[]{info[0], info[1], info[2], t - 1});
                    }
                }
            }
        }
        for (int[] info : open.values()) {
            intervals.add(new int[]{info[0], info[1], info[2], q - 1});
        }
        solve(0, q - 1, intervals);
        return answers;
    }

    private void solve(int lo, int hi, List<int[]> active) {
        int snapshot = dsu.snapshot();
        List<int[]> rest = new ArrayList<>();
        for (int[] e : active) {
            if (e[2] <= lo && hi <= e[3]) dsu.union(e[0], e[1]);
            else rest.add(e);
        }
        if (lo == hi) {
            if (ops[lo][0] == 2) answers[lo] = dsu.same(ops[lo][1], ops[lo][2]);
        } else {
            int mid = (lo + hi) >> 1;
            List<int[]> left = new ArrayList<>();
            List<int[]> right = new ArrayList<>();
            for (int[] e : rest) {
                if (e[2] <= mid && e[3] >= lo) left.add(e);
                if (e[2] <= hi && e[3] >= mid + 1) right.add(e);
            }
            solve(lo, mid, left);
            solve(mid + 1, hi, right);
        }
        dsu.rollback(snapshot);
    }
}

public class Solution {
    private static int seed;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static List<int[][]> lcgOpScripts(int trials, int n, int q, int baseSeed) {
        List<int[][]> scripts = new ArrayList<>();
        seed = baseSeed;
        for (int t = 0; t < trials; t++) {
            Map<Long, Integer> counts = new HashMap<>();
            int[][] ops = new int[q][];
            for (int step = 0; step < q; step++) {
                int kind = nextSeed() % 3;
                int u = nextSeed() % n;
                int v = nextSeed() % n;
                if (u == v) v = (v + 1) % n;
                long key = (long) Math.min(u, v) * 1000 + Math.max(u, v);
                if (kind == 1 && counts.isEmpty()) kind = 0;
                if (kind == 0) {
                    ops[step] = new int[]{0, u, v};
                    counts.put(key, counts.getOrDefault(key, 0) + 1);
                } else if (kind == 1) {
                    List<Long> keys = new ArrayList<>(counts.keySet());
                    Collections.sort(keys);
                    long pick = keys.get(nextSeed() % keys.size());
                    int a = (int) (pick / 1000);
                    int b = (int) (pick % 1000);
                    ops[step] = new int[]{1, a, b};
                    int c = counts.get(pick) - 1;
                    if (c == 0) counts.remove(pick);
                    else counts.put(pick, c);
                } else {
                    ops[step] = new int[]{2, u, v};
                }
            }
            scripts.add(ops);
        }
        return scripts;
    }

    private static Boolean[] bruteAnswers(int n, int[][] ops) {
        Map<Long, Integer> counts = new HashMap<>();
        Boolean[] answers = new Boolean[ops.length];
        for (int t = 0; t < ops.length; t++) {
            int kind = ops[t][0], u = ops[t][1], v = ops[t][2];
            long key = (long) Math.min(u, v) * 1000 + Math.max(u, v);
            if (kind == 0) {
                counts.put(key, counts.getOrDefault(key, 0) + 1);
            } else if (kind == 1) {
                int c = counts.get(key) - 1;
                if (c == 0) counts.remove(key);
                else counts.put(key, c);
            } else {
                List<List<Integer>> adjacency = new ArrayList<>();
                for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
                for (long e : counts.keySet()) {
                    int a = (int) (e / 1000);
                    int b = (int) (e % 1000);
                    adjacency.get(a).add(b);
                    adjacency.get(b).add(a);
                }
                boolean[] seen = new boolean[n];
                seen[u] = true;
                Deque<Integer> stack = new ArrayDeque<>();
                stack.push(u);
                while (!stack.isEmpty()) {
                    int x = stack.pop();
                    for (int y : adjacency.get(x)) {
                        if (!seen[y]) {
                            seen[y] = true;
                            stack.push(y);
                        }
                    }
                }
                answers[t] = seen[v];
            }
        }
        return answers;
    }

    public static void main(String[] args) {
        int[][] ops = {{0, 0, 1}, {2, 0, 1}, {0, 1, 2}, {2, 0, 2}, {1, 0, 1}, {2, 0, 2}, {2, 0, 0}};
        Boolean[] expected = {null, true, null, true, null, false, true};
        check(Arrays.equals(new OfflineConnectivity(3, ops).run(), expected), "hand script 1");
        int[][] ops2 = {{2, 0, 1}, {0, 0, 1}, {2, 0, 1}, {1, 0, 1}, {2, 0, 1}, {2, 1, 1}};
        Boolean[] expected2 = {false, null, true, null, false, true};
        check(Arrays.equals(new OfflineConnectivity(2, ops2).run(), expected2), "hand script 2");

        int queries = 0;
        for (int[][] script : lcgOpScripts(30, 8, 40, 446)) {
            Boolean[] brute = bruteAnswers(8, script);
            for (Boolean answer : brute) if (answer != null) queries++;
            check(Arrays.equals(new OfflineConnectivity(8, script).run(), brute), "random script n=8");
        }
        check(queries > 100, "query coverage");

        for (int[][] script : lcgOpScripts(10, 12, 60, 777)) {
            check(Arrays.equals(new OfflineConnectivity(12, script).run(), bruteAnswers(12, script)),
                    "random script n=12");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
