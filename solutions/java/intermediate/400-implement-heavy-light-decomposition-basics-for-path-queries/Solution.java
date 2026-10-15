/** Task 400 - Implement heavy-light decomposition basics for path queries. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class Fenwick {
    private final long[] tree;

    Fenwick(int size) {
        tree = new long[size + 1];
    }

    void add(int index, long delta) {
        int i = index + 1;
        while (i < tree.length) {
            tree[i] += delta;
            i += i & -i;
        }
    }

    private long prefix(int count) {
        long total = 0;
        int i = count;
        while (i > 0) {
            total += tree[i];
            i -= i & -i;
        }
        return total;
    }

    long rangeSum(int left, int right) {
        return prefix(right + 1) - prefix(left);
    }
}

class HeavyLight {
    final int[] parent, depth, size, heavy, head, pos;
    final long[] values;
    final Fenwick fenwick;

    HeavyLight(int n, int[][] edges, int[] nodeValues) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        parent = new int[n];
        depth = new int[n];
        Arrays.fill(parent, -1);
        List<Integer> order = new ArrayList<>();
        order.add(0);
        for (int i = 0; i < order.size(); i++) {
            int u = order.get(i);
            for (int v : adjacency.get(u)) {
                if (v != parent[u]) {
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    order.add(v);
                }
            }
        }
        size = new int[n];
        heavy = new int[n];
        Arrays.fill(heavy, -1);
        Arrays.fill(size, 1);
        for (int i = order.size() - 1; i >= 0; i--) {
            int u = order.get(i);
            int best = 0;
            for (int v : adjacency.get(u)) {
                if (v != parent[u]) {
                    size[u] += size[v];
                    if (size[v] > best) {
                        best = size[v];
                        heavy[u] = v;
                    }
                }
            }
        }
        head = new int[n];
        pos = new int[n];
        int position = 0;
        for (int u : order) {
            if (u == 0 || heavy[parent[u]] != u) {          // starts a new chain
                int x = u;
                while (x != -1) {
                    head[x] = u;
                    pos[x] = position++;
                    x = heavy[x];
                }
            }
        }
        values = new long[n];
        for (int i = 0; i < n; i++) values[i] = nodeValues[i];
        fenwick = new Fenwick(n);
        for (int node = 0; node < n; node++) fenwick.add(pos[node], values[node]);
    }

    void update(int node, long value) {
        long delta = value - values[node];
        values[node] = value;
        fenwick.add(pos[node], delta);
    }

    long pathSum(int u, int v) {
        long total = 0;
        while (head[u] != head[v]) {
            if (depth[head[u]] < depth[head[v]]) {
                int tmp = u; u = v; v = tmp;
            }
            total += fenwick.rangeSum(pos[head[u]], pos[u]);
            u = parent[head[u]];
        }
        if (depth[u] > depth[v]) {
            int tmp = u; u = v; v = tmp;
        }
        total += fenwick.rangeSum(pos[u], pos[v]);
        return total;
    }
}

public class Solution {
    static long brutePathSum(int n, int[][] edges, long[] values, int u, int v) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        int[] parent = new int[n];
        Arrays.fill(parent, -1);
        parent[u] = u;
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(u);
        while (!queue.isEmpty()) {
            int x = queue.poll();
            if (x == v) break;
            for (int y : adjacency.get(x)) {
                if (parent[y] == -1) {
                    parent[y] = x;
                    queue.add(y);
                }
            }
        }
        long total = 0;
        int x = v;
        while (x != u) {
            total += values[x];
            x = parent[x];
        }
        return total + values[u];
    }

    private static int seed = 400;

    public static void main(String[] args) {
        HeavyLight hld = new HeavyLight(4, new int[][]{{0,1},{1,2},{2,3}}, new int[]{1,2,3,4});
        check(hld.pathSum(0, 3) == 10, "path graph full");
        check(hld.pathSum(0, 0) == 1, "single node");
        check(hld.pathSum(1, 2) == 5, "middle segment");
        hld.update(2, 10);
        check(hld.pathSum(0, 3) == 17, "after update");
        check(hld.pathSum(1, 2) == 12, "middle after update");
        check(hld.pathSum(2, 2) == 10, "single after update");

        HeavyLight star = new HeavyLight(3, new int[][]{{0,1},{0,2}}, new int[]{5,1,1});
        check(star.pathSum(1, 2) == 7, "star leaves");
        check(star.pathSum(0, 0) == 5, "star centre");

        for (int trial = 0; trial < 25; trial++) {
            int n = 2 + seed % 49;
            int[] values = new int[n];
            List<int[]> edges = new ArrayList<>();
            for (int node = 0; node < n; node++) {
                seed = (seed * 75 + 74) % 65537;
                values[node] = seed % 20 - 5;
                if (node > 0) {
                    seed = (seed * 75 + 74) % 65537;
                    edges.add(new int[]{seed % node, node});
                }
            }
            int[][] edgeArray = edges.toArray(new int[0][]);
            long[] longValues = new long[n];
            for (int i = 0; i < n; i++) longValues[i] = values[i];
            HeavyLight random = new HeavyLight(n, edgeArray, values);
            int localSeed = n;
            for (int step = 0; step < 12; step++) {
                localSeed = (localSeed * 75 + 74) % 65537;
                int u = localSeed % n;
                localSeed = (localSeed * 75 + 74) % 65537;
                int v = localSeed % n;
                check(random.pathSum(u, v) == brutePathSum(n, edgeArray, longValues, u, v), "random path");
                localSeed = (localSeed * 75 + 74) % 65537;
                int node = localSeed % n;
                localSeed = (localSeed * 75 + 74) % 65537;
                long newValue = localSeed % 40 - 20;
                longValues[node] = newValue;
                random.update(node, newValue);
                check(random.pathSum(u, v) == brutePathSum(n, edgeArray, longValues, u, v), "random after update");
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
