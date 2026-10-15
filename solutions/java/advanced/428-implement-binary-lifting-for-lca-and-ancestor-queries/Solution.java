/** Task 428 - Implement binary lifting for LCA and ancestor queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class BinaryLifting {
    final int n;
    final int log;
    final int[] depth;
    final int[] parent;
    final int[][] up;

    BinaryLifting(int n, int[][] edges) {
        this.n = n;
        int levels = 1;
        while ((1 << levels) <= n) levels++;
        log = levels;
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        depth = new int[n];
        parent = new int[n];
        boolean[] visited = new boolean[n];
        visited[0] = true;
        List<Integer> order = new ArrayList<>();
        order.add(0);
        for (int i = 0; i < order.size(); i++) {
            int u = order.get(i);
            for (int v : adjacency.get(u)) {
                if (!visited[v]) {
                    visited[v] = true;
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    order.add(v);
                }
            }
        }
        up = new int[log][];
        up[0] = parent.clone();
        for (int j = 1; j < log; j++) {
            int[] previous = up[j - 1];
            int[] current = new int[n];
            for (int u = 0; u < n; u++) current[u] = previous[previous[u]];
            up[j] = current;
        }
    }

    int kthAncestor(int node, int k) {
        if (k > depth[node]) return -1;
        int bit = 0;
        while (k != 0) {
            if ((k & 1) != 0) node = up[bit][node];
            k >>= 1;
            bit++;
        }
        return node;
    }

    int lca(int a, int b) {
        if (depth[a] < depth[b]) { int tmp = a; a = b; b = tmp; }
        a = kthAncestor(a, depth[a] - depth[b]);
        if (a == b) return a;
        for (int j = log - 1; j >= 0; j--) {
            if (up[j][a] != up[j][b]) {
                a = up[j][a];
                b = up[j][b];
            }
        }
        return parent[a];
    }
}

public class Solution {
    static int[][] rootedTree(int n, int[][] edges) {
        List<List<Integer>> adjacency = new ArrayList<>();
        for (int i = 0; i < n; i++) adjacency.add(new ArrayList<>());
        for (int[] edge : edges) {
            adjacency.get(edge[0]).add(edge[1]);
            adjacency.get(edge[1]).add(edge[0]);
        }
        int[] parent = new int[n];
        int[] depth = new int[n];
        boolean[] visited = new boolean[n];
        visited[0] = true;
        List<Integer> order = new ArrayList<>();
        order.add(0);
        for (int i = 0; i < order.size(); i++) {
            int u = order.get(i);
            for (int v : adjacency.get(u)) {
                if (!visited[v]) {
                    visited[v] = true;
                    parent[v] = u;
                    depth[v] = depth[u] + 1;
                    order.add(v);
                }
            }
        }
        return new int[][]{parent, depth};
    }

    static int naiveLca(int[] parent, int[] depth, int a, int b) {
        while (depth[a] > depth[b]) a = parent[a];
        while (depth[b] > depth[a]) b = parent[b];
        while (a != b) {
            a = parent[a];
            b = parent[b];
        }
        return a;
    }

    static int naiveKthAncestor(int[] parent, int[] depth, int node, int k) {
        if (k > depth[node]) return -1;
        for (int i = 0; i < k; i++) node = parent[node];
        return node;
    }

    private static int seed = 428;

    private static Object[] lcgTree(int maxN) {
        int n = 2 + seed % (maxN - 1);
        int[][] edges = new int[n - 1][];
        for (int node = 1; node < n; node++) {
            seed = (seed * 75 + 74) % 65537;
            edges[node - 1] = new int[]{seed % node, node};
        }
        return new Object[]{n, edges};
    }

    public static void main(String[] args) {
        BinaryLifting path = new BinaryLifting(4, new int[][]{{0, 1}, {1, 2}, {2, 3}});
        check(path.lca(0, 3) == 0, "path lca root");
        check(path.lca(2, 3) == 2, "path lca nested");
        check(path.lca(1, 2) == 1, "path lca adjacent");
        check(path.lca(3, 3) == 3, "path lca self");
        check(path.lca(0, 0) == 0, "path lca root self");
        check(path.kthAncestor(3, 0) == 3, "path kth 0");
        check(path.kthAncestor(3, 1) == 2, "path kth 1");
        check(path.kthAncestor(3, 2) == 1, "path kth 2");
        check(path.kthAncestor(3, 3) == 0, "path kth 3");
        check(path.kthAncestor(3, 4) == -1, "path kth beyond root");
        check(path.kthAncestor(0, 1) == -1, "root has no ancestor");

        BinaryLifting star = new BinaryLifting(4, new int[][]{{0, 1}, {0, 2}, {0, 3}});
        check(star.lca(1, 2) == 0, "star lca leaves");
        check(star.lca(1, 3) == 0, "star lca leaves 2");
        check(star.lca(0, 3) == 0, "star lca centre");
        check(star.lca(2, 2) == 2, "star self");
        check(star.kthAncestor(2, 1) == 0, "star kth 1");
        check(star.kthAncestor(2, 2) == -1, "star kth beyond");

        BinaryLifting single = new BinaryLifting(1, new int[][]{});
        check(single.lca(0, 0) == 0, "single lca");
        check(single.kthAncestor(0, 0) == 0, "single kth 0");
        check(single.kthAncestor(0, 1) == -1, "single kth beyond");

        for (int trial = 0; trial < 25; trial++) {
            Object[] tree = lcgTree(30);
            int n = (Integer) tree[0];
            int[][] edges = (int[][]) tree[1];
            BinaryLifting bl = new BinaryLifting(n, edges);
            int[][] rooted = rootedTree(n, edges);
            int[] parent = rooted[0];
            int[] depth = rooted[1];
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    int expected = naiveLca(parent, depth, a, b);
                    check(bl.lca(a, b) == expected, "lcg lca " + trial + "/" + a + "/" + b);
                    check(bl.lca(a, b) == bl.lca(b, a), "lcg lca symmetry");
                }
            }
            for (int u = 0; u < n; u++) {
                for (int k = 0; k <= depth[u] + 2; k++) {
                    int expected = naiveKthAncestor(parent, depth, u, k);
                    check(bl.kthAncestor(u, k) == expected, "lcg kth " + trial + "/" + u + "/" + k);
                }
            }
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
