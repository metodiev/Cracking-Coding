/** Task 419 - Implement dynamic tree (link-cut tree) basic operations. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A link-cut tree represents a forest of rooted trees with splay trees over preferred
 * paths. Nodes are 1-based here, 0 is the null child. Arrays: left/right/parent (the
 * parent entry doubles as the splay parent and the path-parent), rev (lazy subtree flip),
 * val (node value) and total (splay-subtree aggregate = path sum of the preferred path).
 *
 * Splay details:
 *   * isRoot(x): parent[x] == 0 or x is neither the left nor the right child of parent[x]
 *     — that is where a preferred path ends and a path-parent edge starts,
 *   * every splay first pushes the rev flags from the splay root down to x, so rotations
 *     never see stale orientations,
 *   * rotations fix three families of links (b-subtree, p, g) and pull p then x.
 * Operations: access (make the x-root path preferred), makeroot (access + rev toggle),
 * findroot (access + descend leftmost + splay), link/cut (both guarded), connected, and
 * pathSum(u, v) = makeroot(u) + access(v) -> total[v]. Ops in the tests stay in the
 * hundreds; every result is validated against an explicit adjacency oracle.
 */
import java.util.*;

class LinkCutTree {
    private final int[] left, right, parent;
    private final boolean[] rev;
    private final long[] val, total;

    LinkCutTree(long[] values) {
        int n = values.length;
        left = new int[n + 1];
        right = new int[n + 1];
        parent = new int[n + 1];
        rev = new boolean[n + 1];
        val = new long[n + 1];
        total = new long[n + 1];
        for (int i = 1; i <= n; i++) {
            val[i] = values[i - 1];
            total[i] = values[i - 1];
        }
    }

    private boolean isRoot(int x) {
        int p = parent[x];
        return p == 0 || (left[p] != x && right[p] != x);
    }

    private void push(int x) {
        if (rev[x]) {
            int t = left[x];
            left[x] = right[x];
            right[x] = t;
            if (left[x] != 0) rev[left[x]] = !rev[left[x]];
            if (right[x] != 0) rev[right[x]] = !rev[right[x]];
            rev[x] = false;
        }
    }

    private void pull(int x) {
        total[x] = total[left[x]] + total[right[x]] + val[x];
    }

    private void rotate(int x) {
        int p = parent[x], g = parent[p];
        if (left[p] == x) {
            int b = right[x];
            right[x] = p;
            left[p] = b;
            if (b != 0) parent[b] = p;
        } else {
            int b = left[x];
            left[x] = p;
            right[p] = b;
            if (b != 0) parent[b] = p;
        }
        parent[p] = x;
        parent[x] = g;
        if (g != 0) {
            if (left[g] == p) left[g] = x;
            else if (right[g] == p) right[g] = x;
        }
        pull(p);
        pull(x);
    }

    private void splay(int x) {
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(x);
        int y = x;
        while (!isRoot(y)) {
            y = parent[y];
            stack.push(y);
        }
        while (!stack.isEmpty()) push(stack.pop());
        while (!isRoot(x)) {
            int p = parent[x], g = parent[p];
            if (!isRoot(p)) {
                if ((left[p] == x) == (left[g] == p)) rotate(p);
                else rotate(x);
            }
            rotate(x);
        }
    }

    private void access(int x) {
        int last = 0;
        int y = x;
        while (y != 0) {
            splay(y);
            right[y] = last;
            if (last != 0) parent[last] = y;
            pull(y);
            last = y;
            y = parent[y];
        }
        splay(x);
    }

    void makeroot(int x) {
        access(x);
        rev[x] = !rev[x];
    }

    int findroot(int x) {
        access(x);
        while (true) {
            push(x);
            if (left[x] == 0) break;
            x = left[x];
        }
        splay(x);
        return x;
    }

    boolean connected(int u, int v) {
        if (u == v) return true;
        return findroot(u) == findroot(v);
    }

    void link(int u, int v) {
        makeroot(u);
        if (findroot(v) != u) parent[u] = v;
    }

    void cut(int u, int v) {
        makeroot(u);
        access(v);
        if (left[v] == u && right[u] == 0) {
            left[v] = 0;
            parent[u] = 0;
            pull(v);
        }
    }

    long pathSum(int u, int v) {
        makeroot(u);
        access(v);
        return total[v];
    }
}

/** Oracle: explicit adjacency plus BFS for connectivity and path sums. */
class Forest {
    private final List<Set<Integer>> adj;

    Forest(int n) {
        adj = new ArrayList<>();
        for (int i = 0; i <= n; i++) adj.add(new TreeSet<>());
    }

    void link(int u, int v) {
        adj.get(u).add(v);
        adj.get(v).add(u);
    }

    void cut(int u, int v) {
        adj.get(u).remove(v);
        adj.get(v).remove(u);
    }

    boolean hasEdge(int u, int v) {
        return adj.get(u).contains(v);
    }

    /** Unique forest path u -> v as a vertex list, or null when disconnected. */
    int[] path(int u, int v) {
        int n = adj.size() - 1;
        int[] prev = new int[n + 1];
        Arrays.fill(prev, -1);
        prev[u] = 0;
        Deque<Integer> queue = new ArrayDeque<>();
        queue.add(u);
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            if (cur == v) {
                List<Integer> path = new ArrayList<>();
                int node = v;
                while (node != 0) {
                    path.add(node);
                    node = prev[node];
                }
                Collections.reverse(path);
                int[] out = new int[path.size()];
                for (int k = 0; k < out.length; k++) out[k] = path.get(k);
                return out;
            }
            for (int nxt : adj.get(cur)) {
                if (prev[nxt] == -1) {
                    prev[nxt] = cur;
                    queue.add(nxt);
                }
            }
        }
        return null;
    }

    boolean connected(int u, int v) {
        return path(u, v) != null;
    }
}

public class Solution {
    private static int seed;

    private static int nextRand() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static long pathSumOracle(Forest forest, long[] values, int u, int v) {
        int[] path = forest.path(u, v);
        long sum = 0;
        for (int node : path) sum += values[node - 1];
        return sum;
    }

    public static void main(String[] args) {
        long[] values = {3, 1, 4, 1, 5, 9};
        LinkCutTree lct = new LinkCutTree(values);

        // isolated vertices: every node is its own root, no paths connect them
        for (int v = 1; v <= 6; v++) check(lct.findroot(v) == v, "isolated root " + v);
        check(!lct.connected(1, 2), "isolated 1 2");
        check(!lct.connected(2, 3) && !lct.connected(4, 5), "isolated pairs");

        // link two singletons and read the path
        lct.link(1, 2);
        check(lct.connected(1, 2) && lct.connected(2, 1), "link pair");
        check(lct.pathSum(1, 2) == 4 && lct.pathSum(2, 1) == 4, "pair sum");   // 3 + 1
        check(lct.findroot(2) == lct.findroot(1), "pair root");
        check(lct.pathSum(1, 1) == 3, "single value");

        // extend to a chain 1 - 2 - 3 and query across it
        lct.link(2, 3);
        check(lct.pathSum(1, 3) == 8, "chain sum");                            // 3 + 1 + 4
        check(lct.pathSum(3, 1) == 8, "chain sum reversed");
        check(lct.pathSum(2, 2) == 1, "middle value");
        // pathSum calls makeroot internally; an explicit makeroot(3) makes 3 the root
        lct.makeroot(3);
        check(lct.findroot(1) == 3 && lct.findroot(3) == 3, "makeroot 3");

        // cutting an existing edge splits the tree; missing edges are left untouched
        lct.cut(2, 3);
        check(!lct.connected(1, 3), "cut 2 3");
        check(lct.connected(1, 2), "still linked 1 2");
        check(lct.findroot(3) == 3, "3 isolated after cut");
        lct.cut(1, 3);                    // no such edge: guarded no-op
        check(lct.connected(1, 2) && !lct.connected(1, 3), "cut guard");

        // build two trees, then join them through single edges
        lct.link(3, 4);
        lct.link(5, 6);
        check(lct.connected(3, 4) && lct.connected(5, 6), "two trees");
        check(!lct.connected(1, 4), "trees apart");
        lct.link(2, 3);                    // re-join {1,2} with {3,4}
        check(lct.connected(1, 4), "joined 1 4");
        check(lct.pathSum(1, 4) == 9, "sum 1 4");                              // 3+1+4+1
        lct.link(4, 5);
        check(lct.connected(1, 6), "joined 1 6");
        check(lct.pathSum(1, 6) == 23, "sum 1 6");                             // 3+1+4+1+5+9
        check(lct.pathSum(6, 1) == 23, "sum 6 1");
        check(lct.pathSum(3, 5) == 10, "sum 3 5");                             // 4+1+5

        // cutting the bridge splits the forest again
        lct.cut(4, 5);
        check(lct.connected(1, 4) && lct.connected(5, 6), "post bridge cut");
        check(!lct.connected(1, 5), "bridge gone");

        // LCG-driven script against the adjacency oracle (hundreds of operations)
        int n = 30;
        seed = 4190;
        long[] vals = new long[n];
        for (int i = 0; i < n; i++) vals[i] = 1 + nextRand() % 100;
        lct = new LinkCutTree(vals);
        Forest forest = new Forest(n);

        for (int step = 0; step < 350; step++) {
            int kind = nextRand() % 4;
            if (kind == 0) {
                int u = 1 + nextRand() % n;
                int v = 1 + nextRand() % n;
                if (u != v && forest.connected(u, v) && forest.hasEdge(u, v)) {
                    lct.cut(u, v);
                    forest.cut(u, v);
                    check(!lct.connected(u, v), "script cut " + step);
                }
            } else if (kind == 1) {
                int u = 1 + nextRand() % n;
                int v = 1 + nextRand() % n;
                if (u != v && !forest.connected(u, v)) {
                    lct.link(u, v);
                    forest.link(u, v);
                    check(lct.connected(u, v), "script link " + step);
                }
            } else if (kind == 2) {
                int u = 1 + nextRand() % n;
                int v = 1 + nextRand() % n;
                check(lct.connected(u, v) == forest.connected(u, v), "script conn " + step);
                if (forest.connected(u, v)) {
                    check(lct.pathSum(u, v) == pathSumOracle(forest, vals, u, v), "script sum " + step);
                    check(lct.pathSum(v, u) == lct.pathSum(u, v), "script sum sym " + step);
                }
            } else {
                // full pairwise connectivity audit
                for (int u = 1; u <= n; u++) {
                    for (int v = u + 1; v <= n; v++) {
                        check(lct.connected(u, v) == forest.connected(u, v), "audit " + step + " " + u + " " + v);
                    }
                }
            }
        }

        // final audit: every path sum across the whole forest matches the oracle
        int checks = 0;
        for (int u = 1; u <= n; u++) {
            for (int v = u; v <= n; v++) {
                check(lct.connected(u, v) == forest.connected(u, v), "final conn");
                if (forest.connected(u, v)) {
                    check(lct.pathSum(u, v) == pathSumOracle(forest, vals, u, v), "final sum " + u + " " + v);
                    checks++;
                }
            }
        }
        check(checks > n, "final coverage");

        // a star built by links from one hub answers all leaf-to-leaf paths through it
        seed = 777;
        long[] hubValues = new long[9];
        for (int i = 0; i < 9; i++) hubValues[i] = 1 + nextRand() % 9;
        lct = new LinkCutTree(hubValues);
        for (int leaf = 2; leaf <= 9; leaf++) lct.link(1, leaf);
        for (int leaf = 2; leaf <= 9; leaf++) {
            check(lct.pathSum(1, leaf) == hubValues[0] + hubValues[leaf - 1], "star hub " + leaf);
        }
        for (int a = 2; a <= 9; a++) {
            for (int b = a + 1; b <= 9; b++) {
                long want = hubValues[0] + hubValues[a - 1] + hubValues[b - 1];
                check(lct.pathSum(a, b) == want, "star leaf " + a + " " + b);
            }
        }
        // each path query re-roots through makeroot; after makeroot(9), 9 is the tree root
        lct.makeroot(9);
        check(lct.findroot(1) == 9 && lct.findroot(9) == 9, "star root");

        // cutting every leaf leaves a single node
        for (int leaf = 2; leaf <= 9; leaf++) lct.cut(1, leaf);
        for (int leaf = 2; leaf <= 9; leaf++) {
            check(!lct.connected(1, leaf), "leaf cut " + leaf);
            check(lct.findroot(leaf) == leaf, "leaf root " + leaf);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
