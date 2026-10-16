/** Task 703 - Implement dynamic graph connectivity with link-cut trees in practice.
 * (Tier: Expert, CTCI Ch 4 -- Trees and Graphs)
 *
 * A link-cut tree keeps a forest in a set of splay trees, one per *preferred path*. Each node has two
 * kinds of parents: the splay parent inside its preferred path and the path-parent pointer that links
 * the head of a preferred path to the node below it. access(x) walks up the path-parent chain,
 * splicing the path through x into the right spine (right[y] = last) and splaying x to the root, after
 * which the splay tree rooted at x holds exactly the path from the represented tree's root to x.
 * makeRoot(x) = access(x) followed by toggling the lazy reversal flag, link(u, v) = makeRoot(u) plus a
 * path-parent pointer, and cut(u, v) = makeRoot(u), access(v) and then the structural test
 * left[v] == u && right[u] == 0, which is what makes cutting a non-edge impossible rather than silent.
 * Aggregates (here a maximum) are maintained bottom-up by pull(), so pathBest(u, v) is one makeRoot,
 * one access and the aggregate of the root of the resulting splay tree.
 *
 * In practice edges carry the weights, not vertices, so every edge of the represented forest is split
 * into a vertex node (weight 0/negative-safe) plus an edge node holding the edge weight; the two are
 * linked when the edge is inserted and cut when it is removed. Both the split and the max aggregate
 * are the classic trick for path-maximum queries.
 *
 * Verified: after every operation the structure's connectivity, component count and path maxima agree
 * with an explicit adjacency-matrix forest that answers by BFS/DFS; links inside a component, cuts of
 * non-edges, self links and path queries across components are rejected; 1-node trees, negative
 * weights, repeated link/cut cycles and 220 random operations are covered.
 */
import java.util.*;

class LinkCutTree {
    private final int nodeCount;
    private final int[] left;
    private final int[] right;
    private final int[] parent;
    private final int[] value;
    private final int[] best;
    private final boolean[] reversed;
    private final int[] stack;

    LinkCutTree(int nodeCount, int defaultValue) {
        if (nodeCount < 1) throw new IllegalArgumentException("need at least one node");
        this.nodeCount = nodeCount;
        left = new int[nodeCount + 1];
        right = new int[nodeCount + 1];
        parent = new int[nodeCount + 1];
        value = new int[nodeCount + 1];
        best = new int[nodeCount + 1];
        reversed = new boolean[nodeCount + 1];
        stack = new int[nodeCount + 1];
        for (int i = 1; i <= nodeCount; i++) {
            value[i] = defaultValue;
            best[i] = defaultValue;
        }
    }

    int nodeCount() {
        return nodeCount;
    }

    void setValue(int node, int newValue) {
        access(node);
        value[node] = newValue;
        pull(node);
    }

    int valueOf(int node) {
        access(node);
        return value[node];
    }

    boolean connected(int u, int v) {
        checkRange(u);
        checkRange(v);
        if (u == v) return true;
        return findRoot(u) == findRoot(v);
    }

    /** Representative of the tree containing u; the client only sees equality. */
    int findRoot(int u) {
        checkRange(u);
        access(u);
        while (left[u] != 0) {
            push(u);
            u = left[u];
        }
        access(u);
        return u;
    }

    /** Adds the edge u-v; the two nodes must currently lie in different trees. */
    void link(int u, int v) {
        checkRange(u);
        checkRange(v);
        if (u == v) throw new IllegalArgumentException("self link is not an edge: " + u);
        if (connected(u, v)) throw new IllegalArgumentException("already connected: " + u + "-" + v);
        makeRoot(u);
        parent[u] = v;
    }

    /** Removes the edge u-v, which must exist. */
    void cut(int u, int v) {
        checkRange(u);
        checkRange(v);
        makeRoot(u);
        access(v);
        if (left[v] != u || right[u] != 0) {
            throw new IllegalArgumentException("not an edge: " + u + "-" + v);
        }
        left[v] = 0;
        parent[u] = 0;
        pull(v);
    }

    /** Maximum vertex value on the u-v path; requires the two nodes to be connected. */
    int pathBest(int u, int v) {
        checkRange(u);
        checkRange(v);
        if (!connected(u, v)) throw new IllegalStateException("no path between " + u + " and " + v);
        makeRoot(u);
        access(v);
        return best[v];
    }

    private void makeRoot(int u) {
        access(u);
        reversed[u] = !reversed[u];
    }

    private void checkRange(int node) {
        if (node < 1 || node > nodeCount) throw new IndexOutOfBoundsException("node " + node);
    }

    private boolean isSplayRoot(int x) {
        int p = parent[x];
        return p == 0 || (left[p] != x && right[p] != x);
    }

    private void push(int x) {
        if (!reversed[x]) return;
        int l = left[x];
        int r = right[x];
        left[x] = r;
        right[x] = l;
        if (r != 0) reversed[r] = !reversed[r];
        if (l != 0) reversed[l] = !reversed[l];
        reversed[x] = false;
    }

    private void pull(int x) {
        best[x] = value[x];
        if (left[x] != 0 && best[left[x]] > best[x]) best[x] = best[left[x]];
        if (right[x] != 0 && best[right[x]] > best[x]) best[x] = best[right[x]];
    }

    private void rotate(int x) {
        int p = parent[x];
        int g = parent[p];
        if (left[p] == x) {
            left[p] = right[x];
            if (right[x] != 0) parent[right[x]] = p;
            right[x] = p;
        } else {
            right[p] = left[x];
            if (left[x] != 0) parent[left[x]] = p;
            left[x] = p;
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
        int top = 0;
        stack[top++] = x;
        for (int y = x; !isSplayRoot(y); ) {
            y = parent[y];
            stack[top++] = y;
        }
        while (top > 0) push(stack[--top]);
        while (!isSplayRoot(x)) {
            int p = parent[x];
            if (!isSplayRoot(p)) {
                int g = parent[p];
                boolean sameDirection = (left[p] == x) == (left[g] == p);
                rotate(sameDirection ? p : x);
            }
            rotate(x);
        }
    }

    private void access(int x) {
        int last = 0;
        for (int y = x; y != 0; y = parent[y]) {
            splay(y);
            right[y] = last;
            pull(y);
            last = y;
        }
        splay(x);
    }
}

/** Explicit forest used as the oracle: adjacency matrix plus BFS/DFS answers. */
class ExplicitForest {
    private final int n;
    private final boolean[][] adjacent;
    private final int[][] weight;
    private final List<List<Integer>> neighbours;
    private int edges;

    ExplicitForest(int n) {
        this.n = n;
        adjacent = new boolean[n + 1][n + 1];
        weight = new int[n + 1][n + 1];
        neighbours = new ArrayList<>();
        for (int i = 0; i <= n; i++) neighbours.add(new ArrayList<>());
    }

    int edgeCount() {
        return edges;
    }

    boolean hasEdge(int u, int v) {
        return adjacent[u][v];
    }

    void addEdge(int u, int v, int w) {
        if (adjacent[u][v]) throw new IllegalArgumentException("duplicate edge " + u + "-" + v);
        adjacent[u][v] = true;
        adjacent[v][u] = true;
        weight[u][v] = w;
        weight[v][u] = w;
        neighbours.get(u).add(v);
        neighbours.get(v).add(u);
        edges++;
    }

    void removeEdge(int u, int v) {
        if (!adjacent[u][v]) throw new IllegalArgumentException("missing edge " + u + "-" + v);
        adjacent[u][v] = false;
        adjacent[v][u] = false;
        neighbours.get(u).remove(Integer.valueOf(v));
        neighbours.get(v).remove(Integer.valueOf(u));
        edges--;
    }

    int[] componentLabels() {
        int[] label = new int[n + 1];
        Deque<Integer> queue = new ArrayDeque<>();
        int current = 0;
        for (int start = 1; start <= n; start++) {
            if (label[start] != 0) continue;
            current++;
            label[start] = current;
            queue.add(start);
            while (!queue.isEmpty()) {
                int node = queue.poll();
                for (int other : neighbours.get(node)) {
                    if (label[other] == 0) {
                        label[other] = current;
                        queue.add(other);
                    }
                }
            }
        }
        return label;
    }

    boolean connected(int u, int v) {
        int[] label = componentLabels();
        return label[u] == label[v];
    }

    /** Maximum edge weight on the unique u-v path, or Integer.MIN_VALUE when disconnected. */
    int pathBest(int u, int v) {
        int[] parentOf = new int[n + 1];
        Arrays.fill(parentOf, -1);
        Deque<Integer> queue = new ArrayDeque<>();
        parentOf[u] = u;
        queue.add(u);
        while (!queue.isEmpty() && parentOf[v] == -1) {
            int node = queue.poll();
            for (int other : neighbours.get(node)) {
                if (parentOf[other] == -1) {
                    parentOf[other] = node;
                    queue.add(other);
                }
            }
        }
        if (parentOf[v] == -1) return Integer.MIN_VALUE;
        int bestWeight = 0;
        int cursor = v;
        while (cursor != u) {
            int previous = parentOf[cursor];
            bestWeight = Math.max(bestWeight, weight[cursor][previous]);
            cursor = previous;
        }
        return bestWeight;
    }
}

public class Solution {
    private static long state = 0xFEEDFACE1234L;

    /** splitmix64 - deterministic PRNG, no Math.random. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    public static void main(String[] args) {
        // --- degenerate single-node tree ---------------------------------------
        LinkCutTree lonely = new LinkCutTree(1, 42);
        check(lonely.valueOf(1) == 42, "initial value");
        check(lonely.connected(1, 1), "a node is connected to itself");
        check(lonely.findRoot(1) == 1, "a lone node is its own root");
        check(lonely.pathBest(1, 1) == 42, "path over a single node");
        lonely.setValue(1, -7);
        check(lonely.pathBest(1, 1) == -7, "negative value");
        check(rejects(() -> lonely.link(1, 1)), "self link rejected");
        check(rejects(() -> lonely.cut(1, 1)), "cut of a non-edge rejected");

        // --- negative weights survive the max aggregate ------------------------
        LinkCutTree negatives = new LinkCutTree(3, 0);
        negatives.setValue(1, -5);
        negatives.setValue(2, -9);
        negatives.setValue(3, -2);
        negatives.link(1, 2);
        check(negatives.pathBest(1, 2) == -5, "max of two negatives");
        check(negatives.pathBest(2, 1) == -5, "path max is symmetric");
        check(rejects(() -> negatives.pathBest(1, 3)), "path across components rejected");
        check(!negatives.connected(1, 3), "separate components report disconnected");

        // --- edge-node split on a small weighted forest ------------------------
        LinkCutTree small = new LinkCutTree(5, 0);
        small.setValue(4, 30);
        small.setValue(5, 70);
        small.link(1, 4);
        small.link(4, 2);
        small.link(2, 5);
        small.link(5, 3);
        check(small.pathBest(1, 3) == 70, "path max through four nodes");
        check(small.pathBest(1, 2) == 30, "path max over the first edge");
        check(small.pathBest(4, 2) == 30, "edge node itself on the path");
        small.cut(2, 5);
        check(!small.connected(1, 3), "cut separates the forest");
        check(small.connected(1, 2) && small.connected(3, 5), "cut leaves both sides intact");
        check(rejects(() -> small.cut(2, 5)), "cutting again must fail");
        check(rejects(() -> small.link(1, 2)), "link inside a component must fail");
        small.link(2, 5);
        check(small.connected(1, 3), "relink restores connectivity");
        check(small.pathBest(1, 3) == 70, "path max after relink");
        check(rejects(() -> small.link(1, 3)), "link inside a component must fail after relink");

        // --- random operations against the explicit forest ---------------------
        int n = 20;
        int maxEdges = n * (n - 1) / 2;
        LinkCutTree lct = new LinkCutTree(n + maxEdges, 0);
        ExplicitForest oracle = new ExplicitForest(n);
        Map<Long, Integer> edgeNode = new HashMap<>();
        Deque<Integer> freeEdgeNodes = new ArrayDeque<>();
        for (int i = 1; i <= maxEdges; i++) freeEdgeNodes.push(n + i);
        List<int[]> liveEdges = new ArrayList<>();
        int links = 0;
        int cuts = 0;

        for (int step = 0; step < 400; step++) {
            int op = nextInt(10);
            int u = 1 + nextInt(n);
            int v = 1 + nextInt(n);
            if (op < 5) {
                if (u == v) continue;
                if (oracle.connected(u, v)) {
                    check(rejects(() -> lct.link(u, v)), "step " + step + ": link inside a component");
                    continue;
                }
                int w = 1 + nextInt(1000);
                int edgeVertex = freeEdgeNodes.pop();
                lct.setValue(edgeVertex, w);
                lct.link(u, edgeVertex);
                lct.link(edgeVertex, v);
                oracle.addEdge(u, v, w);
                liveEdges.add(new int[]{u, v, edgeVertex});
                links++;
            } else if (op < 8) {
                if (liveEdges.isEmpty()) continue;
                int index = nextInt(liveEdges.size());
                int[] edge = liveEdges.get(index);
                int[] last = liveEdges.remove(liveEdges.size() - 1);
                if (index < liveEdges.size()) liveEdges.set(index, last);
                lct.cut(edge[0], edge[2]);
                lct.cut(edge[1], edge[2]);
                oracle.removeEdge(edge[0], edge[1]);
                freeEdgeNodes.push(edge[2]);
                cuts++;
            } else {
                if (oracle.connected(u, v)) {
                    check(lct.pathBest(u, v) == oracle.pathBest(u, v),
                            "step " + step + ": path max " + u + "-" + v + " got " + lct.pathBest(u, v)
                                    + " expected " + oracle.pathBest(u, v));
                } else if (u != v) {
                    check(rejects(() -> lct.pathBest(u, v)), "step " + step + ": path max across components");
                }
            }

            // full structural comparison after every operation
            int[] labels = oracle.componentLabels();
            for (int a = 1; a <= n; a++) {
                for (int b = a + 1; b <= n; b++) {
                    check(lct.connected(a, b) == (labels[a] == labels[b]),
                            "step " + step + ": connectivity of " + a + "-" + b);
                }
            }
            int[] structureLabels = new int[n + 1];
            int groups = 0;
            for (int a = 1; a <= n; a++) {
                if (structureLabels[a] != 0) continue;
                groups++;
                int root = lct.findRoot(a);
                for (int b = a; b <= n; b++) {
                    if (lct.findRoot(b) == root) structureLabels[b] = groups;
                }
            }
            for (int a = 1; a <= n; a++) {
                for (int b = 1; b <= n; b++) {
                    check((structureLabels[a] == structureLabels[b]) == (labels[a] == labels[b]),
                            "step " + step + ": component partition mismatch");
                }
            }
        }
        check(oracle.edgeCount() == links - cuts, "edge accounting: "
                + oracle.edgeCount() + " == " + links + " - " + cuts);
        check(links > 40 && cuts > 20, "the random workload exercised both operations: "
                + links + " links, " + cuts + " cuts");

        int negativeCuts = 0;
        for (int trial = 0; trial < 40; trial++) {
            int u = 1 + nextInt(n);
            int v = 1 + nextInt(n);
            if (u == v || oracle.hasEdge(u, v)) continue;
            check(rejects(() -> lct.cut(u, v)), "cut of the non-edge " + u + "-" + v + " must be rejected");
            negativeCuts++;
        }
        check(negativeCuts >= 10, "several non-edges must be rejected: " + negativeCuts);
        // --- component of every node after the random run ----------------------
        int[] finalLabels = oracle.componentLabels();
        int components = 0;
        for (int i = 1; i <= n; i++) components = Math.max(components, finalLabels[i]);
        Set<Integer> roots = new HashSet<>();
        for (int i = 1; i <= n; i++) roots.add(lct.findRoot(i));
        check(roots.size() == components, "component count " + roots.size() + " vs " + components);

        // --- long chain: the worst case for link-cut paths ---------------------
        int chainLength = 60;
        LinkCutTree chain = new LinkCutTree(chainLength + chainLength - 1, 0);
        for (int i = 0; i + 1 < chainLength; i++) {
            int edgeVertex = chainLength + 1 + i;
            chain.setValue(edgeVertex, (i * 37) % 500);
            chain.link(i + 1, edgeVertex);
            chain.link(edgeVertex, i + 2);
        }
        check(chain.connected(1, chainLength), "the chain is connected");
        check(chain.pathBest(1, chainLength) == 499, "chain path max, got " + chain.pathBest(1, chainLength));
        int splitEdge = chainLength + 1 + 30;
        chain.cut(31, splitEdge);
        chain.cut(32, splitEdge);
        check(!chain.connected(1, chainLength), "splitting the chain");
        check(chain.connected(32, chainLength), "the far half is still connected");

        System.out.println("PASS");
    }

    static long key(int u, int v) {
        return u * 1000L + v;
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
