/** Task 459 - Implement virtual tree building from set of nodes (for queries). (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static class TreeData {
        final List<List<Integer>> children;
        final int[] tin, depth, dist;

        TreeData(List<List<Integer>> children, int[] tin, int[] depth, int[] dist) {
            this.children = children;
            this.tin = tin;
            this.depth = depth;
            this.dist = dist;
        }
    }

    static TreeData buildTreeData(int n, int[] parent, int[] edgeWeight) {
        List<List<Integer>> children = new ArrayList<>();
        for (int i = 0; i < n; i++) children.add(new ArrayList<>());
        for (int v = 1; v < n; v++) children.get(parent[v]).add(v);
        int[] tin = new int[n];
        int[] depth = new int[n];
        int[] dist = new int[n];
        int order = 0;
        ArrayDeque<Integer> stack = new ArrayDeque<>();
        stack.push(0);
        while (!stack.isEmpty()) {
            int u = stack.pop();
            tin[u] = order++;
            List<Integer> childList = children.get(u);
            for (int i = childList.size() - 1; i >= 0; i--) {
                int child = childList.get(i);
                depth[child] = depth[u] + 1;
                dist[child] = dist[u] + edgeWeight[child];
                stack.push(child);
            }
        }
        return new TreeData(children, tin, depth, dist);
    }

    static int lca(int[] parent, int[] depth, int u, int v) {
        while (depth[u] > depth[v]) u = parent[u];
        while (depth[v] > depth[u]) v = parent[v];
        while (u != v) {
            u = parent[u];
            v = parent[v];
        }
        return u;
    }

    static boolean isAncestor(int[] parent, int candidate, int node) {
        while (node != -1) {
            if (node == candidate) return true;
            node = parent[node];
        }
        return false;
    }

    static int naiveDistance(int[] parent, int[] depth, int[] dist, int u, int v) {
        return dist[u] + dist[v] - 2 * dist[lca(parent, depth, u, v)];
    }

    static class VtResult {
        final int[] nodes;
        final List<int[]> edges;

        VtResult(int[] nodes, List<int[]> edges) {
            this.nodes = nodes;
            this.edges = edges;
        }
    }

    /** Returns the virtual tree as sorted nodes plus edges {vtParent, vtChild, weight}. */
    static VtResult buildVirtualTree(int[] parent, int[] depth, int[] dist, int[] tin, int[] selected) {
        Integer[] order = new Integer[selected.length];
        for (int i = 0; i < selected.length; i++) order[i] = selected[i];
        Arrays.sort(order, Comparator.comparingInt(a -> tin[a]));
        List<Integer> stack = new ArrayList<>();
        List<int[]> edges = new ArrayList<>();
        if (order.length == 0) return new VtResult(new int[0], edges);
        stack.add(order[0]);
        for (int idx = 1; idx < order.length; idx++) {
            int v = order[idx];
            int l = lca(parent, depth, v, stack.get(stack.size() - 1));
            if (l == stack.get(stack.size() - 1)) {
                stack.add(v);
                continue;
            }
            while (stack.size() >= 2 && depth[stack.get(stack.size() - 2)] >= depth[l]) {
                int child = stack.remove(stack.size() - 1);
                int par = stack.get(stack.size() - 1);
                edges.add(new int[]{par, child, dist[child] - dist[par]});
            }
            if (stack.get(stack.size() - 1) != l) {
                int child = stack.remove(stack.size() - 1);
                edges.add(new int[]{l, child, dist[child] - dist[l]});
                stack.add(l);
            }
            stack.add(v);
        }
        while (stack.size() >= 2) {
            int child = stack.remove(stack.size() - 1);
            int par = stack.get(stack.size() - 1);
            edges.add(new int[]{par, child, dist[child] - dist[par]});
        }
        TreeSet<Integer> nodes = new TreeSet<>();
        for (int v : selected) nodes.add(v);
        for (int[] e : edges) {
            nodes.add(e[0]);
            nodes.add(e[1]);
        }
        int[] nodeArray = new int[nodes.size()];
        int i = 0;
        for (int v : nodes) nodeArray[i++] = v;
        return new VtResult(nodeArray, edges);
    }

    static int virtualTreePathWeight(int[] nodes, List<int[]> edges, int source, int target) {
        Map<Integer, List<int[]>> adj = new HashMap<>();
        for (int v : nodes) adj.put(v, new ArrayList<>());
        for (int[] e : edges) {
            adj.get(e[0]).add(new int[]{e[1], e[2]});
            adj.get(e[1]).add(new int[]{e[0], e[2]});
        }
        Set<Integer> seen = new HashSet<>();
        seen.add(source);
        ArrayDeque<int[]> stack = new ArrayDeque<>();
        stack.push(new int[]{source, 0});
        while (!stack.isEmpty()) {
            int[] cur = stack.pop();
            if (cur[0] == target) return cur[1];
            for (int[] nxt : adj.get(cur[0])) {
                if (!seen.contains(nxt[0])) {
                    seen.add(nxt[0]);
                    stack.push(new int[]{nxt[0], cur[1] + nxt[1]});
                }
            }
        }
        return -1;
    }

    static void checkVirtualTree(int n, int[] parent, int[] selected, TreeData data) {
        VtResult built = buildVirtualTree(parent, data.depth, data.dist, data.tin, selected);
        int[] nodes = built.nodes;
        List<int[]> edges = built.edges;
        Set<Integer> nodeSet = new HashSet<>();
        for (int v : nodes) nodeSet.add(v);
        for (int s : selected) check(nodeSet.contains(s), "selected present");
        for (int v : nodes) {
            boolean ok = false;
            for (int s : selected) ok |= isAncestor(parent, v, s);
            check(ok, "ancestor-or-selected");
        }
        check(edges.size() == nodes.length - 1, "tree edge count");
        check(nodes.length <= 2 * selected.length - 1, "node bound");
        for (int[] e : edges) {
            check(e[2] == data.dist[e[1]] - data.dist[e[0]] && e[2] > 0, "edge weight");
        }
        for (int i = 0; i < selected.length; i++) {
            for (int j = i + 1; j < selected.length; j++) {
                int got = virtualTreePathWeight(nodes, edges, selected[i], selected[j]);
                int want = naiveDistance(parent, data.depth, data.dist, selected[i], selected[j]);
                check(got == want, "pair distance " + selected[i] + "," + selected[j]);
            }
        }
    }

    static class TreeInstance {
        final int n;
        final int[] parent;
        final int[] edgeWeight;

        TreeInstance(int n, int[] parent, int[] edgeWeight) {
            this.n = n;
            this.parent = parent;
            this.edgeWeight = edgeWeight;
        }
    }

    static List<TreeInstance> lcgTrees(int count, int seed) {
        List<TreeInstance> trees = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 8 + seed % 5;
            int[] parent = new int[n];
            int[] edgeWeight = new int[n];
            parent[0] = -1;
            for (int v = 1; v < n; v++) {
                seed = (seed * 75 + 74) % 65537;
                parent[v] = seed % v;
                seed = (seed * 75 + 74) % 65537;
                edgeWeight[v] = 1 + seed % 10;
            }
            trees.add(new TreeInstance(n, parent, edgeWeight));
        }
        return trees;
    }

    static int[] lcgSelection(int n, int seed, int size) {
        seed = (seed * 75 + 74) % 65537;
        List<Integer> pool = new ArrayList<>();
        for (int i = 0; i < n; i++) pool.add(i);
        int count = Math.min(size, n);
        int[] selected = new int[count];
        for (int k = 0; k < count; k++) {
            seed = (seed * 75 + 74) % 65537;
            int idx = seed % pool.size();
            selected[k] = pool.get(idx);
            pool.set(idx, pool.get(pool.size() - 1));
            pool.remove(pool.size() - 1);
            seed = (seed * 75 + 74) % 65537;
        }
        return selected;
    }

    public static void main(String[] args) {
        int[] chainParent = {-1, 0, 1, 2, 3};
        int[] chainWeight = {0, 1, 1, 1, 1};
        TreeData chain = buildTreeData(5, chainParent, chainWeight);
        checkVirtualTree(5, chainParent, new int[]{2, 4}, chain);
        VtResult built = buildVirtualTree(chainParent, chain.depth, chain.dist, chain.tin, new int[]{2, 4});
        check(Arrays.equals(built.nodes, new int[]{2, 4}), "chain nodes");
        check(built.edges.size() == 1 && built.edges.get(0)[2] == 2, "chain edge weight 2");
        checkVirtualTree(5, chainParent, new int[]{1, 3}, chain);
        checkVirtualTree(5, chainParent, new int[]{0, 3}, chain);
        checkVirtualTree(5, chainParent, new int[]{1}, chain);
        built = buildVirtualTree(chainParent, chain.depth, chain.dist, chain.tin, new int[]{1});
        check(built.edges.isEmpty(), "single node no edges");

        int[] starParent = {-1, 0, 0, 0};
        int[] starWeight = {0, 1, 1, 1};
        TreeData star = buildTreeData(4, starParent, starWeight);
        checkVirtualTree(4, starParent, new int[]{1, 2}, star);
        built = buildVirtualTree(starParent, star.depth, star.dist, star.tin, new int[]{1, 2});
        check(Arrays.equals(built.nodes, new int[]{0, 1, 2}), "star adds LCA root");
        checkVirtualTree(4, starParent, new int[]{1, 2, 3}, star);

        int[] balParent = {-1, 0, 0, 1, 1, 2, 2};
        int[] balWeight = {0, 3, 4, 5, 6, 7, 8};
        TreeData balanced = buildTreeData(7, balParent, balWeight);
        checkVirtualTree(7, balParent, new int[]{3, 4, 5, 6}, balanced);

        int checked = 0, trees = 0;
        for (int seed : new int[]{4591, 4592, 4593, 4594}) {
            for (TreeInstance tree : lcgTrees(4, seed)) {
                TreeData data = buildTreeData(tree.n, tree.parent, tree.edgeWeight);
                int seed2 = seed + tree.n;
                for (int extra = 0; extra < 2; extra++) {
                    int size = 2 + seed2 % 4;
                    int[] selected = lcgSelection(tree.n, seed2 + extra * 3, size);
                    checkVirtualTree(tree.n, tree.parent, selected, data);
                    checked++;
                }
                trees++;
            }
        }
        check(trees == 16, "tree count");
        check(checked == 32, "selection count");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
