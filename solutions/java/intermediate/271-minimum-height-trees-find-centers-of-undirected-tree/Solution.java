/** Task 271 - Minimum height trees (tree centers). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static List<Integer> minHeightTrees(int n, int[][] edges) {
        if (n == 1) {
            List<Integer> single = new ArrayList<>();
            single.add(0);
            return single;
        }
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        int[] degree = new int[n];
        for (int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
            degree[edge[0]]++;
            degree[edge[1]]++;
        }

        Deque<Integer> leaves = new ArrayDeque<>();
        for (int i = 0; i < n; i++) if (degree[i] == 1) leaves.add(i);
        int remaining = n;
        while (remaining > 2) {
            int size = leaves.size();
            remaining -= size;
            for (int i = 0; i < size; i++) {
                int leaf = leaves.poll();
                for (int neighbour : adj.get(leaf)) {
                    if (--degree[neighbour] == 1) leaves.add(neighbour);
                }
            }
        }
        List<Integer> out = new ArrayList<>(leaves);
        Collections.sort(out);
        return out;
    }

    /** Independent oracle: BFS height from every root, keep the minimizers. */
    static int[] bruteForceCenters(int n, int[][] edges) {
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (int[] edge : edges) {
            adj.get(edge[0]).add(edge[1]);
            adj.get(edge[1]).add(edge[0]);
        }
        int[] heights = new int[n];
        for (int root = 0; root < n; root++) {
            boolean[] seen = new boolean[n];
            seen[root] = true;
            List<Integer> frontier = new ArrayList<>();
            frontier.add(root);
            int levels = -1;
            while (!frontier.isEmpty()) {
                levels++;
                List<Integer> next = new ArrayList<>();
                for (int u : frontier) {
                    for (int v : adj.get(u)) {
                        if (!seen[v]) { seen[v] = true; next.add(v); }
                    }
                }
                frontier = next;
            }
            heights[root] = levels;
        }
        int best = Integer.MAX_VALUE;
        for (int h : heights) best = Math.min(best, h);
        List<Integer> centers = new ArrayList<>();
        for (int i = 0; i < n; i++) if (heights[i] == best) centers.add(i);
        int[] out = new int[centers.size() + 1];
        for (int i = 0; i < centers.size(); i++) out[i] = centers.get(i);
        out[centers.size()] = best;                       // last slot carries the best height
        return out;
    }

    static String join(List<Integer> values) {
        StringBuilder sb = new StringBuilder();
        for (int v : values) {
            if (sb.length() > 0) sb.append(',');
            sb.append(v);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(join(minHeightTrees(6, new int[][]{{0,1},{1,2},{2,3},{3,4},{4,5}})).equals("2,3"), "path of 6");
        check(join(minHeightTrees(4, new int[][]{{0,1},{0,2},{0,3}})).equals("0"), "star");
        check(join(minHeightTrees(1, new int[][]{})).equals("0"), "single node");
        check(join(minHeightTrees(2, new int[][]{{0,1}})).equals("0,1"), "two nodes");

        int[][] sample = {{0,1},{1,2},{1,3},{3,4},{4,5},{5,6}};
        int[] oracle = bruteForceCenters(7, sample);
        List<Integer> centers = new ArrayList<>();
        for (int i = 0; i < oracle.length - 1; i++) centers.add(oracle[i]);
        check(minHeightTrees(7, sample).equals(centers), "matches brute force centres");
        check(oracle[oracle.length - 1] == 3, "minimum height is 3");
        check(minHeightTrees(7, sample).size() <= 2, "at most two centres");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
