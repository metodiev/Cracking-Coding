/** Task 458 - Implement dynamic programming on trees with knapsack per node. (Tier: Advanced, CTCI Ch 4) */
import java.util.*;

public class Solution {
    static final int NEG_INF = -1_000_000_000;

    /** Best sum of <= k chosen vertices with the parent-before-child closure rule. */
    static int treeKnapsack(int n, int[] parent, int[] weight, int k) {
        if (k <= 0 || n == 0) return 0;
        List<List<Integer>> children = new ArrayList<>();
        for (int i = 0; i < n; i++) children.add(new ArrayList<>());
        for (int v = 1; v < n; v++) children.get(parent[v]).add(v);
        DpResult result = dfsFull(0, k, children, weight);
        int best = 0;
        for (int j = 1; j <= k; j++) best = Math.max(best, result.dp[j]);
        return best;
    }

    static class DpResult {
        final int[] dp;
        final int size;

        DpResult(int[] dp, int size) {
            this.dp = dp;
            this.size = size;
        }
    }

    static DpResult dfsFull(int u, int k, List<List<Integer>> children, int[] weight) {
        int subtreeSize = 1;
        int[] dp = new int[k + 1];
        Arrays.fill(dp, NEG_INF);
        dp[1] = weight[u];
        for (int child : children.get(u)) {
            DpResult childResult = dfsFull(child, k, children, weight);
            int[] cdp = childResult.dp;
            int childSize = childResult.size;
            int[] newDp = dp.clone();
            for (int i = 1; i <= Math.min(k, subtreeSize); i++) {
                if (dp[i] == NEG_INF) continue;
                for (int j = 1; j <= Math.min(childSize, k - i); j++) {
                    if (cdp[j] == NEG_INF) continue;
                    newDp[i + j] = Math.max(newDp[i + j], dp[i] + cdp[j]);
                }
            }
            dp = newDp;
            subtreeSize += childSize;
        }
        return new DpResult(dp, subtreeSize);
    }

    /** Exhaustive over all parent-closed subsets of size <= k (empty set allowed). */
    static int bruteTreeKnapsack(int n, int[] parent, int[] weight, int k) {
        int best = 0;
        for (int mask = 0; mask < (1 << n); mask++) {
            if (Integer.bitCount(mask) > k) continue;
            boolean ok = true;
            for (int v = 1; v < n; v++) {
                if (((mask >> v) & 1) == 1 && ((mask >> parent[v]) & 1) == 0) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            int total = 0;
            for (int v = 0; v < n; v++) if (((mask >> v) & 1) == 1) total += weight[v];
            best = Math.max(best, total);
        }
        return best;
    }

    static class TreeInstance {
        final int n;
        final int[] parent;
        final int[] weight;

        TreeInstance(int n, int[] parent, int[] weight) {
            this.n = n;
            this.parent = parent;
            this.weight = weight;
        }
    }

    static List<TreeInstance> lcgTrees(int count, int seed) {
        List<TreeInstance> trees = new ArrayList<>();
        for (int g = 0; g < count; g++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 6 + seed % 4;
            int[] parent = new int[n];
            int[] weight = new int[n];
            parent[0] = -1;
            seed = (seed * 75 + 74) % 65537;
            weight[0] = seed % 31 - 15;
            for (int v = 1; v < n; v++) {
                seed = (seed * 75 + 74) % 65537;
                parent[v] = seed % v;
                seed = (seed * 75 + 74) % 65537;
                weight[v] = seed % 31 - 15;
            }
            trees.add(new TreeInstance(n, parent, weight));
        }
        return trees;
    }

    public static void main(String[] args) {
        int[] p1 = {-1, 0, 0, 1};
        int[] w1 = {5, 3, 4, -10};
        check(treeKnapsack(4, p1, w1, 1) == 5, "tree1 K=1");
        check(treeKnapsack(4, p1, w1, 2) == 9, "tree1 K=2");
        check(treeKnapsack(4, p1, w1, 3) == 12, "tree1 K=3");
        check(treeKnapsack(4, p1, w1, 4) == 12, "tree1 K=4");
        check(treeKnapsack(4, p1, w1, 0) == 0, "tree1 K=0");
        for (int k = 0; k <= 4; k++) check(treeKnapsack(4, p1, w1, k) == bruteTreeKnapsack(4, p1, w1, k), "tree1 brute " + k);

        int[] p2 = {-1, 0, 0, 1, 1};
        int[] w2 = {10, 8, 5, 7, 6};
        check(treeKnapsack(5, p2, w2, 3) == 25, "tree2 K=3");
        for (int k = 0; k <= 5; k++) check(treeKnapsack(5, p2, w2, k) == bruteTreeKnapsack(5, p2, w2, k), "tree2 brute " + k);

        int[] p3 = {-1, 0};
        int[] w3 = {-2, -3};
        check(treeKnapsack(2, p3, w3, 2) == 0, "negative K=2");
        check(treeKnapsack(2, p3, w3, 1) == 0, "negative K=1");

        int[] p4 = {-1, 0, 1};
        int[] w4 = {1, -5, 10};
        check(treeKnapsack(3, p4, w4, 1) == 1, "chain K=1");
        check(treeKnapsack(3, p4, w4, 2) == 1, "chain K=2");
        check(treeKnapsack(3, p4, w4, 3) == 6, "chain K=3");
        for (int k = 0; k <= 3; k++) check(treeKnapsack(3, p4, w4, k) == bruteTreeKnapsack(3, p4, w4, k), "chain brute " + k);

        int checked = 0, comparisons = 0, expectedComparisons = 0;
        for (int seed : new int[]{4581, 4582, 4583}) {
            for (TreeInstance tree : lcgTrees(5, seed)) {
                int previous = 0;
                for (int k = 0; k <= tree.n; k++) {
                    int got = treeKnapsack(tree.n, tree.parent, tree.weight, k);
                    int want = bruteTreeKnapsack(tree.n, tree.parent, tree.weight, k);
                    check(got == want, "LCG tree K=" + k);
                    check(got >= previous, "monotone in K");
                    previous = got;
                    comparisons++;
                }
                expectedComparisons += tree.n + 1;
                checked++;
            }
        }
        check(checked == 15, "tree count");
        check(comparisons == expectedComparisons, "comparison count");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
