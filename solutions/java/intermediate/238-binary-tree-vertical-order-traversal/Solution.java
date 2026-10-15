/** Task 238 - Binary tree vertical order traversal. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int v) { val = v; }
}

public class Solution {
    /** Build a binary tree from level-order values (null marks a missing child). */
    static TreeNode buildTree(Integer... values) {
        if (values.length == 0 || values[0] == null) return null;
        TreeNode root = new TreeNode(values[0]);
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        int i = 1;
        while (!q.isEmpty() && i < values.length) {
            TreeNode node = q.poll();
            if (i < values.length && values[i] != null) {
                node.left = new TreeNode(values[i]);
                q.add(node.left);
            }
            i++;
            if (i < values.length && values[i] != null) {
                node.right = new TreeNode(values[i]);
                q.add(node.right);
            }
            i++;
        }
        return root;
    }

    /**
     * Columns from the root (root = 0, left = -1, right = +1). BFS with columns reads
     * each column top-to-bottom and visits same-row nodes left-to-right (tie rule);
     * a sorted map emits the columns in increasing order. O(n log n) time, O(n) space.
     */
    static List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) return result;
        Map<Integer, List<Integer>> columns = new TreeMap<>();
        Deque<TreeNode> nodes = new ArrayDeque<>();
        Deque<Integer> cols = new ArrayDeque<>();
        nodes.add(root);
        cols.add(0);
        while (!nodes.isEmpty()) {
            TreeNode node = nodes.poll();
            int col = cols.poll();
            columns.computeIfAbsent(col, k -> new ArrayList<>()).add(node.val);
            if (node.left != null) { nodes.add(node.left); cols.add(col - 1); }
            if (node.right != null) { nodes.add(node.right); cols.add(col + 1); }
        }
        return new ArrayList<>(columns.values());
    }

    /** Independent helper: sort (column, row, preorder index) then group by column. */
    static List<List<Integer>> verticalOrderSorted(TreeNode root) {
        List<int[]> items = new ArrayList<>();     // {column, row, preorder index, value}
        dfs(root, 0, 0, items);
        items.sort((a, b) -> {
            if (a[0] != b[0]) return Integer.compare(a[0], b[0]);
            if (a[1] != b[1]) return Integer.compare(a[1], b[1]);
            return Integer.compare(a[2], b[2]);
        });
        List<List<Integer>> result = new ArrayList<>();
        int lastCol = Integer.MIN_VALUE;
        for (int[] item : items) {
            if (result.isEmpty() || item[0] != lastCol) {
                result.add(new ArrayList<>());
                lastCol = item[0];
            }
            result.get(result.size() - 1).add(item[3]);
        }
        return result;
    }

    private static void dfs(TreeNode node, int col, int row, List<int[]> items) {
        if (node == null) return;
        items.add(new int[]{col, row, items.size(), node.val});
        dfs(node.left, col - 1, row + 1, items);
        dfs(node.right, col + 1, row + 1, items);
    }

    private static TreeNode prune(TreeNode node, Random rng) {
        if (node == null || rng.nextDouble() < 0.25) return null;
        node.left = prune(node.left, rng);
        node.right = prune(node.right, rng);
        return node;
    }

    public static void main(String[] args) {
        TreeNode classic = buildTree(3, 9, 20, null, null, 15, 7);
        check(verticalOrder(classic).equals(Arrays.asList(
                Arrays.asList(9), Arrays.asList(3, 15), Arrays.asList(20), Arrays.asList(7))),
                "classic");
        check(verticalOrderSorted(classic).equals(verticalOrder(classic)), "classic sorted agrees");

        TreeNode perfect = buildTree(1, 2, 3, 4, 5, 6, 7);
        check(verticalOrder(perfect).equals(Arrays.asList(
                Arrays.asList(4), Arrays.asList(2), Arrays.asList(1, 5, 6),
                Arrays.asList(3), Arrays.asList(7))),
                "perfect tree with left-to-right tie");
        check(verticalOrderSorted(perfect).equals(verticalOrder(perfect)), "perfect sorted agrees");

        TreeNode deeper = buildTree(3, 9, 20, null, null, 15, 7, 8, 11);
        check(verticalOrder(deeper).equals(Arrays.asList(
                Arrays.asList(9, 8), Arrays.asList(3, 15),
                Arrays.asList(20, 11), Arrays.asList(7))),
                "stacked columns");
        check(verticalOrderSorted(deeper).equals(verticalOrder(deeper)), "deeper sorted agrees");

        TreeNode chain = buildTree(1, 2, null, 3, null, 4);
        check(verticalOrder(chain).equals(Arrays.asList(
                Arrays.asList(4), Arrays.asList(3), Arrays.asList(2), Arrays.asList(1))),
                "left chain");
        check(verticalOrder(buildTree(5)).equals(Arrays.asList(Arrays.asList(5))), "single node");
        check(verticalOrder(null).isEmpty(), "empty");

        Random rng = new Random(238);
        for (int test = 0; test < 200; test++) {    // BFS order == sorted-DFS order
            int n = rng.nextInt(15);
            TreeNode[] nodes = new TreeNode[n];
            for (int i = 0; i < n; i++) nodes[i] = new TreeNode(i + 1);
            for (int i = 0; i < n; i++) {
                if (2 * i + 1 < n) nodes[i].left = nodes[2 * i + 1];
                if (2 * i + 2 < n) nodes[i].right = nodes[2 * i + 2];
            }
            TreeNode root = n == 0 ? null : prune(nodes[0], rng);
            check(verticalOrder(root).equals(verticalOrderSorted(root)), "random agreement");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
