/** Task 236 - Count nodes at distance K from target in binary tree. (Tier: Intermediate, CTCI Ch 4) */
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
     * Distance counts edges, so k = 0 is the target. A BFS records every node's parent;
     * a second BFS radiates outward from the target k steps and counts the frontier.
     * O(n) time. A missing target, empty tree or k beyond the diameter return 0; for
     * repeated values the first node in level order is the target.
     */
    static int countNodesAtDistance(TreeNode root, int targetVal, int k) {
        if (root == null || k < 0) return 0;
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        parent.put(root, null);
        TreeNode target = null;
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (target == null && node.val == targetVal) target = node;
            if (node.left != null) { parent.put(node.left, node); q.add(node.left); }
            if (node.right != null) { parent.put(node.right, node); q.add(node.right); }
        }
        if (target == null) return 0;
        Set<TreeNode> seen = new HashSet<>();
        seen.add(target);
        List<TreeNode> frontier = new ArrayList<>();
        frontier.add(target);                      // nodes exactly `step` edges away
        for (int step = 0; step < k; step++) {
            List<TreeNode> next = new ArrayList<>();
            for (TreeNode node : frontier) {
                for (TreeNode nxt : new TreeNode[]{node.left, node.right, parent.get(node)}) {
                    if (nxt != null && seen.add(nxt)) next.add(nxt);
                }
            }
            frontier = next;
            if (frontier.isEmpty()) return 0;
        }
        return frontier.size();
    }

    /** O(n * h) reference: distance = root paths without their common prefix. */
    static int bruteCount(TreeNode root, int targetVal, int k) {
        if (root == null || k < 0) return 0;
        List<TreeNode> targetPath = findPath(root, targetVal);
        if (targetPath == null) return 0;
        return bruteDfs(root, new ArrayList<>(), targetPath, k);
    }

    private static int bruteDfs(TreeNode node, List<TreeNode> path, List<TreeNode> targetPath, int k) {
        path.add(node);
        int common = 0;
        while (common < path.size() && common < targetPath.size()
                && path.get(common) == targetPath.get(common)) common++;
        int count = ((path.size() - common) + (targetPath.size() - common) == k) ? 1 : 0;
        if (node.left != null) count += bruteDfs(node.left, path, targetPath, k);
        if (node.right != null) count += bruteDfs(node.right, path, targetPath, k);
        path.remove(path.size() - 1);
        return count;
    }

    /** Root-to-target path via BFS parent pointers (same target rule as the main one). */
    static List<TreeNode> findPath(TreeNode root, int targetVal) {
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        parent.put(root, null);
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        TreeNode target = null;
        while (!q.isEmpty() && target == null) {
            TreeNode node = q.poll();
            if (node.val == targetVal) { target = node; break; }
            if (node.left != null) { parent.put(node.left, node); q.add(node.left); }
            if (node.right != null) { parent.put(node.right, node); q.add(node.right); }
        }
        if (target == null) return null;
        LinkedList<TreeNode> path = new LinkedList<>();
        for (TreeNode n = target; n != null; n = parent.get(n)) path.addFirst(n);
        return path;
    }

    private static TreeNode prune(TreeNode node, Random rng) {
        if (node == null || rng.nextDouble() < 0.25) return null;
        node.left = prune(node.left, rng);
        node.right = prune(node.right, rng);
        return node;
    }

    public static void main(String[] args) {
        TreeNode classic = buildTree(3, 5, 1, 6, 2, 0, 8, null, null, 7, 4);
        check(countNodesAtDistance(classic, 5, 0) == 1, "k=0 is the target");
        check(countNodesAtDistance(classic, 5, 1) == 3, "{3, 6, 2}");
        check(countNodesAtDistance(classic, 5, 2) == 3, "{1, 7, 4}");
        check(countNodesAtDistance(classic, 5, 3) == 2, "{0, 8}");
        check(countNodesAtDistance(classic, 5, 4) == 0, "k past distance from target");
        check(countNodesAtDistance(classic, 3, 0) == 1, "root k=0");
        check(countNodesAtDistance(classic, 3, 2) == 4, "root level 2");
        check(countNodesAtDistance(classic, 3, 3) == 2, "root reaches deepest nodes");
        check(countNodesAtDistance(classic, 99, 1) == 0, "missing target");

        TreeNode skewed = buildTree(1, 2, null, 3, null, 4);
        check(countNodesAtDistance(skewed, 1, 3) == 1, "root to deepest in chain");
        check(countNodesAtDistance(skewed, 1, 4) == 0, "chain k past diameter");
        check(countNodesAtDistance(buildTree(7), 7, 0) == 1, "single node k=0");
        check(countNodesAtDistance(buildTree(7), 7, 1) == 0, "single node k=1");
        check(countNodesAtDistance(null, 1, 0) == 0, "empty tree");
        check(countNodesAtDistance(buildTree(1, 2, 1), 1, 1) == 2, "duplicate target, level order first");
        check(countNodesAtDistance(buildTree(1, 2, 1), 1, 2) == 0, "duplicate target explored");

        Random rng = new Random(236);
        for (int test = 0; test < 200; test++) {   // cross-check vs path-based brute force
            int n = rng.nextInt(13);
            TreeNode[] nodes = new TreeNode[n];
            for (int i = 0; i < n; i++) nodes[i] = new TreeNode(i + 1);
            for (int i = 0; i < n; i++) {
                if (2 * i + 1 < n) nodes[i].left = nodes[2 * i + 1];
                if (2 * i + 2 < n) nodes[i].right = nodes[2 * i + 2];
            }
            TreeNode root = n == 0 ? null : prune(nodes[0], rng);
            List<TreeNode> reachable = new ArrayList<>();
            if (root != null) {
                Deque<TreeNode> stack = new ArrayDeque<>();
                stack.push(root);
                while (!stack.isEmpty()) {
                    TreeNode node = stack.pop();
                    reachable.add(node);
                    if (node.left != null) stack.push(node.left);
                    if (node.right != null) stack.push(node.right);
                }
            }
            int targetVal = reachable.isEmpty() ? 1 : reachable.get(rng.nextInt(reachable.size())).val;
            for (int k = 0; k <= n + 2; k++) {
                check(countNodesAtDistance(root, targetVal, k) == bruteCount(root, targetVal, k),
                        "matches brute force: target=" + targetVal + " k=" + k);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
