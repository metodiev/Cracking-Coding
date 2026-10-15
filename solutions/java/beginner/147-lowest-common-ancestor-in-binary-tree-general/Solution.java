/** Task 147 - Lowest common ancestor in binary tree (general). (Tier: Beginner, CTCI Ch 4)
 *
 * No BST ordering is available, so a postorder search is used: a subtree reports
 * the first key it finds from below; the node where both subtrees report a key is
 * the split point and therefore the LCA. When either key is not part of the tree
 * at all, the answer is null.
 */
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

    /** Plain depth-first search: a general tree has no ordering to exploit. */
    static TreeNode find(TreeNode root, int val) {
        if (root == null || root.val == val) return root;
        TreeNode hit = find(root.left, val);
        return hit != null ? hit : find(root.right, val);
    }

    /** Identity search: is that very node object part of this tree? */
    static boolean contains(TreeNode root, TreeNode target) {
        if (root == null) return false;
        return root == target || contains(root.left, target) || contains(root.right, target);
    }

    /** Postorder LCA; null when either node is missing from the tree. */
    static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null) return null;
        if (!contains(root, p) || !contains(root, q)) return null;
        return search(root, p, q);
    }

    /** Classic postorder: report a target from below, else report the split node. */
    private static TreeNode search(TreeNode node, TreeNode p, TreeNode q) {
        if (node == null || node == p || node == q) return node;
        TreeNode left = search(node.left, p, q);
        TreeNode right = search(node.right, p, q);
        if (left != null && right != null) return node;   // p and q in different subtrees
        return left != null ? left : right;
    }

    public static void main(String[] args) {
        TreeNode tree = buildTree(3, 5, 1, 6, 2, 0, 8, null, null, 7, 4);
        check(lowestCommonAncestor(tree, find(tree, 5), find(tree, 1)).val == 3, "root splits them");
        check(lowestCommonAncestor(tree, find(tree, 5), find(tree, 4)).val == 5, "one key is an ancestor");
        check(lowestCommonAncestor(tree, find(tree, 6), find(tree, 7)).val == 5, "both in the left branch");
        check(lowestCommonAncestor(tree, find(tree, 6), find(tree, 4)).val == 5, "deeper split");
        check(lowestCommonAncestor(tree, find(tree, 7), find(tree, 4)).val == 2, "siblings under 2");
        check(lowestCommonAncestor(tree, find(tree, 0), find(tree, 4)).val == 3, "different subtrees");
        check(lowestCommonAncestor(tree, find(tree, 3), find(tree, 5)).val == 3, "one key is the root");
        check(lowestCommonAncestor(tree, find(tree, 7), find(tree, 7)).val == 7, "node with itself");
        check(lowestCommonAncestor(tree, find(tree, 0), find(tree, 99)) == null, "missing key");
        check(lowestCommonAncestor(tree, null, find(tree, 5)) == null, "null key");
        check(lowestCommonAncestor(null, find(tree, 5), find(tree, 1)) == null, "no tree");
        // same value, but a different node object is not part of this tree
        check(lowestCommonAncestor(tree, find(tree, 3), new TreeNode(5)) == null, "detached node");
        TreeNode chain = buildTree(1, null, 2, null, 3);
        check(lowestCommonAncestor(chain, find(chain, 1), find(chain, 3)).val == 1, "right chain");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
