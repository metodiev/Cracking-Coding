/** Task 146 - Lowest common ancestor in BST (iterative). (Tier: Beginner, CTCI Ch 4)
 *
 * The BST ordering turns the search into a walk: while both keys sit on the same
 * side of the current node, step that way. The first node where the keys split
 * (or where one key IS the node) is the lowest common ancestor.
 * O(h) steps, O(1) extra space.
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

    /** Locate a node by value using the BST ordering; null when it is absent. */
    static TreeNode find(TreeNode root, int val) {
        while (root != null && root.val != val) {
            root = val < root.val ? root.left : root.right;
        }
        return root;
    }

    /** Iterative split-point walk; null when a key is missing from this BST. */
    static TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null) return null;
        TreeNode node = root;
        while (node != null) {
            if (p.val < node.val && q.val < node.val) {
                node = node.left;                 // both keys live in the left subtree
            } else if (p.val > node.val && q.val > node.val) {
                node = node.right;                // both keys live in the right subtree
            } else {
                return node;                      // split point, or one key is node itself
            }
        }
        return null;
    }

    public static void main(String[] args) {
        TreeNode tree = buildTree(6, 2, 8, 0, 4, 7, 9, null, null, 3, 5);
        check(lowestCommonAncestor(tree, find(tree, 2), find(tree, 8)).val == 6, "split at the root");
        check(lowestCommonAncestor(tree, find(tree, 2), find(tree, 4)).val == 2, "ancestor == key");
        check(lowestCommonAncestor(tree, find(tree, 0), find(tree, 5)).val == 2, "split on the left");
        check(lowestCommonAncestor(tree, find(tree, 4), find(tree, 5)).val == 4, "both left, then right");
        check(lowestCommonAncestor(tree, find(tree, 7), find(tree, 9)).val == 8, "both right of the root");
        check(lowestCommonAncestor(tree, find(tree, 2), find(tree, 2)).val == 2, "node with itself");
        check(lowestCommonAncestor(tree, find(tree, 0), find(tree, 0)).val == 0, "leaf with itself");
        check(lowestCommonAncestor(tree, find(tree, 0), find(tree, 3)).val == 2, "both keys left");
        check(lowestCommonAncestor(tree, find(tree, 0), find(tree, 99)) == null, "missing key");
        check(lowestCommonAncestor(tree, null, find(tree, 8)) == null, "null key");
        check(lowestCommonAncestor(null, find(tree, 2), find(tree, 8)) == null, "no tree");
        // 2 and 4 sit in opposite subtrees of the root 3; 5 is 4's ancestor
        TreeNode chain = buildTree(3, 1, 5, null, 2, 4);
        check(lowestCommonAncestor(chain, find(chain, 2), find(chain, 4)).val == 3, "different subtrees");
        check(lowestCommonAncestor(chain, find(chain, 1), find(chain, 2)).val == 1, "both below 1");
        check(lowestCommonAncestor(chain, find(chain, 4), find(chain, 5)).val == 5, "ancestor == key");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
