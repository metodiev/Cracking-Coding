/** Task 156 - Implement iterator for a binary tree (inorder). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int v) { val = v; }
}

/**
 * In-order iterator with an explicit stack: push the left spine of the current
 * subtree; next() pops one node, then descends that node's right subtree's left
 * spine. O(1) amortised per step, O(h) extra memory.
 */
class InOrderIterator {
    private final Deque<TreeNode> stack = new ArrayDeque<>();

    InOrderIterator(TreeNode root) {
        pushLeftSpine(root);
    }

    private void pushLeftSpine(TreeNode node) {
        while (node != null) {
            stack.push(node);
            node = node.left;
        }
    }

    boolean hasNext() {
        return !stack.isEmpty();
    }

    /** Next in-order value; throws NoSuchElementException once exhausted. */
    int next() {
        if (stack.isEmpty()) throw new NoSuchElementException("iterator exhausted");
        TreeNode node = stack.pop();
        pushLeftSpine(node.right);
        return node.val;
    }
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

    public static void main(String[] args) {
        // valid BST: in-order must be ascending
        TreeNode bst = buildTree(8, 3, 10, 1, 6, null, 14, null, null, 4, 7, 13);
        InOrderIterator it = new InOrderIterator(bst);
        List<Integer> values = new ArrayList<>();
        while (it.hasNext()) values.add(it.next());
        check(values.toString().equals("[1, 3, 4, 6, 7, 8, 10, 13, 14]"), "BST in-order");
        for (int i = 1; i < values.size(); i++) {
            check(values.get(i - 1) < values.get(i), "BST ascending");
        }

        // exhaustion: hasNext() is false and next() throws
        check(!it.hasNext(), "exhausted");
        boolean threw = false;
        try { it.next(); } catch (NoSuchElementException e) { threw = true; }
        check(threw, "next after exhaustion throws");

        // empty tree is immediately exhausted
        InOrderIterator empty = new InOrderIterator(null);
        check(!empty.hasNext(), "empty exhausted");
        threw = false;
        try { empty.next(); } catch (NoSuchElementException e) { threw = true; }
        check(threw, "empty next throws");

        // hasNext is idempotent (peeks without consuming) and works one step at a time
        InOrderIterator it2 = new InOrderIterator(buildTree(1, 2, 3));
        check(it2.hasNext() && it2.hasNext(), "hasNext idempotent");
        check(it2.next() == 2 && it2.next() == 1, "left-most then root");
        check(it2.hasNext() && it2.next() == 3 && !it2.hasNext(), "right child last");

        // single node
        InOrderIterator it3 = new InOrderIterator(buildTree(5));
        check(it3.hasNext() && it3.next() == 5 && !it3.hasNext(), "single node");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
