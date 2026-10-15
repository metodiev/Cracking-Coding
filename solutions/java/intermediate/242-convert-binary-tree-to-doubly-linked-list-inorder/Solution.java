/** Task 242 - Convert binary tree to doubly linked list (inorder). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class TreeNode {
    int val;
    TreeNode left, right;
    TreeNode(int v) { val = v; }
}

public class Solution {
    static TreeNode buildTree(Integer... values) {
        if (values.length == 0 || values[0] == null) return null;
        TreeNode root = new TreeNode(values[0]);
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        int i = 1;
        while (!q.isEmpty() && i < values.length) {
            TreeNode node = q.poll();
            if (i < values.length && values[i] != null) { node.left = new TreeNode(values[i]); q.add(node.left); }
            i++;
            if (i < values.length && values[i] != null) { node.right = new TreeNode(values[i]); q.add(node.right); }
            i++;
        }
        return root;
    }

    static List<Integer> preorder(TreeNode node) {
        List<Integer> out = new ArrayList<>();
        pre(node, out);
        return out;
    }

    private static void pre(TreeNode node, List<Integer> out) {
        if (node == null) return;
        out.add(node.val);
        pre(node.left, out);
        pre(node.right, out);
    }

    static List<Integer> inorder(TreeNode node) {
        List<Integer> out = new ArrayList<>();
        in(node, out);
        return out;
    }

    private static void in(TreeNode node, List<Integer> out) {
        if (node == null) return;
        in(node.left, out);
        out.add(node.val);
        in(node.right, out);
    }

    private static TreeNode last, head;

    static TreeNode toDoublyLinkedList(TreeNode root) {
        last = null;
        head = null;
        visit(root);
        return head;
    }

    private static void visit(TreeNode node) {
        if (node == null) return;
        visit(node.left);
        node.left = last;
        if (last != null) last.right = node;
        else head = node;
        last = node;
        visit(node.right);
    }

    static List<Integer> toValues(TreeNode head) {
        List<Integer> out = new ArrayList<>();
        for (TreeNode node = head; node != null; node = node.right) out.add(node.val);
        return out;
    }

    static List<Integer> toValuesReverse(TreeNode head) {
        List<Integer> out = new ArrayList<>();
        if (head == null) return out;
        TreeNode node = head;
        while (node.right != null) node = node.right;
        for (; node != null; node = node.left) out.add(node.val);
        return out;
    }

    public static void main(String[] args) {
        TreeNode head = toDoublyLinkedList(buildTree(4, 2, 5, 1, 3));
        check(toValues(head).toString().equals("[1, 2, 3, 4, 5]"), "inorder order");
        check(toValuesReverse(head).toString().equals("[5, 4, 3, 2, 1]"), "prev links");
        check(toDoublyLinkedList(null) == null, "empty");
        TreeNode single = toDoublyLinkedList(buildTree(7));
        check(toValues(single).toString().equals("[7]") && single.left == null && single.right == null, "single");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
