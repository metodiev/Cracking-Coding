/** Task 239 - Connect next pointers in each node (perfect binary tree). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class Node {
    int val;
    Node left, right, next;
    Node(int v) { val = v; }
}

public class Solution {
    /** Build a binary tree from level-order values (null marks a missing child). */
    static Node buildTree(Integer... values) {
        if (values.length == 0 || values[0] == null) return null;
        Node root = new Node(values[0]);
        Deque<Node> q = new ArrayDeque<>();
        q.add(root);
        int i = 1;
        while (!q.isEmpty() && i < values.length) {
            Node node = q.poll();
            if (i < values.length && values[i] != null) {
                node.left = new Node(values[i]);
                q.add(node.left);
            }
            i++;
            if (i < values.length && values[i] != null) {
                node.right = new Node(values[i]);
                q.add(node.right);
            }
            i++;
        }
        return root;
    }

    /**
     * Every node gets next -> its right neighbour on the same level (null at level
     * ends). O(1) extra space: descend level by level through the already-set next
     * pointers of the parent level; for a parent p, p.left.next = p.right and, when
     * p.next exists, p.right.next = p.next.left. Returns the root.
     */
    static Node connect(Node root) {
        if (root == null) return null;
        for (Node leftmost = root; leftmost.left != null; leftmost = leftmost.left) {
            for (Node node = leftmost; node != null; node = node.next) {
                node.left.next = node.right;
                if (node.next != null) node.right.next = node.next.left;
            }
        }
        return root;
    }

    /** Read the levels through the next pointers (fails fast if a chain loops). */
    static List<List<Integer>> levelsViaNext(Node root) {
        List<List<Integer>> levels = new ArrayList<>();
        for (Node leftmost = root; leftmost != null; leftmost = leftmost.left) {
            List<Integer> level = new ArrayList<>();
            for (Node node = leftmost; node != null; node = node.next) {
                level.add(node.val);
                if (level.size() > 1000) throw new AssertionError("next chain longer than expected");
            }
            levels.add(level);
        }
        return levels;
    }

    public static void main(String[] args) {
        Node root = connect(buildTree(1, 2, 3, 4, 5, 6, 7));
        check(levelsViaNext(root).equals(Arrays.asList(
                Arrays.asList(1), Arrays.asList(2, 3), Arrays.asList(4, 5, 6, 7))),
                "classic levels");
        check(root.next == null, "root boundary");
        check(root.left.next == root.right, "2 -> 3");
        check(root.right.next == null, "3 boundary");
        check(root.left.left.next == root.left.right, "4 -> 5");
        check(root.left.right.next == root.right.left, "5 -> 6");
        check(root.right.left.next == root.right.right, "6 -> 7");
        check(root.right.right.next == null, "7 boundary");

        Node single = connect(buildTree(1));
        check(levelsViaNext(single).equals(Arrays.asList(Arrays.asList(1))), "single level");
        check(single.next == null, "single boundary");
        check(connect(null) == null, "empty");

        Node big = connect(buildTree(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
        check(levelsViaNext(big).equals(Arrays.asList(
                Arrays.asList(1), Arrays.asList(2, 3), Arrays.asList(4, 5, 6, 7),
                Arrays.asList(8, 9, 10, 11, 12, 13, 14, 15))),
                "four levels");

        Node again = connect(root);               // connecting twice changes nothing
        check(again == root, "idempotent identity");
        check(levelsViaNext(again).equals(Arrays.asList(
                Arrays.asList(1), Arrays.asList(2, 3), Arrays.asList(4, 5, 6, 7))),
                "idempotent levels");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
