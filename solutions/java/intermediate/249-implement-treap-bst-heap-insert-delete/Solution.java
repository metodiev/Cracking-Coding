/** Task 249 - Implement treap (BST + heap) insert/delete. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class TreapNode {
    int key, pri;
    TreapNode left, right;
    TreapNode(int key, int pri) { this.key = key; this.pri = pri; }
}

public class Solution {
    static TreapNode rotateRight(TreapNode y) {
        TreapNode x = y.left;
        y.left = x.right;
        x.right = y;
        return x;
    }

    static TreapNode rotateLeft(TreapNode x) {
        TreapNode y = x.right;
        x.right = y.left;
        y.left = x;
        return y;
    }

    /** Returns {new root} and records whether the key was inserted in inserted[0]. */
    static TreapNode insert(TreapNode node, int key, int pri, boolean[] inserted) {
        if (node == null) { inserted[0] = true; return new TreapNode(key, pri); }
        if (key == node.key) { inserted[0] = false; return node; }
        if (key < node.key) {
            node.left = insert(node.left, key, pri, inserted);
            if (node.left.pri > node.pri) node = rotateRight(node);
        } else {
            node.right = insert(node.right, key, pri, inserted);
            if (node.right.pri > node.pri) node = rotateLeft(node);
        }
        return node;
    }

    static TreapNode erase(TreapNode node, int key, boolean[] removed) {
        if (node == null) { removed[0] = false; return null; }
        if (key < node.key) {
            node.left = erase(node.left, key, removed);
        } else if (key > node.key) {
            node.right = erase(node.right, key, removed);
        } else {
            removed[0] = true;
            if (node.left == null) return node.right;
            if (node.right == null) return node.left;
            if (node.left.pri > node.right.pri) {
                node = rotateRight(node);
                node.right = erase(node.right, key, removed);
            } else {
                node = rotateLeft(node);
                node.left = erase(node.left, key, removed);
            }
        }
        return node;
    }

    static TreapNode build(int[][] entries) {
        TreapNode root = null;
        for (int[] entry : entries) {
            boolean[] inserted = new boolean[1];
            root = insert(root, entry[0], entry[1], inserted);
        }
        return root;
    }

    static List<Integer> inorder(TreapNode node) {
        List<Integer> out = new ArrayList<>();
        in(node, out);
        return out;
    }

    private static void in(TreapNode node, List<Integer> out) {
        if (node == null) return;
        in(node.left, out);
        out.add(node.key);
        in(node.right, out);
    }

    static List<Integer> preorder(TreapNode node) {
        List<Integer> out = new ArrayList<>();
        pre(node, out);
        return out;
    }

    private static void pre(TreapNode node, List<Integer> out) {
        if (node == null) return;
        out.add(node.key);
        pre(node.left, out);
        pre(node.right, out);
    }

    static boolean isHeap(TreapNode node) {
        if (node == null) return true;
        if (node.left != null && node.left.pri > node.pri) return false;
        if (node.right != null && node.right.pri > node.pri) return false;
        return isHeap(node.left) && isHeap(node.right);
    }

    static boolean contains(TreapNode node, int key) {
        while (node != null) {
            if (key == node.key) return true;
            node = key < node.key ? node.left : node.right;
        }
        return false;
    }

    public static void main(String[] args) {
        int[][] entries = {{50, 40}, {30, 25}, {70, 60}, {20, 15}, {40, 35}, {60, 50}, {80, 70}};
        TreapNode root = build(entries);
        check(root.key == 80 && root.pri == 70, "highest priority at root");
        check(preorder(root).toString().equals("[80, 70, 60, 50, 40, 30, 20]"), "preorder shape");
        check(inorder(root).toString().equals("[20, 30, 40, 50, 60, 70, 80]"), "inorder sorted");
        check(isHeap(root), "max-heap on priorities");

        boolean[] inserted = new boolean[1];
        insert(build(new int[][]{{1, 5}}), 1, 9, inserted);
        check(!inserted[0], "duplicate rejected");

        boolean[] removed = new boolean[1];
        root = erase(root, 30, removed);
        check(removed[0] && inorder(root).toString().equals("[20, 40, 50, 60, 70, 80]") && isHeap(root), "erase leaf");
        root = erase(root, 80, removed);
        check(removed[0] && root.key == 70 && isHeap(root), "erase root");
        root = erase(root, 999, removed);
        check(!removed[0] && inorder(root).toString().equals("[20, 40, 50, 60, 70]"), "erase missing");
        for (int key : new int[]{20, 40, 50, 60, 70}) check(contains(root, key), "contains " + key);
        check(!contains(root, 30), "30 gone");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
