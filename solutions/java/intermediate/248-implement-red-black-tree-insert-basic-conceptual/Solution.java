/** Task 248 - Implement red-black tree insert (basic). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class RBNode {
    int key;
    boolean red;
    RBNode left, right, parent;
    RBNode(int key) { this.key = key; this.red = true; }
}

class RedBlackTree {
    RBNode root;

    void rotateLeft(RBNode x) {
        RBNode y = x.right;
        x.right = y.left;
        if (y.left != null) y.left.parent = x;
        y.parent = x.parent;
        if (x.parent == null) root = y;
        else if (x == x.parent.left) x.parent.left = y;
        else x.parent.right = y;
        y.left = x;
        x.parent = y;
    }

    void rotateRight(RBNode x) {
        RBNode y = x.left;
        x.left = y.right;
        if (y.right != null) y.right.parent = x;
        y.parent = x.parent;
        if (x.parent == null) root = y;
        else if (x == x.parent.right) x.parent.right = y;
        else x.parent.left = y;
        y.right = x;
        x.parent = y;
    }

    boolean insert(int key) {
        RBNode parent = null;
        RBNode node = root;
        while (node != null) {
            parent = node;
            if (key < node.key) node = node.left;
            else if (key > node.key) node = node.right;
            else return false;
        }
        RBNode z = new RBNode(key);
        z.parent = parent;
        if (parent == null) root = z;
        else if (key < parent.key) parent.left = z;
        else parent.right = z;
        fixup(z);
        return true;
    }

    private void fixup(RBNode z) {
        while (z.parent != null && z.parent.red) {
            RBNode parent = z.parent;
            RBNode grand = parent.parent;
            if (parent == grand.left) {
                RBNode uncle = grand.right;
                if (uncle != null && uncle.red) {
                    parent.red = false;
                    uncle.red = false;
                    grand.red = true;
                    z = grand;
                } else {
                    if (z == parent.right) {
                        z = parent;
                        rotateLeft(z);
                        parent = z.parent;
                        grand = parent.parent;
                    }
                    parent.red = false;
                    grand.red = true;
                    rotateRight(grand);
                }
            } else {
                RBNode uncle = grand.left;
                if (uncle != null && uncle.red) {
                    parent.red = false;
                    uncle.red = false;
                    grand.red = true;
                    z = grand;
                } else {
                    if (z == parent.left) {
                        z = parent;
                        rotateRight(z);
                        parent = z.parent;
                        grand = parent.parent;
                    }
                    parent.red = false;
                    grand.red = true;
                    rotateLeft(grand);
                }
            }
        }
        root.red = false;
    }
}

public class Solution {
    static List<Integer> inorder(RBNode node) {
        List<Integer> out = new ArrayList<>();
        in(node, out);
        return out;
    }

    private static void in(RBNode node, List<Integer> out) {
        if (node == null) return;
        in(node.left, out);
        out.add(node.key);
        in(node.right, out);
    }

    static boolean noRedRed(RBNode node) {
        if (node == null) return true;
        if (node.red) {
            if (node.left != null && node.left.red) return false;
            if (node.right != null && node.right.red) return false;
        }
        return noRedRed(node.left) && noRedRed(node.right);
    }

    static Set<Integer> blackHeights(RBNode node, int measure) {
        Set<Integer> out = new HashSet<>();
        if (node == null) { out.add(measure); return out; }
        out.addAll(blackHeights(node.left, measure));
        out.addAll(blackHeights(node.right, measure));
        int extra = node.red ? 0 : 1;
        Set<Integer> shifted = new HashSet<>();
        for (int value : out) shifted.add(value + extra);
        return shifted;
    }

    public static void main(String[] args) {
        RedBlackTree tree = new RedBlackTree();
        for (int key : new int[]{10, 20, 30}) check(tree.insert(key), "insert " + key);
        check(tree.root.key == 20 && !tree.root.red, "root 20 black");
        check(tree.root.left.key == 10 && tree.root.left.red, "left 10 red");
        check(tree.root.right.key == 30 && tree.root.right.red, "right 30 red");

        int[][] orders = {
            {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            {7, 3, 9, 1, 5, 8, 10, 2, 4, 6},
            {50, 20, 60, 10, 30, 55, 70, 5, 15, 25, 35}
        };
        for (int[] order : orders) {
            tree = new RedBlackTree();
            for (int key : order) check(tree.insert(key), "insert " + key);
            int[] sorted = order.clone();
            Arrays.sort(sorted);
            check(inorder(tree.root).toString().equals(Arrays.toString(sorted)), "BST order");
            check(!tree.root.red, "root is black");
            check(noRedRed(tree.root), "no red-red violation");
            check(blackHeights(tree.root, 0).size() == 1, "uniform black height");
        }
        check(!tree.insert(50), "duplicate rejected");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
