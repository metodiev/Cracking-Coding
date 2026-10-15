/** Task 247 - Implement AVL tree insert (rotations + balance). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class AVLNode {
    int key, height = 1;
    AVLNode left, right;
    AVLNode(int key) { this.key = key; }
}

public class Solution {
    static int height(AVLNode node) { return node == null ? 0 : node.height; }

    static void update(AVLNode node) { node.height = 1 + Math.max(height(node.left), height(node.right)); }

    static int balanceFactor(AVLNode node) { return height(node.left) - height(node.right); }

    static AVLNode rotateRight(AVLNode y) {
        AVLNode x = y.left;
        y.left = x.right;
        x.right = y;
        update(y);
        update(x);
        return x;
    }

    static AVLNode rotateLeft(AVLNode x) {
        AVLNode y = x.right;
        x.right = y.left;
        y.left = x;
        update(x);
        update(y);
        return y;
    }

    static AVLNode insert(AVLNode node, int key) {
        if (node == null) return new AVLNode(key);
        if (key < node.key) node.left = insert(node.left, key);
        else if (key > node.key) node.right = insert(node.right, key);
        else return node;

        update(node);
        int bf = balanceFactor(node);
        if (bf > 1 && key < node.left.key) return rotateRight(node);
        if (bf < -1 && key > node.right.key) return rotateLeft(node);
        if (bf > 1 && key > node.left.key) { node.left = rotateLeft(node.left); return rotateRight(node); }
        if (bf < -1 && key < node.right.key) { node.right = rotateRight(node.right); return rotateLeft(node); }
        return node;
    }

    static AVLNode build(int... keys) {
        AVLNode root = null;
        for (int key : keys) root = insert(root, key);
        return root;
    }

    static List<Integer> inorder(AVLNode node) {
        List<Integer> out = new ArrayList<>();
        in(node, out);
        return out;
    }

    private static void in(AVLNode node, List<Integer> out) {
        if (node == null) return;
        in(node.left, out);
        out.add(node.key);
        in(node.right, out);
    }

    static boolean isBalanced(AVLNode node) {
        if (node == null) return true;
        if (Math.abs(balanceFactor(node)) > 1) return false;
        return isBalanced(node.left) && isBalanced(node.right);
    }

    static int count(AVLNode node) { return node == null ? 0 : 1 + count(node.left) + count(node.right); }

    public static void main(String[] args) {
        AVLNode root = build(10, 20, 30);
        check(root.key == 20 && root.left.key == 10 && root.right.key == 30, "LL rotation");
        root = build(30, 20, 10);
        check(root.key == 20 && root.left.key == 10 && root.right.key == 30, "RR rotation");
        root = build(30, 10, 20);
        check(root.key == 20 && root.left.key == 10 && root.right.key == 30, "LR rotation");
        root = build(10, 30, 20);
        check(root.key == 20 && root.left.key == 10 && root.right.key == 30, "RL rotation");

        AVLNode asc = build(1, 2, 3, 4, 5, 6, 7);
        check(inorder(asc).toString().equals("[1, 2, 3, 4, 5, 6, 7]"), "inorder sorted");
        check(isBalanced(asc) && asc.height == 3, "height 3 for 7 ascending keys");

        int[] keys10 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        AVLNode ten = build(keys10);
        check(count(ten) == 10 && isBalanced(ten) && ten.height == 4, "height 4 for 10 ascending keys");
        check(count(build(5, 5, 5)) == 1, "duplicates ignored");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
