/** Task 250 - Implement splay tree access operation (basic). (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class SplayNode {
    int key;
    SplayNode left, right, parent;
    SplayNode(int key) { this.key = key; }
}

class SplayTree {
    SplayNode root;

    private void replaceChild(SplayNode parent, SplayNode old, SplayNode fresh) {
        if (parent == null) root = fresh;
        else if (parent.left == old) parent.left = fresh;
        else parent.right = fresh;
        if (fresh != null) fresh.parent = parent;
    }

    private void rotateLeft(SplayNode x) {
        SplayNode y = x.right;
        x.right = y.left;
        if (y.left != null) y.left.parent = x;
        SplayNode parent = x.parent;
        replaceChild(parent, x, y);
        y.left = x;
        x.parent = y;
    }

    private void rotateRight(SplayNode x) {
        SplayNode y = x.left;
        x.left = y.right;
        if (y.right != null) y.right.parent = x;
        SplayNode parent = x.parent;
        replaceChild(parent, x, y);
        y.right = x;
        x.parent = y;
    }

    void splay(SplayNode x) {
        while (x.parent != null) {
            SplayNode parent = x.parent;
            SplayNode grand = parent.parent;
            if (grand == null) {
                if (x == parent.left) rotateRight(parent); else rotateLeft(parent);
            } else if (x == parent.left && parent == grand.left) {
                rotateRight(grand);
                rotateRight(parent);
            } else if (x == parent.right && parent == grand.right) {
                rotateLeft(grand);
                rotateLeft(parent);
            } else if (x == parent.right && parent == grand.left) {
                rotateLeft(parent);
                rotateRight(grand);
            } else {
                rotateRight(parent);
                rotateLeft(grand);
            }
        }
        root = x;
    }

    boolean insert(int key) {
        SplayNode parent = null;
        SplayNode node = root;
        while (node != null) {
            parent = node;
            if (key < node.key) node = node.left;
            else if (key > node.key) node = node.right;
            else { splay(node); return false; }
        }
        SplayNode fresh = new SplayNode(key);
        fresh.parent = parent;
        if (parent == null) root = fresh;
        else if (key < parent.key) parent.left = fresh;
        else parent.right = fresh;
        splay(fresh);
        return true;
    }

    /** Splays key to the root if it exists; returns whether it was found. */
    boolean access(int key) {
        SplayNode node = root;
        while (node != null) {
            if (key == node.key) { splay(node); return true; }
            node = key < node.key ? node.left : node.right;
        }
        return false;
    }

    List<Integer> inorder() {
        List<Integer> out = new ArrayList<>();
        walk(root, out);
        return out;
    }

    private void walk(SplayNode node, List<Integer> out) {
        if (node == null) return;
        walk(node.left, out);
        out.add(node.key);
        walk(node.right, out);
    }

    boolean isBst() {
        List<Integer> keys = inorder();
        for (int i = 1; i < keys.size(); i++) if (keys.get(i - 1) >= keys.get(i)) return false;
        return parentsConsistent(root);
    }

    private boolean parentsConsistent(SplayNode node) {
        if (node == null) return true;
        if (node.left != null && node.left.parent != node) return false;
        if (node.right != null && node.right.parent != node) return false;
        return parentsConsistent(node.left) && parentsConsistent(node.right);
    }
}

public class Solution {
    public static void main(String[] args) {
        SplayTree tree = new SplayTree();
        for (int key : new int[]{50, 30, 70, 20, 40, 60, 80}) check(tree.insert(key), "insert " + key);
        check(tree.root.key == 80, "last insert at root");
        check(tree.inorder().toString().equals("[20, 30, 40, 50, 60, 70, 80]"), "sorted");
        check(tree.isBst(), "bst + parent links");

        check(tree.access(20) && tree.root.key == 20, "zig-zig to root");
        check(tree.isBst(), "still bst after splay 20");

        check(tree.access(80) && tree.root.key == 80, "mirror zig-zig");
        check(tree.isBst(), "still bst after splay 80");

        check(tree.access(50) && tree.root.key == 50, "zig-zag to root");
        check(tree.inorder().toString().equals("[20, 30, 40, 50, 60, 70, 80]"), "sorted after splays");
        check(tree.isBst(), "bst + parent links after splays");

        check(!tree.access(999) && tree.root.key == 50, "miss leaves root alone");
        check(!tree.insert(50) && tree.root.key == 50, "duplicate splayed, not inserted");
        check(tree.inorder().toString().equals("[20, 30, 40, 50, 60, 70, 80]"), "no duplicate stored");

        SplayTree single = new SplayTree();
        check(single.insert(7), "single insert");
        check(single.access(7) && single.root.key == 7 && single.root.parent == null, "single access");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
