/** Task 063 - Implement doubly linked list insert/delete. (Tier: Beginner, CTCI Ch 2) */
import java.util.*;

class DNode {
    int val;
    DNode prev, next;
    DNode(int v) { val = v; }
}

class DoublyLinkedList {
    DNode head, tail;

    DNode append(int val) {
        DNode node = new DNode(val);
        if (tail == null) head = tail = node;
        else {
            node.prev = tail;
            tail.next = node;
            tail = node;
        }
        return node;
    }

    DNode insertBefore(DNode node, int val) {
        DNode fresh = new DNode(val);
        fresh.prev = node.prev;
        fresh.next = node;
        if (node.prev != null) node.prev.next = fresh;
        else head = fresh;
        node.prev = fresh;
        return fresh;
    }

    void delete(DNode node) {
        if (node.prev != null) node.prev.next = node.next;
        else head = node.next;
        if (node.next != null) node.next.prev = node.prev;
        else tail = node.prev;
        node.prev = node.next = null;
    }

    List<Integer> toList() {
        List<Integer> out = new ArrayList<>();
        for (DNode p = head; p != null; p = p.next) out.add(p.val);
        return out;
    }

    List<Integer> toListReversed() {
        List<Integer> out = new ArrayList<>();
        for (DNode p = tail; p != null; p = p.prev) out.add(p.val);
        return out;
    }
}

public class Solution {
    public static void main(String[] args) {
        DoublyLinkedList d = new DoublyLinkedList();
        DNode a = d.append(1), b = d.append(2), c = d.append(3);
        check(d.toList().toString().equals("[1, 2, 3]"), "append");
        check(d.toListReversed().toString().equals("[3, 2, 1]"), "reversed links");

        d.insertBefore(b, 9);
        check(d.toList().toString().equals("[1, 9, 2, 3]"), "insert before middle");
        d.insertBefore(a, 0);
        check(d.toList().toString().equals("[0, 1, 9, 2, 3]"), "insert before head");
        check(d.toListReversed().toString().equals("[3, 2, 9, 1, 0]"), "reverse after inserts");

        d.delete(a);
        check(d.toList().toString().equals("[0, 9, 2, 3]"), "delete middle");
        d.delete(c);
        check(d.toList().toString().equals("[0, 9, 2]"), "delete tail");
        d.delete(b);
        check(d.toList().toString().equals("[0, 9]"), "delete node");
        check(d.toListReversed().toString().equals("[9, 0]"), "reverse after deletes");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
