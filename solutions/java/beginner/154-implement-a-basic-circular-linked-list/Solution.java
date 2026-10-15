/** Task 154 - Implement a basic circular linked list. (Tier: Beginner, CTCI Ch 2) */
import java.util.*;

class CircularNode {
    int value;
    CircularNode next;
    CircularNode(int value) { this.value = value; }
}

/** Singly linked circular list: the tail always points back at the head. */
class CircularLinkedList {
    CircularNode head;
    int size;

    /** Append at the tail; the new tail points back at the head. */
    void append(int value) {
        CircularNode node = new CircularNode(value);
        if (head == null) {
            node.next = node;
            head = node;
        } else {
            CircularNode tail = tail();
            tail.next = node;
            node.next = head;
        }
        size++;
    }

    /** Insert before position `index` (0 <= index <= size); size appends. */
    void insert(int index, int value) {
        if (index < 0 || index > size) throw new IndexOutOfBoundsException("insert index out of range");
        if (index == size) { append(value); return; }
        CircularNode node = new CircularNode(value);
        if (index == 0) {
            CircularNode tail = tail();
            node.next = head;
            tail.next = node;
            head = node;
        } else {
            CircularNode prev = head;
            for (int i = 0; i < index - 1; i++) prev = prev.next;
            node.next = prev.next;
            prev.next = node;
        }
        size++;
    }

    /** Delete the first node holding `value`; returns true if one was removed. */
    boolean delete(int value) {
        if (head == null) return false;
        if (head.value == value) {
            if (size == 1) {
                head = null;
            } else {
                CircularNode tail = tail();
                head = head.next;
                tail.next = head;
            }
            size--;
            return true;
        }
        CircularNode prev = head;
        while (prev.next != head && prev.next.value != value) prev = prev.next;
        if (prev.next == head) return false;
        prev.next = prev.next.next;
        size--;
        return true;
    }

    /** The node whose next is the head (null for an empty list). */
    CircularNode tail() {
        if (head == null) return null;
        CircularNode node = head;
        while (node.next != head) node = node.next;
        return node;
    }

    /** One lap: exactly `size` steps starting at the head (terminates). */
    List<Integer> toList() {
        List<Integer> out = new ArrayList<>();
        CircularNode node = head;
        for (int i = 0; i < size; i++) {
            out.add(node.value);
            node = node.next;
        }
        return out;
    }
}

public class Solution {
    public static void main(String[] args) {
        CircularLinkedList lst = new CircularLinkedList();
        check(lst.toList().isEmpty() && lst.head == null && lst.size == 0, "empty list");

        for (int v : new int[] {1, 2, 3}) lst.append(v);
        check(lst.toList().toString().equals("[1, 2, 3]") && lst.size == 3, "appended three");
        check(lst.tail().next == lst.head, "tail wraps to head");
        check(lst.head.next.next.next == lst.head, "third node wraps to head");

        lst.insert(0, 0);
        check(lst.toList().toString().equals("[0, 1, 2, 3]") && lst.head.value == 0, "insert at head");
        lst.insert(2, 99);
        check(lst.toList().toString().equals("[0, 1, 99, 2, 3]"), "insert in middle");
        lst.insert(lst.size, 4);
        check(lst.toList().toString().equals("[0, 1, 99, 2, 3, 4]"), "insert at tail");
        check(lst.tail().value == 4 && lst.tail().next == lst.head, "tail still wraps");

        boolean threw = false;
        try { lst.insert(99, 0); } catch (IndexOutOfBoundsException e) { threw = true; }
        check(threw, "insert out of range throws");

        check(lst.delete(99) && lst.toList().toString().equals("[0, 1, 2, 3, 4]"), "delete middle");
        check(lst.delete(0) && lst.head.value == 1 && lst.toList().toString().equals("[1, 2, 3, 4]"),
              "delete head");
        check(lst.tail().next == lst.head, "ring still closed after head delete");
        check(lst.delete(4) && lst.toList().toString().equals("[1, 2, 3]"), "delete tail");
        check(!lst.delete(42) && lst.toList().toString().equals("[1, 2, 3]"), "missing value");

        check(lst.delete(1) && lst.delete(2) && lst.delete(3), "drain to empty");
        check(lst.toList().isEmpty() && lst.size == 0 && lst.head == null && lst.tail() == null, "empty again");
        check(!lst.delete(1), "delete on empty");
        lst.append(7);
        check(lst.toList().toString().equals("[7]") && lst.head.next == lst.head, "single node self-loop");
        check(lst.delete(7) && lst.toList().isEmpty() && lst.head == null, "delete last node");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
