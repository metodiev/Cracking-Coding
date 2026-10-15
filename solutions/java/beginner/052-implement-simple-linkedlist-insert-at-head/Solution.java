/** Task 052 - Implement simple LinkedList insert at head. (Tier: Beginner, CTCI Ch 2) */


import java.util.*;

class ListNode {
    int val;
    ListNode next;
    ListNode(int v) { val = v; }
}

public class Solution {
    static ListNode fromList(int... vals) {
        ListNode head = null;
        for (int i = vals.length - 1; i >= 0; i--) {
            ListNode n = new ListNode(vals[i]);
            n.next = head;
            head = n;
        }
        return head;
    }

    static List<Integer> toList(ListNode head) {
        List<Integer> out = new ArrayList<>();
        for (ListNode p = head; p != null; p = p.next) out.add(p.val);
        return out;
    }
    static ListNode insertHead(ListNode head, int value) {
        ListNode n = new ListNode(value);
        n.next = head;
        return n;
    }

    public static void main(String[] args) {
        ListNode head = null;
        for (int v : new int[]{1, 2, 3}) head = insertHead(head, v);
        check(toList(head).toString().equals("[3, 2, 1]"), "reversed by head inserts");
        check(toList(insertHead(null, 9)).toString().equals("[9]"), "into empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
