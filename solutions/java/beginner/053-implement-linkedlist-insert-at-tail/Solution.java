/** Task 053 - Implement LinkedList insert at tail. (Tier: Beginner, CTCI Ch 2) */


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
    static ListNode insertTail(ListNode head, int value) {
        ListNode node = new ListNode(value);
        if (head == null) return node;
        ListNode p = head;
        while (p.next != null) p = p.next;
        p.next = node;
        return head;
    }

    public static void main(String[] args) {
        ListNode head = null;
        for (int v : new int[]{1, 2, 3}) head = insertTail(head, v);
        check(toList(head).toString().equals("[1, 2, 3]"), "appended");
        check(toList(insertTail(null, 7)).toString().equals("[7]"), "into empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
