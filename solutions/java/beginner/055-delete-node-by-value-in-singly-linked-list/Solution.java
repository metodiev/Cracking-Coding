/** Task 055 - Delete node by value in singly linked list. (Tier: Beginner, CTCI Ch 2) */


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
    static ListNode deleteValue(ListNode head, int value) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode prev = dummy;
        while (prev.next != null) {
            if (prev.next.val == value) {
                prev.next = prev.next.next;
                break;
            }
            prev = prev.next;
        }
        return dummy.next;
    }

    public static void main(String[] args) {
        check(toList(deleteValue(fromList(1, 2, 3), 2)).toString().equals("[1, 3]"), "middle");
        check(toList(deleteValue(fromList(1, 2, 3), 1)).toString().equals("[2, 3]"), "head");
        check(toList(deleteValue(fromList(1), 1)).isEmpty(), "only node");
        check(toList(deleteValue(fromList(1, 2), 9)).toString().equals("[1, 2]"), "missing no-op");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
