/** Task 054 - Traverse linked list and print nodes. (Tier: Beginner, CTCI Ch 2) */


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
    static List<Integer> traverse(ListNode head) {
        return toList(head);
    }

    public static void main(String[] args) {
        check(traverse(fromList(1, 2, 3)).toString().equals("[1, 2, 3]"), "three nodes");
        check(traverse(null).isEmpty(), "empty");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
