/** Task 056 - Find length of linked list. (Tier: Beginner, CTCI Ch 2) */


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
    static int length(ListNode head) {
        int count = 0;
        for (ListNode p = head; p != null; p = p.next) count++;
        return count;
    }

    public static void main(String[] args) {
        check(length(fromList(1, 2, 3, 4)) == 4, "four");
        check(length(null) == 0, "empty");
        check(length(fromList(7)) == 1, "single");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
