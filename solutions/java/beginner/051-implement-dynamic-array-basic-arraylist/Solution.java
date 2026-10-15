/** Task 051 - Implement dynamic array (basic ArrayList). (Tier: Beginner, CTCI Ch 1) */


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
    private static class DynamicArray {
        private int[] data = new int[2];
        private int size = 0;

        void pushBack(int value) {
            if (size == data.length) grow();
            data[size++] = value;
        }

        private void grow() {
            int[] bigger = new int[data.length * 2];
            System.arraycopy(data, 0, bigger, 0, size);
            data = bigger;
        }

        int get(int index) { check(index); return data[index]; }
        void set(int index, int value) { check(index); data[index] = value; }
        int popBack() {
            if (size == 0) throw new IllegalStateException("empty");
            return data[--size];
        }
        private void check(int index) {
            if (index < 0 || index >= size) throw new IndexOutOfBoundsException();
        }
        int size() { return size; }
        int capacity() { return data.length; }
    }

    public static void main(String[] args) {
        DynamicArray a = new DynamicArray();
        for (int i = 0; i < 5; i++) a.pushBack(i * 10);
        check(a.size() == 5, "size 5");
        check(a.capacity() == 8, "capacity doubled");
        check(a.get(3) == 30, "get");
        a.set(0, 99);
        check(a.get(0) == 99, "set");
        check(a.popBack() == 40 && a.size() == 4, "pop");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
