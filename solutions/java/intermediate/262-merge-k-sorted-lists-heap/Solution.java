/** Task 262 - Merge K sorted lists (heap). (Tier: Intermediate, CTCI Ch 10) */
import java.util.*;

class ListNode {
    int val;
    ListNode next;
    ListNode(int v) { val = v; }
}

public class Solution {
    /** Heap entry: value ordering first, then an insertion serial for a deterministic tie-break. */
    static class Entry {
        final ListNode node;
        final int serial;
        Entry(ListNode node, int serial) { this.node = node; this.serial = serial; }
    }

    static ListNode buildList(int... vals) {
        ListNode head = null;
        for (int i = vals.length - 1; i >= 0; i--) {
            ListNode n = new ListNode(vals[i]);
            n.next = head;
            head = n;
        }
        return head;
    }

    static List<Integer> toArray(ListNode head) {
        List<Integer> out = new ArrayList<>();
        for (ListNode p = head; p != null; p = p.next) out.add(p.val);
        return out;
    }

    /**
     * Min-heap over the current heads: each of the n nodes enters the heap once, so
     * O(n log k) time / O(k) space; tails are rewired in place. Returns null for k = 0
     * and the same head object when k = 1.
     */
    static ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<Entry> pq = new PriorityQueue<>((e1, e2) ->
                e1.node.val != e2.node.val ? Integer.compare(e1.node.val, e2.node.val)
                                           : Integer.compare(e1.serial, e2.serial));
        int serial = 0;
        for (ListNode node : lists) if (node != null) pq.add(new Entry(node, serial++));
        ListNode dummy = new ListNode(0);
        ListNode tail = dummy;
        while (!pq.isEmpty()) {
            ListNode node = pq.poll().node;
            tail.next = node;
            tail = node;
            if (node.next != null) pq.add(new Entry(node.next, serial++));
        }
        tail.next = null;
        return dummy.next;
    }

    /** Flat sort reference used only for cross-checks; call BEFORE merging (merging rewires tails). */
    static int[] mergeOracle(ListNode[] lists) {
        List<Integer> values = new ArrayList<>();
        for (ListNode node : lists) values.addAll(toArray(node));
        Collections.sort(values);
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
    }

    static void checkArray(ListNode head, int[] expected, String label) {
        List<Integer> got = toArray(head);
        boolean ok = got.size() == expected.length;
        for (int i = 0; ok && i < expected.length; i++) ok = got.get(i) == expected[i];
        check(ok, label + " got=" + got);
    }

    public static void main(String[] args) {
        ListNode merged = mergeKLists(new ListNode[]{buildList(1, 4, 5), buildList(1, 3, 4), buildList(2, 6)});
        checkArray(merged, new int[]{1, 1, 2, 3, 4, 4, 5, 6}, "classic");

        check(mergeKLists(new ListNode[0]) == null, "k = 0 -> empty");
        check(mergeKLists(new ListNode[]{null, null}) == null, "only empty lists");

        ListNode single = buildList(1, 2, 3);
        ListNode same = mergeKLists(new ListNode[]{single});
        check(same == single, "k = 1 returns the same list");
        checkArray(same, new int[]{1, 2, 3}, "single list kept");

        checkArray(mergeKLists(new ListNode[]{buildList(1), buildList(2, 3, 4, 5), buildList(6)}),
                new int[]{1, 2, 3, 4, 5, 6}, "unequal lengths");
        checkArray(mergeKLists(new ListNode[]{buildList(1, 1), buildList(1, 1)}),
                new int[]{1, 1, 1, 1}, "duplicates");
        checkArray(mergeKLists(new ListNode[]{null, buildList(1), null, buildList()}),
                new int[]{1}, "empty lists mixed in");
        checkArray(mergeKLists(new ListNode[]{buildList(-3, -1), buildList(-2, 0)}),
                new int[]{-3, -2, -1, 0}, "negatives");
        checkArray(mergeKLists(new ListNode[]{buildList(5, 5), buildList(5), buildList()}),
                new int[]{5, 5, 5}, "repeats with an empty tail");

        Random rng = new Random(262);
        for (int test = 0; test < 300; test++) {        // cross-check vs the flat-sort oracle
            int k = rng.nextInt(6);
            ListNode[] lists = new ListNode[k];
            for (int i = 0; i < k; i++) {
                int len = rng.nextInt(6);
                int[] values = new int[len];
                for (int j = 0; j < len; j++) values[j] = rng.nextInt(13) - 6;
                Arrays.sort(values);
                lists[i] = buildList(values);
            }
            int[] expected = mergeOracle(lists);
            checkArray(mergeKLists(lists), expected, "random agreement");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
