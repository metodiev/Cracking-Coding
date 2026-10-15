/** Task 251 - Implement skip list insert/search/delete. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class SkipNode {
    int key;
    int level;
    SkipNode[] next;

    SkipNode(int key, int level) {
        this.key = key;
        this.level = level;
        this.next = new SkipNode[level];
    }
}

class SkipList {
    static final int MAX_LEVEL = 6;
    final SkipNode head = new SkipNode(-1, MAX_LEVEL);
    int size = 0;
    long state;

    SkipList(long seed) {
        state = seed;
    }

    /** Deterministic coin: 31-bit LCG, bit 16 is the flip. */
    private int coin() {
        state = (state * 1103515245L + 12345) & 0x7FFFFFFFL;
        return (int) ((state >> 16) & 1);
    }

    private int randomLevel() {
        int level = 1;
        while (level < MAX_LEVEL && coin() == 1) level++;
        return level;
    }

    /** Last node with key < target at every level. */
    private SkipNode[] find(int key) {
        SkipNode[] update = new SkipNode[MAX_LEVEL];
        SkipNode node = head;
        for (int i = MAX_LEVEL - 1; i >= 0; i--) {
            while (node.next[i] != null && node.next[i].key < key) node = node.next[i];
            update[i] = node;
        }
        return update;
    }

    boolean insert(int key) {
        SkipNode[] update = find(key);
        SkipNode after = update[0].next[0];
        if (after != null && after.key == key) return false;
        SkipNode node = new SkipNode(key, randomLevel());
        for (int i = 0; i < node.level; i++) {
            node.next[i] = update[i].next[i];
            update[i].next[i] = node;
        }
        size++;
        return true;
    }

    boolean search(int key) {
        SkipNode node = head;
        for (int i = MAX_LEVEL - 1; i >= 0; i--) {
            while (node.next[i] != null && node.next[i].key < key) node = node.next[i];
            if (node.next[i] != null && node.next[i].key == key) return true;
        }
        return false;
    }

    boolean delete(int key) {
        SkipNode[] update = find(key);
        SkipNode target = update[0].next[0];
        if (target == null || target.key != key) return false;
        for (int i = 0; i < target.level; i++) update[i].next[i] = target.next[i];
        size--;
        return true;
    }

    List<Integer> walk(int level) {
        List<Integer> out = new ArrayList<>();
        for (SkipNode node = head.next[level]; node != null; node = node.next[level]) out.add(node.key);
        return out;
    }

    /** Heights of the level-0 nodes, in sorted walk order. */
    int[] walkHeights() {
        List<Integer> heights = new ArrayList<>();
        for (SkipNode node = head.next[0]; node != null; node = node.next[0]) heights.add(node.level);
        int[] out = new int[heights.size()];
        for (int i = 0; i < out.length; i++) out[i] = heights.get(i);
        return out;
    }

    /** {key -> height} joined for message output (not used in assertions). */
    Map<Integer, Integer> heights() {
        Map<Integer, Integer> out = new LinkedHashMap<>();
        for (SkipNode node = head.next[0]; node != null; node = node.next[0]) out.put(node.key, node.level);
        return out;
    }
}

public class Solution {
    static final int[] KEYS = {3, 6, 7, 9, 12, 19, 17, 26, 21, 25};
    // Deterministic promotion bits for seed 20251 over KEYS, in sorted walk order.
    static final int[] HEIGHTS = {1, 1, 1, 1, 1, 4, 1, 6, 1, 4};

    public static void main(String[] args) {
        SkipList sl = new SkipList(20251);
        for (int key : KEYS) check(sl.insert(key), "insert " + key);
        check(sl.size == 10, "size after inserts");
        check(Arrays.equals(sl.walkHeights(), HEIGHTS), "identical structure everywhere");
        check(Arrays.stream(HEIGHTS).max().getAsInt() == SkipList.MAX_LEVEL, "top lane used");
        check(sl.walk(0).equals(Arrays.asList(3, 6, 7, 9, 12, 17, 19, 21, 25, 26)),
                "level 0 walk sorted");

        for (int key : KEYS) check(sl.search(key), "search hit " + key);
        for (int key : new int[]{-1, 0, 5, 10, 20, 22, 30}) check(!sl.search(key), "search miss " + key);

        int[] heights = sl.walkHeights();
        List<Integer> walk0 = sl.walk(0);
        for (int i = 0; i < SkipList.MAX_LEVEL; i++) {          // index consistency
            List<Integer> expected = new ArrayList<>();
            for (int j = 0; j < walk0.size(); j++) if (heights[j] > i) expected.add(walk0.get(j));
            List<Integer> chain = sl.walk(i);
            check(chain.equals(expected), "level " + i + " is a sorted subsequence of level 0");
        }

        check(!sl.insert(3) && sl.size == 10, "duplicates rejected");

        check(sl.delete(19) && sl.size == 9, "delete 19");
        check(!sl.search(19), "19 gone");
        check(!sl.delete(19), "second delete fails");
        check(!sl.delete(4) && !sl.delete(100), "missing deletes fail");
        heights = sl.walkHeights();
        walk0 = sl.walk(0);
        for (int i = 0; i < SkipList.MAX_LEVEL; i++) {
            List<Integer> expected = new ArrayList<>();
            for (int j = 0; j < walk0.size(); j++) if (heights[j] > i) expected.add(walk0.get(j));
            check(sl.walk(i).equals(expected), "consistency after delete");
        }

        check(sl.delete(3) && sl.delete(25), "head-adjacent and near-tail keys");
        check(sl.delete(21), "top-lane node");
        for (int key : new int[]{6, 7, 9, 12, 17, 26}) check(sl.delete(key), "delete " + key);
        check(sl.size == 0, "empty");
        for (int i = 0; i < SkipList.MAX_LEVEL; i++) check(sl.walk(i).isEmpty(), "lane drained");
        for (int key : KEYS) check(!sl.search(key), "no stale key " + key);

        check(sl.insert(42) && sl.search(42) && sl.size == 1, "reuse after empty");
        check(sl.delete(42) && sl.size == 0, "drain again");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
