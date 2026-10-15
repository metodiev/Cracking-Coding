/** Task 160 - Implement LRU cache using linked list + hashmap. (Tier: Beginner, CTCI Ch 7) */
import java.util.*;

class LRUNode {
    int key, value;
    LRUNode prev, next;
    LRUNode(int key, int value) { this.key = key; this.value = value; }
}

/**
 * O(1) get/put with a hash map from key to node plus a hand-rolled doubly linked
 * list: most recently used at the front, least recently used at the back.
 * Sentinel head/tail nodes remove all empty-list edge cases; no built-in ordered
 * map is used - the list is implemented from scratch.
 */
class LRUCache {
    private final int capacity;
    private final Map<Integer, LRUNode> map = new HashMap<>();
    private final LRUNode head = new LRUNode(0, 0);   // most-recent sentinel
    private final LRUNode tail = new LRUNode(0, 0);   // least-recent sentinel

    LRUCache(int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be >= 1");
        this.capacity = capacity;
        head.next = tail;
        tail.prev = head;
    }

    /** Return the value, or -1 if missing; a hit marks the key most recently used. */
    int get(int key) {
        LRUNode node = map.get(key);
        if (node == null) return -1;
        unlink(node);
        pushFront(node);
        return node.value;
    }

    /** Insert or update; updates refresh recency; a full cache evicts the LRU entry. */
    void put(int key, int value) {
        LRUNode node = map.get(key);
        if (node != null) {
            node.value = value;
            unlink(node);
            pushFront(node);
            return;
        }
        if (map.size() >= capacity) {
            LRUNode lru = tail.prev;
            unlink(lru);
            map.remove(lru.key);
        }
        node = new LRUNode(key, value);
        map.put(key, node);
        pushFront(node);
    }

    int size() {
        return map.size();
    }

    private void unlink(LRUNode node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void pushFront(LRUNode node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }
}

public class Solution {
    public static void main(String[] args) {
        LRUCache cache = new LRUCache(2);
        cache.put(1, 1);
        cache.put(2, 2);
        check(cache.get(1) == 1, "get 1");
        cache.put(3, 3);                                  // evicts key 2
        check(cache.get(2) == -1, "2 evicted");
        cache.put(4, 4);                                  // evicts key 1
        check(cache.get(1) == -1, "1 evicted");
        check(cache.get(3) == 3, "get 3");
        check(cache.get(4) == 4, "get 4");

        // updating an existing key refreshes recency and does not grow the cache
        LRUCache c2 = new LRUCache(2);
        c2.put(1, 1);
        c2.put(2, 2);
        c2.put(1, 10);                                    // key 1 is now most recent
        check(c2.size() == 2, "size unchanged after update");
        c2.put(3, 3);                                     // evicts key 2, NOT key 1
        check(c2.get(2) == -1, "2 evicted after update");
        check(c2.get(1) == 10, "1 kept with new value");
        check(c2.get(3) == 3, "3 present");

        // capacity 1 keeps only the newest
        LRUCache c3 = new LRUCache(1);
        c3.put(1, 1);
        c3.put(2, 2);
        check(c3.get(1) == -1 && c3.get(2) == 2, "capacity 1");

        // get() refreshes recency too
        LRUCache c4 = new LRUCache(2);
        c4.put(1, 1);
        c4.put(2, 2);
        check(c4.get(1) == 1, "hit refreshes");
        c4.put(3, 3);                                     // evicts key 2
        check(c4.get(2) == -1 && c4.get(1) == 1 && c4.get(3) == 3, "get refreshed order");

        // a miss returns -1 and does not disturb the recency order
        LRUCache c5 = new LRUCache(2);
        c5.put(1, 1);
        c5.put(2, 2);
        check(c5.get(99) == -1, "miss is -1");
        c5.put(3, 3);                                     // still evicts key 1
        check(c5.get(1) == -1 && c5.get(2) == 2 && c5.get(3) == 3, "miss did not refresh");

        boolean threw = false;
        try { new LRUCache(0); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "capacity 0 throws");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
