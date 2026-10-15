/** Task 160 - Implement LRU cache using linked list + hashmap. (Tier: Beginner, CTCI Ch 7) */
"use strict";

class LRUNode {
  constructor(key, value) {
    this.key = key;
    this.value = value;
    this.prev = null;
    this.next = null;
  }
}

/**
 * O(1) get/put with a hash map from key to node plus a hand-rolled doubly linked
 * list: most recently used at the front, least recently used at the back.
 * Sentinel head/tail nodes remove all empty-list edge cases; no built-in ordered
 * map is used - the list is implemented from scratch.
 */
class LRUCache {
  constructor(capacity) {
    if (capacity <= 0) throw new RangeError("capacity must be >= 1");
    this.capacity = capacity;
    this.map = new Map();
    this.head = new LRUNode(0, 0);        // most-recent sentinel
    this.tail = new LRUNode(0, 0);        // least-recent sentinel
    this.head.next = this.tail;
    this.tail.prev = this.head;
  }

  /** Return the value, or -1 if missing; a hit marks the key most recently used. */
  get(key) {
    const node = this.map.get(key);
    if (!node) return -1;
    this.unlink(node);
    this.pushFront(node);
    return node.value;
  }

  /** Insert or update; updates refresh recency; a full cache evicts the LRU entry. */
  put(key, value) {
    let node = this.map.get(key);
    if (node) {
      node.value = value;
      this.unlink(node);
      this.pushFront(node);
      return;
    }
    if (this.map.size >= this.capacity) {
      const lru = this.tail.prev;
      this.unlink(lru);
      this.map.delete(lru.key);
    }
    node = new LRUNode(key, value);
    this.map.set(key, node);
    this.pushFront(node);
  }

  get size() {
    return this.map.size;
  }

  unlink(node) {
    node.prev.next = node.next;
    node.next.prev = node.prev;
  }

  pushFront(node) {
    node.prev = this.head;
    node.next = this.head.next;
    this.head.next.prev = node;
    this.head.next = node;
  }
}

module.exports = { LRUNode, LRUCache };

if (require.main === module) {
  const assert = require("node:assert");
  const cache = new LRUCache(2);
  cache.put(1, 1);
  cache.put(2, 2);
  assert.strictEqual(cache.get(1), 1);
  cache.put(3, 3);                                    // evicts key 2
  assert.strictEqual(cache.get(2), -1);
  cache.put(4, 4);                                    // evicts key 1
  assert.strictEqual(cache.get(1), -1);
  assert.strictEqual(cache.get(3), 3);
  assert.strictEqual(cache.get(4), 4);

  // updating an existing key refreshes recency and does not grow the cache
  const c2 = new LRUCache(2);
  c2.put(1, 1);
  c2.put(2, 2);
  c2.put(1, 10);                                      // key 1 is now most recent
  assert.strictEqual(c2.size, 2);
  c2.put(3, 3);                                       // evicts key 2, NOT key 1
  assert.strictEqual(c2.get(2), -1);
  assert.strictEqual(c2.get(1), 10);
  assert.strictEqual(c2.get(3), 3);

  // capacity 1 keeps only the newest
  const c3 = new LRUCache(1);
  c3.put(1, 1);
  c3.put(2, 2);
  assert.strictEqual(c3.get(1), -1);
  assert.strictEqual(c3.get(2), 2);

  // get() refreshes recency too
  const c4 = new LRUCache(2);
  c4.put(1, 1);
  c4.put(2, 2);
  assert.strictEqual(c4.get(1), 1);                   // key 2 becomes LRU
  c4.put(3, 3);                                       // evicts key 2
  assert.strictEqual(c4.get(2), -1);
  assert.strictEqual(c4.get(1), 1);
  assert.strictEqual(c4.get(3), 3);

  // a miss returns -1 and does not disturb the recency order
  const c5 = new LRUCache(2);
  c5.put(1, 1);
  c5.put(2, 2);
  assert.strictEqual(c5.get(99), -1);
  c5.put(3, 3);                                       // still evicts key 1
  assert.strictEqual(c5.get(1), -1);
  assert.strictEqual(c5.get(2), 2);
  assert.strictEqual(c5.get(3), 3);

  assert.throws(() => new LRUCache(0), RangeError);
  console.log("PASS");
}
