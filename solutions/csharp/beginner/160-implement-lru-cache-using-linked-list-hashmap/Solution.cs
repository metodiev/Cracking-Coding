// Task 160 - Implement LRU cache using linked list + hashmap. (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;

public class LRUNode
{
    public int Key, Value;
    public LRUNode? Prev, Next;
    public LRUNode(int key, int value) { Key = key; Value = value; }
}

/// <summary>
/// O(1) get/put with a hash map from key to node plus a hand-rolled doubly linked
/// list: most recently used at the front, least recently used at the back.
/// Sentinel head/tail nodes remove all empty-list edge cases; no built-in ordered
/// map is used - the list is implemented from scratch.
/// </summary>
public class LRUCache
{
    private readonly int capacity;
    private readonly Dictionary<int, LRUNode> map = new Dictionary<int, LRUNode>();
    private readonly LRUNode head = new LRUNode(0, 0);   // most-recent sentinel
    private readonly LRUNode tail = new LRUNode(0, 0);   // least-recent sentinel

    public LRUCache(int capacity)
    {
        if (capacity <= 0) throw new ArgumentException("capacity must be >= 1");
        this.capacity = capacity;
        head.Next = tail;
        tail.Prev = head;
    }

    /// <summary>Return the value, or -1 if missing; a hit marks the key most recently used.</summary>
    public int Get(int key)
    {
        if (!map.TryGetValue(key, out var node)) return -1;
        Unlink(node);
        PushFront(node);
        return node.Value;
    }

    /// <summary>Insert or update; updates refresh recency; a full cache evicts the LRU entry.</summary>
    public void Put(int key, int value)
    {
        if (map.TryGetValue(key, out var node))
        {
            node.Value = value;
            Unlink(node);
            PushFront(node);
            return;
        }
        if (map.Count >= capacity)
        {
            var lru = tail.Prev!;
            Unlink(lru);
            map.Remove(lru.Key);
        }
        node = new LRUNode(key, value);
        map[key] = node;
        PushFront(node);
    }

    public int Size => map.Count;

    private void Unlink(LRUNode node)
    {
        node.Prev!.Next = node.Next;
        node.Next!.Prev = node.Prev;
    }

    private void PushFront(LRUNode node)
    {
        node.Prev = head;
        node.Next = head.Next;
        head.Next!.Prev = node;
        head.Next = node;
    }
}

public static class Solution
{
    public static void Main()
    {
        var cache = new LRUCache(2);
        cache.Put(1, 1);
        cache.Put(2, 2);
        Check(cache.Get(1) == 1, "get 1");
        cache.Put(3, 3);                                  // evicts key 2
        Check(cache.Get(2) == -1, "2 evicted");
        cache.Put(4, 4);                                  // evicts key 1
        Check(cache.Get(1) == -1, "1 evicted");
        Check(cache.Get(3) == 3, "get 3");
        Check(cache.Get(4) == 4, "get 4");

        // updating an existing key refreshes recency and does not grow the cache
        var c2 = new LRUCache(2);
        c2.Put(1, 1);
        c2.Put(2, 2);
        c2.Put(1, 10);                                    // key 1 is now most recent
        Check(c2.Size == 2, "size unchanged after update");
        c2.Put(3, 3);                                     // evicts key 2, NOT key 1
        Check(c2.Get(2) == -1, "2 evicted after update");
        Check(c2.Get(1) == 10, "1 kept with new value");
        Check(c2.Get(3) == 3, "3 present");

        // capacity 1 keeps only the newest
        var c3 = new LRUCache(1);
        c3.Put(1, 1);
        c3.Put(2, 2);
        Check(c3.Get(1) == -1 && c3.Get(2) == 2, "capacity 1");

        // Get() refreshes recency too
        var c4 = new LRUCache(2);
        c4.Put(1, 1);
        c4.Put(2, 2);
        Check(c4.Get(1) == 1, "hit refreshes");
        c4.Put(3, 3);                                     // evicts key 2
        Check(c4.Get(2) == -1 && c4.Get(1) == 1 && c4.Get(3) == 3, "get refreshed order");

        // a miss returns -1 and does not disturb the recency order
        var c5 = new LRUCache(2);
        c5.Put(1, 1);
        c5.Put(2, 2);
        Check(c5.Get(99) == -1, "miss is -1");
        c5.Put(3, 3);                                     // still evicts key 1
        Check(c5.Get(1) == -1 && c5.Get(2) == 2 && c5.Get(3) == 3, "miss did not refresh");

        bool threw = false;
        try { new LRUCache(0); } catch (ArgumentException) { threw = true; }
        Check(threw, "capacity 0 throws");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
