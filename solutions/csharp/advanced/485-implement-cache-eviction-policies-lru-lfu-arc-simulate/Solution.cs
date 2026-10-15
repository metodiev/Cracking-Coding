// Task 485 - Implement cache eviction policies: LRU, LFU, ARC (simulate). (Tier: Advanced, CTCI Ch 14)
//
// Documented capacity semantics: `capacity` c is the maximum number of distinct keys the cache may
// hold (cache-as-main-memory, no separate disk tiers here). Every access returns hit/miss and the
// cache size never exceeds c.
//
// Policy definitions (the "simulator"):
// - LRU: a hit moves the key to the most-recently-used end; a miss inserts at MRU and, when the
//   cache is over capacity, evicts the least-recently-used key.
// - LFU: every hit increments the key's count and refreshes its recency; a miss on a full cache
//   evicts the key with the smallest count, ties broken by least-recently-used (documented rule).
// - ARC (Megiddo & Modha): p-adaptive split of T1 (recent) / T2 (frequent); misses that hit ghost
//   lists B1/B2 adjust p, and Replace demotes within real lists. Cache contents = T1 + T2 with
//   |T1| + |T2| <= c; B1/B2 hold only ghosts.
//
// Each policy is run against an independent naive reference implementation (Lists + linear scans)
// on three fixed LCG-derived traces; hit totals must be identical, and the totals are also pinned
// as constants verified by the oracle.
using System;
using System.Collections.Generic;

internal interface ICachePolicy
{
    string Name();
    bool Access(int key);
    int Size();
    List<int> Contents();
    int Hits();
}

internal class LRUCache : ICachePolicy
{
    private readonly int capacity;
    private readonly LinkedList<int> order = new LinkedList<int>();
    private int hitCount;

    public LRUCache(int capacity)
    {
        this.capacity = capacity;
    }

    public string Name() => "LRU";

    public bool Access(int key)
    {
        LinkedListNode<int> node = FindNode(key);
        if (node != null)
        {
            order.Remove(node);
            order.AddLast(key);
            hitCount++;
            return true;
        }
        order.AddLast(key);
        if (order.Count > capacity) order.RemoveFirst();
        return false;
    }

    private LinkedListNode<int> FindNode(int key)
    {
        for (LinkedListNode<int> node = order.First; node != null; node = node.Next)
        {
            if (node.Value == key) return node;
        }
        return null;
    }

    public int Size() => order.Count;

    public List<int> Contents()
    {
        var keys = new List<int>(order);
        keys.Sort();
        return keys;
    }

    public int Hits() => hitCount;
}

internal class LFUCache : ICachePolicy
{
    private readonly int capacity;
    private readonly Dictionary<int, int> counts = new Dictionary<int, int>();
    private readonly Dictionary<int, int> recency = new Dictionary<int, int>();
    private int tick;
    private int hitCount;

    public LFUCache(int capacity)
    {
        this.capacity = capacity;
    }

    public string Name() => "LFU";

    public bool Access(int key)
    {
        tick++;
        if (counts.ContainsKey(key))
        {
            counts[key]++;
            recency[key] = tick;
            hitCount++;
            return true;
        }
        if (counts.Count >= capacity)
        {
            int victim = 0;
            long best = long.MaxValue;
            foreach (int candidate in counts.Keys)
            {
                long score = counts[candidate] * 1000000L + recency[candidate];
                if (score < best)
                {
                    best = score;
                    victim = candidate;
                }
            }
            counts.Remove(victim);
            recency.Remove(victim);
        }
        counts[key] = 1;
        recency[key] = tick;
        return false;
    }

    public int Size() => counts.Count;

    public List<int> Contents()
    {
        var keys = new List<int>(counts.Keys);
        keys.Sort();
        return keys;
    }

    public int Hits() => hitCount;
}

internal class ARCCache : ICachePolicy
{
    private readonly int capacity;
    private int p;
    private readonly LinkedList<int> t1 = new LinkedList<int>();
    private readonly LinkedList<int> t2 = new LinkedList<int>();
    private readonly LinkedList<int> b1 = new LinkedList<int>();
    private readonly LinkedList<int> b2 = new LinkedList<int>();
    private int hitCount;

    public ARCCache(int capacity)
    {
        this.capacity = capacity;
    }

    public string Name() => "ARC";

    private static bool Contains(LinkedList<int> list, int key)
    {
        for (LinkedListNode<int> node = list.First; node != null; node = node.Next)
        {
            if (node.Value == key) return true;
        }
        return false;
    }

    private static void RemoveKey(LinkedList<int> list, int key)
    {
        for (LinkedListNode<int> node = list.First; node != null; node = node.Next)
        {
            if (node.Value == key)
            {
                list.Remove(node);
                return;
            }
        }
    }

    private void Replace(int key)
    {
        if (t1.Count > 0 && ((Contains(b2, key) && t1.Count == p) || t1.Count > p))
        {
            b1.AddLast(t1.First.Value);
            t1.RemoveFirst();
        }
        else if (t2.Count > 0)
        {
            b2.AddLast(t2.First.Value);
            t2.RemoveFirst();
        }
    }

    public bool Access(int key)
    {
        if (Contains(t1, key))
        {
            RemoveKey(t1, key);
            t2.AddLast(key);
            hitCount++;
            return true;
        }
        if (Contains(t2, key))
        {
            RemoveKey(t2, key);
            t2.AddLast(key);
            hitCount++;
            return true;
        }
        if (Contains(b1, key))
        {
            p = Math.Min(capacity, p + Math.Max(1, b2.Count / b1.Count));
            Replace(key);
            RemoveKey(b1, key);
            t2.AddLast(key);
        }
        else if (Contains(b2, key))
        {
            p = Math.Max(0, p - Math.Max(1, b1.Count / b2.Count));
            Replace(key);
            RemoveKey(b2, key);
            t2.AddLast(key);
        }
        else
        {
            if (t1.Count + b1.Count == capacity)
            {
                if (t1.Count < capacity)
                {
                    b1.RemoveFirst();
                    Replace(key);
                }
                else
                {
                    t1.RemoveFirst();
                }
            }
            else if (t1.Count + b1.Count < capacity)
            {
                int total = t1.Count + t2.Count + b1.Count + b2.Count;
                if (total >= capacity)
                {
                    if (total == 2 * capacity) b2.RemoveFirst();
                    Replace(key);
                }
            }
            t1.AddLast(key);
        }
        return false;
    }

    public int Size() => t1.Count + t2.Count;

    public List<int> Contents()
    {
        var keys = new List<int>(t1);
        keys.AddRange(t2);
        keys.Sort();
        return keys;
    }

    public int Hits() => hitCount;
}

public static class Solution
{
    private static int seed = 485;

    private static int[] UniformTrace(int count, int keyCount)
    {
        var trace = new int[count];
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            trace[i] = seed % keyCount;
        }
        return trace;
    }

    private static int[] LocalityTrace(int count, int keyCount, int repeatMod)
    {
        var trace = new int[count];
        for (int i = 0; i < count; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            if (i > 0 && seed % 10 < repeatMod) trace[i] = trace[i - 1];
            else trace[i] = seed % keyCount;
        }
        return trace;
    }

    private static int[] HotScanTrace(int rounds, int hot, int scanWidth)
    {
        var trace = new int[rounds * (2 * hot + scanWidth)];
        int at = 0;
        for (int block = 0; block < rounds; block++)
        {
            for (int i = 0; i < hot; i++) trace[at++] = i;
            for (int i = 0; i < scanWidth; i++) trace[at++] = 100 + block * scanWidth + i;
            for (int i = 0; i < hot; i++) trace[at++] = i;
        }
        return trace;
    }

    private static int LruReference(int[] trace, int capacity)
    {
        var order = new List<int>();
        int hits = 0;
        foreach (int key in trace)
        {
            if (order.Contains(key))
            {
                order.Remove(key);
                order.Add(key);
                hits++;
            }
            else
            {
                order.Add(key);
                if (order.Count > capacity) order.RemoveAt(0);
            }
        }
        return hits;
    }

    private static int LfuReference(int[] trace, int capacity)
    {
        var counts = new Dictionary<int, int>();
        var recency = new Dictionary<int, int>();
        int tick = 0, hits = 0;
        foreach (int key in trace)
        {
            tick++;
            if (counts.ContainsKey(key))
            {
                counts[key]++;
                recency[key] = tick;
                hits++;
            }
            else
            {
                if (counts.Count >= capacity)
                {
                    int victim = 0;
                    long best = long.MaxValue;
                    foreach (int candidate in counts.Keys)
                    {
                        long score = counts[candidate] * 1000000L + recency[candidate];
                        if (score < best)
                        {
                            best = score;
                            victim = candidate;
                        }
                    }
                    counts.Remove(victim);
                    recency.Remove(victim);
                }
                counts[key] = 1;
                recency[key] = tick;
            }
        }
        return hits;
    }

    private static int ArcReference(int[] trace, int capacity)
    {
        int p = 0, hits = 0;
        var t1 = new List<int>();
        var t2 = new List<int>();
        var b1 = new List<int>();
        var b2 = new List<int>();
        foreach (int key in trace)
        {
            if (t1.Contains(key))
            {
                t1.Remove(key);
                t2.Add(key);
                hits++;
            }
            else if (t2.Contains(key))
            {
                t2.Remove(key);
                t2.Add(key);
                hits++;
            }
            else
            {
                if (b1.Contains(key))
                {
                    p = Math.Min(capacity, p + Math.Max(1, b2.Count / b1.Count));
                    if (t1.Count > 0 && ((b2.Contains(key) && t1.Count == p) || t1.Count > p))
                    {
                        b1.Add(t1[0]);
                        t1.RemoveAt(0);
                    }
                    else if (t2.Count > 0)
                    {
                        b2.Add(t2[0]);
                        t2.RemoveAt(0);
                    }
                    b1.Remove(key);
                    t2.Add(key);
                }
                else if (b2.Contains(key))
                {
                    p = Math.Max(0, p - Math.Max(1, b1.Count / b2.Count));
                    if (t1.Count > 0 && ((b2.Contains(key) && t1.Count == p) || t1.Count > p))
                    {
                        b1.Add(t1[0]);
                        t1.RemoveAt(0);
                    }
                    else if (t2.Count > 0)
                    {
                        b2.Add(t2[0]);
                        t2.RemoveAt(0);
                    }
                    b2.Remove(key);
                    t2.Add(key);
                }
                else
                {
                    if (t1.Count + b1.Count == capacity)
                    {
                        if (t1.Count < capacity)
                        {
                            b1.RemoveAt(0);
                            if (t1.Count > 0 && t1.Count > p)
                            {
                                b1.Add(t1[0]);
                                t1.RemoveAt(0);
                            }
                            else if (t2.Count > 0)
                            {
                                b2.Add(t2[0]);
                                t2.RemoveAt(0);
                            }
                        }
                        else
                        {
                            t1.RemoveAt(0);
                        }
                    }
                    else if (t1.Count + b1.Count < capacity)
                    {
                        int total = t1.Count + t2.Count + b1.Count + b2.Count;
                        if (total >= capacity)
                        {
                            if (total == 2 * capacity) b2.RemoveAt(0);
                            if (t1.Count > 0 && ((b2.Contains(key) && t1.Count == p) || t1.Count > p))
                            {
                                b1.Add(t1[0]);
                                t1.RemoveAt(0);
                            }
                            else if (t2.Count > 0)
                            {
                                b2.Add(t2[0]);
                                t2.RemoveAt(0);
                            }
                        }
                    }
                    t1.Add(key);
                }
            }
        }
        return hits;
    }

    private static void RunTrace(ICachePolicy cache, int[] trace, int expected, string label)
    {
        int overflows = 0, lastSize = 0, maxSize = 0, summedHits = 0;
        foreach (int key in trace)
        {
            if (cache.Access(key)) summedHits++;
            int size = cache.Size();
            lastSize = size;
            maxSize = Math.Max(maxSize, size);
            if (size > 4) overflows++;
        }
        Check(cache.Hits() == expected, label + " hits " + cache.Hits());
        Check(summedHits == cache.Hits(), label + " hit vector");
        Check(overflows == 0, label + " capacity respected");
        Check(maxSize == 4 && lastSize == 4, label + " warm capacity");
        var contents = cache.Contents();
        Check(contents.Count == 4 && new HashSet<int>(contents).Count == 4, label + " contents");
    }

    public static void Main()
    {
        var hand = new[] { 1, 2, 1, 3, 1 };
        var expected = new[] { false, false, true, false, true };
        foreach (ICachePolicy cache in new ICachePolicy[] { new LRUCache(2), new LFUCache(2), new ARCCache(2) })
        {
            var observed = new bool[hand.Length];
            for (int i = 0; i < hand.Length; i++) observed[i] = cache.Access(hand[i]);
            Check(observed[0] == expected[0] && observed[1] == expected[1] && observed[2] == expected[2]
                  && observed[3] == expected[3] && observed[4] == expected[4], cache.Name() + " hand vector");
            Check(cache.Hits() == 2, cache.Name() + " hand hits");
            var contents = cache.Contents();
            Check(contents.Count == 2 && contents[0] == 1 && contents[1] == 3, cache.Name() + " hand contents");
        }
        Check(LruReference(hand, 2) == 2 && LfuReference(hand, 2) == 2 && ArcReference(hand, 2) == 2, "hand refs");

        seed = 485;
        var uniform = UniformTrace(400, 20);
        seed = 486;
        var locality = LocalityTrace(400, 24, 7);
        var hotScan = HotScanTrace(24, 4, 8);

        RunTrace(new LRUCache(4), uniform, 70, "uniform LRU");
        RunTrace(new LFUCache(4), uniform, 64, "uniform LFU");
        RunTrace(new ARCCache(4), uniform, 64, "uniform ARC");
        RunTrace(new LRUCache(4), locality, 296, "locality LRU");
        RunTrace(new LFUCache(4), locality, 291, "locality LFU");
        RunTrace(new ARCCache(4), locality, 290, "locality ARC");
        RunTrace(new LRUCache(4), hotScan, 92, "hot-scan LRU");
        RunTrace(new LFUCache(4), hotScan, 161, "hot-scan LFU");
        RunTrace(new ARCCache(4), hotScan, 161, "hot-scan ARC");

        Check(LruReference(uniform, 4) == 70, "uniform LRU ref");
        Check(LfuReference(uniform, 4) == 64, "uniform LFU ref");
        Check(ArcReference(uniform, 4) == 64, "uniform ARC ref");
        Check(LruReference(locality, 4) == 296, "locality LRU ref");
        Check(LfuReference(locality, 4) == 291, "locality LFU ref");
        Check(ArcReference(locality, 4) == 290, "locality ARC ref");
        Check(LruReference(hotScan, 4) == 92, "hot-scan LRU ref");
        Check(LfuReference(hotScan, 4) == 161, "hot-scan LFU ref");
        Check(ArcReference(hotScan, 4) == 161, "hot-scan ARC ref");
        Check(161 > 92, "ARC beats LRU on hot-scan");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
