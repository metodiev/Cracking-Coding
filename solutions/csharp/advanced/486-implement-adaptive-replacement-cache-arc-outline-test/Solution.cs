// Task 486 - Implement adaptive replacement cache (ARC) outline & test. (Tier: Advanced, CTCI Ch 14)
//
// ARC outline (Megiddo & Modha 2003), capacity c:
// - Real lists: T1 (keys seen once recently), T2 (keys seen at least twice). Cache contents = T1+T2,
//   with |T1| + |T2| <= c.
// - Ghost lists: B1 (recently evicted from T1), B2 (evicted from T2); |T1| + |B1| <= c and
//   |T1|+|T2|+|B1|+|B2| <= 2c.
// - A request found in T1 is promoted to T2; in T2 it is refreshed (MRU). A request found in a
//   ghost list adapts the target size p for T1: p grows on a B1 hit, shrinks on a B2 hit, always
//   clamped to [0, c]; the hit key is admitted to T2. A brand-new request is admitted to T1, first
//   discarding a ghost or demoting the LRU of T1/T2 via Replace (T1 when |T1| > p, else T2).
//
// Adaptivity test:
// - Trace S ("scan-resistance", 20 rounds of [hot 0..3, 8-key scan, hot 0..3], scan keys cycling
//   mod 8): LRU thrashes on the scans; pinned ARC 133 vs LRU 76 hits.
// - Trace A ("adaptivity", same rounds but 4 rotating scan keys): ghost hits drive p up to c = 4
//   and back down; pinned p_max = 4, 4 B1-hits, 141 B2-hits, ARC 79 vs LRU 76.
//
// Correctness: on every access the requested key is in the cache, the four lists are pairwise
// disjoint with the documented size bounds honoured, and per-step cache contents/hits match an
// independent list-based ARC reference implementation.
using System;
using System.Collections.Generic;

internal class ARCCache
{
    public readonly int C;
    public int P;
    public readonly LinkedList<int> T1 = new LinkedList<int>();
    public readonly LinkedList<int> T2 = new LinkedList<int>();
    public readonly LinkedList<int> B1 = new LinkedList<int>();
    public readonly LinkedList<int> B2 = new LinkedList<int>();
    public int Hits;
    public int PMax;
    public int PMin;
    public int B1Hits;
    public int B2Hits;

    public ARCCache(int capacity)
    {
        C = capacity;
    }

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

    public void Replace(int key)
    {
        if (T1.Count > 0 && ((Contains(B2, key) && T1.Count == P) || T1.Count > P))
        {
            B1.AddLast(T1.First.Value);
            T1.RemoveFirst();
        }
        else if (T2.Count > 0)
        {
            B2.AddLast(T2.First.Value);
            T2.RemoveFirst();
        }
    }

    public bool Access(int key)
    {
        if (Contains(T1, key))
        {
            RemoveKey(T1, key);
            T2.AddLast(key);
            Hits++;
            return true;
        }
        if (Contains(T2, key))
        {
            RemoveKey(T2, key);
            T2.AddLast(key);
            Hits++;
            return true;
        }
        if (Contains(B1, key))
        {
            P = Math.Min(C, P + Math.Max(1, B2.Count / B1.Count));
            B1Hits++;
            Replace(key);
            RemoveKey(B1, key);
            T2.AddLast(key);
        }
        else if (Contains(B2, key))
        {
            P = Math.Max(0, P - Math.Max(1, B1.Count / B2.Count));
            B2Hits++;
            Replace(key);
            RemoveKey(B2, key);
            T2.AddLast(key);
        }
        else
        {
            if (T1.Count + B1.Count == C)
            {
                if (T1.Count < C)
                {
                    B1.RemoveFirst();
                    Replace(key);
                }
                else
                {
                    T1.RemoveFirst();
                }
            }
            else if (T1.Count + B1.Count < C)
            {
                int total = T1.Count + T2.Count + B1.Count + B2.Count;
                if (total >= C)
                {
                    if (total == 2 * C) B2.RemoveFirst();
                    Replace(key);
                }
            }
            T1.AddLast(key);
        }
        PMax = Math.Max(PMax, P);
        PMin = Math.Min(PMin, P);
        return false;
    }

    public List<int> Contents()
    {
        var keys = new List<int>(T1);
        keys.AddRange(T2);
        keys.Sort();
        return keys;
    }

    public void CheckInvariants(int key, int step)
    {
        if (!Contains(T1, key) && !Contains(T2, key)) throw new Exception("FAILED: key absent " + step);
        if (T1.Count + T2.Count > C) throw new Exception("FAILED: cache bound " + step);
        if (T1.Count + B1.Count > C) throw new Exception("FAILED: t1+b1 bound " + step);
        if (T1.Count + T2.Count + B1.Count + B2.Count > 2 * C) throw new Exception("FAILED: total bound " + step);
        if (P < 0 || P > C) throw new Exception("FAILED: p range " + step);
        var lists = new[] { T1, T2, B1, B2 };
        for (int i = 0; i < 4; i++)
        {
            for (int j = i + 1; j < 4; j++)
            {
                foreach (int value in lists[i])
                {
                    if (Contains(lists[j], value)) throw new Exception("FAILED: lists overlap " + step);
                }
            }
        }
    }
}

public static class Solution
{
    private static int ArcReference(int[] trace, int capacity, List<List<int>> contents)
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
            var snapshot = new List<int>(t1);
            snapshot.AddRange(t2);
            snapshot.Sort();
            contents.Add(snapshot);
        }
        return hits;
    }

    private static int LruHits(int[] trace, int capacity)
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

    private static int[] ScanTrace(int rounds, int hot, int scanWidth, int cold)
    {
        var trace = new int[rounds * (2 * hot + scanWidth)];
        int at = 0;
        for (int r = 0; r < rounds; r++)
        {
            for (int i = 0; i < hot; i++) trace[at++] = i;
            for (int j = 0; j < scanWidth; j++) trace[at++] = 100 + (r * scanWidth + j) % cold;
            for (int i = 0; i < hot; i++) trace[at++] = i;
        }
        return trace;
    }

    private static bool SameLists(List<List<int>> a, List<List<int>> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i].Count != b[i].Count) return false;
            for (int j = 0; j < a[i].Count; j++) if (a[i][j] != b[i][j]) return false;
        }
        return true;
    }

    private static ARCCache VerifyTrace(int[] trace, int capacity, int expectedHits, int expectedLru, string label)
    {
        var cache = new ARCCache(capacity);
        var contentsVector = new List<List<int>>();
        int summedHits = 0;
        for (int step = 0; step < trace.Length; step++)
        {
            if (cache.Access(trace[step])) summedHits++;
            cache.CheckInvariants(trace[step], step);
            contentsVector.Add(cache.Contents());
        }
        var referenceContents = new List<List<int>>();
        int referenceHits = ArcReference(trace, capacity, referenceContents);
        if (cache.Hits != referenceHits || cache.Hits != expectedHits)
        {
            throw new Exception("FAILED: " + label + " hits " + cache.Hits + " ref " + referenceHits);
        }
        if (summedHits != cache.Hits) throw new Exception("FAILED: " + label + " hit vector");
        if (!SameLists(contentsVector, referenceContents)) throw new Exception("FAILED: " + label + " contents vector");
        int lru = LruHits(trace, capacity);
        if (lru != expectedLru) throw new Exception("FAILED: " + label + " lru " + lru);
        if (cache.Hits < lru) throw new Exception("FAILED: " + label + " ARC < LRU");
        return cache;
    }

    public static void Main()
    {
        var cache = new ARCCache(2);
        var observed = new bool[5];
        var hand = new[] { 1, 2, 1, 3, 1 };
        for (int i = 0; i < hand.Length; i++) observed[i] = cache.Access(hand[i]);
        var expected = new[] { false, false, true, false, true };
        for (int i = 0; i < expected.Length; i++)
        {
            if (observed[i] != expected[i]) throw new Exception("FAILED: hand vector");
        }
        if (cache.Hits != 2) throw new Exception("FAILED: hand hits");
        var handContents = cache.Contents();
        if (handContents.Count != 2 || handContents[0] != 1 || handContents[1] != 3) throw new Exception("FAILED: hand state");

        var scan = ScanTrace(20, 4, 8, 8);
        if (scan.Length != 320 || scan[0] != 0 || scan[1] != 1 || scan[2] != 2 || scan[3] != 3) throw new Exception("FAILED: scan trace");
        var scanCache = VerifyTrace(scan, 4, 133, 76, "scan-resistance");
        var scanFinal = scanCache.Contents();
        if (scanFinal.Count != 4 || scanFinal[0] != 0 || scanFinal[1] != 1 || scanFinal[2] != 2 || scanFinal[3] != 3)
        {
            throw new Exception("FAILED: scan final cache");
        }

        var adaptive = ScanTrace(20, 4, 4, 4);
        if (adaptive.Length != 240) throw new Exception("FAILED: adaptive length");
        var adaptiveCache = VerifyTrace(adaptive, 4, 79, 76, "adaptivity");
        if (adaptiveCache.PMax != 4 || adaptiveCache.PMin != 0 || adaptiveCache.P != 0) throw new Exception("FAILED: p trajectory");
        if (adaptiveCache.B1Hits != 4 || adaptiveCache.B2Hits != 141) throw new Exception("FAILED: ghost hits");
        var adaptiveFinal = adaptiveCache.Contents();
        if (adaptiveFinal.Count != 4 || adaptiveFinal[0] != 0 || adaptiveFinal[1] != 1 || adaptiveFinal[2] != 2 || adaptiveFinal[3] != 3)
        {
            throw new Exception("FAILED: adaptive final cache");
        }
        Console.WriteLine("PASS");
    }
}
