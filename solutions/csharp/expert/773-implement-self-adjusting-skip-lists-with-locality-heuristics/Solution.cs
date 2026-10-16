// Task 773 - Implement self-adjusting skip lists with locality heuristics. (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
//
// A skip list is a stack of sorted linked lists built from a deterministic coin stream (splitmix64),
// where a node of height h occurs on levels 0..h-1. A plain skip list assigns that height once, at
// insertion, so every key pays roughly the same descent. The self-adjusting variant implemented here
// reacts to the workload using the sentinel locality heuristic: a key v is cheap to reach exactly when
// the node immediately below v (its level-0 sentinel) is present on the high levels, because the
// descent from the top then lands right next to v and the remaining levels need no forward steps at
// all. Every hit therefore credits the sentinel below the hit key and raises it one level, trading
// places with the coldest node whose top level is exactly that level. Demotion plus promotion means
// the population of every level stays exactly what the coin stream produced, so the shape of the
// structure - and the cost of a lookup far away from the hot keys - is preserved.
//
// Invariants checked after every insertion, deletion and access:
//   * every level is strictly increasing, and the set of nodes visible at level l is exactly the set
//     of level-0 nodes with height > l (so level l+1 is a subsequence of level l);
//   * level 0 holds every key exactly once, the head is taller than every node, and no node's height
//     exceeds the configured maximum (the heuristic is capped);
//   * the level-0 traversal equals a sorted reference array maintained by the workload driver.
// Measured properties: the plain and the adaptive list run from the same coin stream over the same
// keys, so they are directly comparable. On an 80%-hot workload the adaptive list must probe strictly
// fewer times in total and reach the hot keys in fewer probes - and those savings must be exactly the
// one-comparison-per-level optimum for a key whose sentinel is tall - while keeping the per-level node
// counts identical to the plain list; on a uniform workload it must stay within a small constant
// factor of the plain list.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Node
{
    internal readonly int Value;
    internal Node[] Next;
    internal long Accesses;

    internal Node(int value, int height)
    {
        Value = value;
        Next = new Node[height];
    }

    internal int Height => Next.Length;
}

internal sealed class SkipList
{
    private readonly int _maxLevel;
    private readonly bool _adaptive;
    private readonly Node _head;
    private ulong _state;
    private int _level = 1;

    internal long Probes;
    internal long Promotions;
    internal long Demotions;
    internal long Swaps;
    internal int Count;

    internal SkipList(int maxLevel, bool adaptive, ulong seed)
    {
        if (maxLevel < 1) throw new ArgumentException("maxLevel must be at least 1");
        _maxLevel = maxLevel;
        _adaptive = adaptive;
        _state = seed;
        _head = new Node(int.MinValue, maxLevel);
    }

    internal int MaxLevel => _maxLevel;
    internal int Level => _level;
    internal int MaxNodeHeight
    {
        get
        {
            int best = 1;
            for (Node node = _head.Next[0]; node != null; node = node.Next[0]) if (node.Height > best) best = node.Height;
            return best;
        }
    }

    /// <summary>Deterministic geometric height: p = 1/2, capped at the configured maximum.</summary>
    private int RandomHeight()
    {
        int height = 1;
        while (height < _maxLevel && (SplitMix64.Next(ref _state) & 1UL) == 1UL) height++;
        return height;
    }

    internal bool Search(int value)
    {
        Node node = Find(value, out long probes, out Node predecessor);
        Probes += probes;
        if (node == null) return false;
        if (_adaptive)
        {
            node.Accesses++;
            // Locality heuristic: a hit key is cheap to reach exactly when the node immediately below
            // it is present on the upper levels, because the descent then lands next to the key and
            // the remaining levels need no forward steps at all. So the sentinel below the hit key
            // earns the credit and is raised one level, trading places with the coldest node whose top
            // level is exactly the level it wants to join (populations per level stay unchanged).
            Node sentinel = predecessor == _head ? node : predecessor;
            sentinel.Accesses++;
            Rebalance(sentinel, node.Accesses);
        }
        return true;
    }

    /// <summary>Descent from the top level; the probe count is the documented cost measure.</summary>
    private Node Find(int value, out long probes, out Node predecessor)
    {
        Node current = _head;
        probes = 0;
        for (int level = _level - 1; level >= 0; level--)
        {
            while (current.Next[level] != null && current.Next[level].Value < value)
            {
                current = current.Next[level];
                probes++;
            }
            probes++;                                        // the comparison that stops the descent
        }
        predecessor = current;
        Node candidate = current.Next[0];
        if (candidate != null && candidate.Value == value) return candidate;
        return null;
    }

    /// <summary>Raise a node one level: extend its forward array and splice it in above.</summary>
    private void RaiseOneLevel(Node node)
    {
        int target = node.Height;                            // the level it is about to join
        if (target >= _maxLevel) return;
        Node current = _head;
        for (int level = _level - 1; level > target; level--)
        {
            while (current.Next[level] != null && current.Next[level].Value < node.Value) current = current.Next[level];
        }
        while (current.Next[target] != null && current.Next[target].Value < node.Value) current = current.Next[target];
        var widened = new Node[target + 1];
        Array.Copy(node.Next, widened, node.Next.Length);
        node.Next = widened;
        node.Next[target] = current.Next[target];
        current.Next[target] = node;
        if (target >= _level) _level = target + 1;
        Promotions++;
    }

    /// <summary>Drop a node's top level: unlink it there and shorten its forward array.</summary>
    private void DemoteTop(Node node)
    {
        if (node.Height < 2) throw new Exception("internal error: a level-1 node cannot be demoted");
        int top = node.Height - 1;
        Node predecessor = _head;
        while (predecessor.Next[top] != null && predecessor.Next[top].Value < node.Value) predecessor = predecessor.Next[top];
        if (predecessor.Next[top] != node) throw new Exception("internal error: the demoted node was not linked at its top level");
        predecessor.Next[top] = node.Next[top];
        var narrowed = new Node[top];
        Array.Copy(node.Next, narrowed, top);
        node.Next = narrowed;
        if (top == _level - 1 && _head.Next[top] == null) _level--;
        Demotions++;
    }

    /// <summary>
    /// Raise a sentinel one level if its demand outweighs the coldest node that can give up the level,
    /// trading places with it. The partner is taken from the same segment of the level above, so the
    /// population of every level, and the spacing of every level inside its segments, is exactly what
    /// the coin stream produced: the swap only decides *which* node represents that gap.
    /// </summary>
    private void Rebalance(Node sentinel, long demand)
    {
        int target = sentinel.Height;
        if (target >= _maxLevel) return;
        if (_head.Next[target + 1] == null) return;              // nothing above to trade with
        Node outer = _head;
        while (outer.Next[target + 1] != null && outer.Next[target + 1].Value < sentinel.Value) outer = outer.Next[target + 1];
        long upperBound = outer.Next[target + 1] == null ? long.MaxValue : outer.Next[target + 1].Value;
        Node coldest = null;
        for (Node current = outer.Next[target]; current != null && current.Value < upperBound; current = current.Next[target])
        {
            if (current == sentinel) return;
            if (current.Value <= outer.Value) continue;
            if (current.Height != target + 1) continue;           // only its top level can be given up
            if (coldest == null || current.Accesses < coldest.Accesses) coldest = current;
        }
        if (coldest == null) return;
        if (demand <= 2 * coldest.Accesses) return;               // the swap must raise the demand potential
        DemoteTop(coldest);
        RaiseOneLevel(sentinel);
        Swaps++;
    }

    internal bool Insert(int value)
    {
        var update = new Node[_maxLevel];
        Node current = _head;
        for (int level = _level - 1; level >= 0; level--)
        {
            while (current.Next[level] != null && current.Next[level].Value < value) current = current.Next[level];
            update[level] = current;
        }
        Node next = current.Next[0];
        if (next != null && next.Value == value) return false;     // duplicates are rejected
        int height = RandomHeight();
        var node = new Node(value, height);
        for (int level = _level; level < height; level++) update[level] = _head;   // new top levels start at the head
        for (int level = 0; level < height; level++)
        {
            node.Next[level] = update[level].Next[level];
            update[level].Next[level] = node;
        }
        if (height > _level) _level = height;
        Count++;
        return true;
    }

    internal bool Delete(int value)
    {
        var update = new Node[_maxLevel];
        Node current = _head;
        for (int level = _level - 1; level >= 0; level--)
        {
            while (current.Next[level] != null && current.Next[level].Value < value) current = current.Next[level];
            update[level] = current;
        }
        Node target = current.Next[0];
        if (target == null || target.Value != value) return false;
        for (int level = 0; level < _level; level++)
        {
            if (update[level].Next[level] != target) continue;
            update[level].Next[level] = target.Next[level];
        }
        while (_level > 1 && _head.Next[_level - 1] == null) _level--;
        Count--;
        return true;
    }

    internal List<int> ToList()
    {
        var values = new List<int>();
        for (Node node = _head.Next[0]; node != null; node = node.Next[0]) values.Add(node.Value);
        return values;
    }

    /// <summary>Height of the node holding a key, or 0 when the key is absent.</summary>
    internal int HeightOf(int value)
    {
        Node node = Find(value, out _, out _);
        return node == null ? 0 : node.Height;
    }

    /// <summary>Height of the level-0 predecessor of a key (or of the key itself when it is the head).</summary>
    internal int SentinelHeight(int value)
    {
        Node node = Find(value, out _, out Node predecessor);
        if (node == null) return 0;
        return predecessor == _head ? node.Height : predecessor.Height;
    }

    /// <summary>Nodes at level l, collected top down, used to check the sublist property.</summary>
    internal List<int> LevelNodes(int level)
    {
        var values = new List<int>();
        if (level >= _level) return values;
        for (Node node = _head.Next[level]; node != null; node = node.Next[level]) values.Add(node.Value);
        return values;
    }

    internal string InvariantViolation()
    {
        if (_level < 1 || _level > _maxLevel) return "level count " + _level + " is outside [1, " + _maxLevel + "]";
        var zero = ToList();
        if (zero.Count != Count) return "level 0 holds " + zero.Count + " nodes but Count is " + Count;
        for (int i = 1; i < zero.Count; i++) if (zero[i - 1] >= zero[i]) return "level 0 is not strictly increasing at " + i;
        for (int level = 0; level < _level; level++)
        {
            var seen = LevelNodes(level);
            var expected = new List<int>();
            for (Node node = _head.Next[0]; node != null; node = node.Next[0]) if (node.Height > level) expected.Add(node.Value);
            if (seen.Count != expected.Count) return "level " + level + " shows " + seen.Count + " nodes, expected " + expected.Count;
            for (int i = 0; i < seen.Count; i++) if (seen[i] != expected[i]) return "level " + level + " diverges from the height filter";
        }
        for (int level = 1; level < _level; level++)
        {
            var below = LevelNodes(level - 1);
            var above = LevelNodes(level);
            int cursor = 0;
            foreach (int value in above)
            {
                while (cursor < below.Count && below[cursor] != value) cursor++;
                if (cursor == below.Count) return "level " + level + " is not a subsequence of level " + (level - 1);
                cursor++;
            }
        }
        for (Node node = _head.Next[0]; node != null; node = node.Next[0])
            if (node.Height > _maxLevel) return "node " + node.Value + " is taller than the cap";
        if (_level > 1 && _head.Next[_level - 1] == null) return "the head advertises a level with no node";
        return null;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Sorted reference array; insertion position found by binary search, not by the skip list.</summary>
    private static void ReferenceInsert(List<int> reference, int value)
    {
        int at = reference.BinarySearch(value);
        if (at >= 0) return;
        reference.Insert(~at, value);
    }

    private static void ReferenceDelete(List<int> reference, int value)
    {
        int at = reference.BinarySearch(value);
        if (at >= 0) reference.RemoveAt(at);
    }

    private static bool SameList(List<int> a, List<int> b)
    {
        if (a.Count != b.Count) return false;
        for (int i = 0; i < a.Count; i++) if (a[i] != b[i]) return false;
        return true;
    }

    public static void Main()
    {
        // ---- degenerate shapes -------------------------------------------------------------------
        {
            var empty = new SkipList(8, true, 1UL);
            Check(!empty.Search(5), "searching an empty skip list fails");
            Check(empty.Count == 0 && empty.ToList().Count == 0, "the empty skip list holds nothing");
            Check(empty.InvariantViolation() == null, "the empty skip list satisfies the invariants");
            Check(!empty.Delete(5), "deleting from an empty skip list is a no-op");
            Check(empty.Insert(5) && empty.Count == 1, "inserting the first key is accepted");

            var single = new SkipList(8, true, 7UL);
            Check(single.Insert(42), "the first key is inserted");
            Check(!single.Insert(42), "a duplicate insert is rejected");
            Check(single.Count == 1 && single.ToList()[0] == 42, "one distinct key is stored");
            for (int i = 0; i < 20; i++) Check(single.Search(42), "the single key keeps being found");
            Check(single.Promotions == 0 && single.Swaps == 0, "a lone node has no colder neighbour to trade with");
            Check(single.InvariantViolation() == null, "the repeatedly hit single node keeps the invariants");
            Check(single.Delete(42), "the last key can be deleted");
            Check(single.Count == 0 && single.ToList().Count == 0 && single.Level == 1, "the list collapses back to level 1");
            Check(!single.Search(42), "the deleted key is gone");
            Check(single.InvariantViolation() == null, "the emptied skip list still satisfies the invariants");

            bool threw = false;
            try { new SkipList(0, false, 1UL); } catch (ArgumentException) { threw = true; }
            Check(threw, "maxLevel = 0 is rejected");

            var flat = new SkipList(1, true, 3UL);
            for (int i = 0; i < 40; i++) flat.Insert(i * 3);
            Check(flat.Level == 1 && flat.MaxNodeHeight == 1, "a maxLevel of 1 degenerates into a plain linked list");
            for (int i = 0; i < 100; i++) flat.Search(i % 120);
            Check(flat.Promotions == 0, "the locality heuristic cannot promote beyond the level cap");
            Check(flat.InvariantViolation() == null, "the flat list satisfies the invariants");
            Check(flat.Delete(0) && flat.Delete(117), "the head and the tail can be deleted from the flat list");
            Check(flat.InvariantViolation() == null, "the flat list stays consistent after deleting both ends");
        }

        // ---- randomized workload against the sorted reference -------------------------------------
        {
            ulong state = 0x5EED773577357735UL;
            var list = new SkipList(16, false, 0xABCDEF0123456789UL);
            var reference = new List<int>();
            var live = new HashSet<int>();
            bool contentsMatch = true, invariantsHold = true;
            string problem = "ok";
            int inserts = 0, deletes = 0, hits = 0, misses = 0;
            for (int step = 0; step < 400; step++)
            {
                int key = SplitMix64.NextInt(ref state, 600);
                int action = SplitMix64.NextInt(ref state, 3);
                if (action == 0)
                {
                    bool added = list.Insert(key);
                    if (added) { ReferenceInsert(reference, key); live.Add(key); inserts++; }
                    else if (!live.Contains(key)) { contentsMatch = false; problem = "duplicate insert reported at step " + step; }
                }
                else if (action == 1)
                {
                    bool removed = list.Delete(key);
                    if (removed) { ReferenceDelete(reference, key); live.Remove(key); deletes++; }
                    else if (live.Contains(key)) { contentsMatch = false; problem = "existing key not deleted at step " + step; }
                }
                else
                {
                    bool found = list.Search(key);
                    if (found != live.Contains(key)) { contentsMatch = false; problem = "membership mismatch at step " + step; }
                    if (found) hits++; else misses++;
                }
                if (!SameList(list.ToList(), reference)) { contentsMatch = false; problem = "level 0 differs from the reference at step " + step; break; }
                string violation = list.InvariantViolation();
                if (violation != null) { invariantsHold = false; problem = "step " + step + ": " + violation; break; }
            }
            Check(contentsMatch, "the skip list matches the sorted reference after all 400 operations: " + problem);
            Check(invariantsHold, "the structural invariants hold after all 400 operations: " + problem);
            Check(inserts > 40 && deletes > 8 && hits > 8 && misses > 8,
                "the workload covered inserts/deletes/hits/misses (" + inserts + "/" + deletes + "/" + hits + "/" + misses + ")");
            Check(list.Level <= list.MaxNodeHeight && list.MaxNodeHeight <= 16, "the level count tracks the tallest node");
            Console.WriteLine("random workload: n=" + list.Count + " level=" + list.Level + " inserts=" + inserts
                + " deletes=" + deletes + " hits=" + hits + " misses=" + misses);
        }

        // ---- height distribution sanity on a larger list ------------------------------------------
        {
            const int Size = 2000;
            ulong state = 0x1122334455667788UL;
            var list = new SkipList(24, false, 0x1122334455667788UL);
            for (int i = 0; i < Size; i++) list.Insert(SplitMix64.NextInt(ref state, 10 * Size));
            double log2 = Math.Log(Size, 2);
            Check(list.Count > Size / 2, "the deterministic coin stream produced a large list");
            Check(list.MaxNodeHeight >= log2 / 2 && list.MaxNodeHeight <= 3 * log2 + 2,
                "the maximum height stays near log2(n): " + list.MaxNodeHeight + " vs log2(" + list.Count + ")=" + log2.ToString("F1"));
            Check(list.Level == list.MaxNodeHeight, "the list level equals the tallest node");
            Check(list.InvariantViolation() == null, "the large list satisfies the invariants");
            Check(list.LevelNodes(0).Count == list.Count, "level 0 lists every key");
            Check(list.LevelNodes(list.Level - 1).Count <= 4, "the top level is nearly empty (" + list.LevelNodes(list.Level - 1).Count + " nodes)");
            Console.WriteLine("height sanity: n=" + list.Count + " maxHeight=" + list.MaxNodeHeight
                + " log2n=" + log2.ToString("F1") + " topLevelNodes=" + list.LevelNodes(list.Level - 1).Count);
        }

        // ---- the locality heuristic: plain versus self-adjusting on a skewed workload -------------
        {
            const int Keys = 500, Hot = 20, Accesses = 6000;
            ulong coin = 0xFEEDFACE00001111UL;
            var plain = new SkipList(20, false, coin);
            var adaptive = new SkipList(20, true, coin);         // identical coin stream
            var keys = new int[Keys];
            for (int i = 0; i < Keys; i++)
            {
                keys[i] = i * 7 + 1;
                Check(plain.Insert(keys[i]), "key " + i + " inserted into the plain list");
                Check(adaptive.Insert(keys[i]), "key " + i + " inserted into the adaptive list");
            }
            Check(SameList(plain.ToList(), adaptive.ToList()), "both variants start with the same key set");

            ulong workload = 0x0BADF00D5EED0000UL;
            int hotAccesses = 0, coldAccesses = 0;
            for (int i = 0; i < Accesses; i++)
            {
                int pick;
                if (SplitMix64.NextInt(ref workload, 5) < 4) { pick = SplitMix64.NextInt(ref workload, Hot); hotAccesses++; }
                else { pick = SplitMix64.NextInt(ref workload, Keys); coldAccesses++; }
                Check(plain.Search(keys[pick]), "the plain list finds a stored key");
                Check(adaptive.Search(keys[pick]), "the adaptive list finds a stored key");
            }
            Check(plain.Promotions == 0, "the plain variant never promotes anything");
            Check(adaptive.Promotions == adaptive.Swaps, "one promotion per swap: " + adaptive.Swaps + " swaps");
            Check(adaptive.Demotions == adaptive.Swaps, "every swap demoted exactly one node");
            bool sameShape = true;
            for (int level = 0; level < Math.Max(plain.Level, adaptive.Level); level++)
            {
                if (plain.LevelNodes(level).Count != adaptive.LevelNodes(level).Count) sameShape = false;
            }
            Check(sameShape, "the level populations stay identical to the plain random structure");
            Check(adaptive.Probes < plain.Probes, "the adaptive list probes less overall (" + adaptive.Probes + " < " + plain.Probes + ")");
            Check(adaptive.InvariantViolation() == null && plain.InvariantViolation() == null, "both variants stay consistent");
            Check(SameList(plain.ToList(), adaptive.ToList()), "promotion never changes the key set or its order");
            long plainHotHeight = 0, adaptiveHotHeight = 0;
            for (int h = 0; h < Hot; h++)
            {
                plainHotHeight += plain.HeightOf(keys[h]);
                adaptiveHotHeight += adaptive.HeightOf(keys[h]);
            }
            Check(adaptiveHotHeight > plainHotHeight,
                "the hot keys really climbed: summed height " + adaptiveHotHeight + " > " + plainHotHeight);
            Check(adaptive.HeightOf(keys[Hot + 5]) > 0, "the cold keys are still present and reachable");

            // the hot keys whose sentinel actually climbed should be the ones that got cheaper: that
            // is the causal link the heuristic claims, and it is what gets asserted here
            var beforeProbes = new long[Hot];
            var probePlain = new long[Hot];
            var probeAdaptive = new long[Hot];
            var sentinelHeights = new int[Hot];
            for (int h = 0; h < Hot; h++)
            {
                beforeProbes[h] = plain.Probes;
                plain.Search(keys[h]);
                long plainCost = plain.Probes - beforeProbes[h];
                long mark = adaptive.Probes;
                adaptive.Search(keys[h]);
                long adaptiveCost = adaptive.Probes - mark;
                probePlain[h] = plainCost;
                probeAdaptive[h] = adaptiveCost;
                sentinelHeights[h] = adaptive.SentinelHeight(keys[h]);
            }
            int better = 0, equal = 0, worse = 0, tallSentinels = 0;
            long totalPlain = 0, totalAdaptive = 0, tallPlain = 0, tallAdaptive = 0;
            for (int h = 0; h < Hot; h++)
            {
                if (probeAdaptive[h] < probePlain[h]) better++;
                else if (probeAdaptive[h] == probePlain[h]) equal++;
                else worse++;
                totalPlain += probePlain[h];
                totalAdaptive += probeAdaptive[h];
                if (sentinelHeights[h] >= 5) { tallSentinels++; tallPlain += probePlain[h]; tallAdaptive += probeAdaptive[h]; }
            }
            Check(better > 0 && worse <= 1, "hot keys got cheaper without making the others worse (" + better + " better, " + equal + " equal, " + worse + " worse)");
            Check(tallSentinels >= 4, "several sentinels climbed to at least five levels (" + tallSentinels + "/" + Hot + ")");
            Check(tallAdaptive < tallPlain, "the keys with a tall sentinel are the cheap ones (" + tallAdaptive + " < " + tallPlain + ")");
            Check(totalAdaptive <= totalPlain, "the hot keys cost no more in total (" + totalAdaptive + " <= " + totalPlain + ")");
            Check(hotAccesses >= 3 * coldAccesses, "the workload really is skewed (" + hotAccesses + " hot vs " + coldAccesses + " cold)");
            Console.WriteLine("skewed workload: plainProbes=" + plain.Probes + " adaptiveProbes=" + adaptive.Probes
                + " swaps=" + adaptive.Swaps + " hotPlain=" + totalPlain + " hotAdaptive=" + totalAdaptive
                + " tallSentinels=" + tallSentinels + " tallProbes " + tallPlain + " -> " + tallAdaptive);
        }

        // ---- a uniform workload must not be hurt much ---------------------------------------------
        {
            const int Keys = 400, Accesses = 5000;
            var plain = new SkipList(20, false, 0x2222333344445555UL);
            var adaptive = new SkipList(20, true, 0x2222333344445555UL);
            var keys = new int[Keys];
            for (int i = 0; i < Keys; i++)
            {
                keys[i] = i * 3 + 2;
                plain.Insert(keys[i]);
                adaptive.Insert(keys[i]);
            }
            ulong workload = 0x9999AAAABBBBCCCCUL;
            for (int i = 0; i < Accesses; i++)
            {
                int pick = SplitMix64.NextInt(ref workload, Keys);
                plain.Search(keys[pick]);
                adaptive.Search(keys[pick]);
            }
            double ratio = (double)adaptive.Probes / plain.Probes;
            Check(ratio <= 1.2, "on uniform accesses the heuristic stays within 20% of the plain list (ratio " + ratio.ToString("F4") + ")");
            Check(ratio > 0, "the ratio is well defined");
            Check(adaptive.InvariantViolation() == null, "the adaptive list is consistent after the uniform workload");
            Console.WriteLine("uniform workload: plainProbes=" + plain.Probes + " adaptiveProbes=" + adaptive.Probes
                + " ratio=" + ratio.ToString("F4") + " promotions=" + adaptive.Promotions);
        }

        // ---- determinism ---------------------------------------------------------------------------
        {
            var a = new SkipList(12, true, 0xDEADBEEFCAFEBABEUL);
            var b = new SkipList(12, true, 0xDEADBEEFCAFEBABEUL);
            ulong state = 0x13572468ACE0BDF1UL;
            long probesA = 0, probesB = 0;
            for (int step = 0; step < 300; step++)
            {
                int key = SplitMix64.NextInt(ref state, 200);
                int action = SplitMix64.NextInt(ref state, 3);
                bool resultA = action == 0 ? a.Insert(key) : action == 1 ? a.Delete(key) : a.Search(key);
                bool resultB = action == 0 ? b.Insert(key) : action == 1 ? b.Delete(key) : b.Search(key);
                if (resultA != resultB) break;
                probesA = a.Probes;
                probesB = b.Probes;
            }
            Check(probesA == probesB, "two lists driven with the same seed perform the identical work");
            Check(SameList(a.ToList(), b.ToList()), "two identically driven lists hold the same keys");
            Check(a.Promotions == b.Promotions, "two identically driven lists promote the same number of nodes");
            Console.WriteLine("determinism: probes=" + probesA + " promotions=" + a.Promotions + " n=" + a.Count);
        }

        Console.WriteLine("PASS");
    }
}
