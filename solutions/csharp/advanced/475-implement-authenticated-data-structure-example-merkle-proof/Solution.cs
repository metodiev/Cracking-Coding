// Task 475 - Implement authenticated data structure example (Merkle+proof). (Tier: Advanced, CTCI Beyond — Security & Privacy)
using System;
using System.Collections.Generic;

internal class AuthenticatedLog
{
    public readonly List<string> Records;
    public readonly List<long[]> Levels;

    public AuthenticatedLog(List<string> values)
    {
        Records = new List<string>(values);
        Levels = new List<long[]>();
        var current = new List<long>();
        foreach (string v in Records) current.Add(HashLeaf(v));
        Levels.Add(current.ToArray());
        while (current.Count > 1)
        {
            var next = new List<long>();
            for (int i = 0; i + 1 < current.Count; i += 2) next.Add(HashNode(current[i], current[i + 1]));
            if (current.Count % 2 == 1) next.Add(current[current.Count - 1]); // promote
            Levels.Add(next.ToArray());
            current = next;
        }
    }

    public long Root() => Levels[Levels.Count - 1][0];

    public List<long[]> Proof(int index)
    {
        var path = new List<long[]>();
        int idx = index;
        for (int li = 0; li < Levels.Count - 1; li++)
        {
            long[] level = Levels[li];
            if (idx % 2 == 1) path.Add(new long[] { level[idx - 1], 0 });
            else if (idx + 1 < level.Length) path.Add(new long[] { level[idx + 1], 1 });
            idx /= 2;
        }
        return path;
    }

    public List<List<long[]>> MultiProof(int[] targets)
    {
        long[] level = Levels[0];
        var needed = new SortedSet<int>(targets);
        var output = new List<List<long[]>>();
        while (level.Length > 1)
        {
            var siblings = new List<long[]>();
            foreach (int i in needed)
            {
                if (i % 2 == 1 && !needed.Contains(i - 1)) siblings.Add(new long[] { i - 1, level[i - 1] });
                else if (i % 2 == 0 && i + 1 < level.Length && !needed.Contains(i + 1)) siblings.Add(new long[] { i + 1, level[i + 1] });
            }
            output.Add(siblings);
            int nxtLen = (level.Length + 1) / 2;
            var next = new long[nxtLen];
            for (int i = 0; i + 1 < level.Length; i += 2) next[i / 2] = HashNode(level[i], level[i + 1]);
            if (level.Length % 2 == 1) next[nxtLen - 1] = level[level.Length - 1];
            level = next;
            var nextNeeded = new SortedSet<int>();
            foreach (int t in needed) nextNeeded.Add(t / 2);
            needed = nextNeeded;
        }
        return output;
    }

    public static long Fnv1a(byte[] data)
    {
        long h = 2166136261L;
        foreach (byte b in data)
        {
            h = ((h ^ b) * 16777619L) & 0xFFFFFFFFL;
        }
        return h;
    }

    public static long HashLeaf(string value)
    {
        var data = new byte[value.Length + 1];
        for (int i = 0; i < value.Length; i++) data[i + 1] = (byte)value[i];
        return Fnv1a(data);
    }

    public static long HashNode(long left, long right)
    {
        var data = new byte[8];
        for (int i = 0; i < 4; i++)
        {
            data[i] = (byte)((left >> (8 * (3 - i))) & 0xFF);
            data[4 + i] = (byte)((right >> (8 * (3 - i))) & 0xFF);
        }
        return Fnv1a(data);
    }

    public static bool Verify(long root, string value, List<long[]> path)
    {
        long h = HashLeaf(value);
        foreach (long[] step in path)
        {
            h = step[1] == 0 ? HashNode(step[0], h) : HashNode(h, step[0]);
        }
        return h == root;
    }

    public static bool VerifyMulti(long root, int n, Dictionary<int, string> targets, List<List<long[]>> proof)
    {
        var known = new Dictionary<int, long>();
        foreach (var pair in targets) known[pair.Key] = HashLeaf(pair.Value);
        int levelLen = n;
        foreach (List<long[]> siblings in proof)
        {
            var sibMap = new Dictionary<int, long>();
            foreach (long[] s in siblings) sibMap[(int)s[0]] = s[1];
            var parents = new Dictionary<int, long>();
            for (int i = 0; i + 1 < levelLen; i += 2)
            {
                bool haveLeft = known.ContainsKey(i);
                bool haveRight = known.ContainsKey(i + 1);
                if (!haveLeft && !haveRight) continue;
                if (haveLeft && haveRight)
                {
                    parents[i / 2] = HashNode(known[i], known[i + 1]);
                }
                else
                {
                    int pos = haveLeft ? i + 1 : i;
                    if (!sibMap.ContainsKey(pos)) return false;
                    long left = haveLeft ? known[i] : sibMap[i];
                    long right = haveRight ? known[i + 1] : sibMap[i + 1];
                    parents[i / 2] = HashNode(left, right);
                }
            }
            if (levelLen % 2 == 1 && known.ContainsKey(levelLen - 1))
            {
                parents[(levelLen - 1) / 2] = known[levelLen - 1];
            }
            known = parents;
            levelLen = (levelLen + 1) / 2;
        }
        return known.TryGetValue(0, out long got) && got == root;
    }

    public static long UpdatedRoot(string newValue, List<long[]> path)
    {
        long h = HashLeaf(newValue);
        foreach (long[] step in path)
        {
            h = step[1] == 0 ? HashNode(step[0], h) : HashNode(h, step[0]);
        }
        return h;
    }
}

public static class Solution
{
    public static void Main()
    {
        var leaves8 = new List<string>();
        for (int i = 0; i < 8; i++) leaves8.Add("tx" + i);
        var log = new AuthenticatedLog(leaves8);
        Check(log.Root() == 955777637L, "n=8 root");

        List<long[]> path2 = log.Proof(2);
        Check(path2.Count == 3 && path2[0][0] == 1947293384L && path2[1][0] == 240787827L && path2[2][0] == 3271786783L,
            "record 2 audit path hashes");
        Check(path2[0][1] == 1 && path2[1][1] == 0 && path2[2][1] == 1, "record 2 path sides");

        var mp01 = log.MultiProof(new[] { 0, 1 });
        Check(mp01.Count == 3 && mp01[0].Count == 0 && mp01[1].Count == 1 && mp01[2].Count == 1, "mp01 shape");
        Check(mp01[1][0][1] == 1737906683L && mp01[2][0][1] == 3271786783L, "mp01 sibling hashes");
        var t01 = new Dictionary<int, string> { { 0, "tx0" }, { 1, "tx1" } };
        Check(AuthenticatedLog.VerifyMulti(log.Root(), 8, t01, mp01), "mp01 verifies");
        Check(2 < log.Proof(0).Count + log.Proof(1).Count, "batching saves hashes");

        var mp03 = log.MultiProof(new[] { 0, 3 });
        Check(mp03[0].Count == 2 && mp03[1].Count == 0 && mp03[2].Count == 1, "mp03 shape");
        Check(AuthenticatedLog.VerifyMulti(log.Root(), 8, new Dictionary<int, string> { { 0, "tx0" }, { 3, "tx3" } }, mp03), "mp03 verifies");

        var mpAll = log.MultiProof(new[] { 0, 1, 2, 3, 4, 5, 6, 7 });
        Check(mpAll[0].Count == 0 && mpAll[1].Count == 0 && mpAll[2].Count == 0, "full batch needs no siblings");
        var allTargets = new Dictionary<int, string>();
        for (int i = 0; i < 8; i++) allTargets[i] = "tx" + i;
        Check(AuthenticatedLog.VerifyMulti(log.Root(), 8, allTargets, mpAll), "full batch verifies");

        var mp246 = log.MultiProof(new[] { 2, 4, 6 });
        Check(mp246[0].Count == 3 && mp246[1].Count == 1 && mp246[2].Count == 0, "mp246 shape");
        Check(AuthenticatedLog.VerifyMulti(log.Root(), 8,
            new Dictionary<int, string> { { 2, "tx2" }, { 4, "tx4" }, { 6, "tx6" } }, mp246), "mp246 verifies");

        var mp5 = log.MultiProof(new[] { 5 });
        var mp5Hashes = new List<long>();
        foreach (List<long[]> lvl in mp5) foreach (long[] s in lvl) mp5Hashes.Add(s[1]);
        var path5Hashes = new List<long>();
        foreach (long[] step in log.Proof(5)) path5Hashes.Add(step[0]);
        mp5Hashes.Sort();
        path5Hashes.Sort();
        Check(mp5Hashes.Count == path5Hashes.Count, "single-target batch equals audit path size");
        for (int i = 0; i < mp5Hashes.Count; i++) Check(mp5Hashes[i] == path5Hashes[i], "single-target batch equals audit path");
        Check(AuthenticatedLog.VerifyMulti(log.Root(), 8, new Dictionary<int, string> { { 5, "tx5" } }, mp5), "mp5 verifies");

        var badSib = new List<List<long[]>>();
        foreach (List<long[]> lvl in mp01)
        {
            var copy = new List<long[]>();
            foreach (long[] s in lvl) copy.Add(new long[] { s[0], s[1] });
            badSib.Add(copy);
        }
        long[] tampered = badSib[1][0];
        badSib[1][0] = new long[] { tampered[0], tampered[1] ^ 1L };
        Check(!AuthenticatedLog.VerifyMulti(log.Root(), 8, t01, badSib), "tampered sibling fails");
        Check(!AuthenticatedLog.VerifyMulti(log.Root(), 8, new Dictionary<int, string> { { 0, "tx0x" }, { 1, "tx1" } }, mp01),
            "tampered target fails");

        var leaves7 = new List<string>();
        for (int i = 0; i < 7; i++) leaves7.Add("tx" + i);
        var log7 = new AuthenticatedLog(leaves7);
        var mp56 = log7.MultiProof(new[] { 5, 6 });
        Check(mp56[0].Count == 1 && mp56[1].Count == 0 && mp56[2].Count == 1, "mp56 shape");
        Check(AuthenticatedLog.VerifyMulti(log7.Root(), 7, new Dictionary<int, string> { { 5, "tx5" }, { 6, "tx6" } }, mp56), "mp56 verifies");
        var mp5b = log7.MultiProof(new[] { 5 });
        Check(mp5b[0].Count == 1 && mp5b[1].Count == 1 && mp5b[2].Count == 1, "mp5b shape");
        Check(AuthenticatedLog.VerifyMulti(log7.Root(), 7, new Dictionary<int, string> { { 5, "tx5" } }, mp5b), "mp5b verifies");

        var updated = new List<string>(leaves8);
        updated[2] = "tx2*";
        var logNew = new AuthenticatedLog(updated);
        Check(logNew.Root() == 3100337682L, "updated root");
        Check(AuthenticatedLog.UpdatedRoot("tx2*", path2) == 3100337682L, "update path yields new root");
        Check(!AuthenticatedLog.Verify(logNew.Root(), "tx2", path2), "stale proof fails after update");
        Check(AuthenticatedLog.Verify(logNew.Root(), "tx2*", logNew.Proof(2)), "new proof verifies");
        Check(!AuthenticatedLog.VerifyMulti(logNew.Root(), 8, t01, mp01), "stale batch fails");
        var mp25New = logNew.MultiProof(new[] { 2, 5 });
        Check(AuthenticatedLog.VerifyMulti(logNew.Root(), 8,
            new Dictionary<int, string> { { 2, "tx2*" }, { 5, "tx5" } }, mp25New), "updated batch verifies");
        Check(!AuthenticatedLog.Verify(logNew.Root(), "tx5", log.Proof(5)), "old proof of unaffected record fails");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
