// Task 387 - Implement minimal perfect hash concept (CHM) small example. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)
using System;
using System.Collections.Generic;

internal sealed class Mphf
{
    public readonly int N;
    public readonly string[] Table;
    public readonly int[] G;
    public readonly int[] Displacement;
    public readonly long Base2;

    public Mphf(int n, string[] table, int[] g, int[] displacement, long base2)
    {
        N = n;
        Table = table;
        G = g;
        Displacement = displacement;
        Base2 = base2;
    }

    /// <summary>Slot of a member key in [0, N); -1 when the first-level bucket stayed empty.</summary>
    public int Lookup(string key)
    {
        int bucket = (int)(Solution.PolyHash(key, Solution.BASE1) % N);
        int shift = Displacement[bucket];
        if (shift < 0) return -1;
        int sigma = (int)((Solution.PolyHash(key, Base2) + shift) % N);
        if (G[sigma] < 0) return -1;
        return G[sigma];
    }
}

public static class Solution
{
    public const long HASH_MOD = 2147483647L;
    public const long BASE1 = 131;
    public const long BASE2 = 1000003;

    private static readonly string[] Keys = { "alice", "bob", "carol", "dave", "erin", "frank",
        "grace", "heidi", "ivan", "judy", "karl", "laura" };
    private static readonly string[] MoreKeys = { "kiwi", "mango", "melon", "olive", "peach",
        "plum", "quince", "raspberry" };

    public static long PolyHash(string text, long baseValue)
    {
        long value = 0;
        foreach (char ch in text)
            value = (value * baseValue + ch) % HASH_MOD;
        return value;
    }

    /// <summary>
    /// Hash-displace build: buckets by h1 mod n; each bucket gets the smallest shift whose
    /// sigmas (h2 + shift) mod n are distinct and unused; final slots are consecutive in
    /// increasing sigma order. Retries the whole build with base2 + attempt on failure.
    /// </summary>
    private static Mphf BuildMphf(string[] keys)
    {
        int n = keys.Length;
        for (int attempt = 0; attempt < 100; attempt++)
        {
            long base2 = BASE2 + attempt;
            var buckets = new List<string>[n];
            for (int b = 0; b < n; b++) buckets[b] = new List<string>();
            foreach (string key in keys) buckets[(int)(PolyHash(key, BASE1) % n)].Add(key);
            var order = new List<int>();
            for (int b = 0; b < n; b++) order.Add(b);
            order.Sort((x, y) => buckets[x].Count != buckets[y].Count
                ? buckets[y].Count.CompareTo(buckets[x].Count)
                : x.CompareTo(y));
            var table = new string[n];
            var g = new int[n];
            var displacement = new int[n];
            for (int i = 0; i < n; i++) { g[i] = -1; displacement[i] = -1; }
            int nextSlot = 0;
            bool feasible = true;
            foreach (int bucket in order)
            {
                var members = buckets[bucket];
                if (members.Count == 0) continue;
                int chosen = -1;
                var sigmas = new int[members.Count];
                for (int shift = 0; shift < n && chosen < 0; shift++)
                {
                    var used = new bool[n];
                    bool good = true;
                    for (int i = 0; i < members.Count && good; i++)
                    {
                        int sigma = (int)((PolyHash(members[i], base2) + shift) % n);
                        if (used[sigma] || g[sigma] != -1) good = false;
                        used[sigma] = true;
                        sigmas[i] = sigma;
                    }
                    if (good) chosen = shift;
                }
                if (chosen < 0)
                {
                    feasible = false;
                    break;
                }
                displacement[bucket] = chosen;
                var idx = new int[members.Count];
                for (int i = 0; i < idx.Length; i++) idx[i] = i;
                int[] finalSigmas = sigmas;
                Array.Sort(idx, (x, y) => finalSigmas[x].CompareTo(finalSigmas[y]));
                for (int rank = 0; rank < idx.Length; rank++)
                {
                    int sigma = sigmas[idx[rank]];
                    g[sigma] = nextSlot + rank;
                    table[nextSlot + rank] = members[idx[rank]];
                }
                nextSlot += members.Count;
            }
            if (feasible) return new Mphf(n, table, g, displacement, base2);
        }
        throw new InvalidOperationException("no displacement assignment found");
    }

    private static string JoinInts(int[] values) => string.Join(",", values);

    public static void Main()
    {
        int n = Keys.Length;
        Check(n == 12, "twelve keys");
        Check(new HashSet<string>(Keys).Count == n, "keys are distinct");

        // deterministic hash sanity (values pin the polynomial hash)
        Check(PolyHash("alice", BASE1) == 893813729, "hash alice base 131");
        Check(PolyHash("alice", BASE2) == 76280623, "hash alice base 1000003");
        Check(PolyHash("alice", BASE1) % n == 5, "alice bucket 5");
        Check(PolyHash("bob", BASE1) % n == 1 && PolyHash("laura", BASE1) % n == 1, "bucket 1 pair");
        Check(PolyHash("frank", BASE1) % n == 8 && PolyHash("judy", BASE1) % n == 8
                && PolyHash("karl", BASE1) % n == 8, "bucket 8 triple");
        int largest = 0;
        for (int b = 0; b < n; b++)
        {
            int count = 0;
            foreach (string key in Keys) if (PolyHash(key, BASE1) % n == b) count++;
            largest = Math.Max(largest, count);
        }
        Check(largest == 3, "largest bucket has 3 keys");

        var mphf = BuildMphf(Keys);
        Check(mphf.Base2 == BASE2, "first attempt already succeeds");
        Check(string.Join(",", mphf.Table) ==
                "frank,judy,karl,laura,bob,ivan,grace,heidi,dave,alice,erin,carol", "expected table");
        Check(JoinInts(mphf.G) == "7,0,5,9,10,11,1,2,3,4,6,8", "g table");
        Check(JoinInts(mphf.Displacement) == "-1,1,7,0,5,8,5,-1,0,-1,6,-1", "displacements");
        foreach (string entry in mphf.Table) Check(entry != null, "no empty table slot");
        foreach (int entry in mphf.G) Check(entry >= 0, "no empty g slot");

        // minimal perfect hash: every key maps to its own slot in [0, n), no collisions
        var seen = new bool[n];
        foreach (string key in Keys)
        {
            int slot = mphf.Lookup(key);
            Check(slot >= 0 && slot < n, "slot in range " + key);
            Check(mphf.Table[slot] == key, "table[slot] = " + key);
            Check(!seen[slot], "slot not reused " + key);
            seen[slot] = true;
        }
        foreach (bool used in seen) Check(used, "all slots used");

        // determinism: rebuilding yields the identical structure
        var again = BuildMphf(Keys);
        bool identical = again.Table.Length == mphf.Table.Length;
        for (int i = 0; identical && i < n; i++) identical = again.Table[i] == mphf.Table[i];
        for (int i = 0; identical && i < n; i++) identical = again.G[i] == mphf.G[i];
        for (int i = 0; identical && i < n; i++) identical = again.Displacement[i] == mphf.Displacement[i];
        Check(identical && again.Base2 == mphf.Base2, "deterministic rebuild");

        // non-member keys are out of scope: any in-range answer is meaningless by construction
        int stranger = mphf.Lookup("stranger");
        Check(stranger == -1 || (stranger >= 0 && stranger < n), "non-member answer in range or -1");

        // a second key set needs one hash retry (documented) and still builds a bijection
        var second = BuildMphf(MoreKeys);
        Check(second.Base2 == BASE2 + 2, "attempt 2 succeeds for the second set");
        Check(string.Join(",", second.Table) ==
                "plum,kiwi,peach,melon,olive,mango,raspberry,quince", "second table");
        Check(JoinInts(second.Displacement) == "0,0,0,-1,1,4,-1,-1", "second displacements");
        var seen3 = new bool[MoreKeys.Length];
        foreach (string key in MoreKeys)
        {
            int slot = second.Lookup(key);
            Check(slot >= 0 && slot < MoreKeys.Length, "second slot in range " + key);
            Check(second.Table[slot] == key, "second table[slot] = " + key);
            Check(!seen3[slot], "second slot not reused " + key);
            seen3[slot] = true;
        }
        foreach (bool used in seen3) Check(used, "second all slots used");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
