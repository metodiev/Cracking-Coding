// Task 081 - Compute prefix sums for an array. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static int[] PrefixSums(int[] a)
    {
        int[] prefix = new int[a.Length + 1];
        for (int i = 0; i < a.Length; i++) prefix[i + 1] = prefix[i] + a[i];
        return prefix;
    }

    public static void Main()
    {
        Check(SeqEqual(PrefixSums(new int[0]), new[] { 0 }), "empty");
        Check(SeqEqual(PrefixSums(new[] { 5 }), new[] { 0, 5 }), "single");
        Check(SeqEqual(PrefixSums(new[] { 1, 2, 3 }), new[] { 0, 1, 3, 6 }), "1,2,3");
        Check(SeqEqual(PrefixSums(new[] { -1, 1, -1 }), new[] { 0, -1, 0, -1 }), "negatives");
        Check(SeqEqual(PrefixSums(new[] { 4, 4, 4, 4 }), new[] { 0, 4, 8, 12, 16 }), "duplicates");
        Check(PrefixSums(new[] { 1, 2, 3, 4, 5 }).Length == 6, "length n+1");
        Console.WriteLine("PASS");
    }

    private static bool SeqEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
