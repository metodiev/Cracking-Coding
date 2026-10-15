// Task 134 - Next lexicographical permutation. (Tier: Beginner, CTCI Ch 8)
using System;

public static class Solution
{
    /// <summary>Mutates in place; returns false (and wraps to sorted order) at the last permutation.</summary>
    public static bool NextPermutation(int[] a)
    {
        int n = a.Length;
        int pivot = n - 2;
        while (pivot >= 0 && a[pivot] >= a[pivot + 1]) pivot--;
        if (pivot < 0)
        {
            Array.Reverse(a);
            return false;
        }
        int successor = n - 1;
        while (a[successor] <= a[pivot]) successor--;
        (a[pivot], a[successor]) = (a[successor], a[pivot]);
        Array.Reverse(a, pivot + 1, n - pivot - 1);
        return true;
    }

    public static void Main()
    {
        var a = new[] { 1, 2, 3 };
        Check(NextPermutation(a) && string.Join(",", a) == "1,3,2", "123 -> 132");
        Check(NextPermutation(a) && string.Join(",", a) == "2,1,3", "132 -> 213");

        var last = new[] { 3, 2, 1 };
        Check(!NextPermutation(last) && string.Join(",", last) == "1,2,3", "wrap");

        var single = new[] { 5 };
        Check(!NextPermutation(single) && single[0] == 5, "single");

        var b = new[] { 1, 2, 3, 4 };
        int seen = 1;
        while (NextPermutation(b)) seen++;
        Check(seen == 24 && b[0] == 1 && b[3] == 4, "24 permutations cycle");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
