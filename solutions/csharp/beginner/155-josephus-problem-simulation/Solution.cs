// Task 155 - Josephus problem simulation. (Tier: Beginner, CTCI Ch 2)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>Eliminate every k-th person in a circle of n; returns the 0-based survivor. O(n^2).</summary>
    public static int JosephusSimulate(int n, int k)
    {
        if (n < 1 || k < 1) throw new ArgumentException("n and k must be >= 1");
        var people = new List<int>();
        for (int i = 0; i < n; i++) people.Add(i);
        int idx = 0;
        while (people.Count > 1)
        {
            idx = (idx + k - 1) % people.Count;   // the k-th person counted from idx
            people.RemoveAt(idx);
        }
        return people[0];
    }

    /// <summary>Bottom-up J(n,k) = (J(n-1,k) + k) % n starting at J(1,k) = 0. O(n) time, O(1) space.</summary>
    public static int JosephusRecurrence(int n, int k)
    {
        if (n < 1 || k < 1) throw new ArgumentException("n and k must be >= 1");
        int survivor = 0;
        for (int i = 2; i <= n; i++) survivor = (survivor + k) % i;
        return survivor;
    }

    public static void Main()
    {
        // known checkpoints
        Check(JosephusSimulate(5, 2) == 2, "J(5,2)");
        Check(JosephusSimulate(7, 3) == 3, "J(7,3)");
        Check(JosephusSimulate(41, 3) == 30, "J(41,3)");

        // both variants agree, including the n = 1 base case
        foreach (int n in new[] { 1, 2, 3, 5, 7, 10, 20, 41 })
        {
            foreach (int k in new[] { 1, 2, 3, 4, 7, 11 })
            {
                Check(JosephusRecurrence(n, k) == JosephusSimulate(n, k), "agree n=" + n + " k=" + k);
            }
        }
        Check(JosephusSimulate(1, 7) == 0 && JosephusRecurrence(1, 999) == 0, "n=1 survivor 0");

        // k = 1 eliminates in order 0, 1, 2, ... so the last index survives
        Check(JosephusSimulate(10, 1) == 9 && JosephusRecurrence(10, 1) == 9, "k=1");

        Check(JosephusRecurrence(5, 2) == 2, "recurrence J(5,2)");
        Check(JosephusRecurrence(7, 3) == 3, "recurrence J(7,3)");
        Check(JosephusRecurrence(41, 3) == 30, "recurrence J(41,3)");

        bool threw = false;
        try { JosephusSimulate(0, 2); } catch (ArgumentException) { threw = true; }
        Check(threw, "n=0 throws");
        threw = false;
        try { JosephusRecurrence(3, -1); } catch (ArgumentException) { threw = true; }
        Check(threw, "k=-1 throws");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
