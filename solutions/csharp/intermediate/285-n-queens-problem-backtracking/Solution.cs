// Task 285 - N-Queens via backtracking. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    /// <summary>
    /// All solutions as lists of column indices per row, in lexicographic order:
    /// queens are placed row by row with columns tried 0..n-1.
    /// </summary>
    public static List<List<int>> SolveNQueens(int n)
    {
        var solutions = new List<List<int>>();
        var columns = new List<int>();
        Place(0, n, columns, solutions);
        return solutions;
    }

    private static void Place(int row, int n, List<int> columns, List<List<int>> solutions)
    {
        if (row == n)
        {
            solutions.Add(new List<int>(columns));
            return;
        }
        for (int col = 0; col < n; col++)
        {
            bool ok = true;
            for (int pr = 0; pr < columns.Count; pr++)
            {
                int pc = columns[pr];
                if (col == pc || Math.Abs(col - pc) == row - pr) { ok = false; break; }
            }
            if (ok)
            {
                columns.Add(col);
                Place(row + 1, n, columns, solutions);
                columns.RemoveAt(columns.Count - 1);
            }
        }
    }

    public static int CountSolutions(int n) => SolveNQueens(n).Count;

    /// <summary>Lexicographically first solution, or null when no placement exists.</summary>
    public static int[]? FirstSolution(int n)
    {
        var solutions = SolveNQueens(n);
        return solutions.Count > 0 ? solutions[0].ToArray() : null;
    }

    /// <summary>One queen per row is structural; check columns are a permutation and no shared diagonal.</summary>
    public static bool IsValidSolution(int[] solution)
    {
        int n = solution.Length;
        var seen = new bool[n];
        foreach (int col in solution)
        {
            if (col < 0 || col >= n || seen[col]) return false;
            seen[col] = true;
        }
        for (int r = 0; r < n; r++)
            for (int s = r + 1; s < n; s++)
                if (Math.Abs(solution[r] - solution[s]) == s - r) return false;
        return true;
    }

    /// <summary>Permutation oracle: count column permutations with no shared diagonal.</summary>
    public static int BruteForceCount(int n)
    {
        var permutation = new int[n];
        var used = new bool[n];
        return CountPermutations(0, n, permutation, used);
    }

    private static int CountPermutations(int row, int n, int[] permutation, bool[] used)
    {
        if (row == n)
        {
            for (int r = 0; r < n; r++)
                for (int s = r + 1; s < n; s++)
                    if (Math.Abs(permutation[r] - permutation[s]) == s - r) return 0;
            return 1;
        }
        int total = 0;
        for (int col = 0; col < n; col++)
        {
            if (used[col]) continue;
            used[col] = true;
            permutation[row] = col;
            total += CountPermutations(row + 1, n, permutation, used);
            used[col] = false;
        }
        return total;
    }

    public static void Main()
    {
        Check(CountSolutions(1) == 1, "n=1");
        Check(CountSolutions(2) == 0, "n=2");
        Check(CountSolutions(3) == 0, "n=3");
        Check(CountSolutions(4) == 2, "n=4");
        Check(CountSolutions(5) == 10, "n=5");
        Check(CountSolutions(6) == 4, "n=6");
        Check(CountSolutions(7) == 40, "n=7");
        Check(CountSolutions(8) == 92, "n=8");

        Check(FirstSolution(1)!.SequenceEqual(new[] { 0 }), "first n=1");
        Check(FirstSolution(4)!.SequenceEqual(new[] { 1, 3, 0, 2 }), "first n=4");
        Check(FirstSolution(2) == null, "first n=2 none");
        Check(FirstSolution(3) == null, "first n=3 none");

        for (int n = 1; n <= 8; n++)
        {
            var solutions = SolveNQueens(n);
            Check(solutions.Count == CountSolutions(n), "count matches list " + n);
            for (int i = 1; i < solutions.Count; i++)
            {
                var previous = solutions[i - 1];
                var current = solutions[i];
                for (int k = 0; k < n; k++)
                {
                    if (current[k] < previous[k]) throw new Exception("FAILED: lexicographic order " + n);
                    if (current[k] > previous[k]) break;
                }
            }
            foreach (var solution in solutions)
            {
                Check(IsValidSolution(solution.ToArray()), "valid " + n + " " + string.Join(",", solution));
            }
            if (solutions.Count > 0) Check(FirstSolution(n)!.SequenceEqual(solutions[0]), "first solution " + n);
            Check(BruteForceCount(n) == solutions.Count, "oracle " + n);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
