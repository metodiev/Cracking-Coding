// Task 321 - Sudoku uniqueness check optimization. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

internal sealed class SudokuCounter
{
    private const int Full = 0x1FF; // nine candidate bits

    private readonly int[] grid = new int[81];
    private readonly int[] rows = new int[9];
    private readonly int[] cols = new int[9];
    private readonly int[] boxes = new int[9];
    private readonly List<(int Cell, int Bit)> trail = new List<(int, int)>();
    private readonly List<int[]> found = new List<int[]>();
    private readonly int limit;

    public SudokuCounter(int[] board, int limit)
    {
        Array.Copy(board, grid, 81);
        this.limit = limit;
    }

    private static int BoxOf(int cell) => (cell / 27) * 3 + (cell % 9) / 3;

    private void Place(int cell, int bit)
    {
        rows[cell / 9] |= bit;
        cols[cell % 9] |= bit;
        boxes[BoxOf(cell)] |= bit;
        grid[cell] = BitLength(bit);
        trail.Add((cell, bit));
    }

    private static int BitLength(int bit)
    {
        int length = 0;
        while (bit > 0)
        {
            length++;
            bit >>= 1;
        }
        return length;
    }

    private void UndoTo(int mark)
    {
        while (trail.Count > mark)
        {
            (int cell, int bit) = trail[trail.Count - 1];
            trail.RemoveAt(trail.Count - 1);
            grid[cell] = 0;
            rows[cell / 9] ^= bit;
            cols[cell % 9] ^= bit;
            boxes[BoxOf(cell)] ^= bit;
        }
    }

    private int Candidates(int cell) => Full & ~(rows[cell / 9] | cols[cell % 9] | boxes[BoxOf(cell)]);

    // Naked singles: a peer left with a single candidate is forced; false on contradiction.
    private bool Propagate(int[] seeds)
    {
        var work = new Stack<int>();
        foreach (int seed in seeds) work.Push(seed);
        while (work.Count > 0)
        {
            int cell = work.Pop();
            for (int peer = 0; peer < 81; peer++)
            {
                if (peer == cell || grid[peer] != 0) continue;
                bool sameUnit = peer / 9 == cell / 9 || peer % 9 == cell % 9 || BoxOf(peer) == BoxOf(cell);
                if (!sameUnit) continue;
                int cand = Candidates(peer);
                if (cand == 0) return false;
                if ((cand & (cand - 1)) == 0) // exactly one bit set -> forced
                {
                    Place(peer, cand);
                    work.Push(peer);
                }
            }
        }
        return true;
    }

    public int Run()
    {
        for (int cell = 0; cell < 81; cell++)
        {
            if (grid[cell] != 0)
            {
                int bit = 1 << (grid[cell] - 1);
                if ((Candidates(cell) & bit) == 0) return 0; // the givens already conflict
                Place(cell, bit);
            }
        }
        var allSeeds = Enumerable.Range(0, 81).ToArray();
        if (!Propagate(allSeeds)) return 0;
        Search();
        return found.Count;
    }

    private void Search()
    {
        int bestCell = -1, bestCand = 0, bestSize = 10;
        for (int cell = 0; cell < 81; cell++)
        {
            if (grid[cell] != 0) continue;
            int cand = Candidates(cell);
            if (cand == 0) return; // dead end
            int size = System.Numerics.BitOperations.PopCount((uint)cand);
            if (size < bestSize)
            {
                bestCell = cell;
                bestCand = cand;
                bestSize = size;
                if (size == 2) break; // cannot do better
            }
        }
        if (bestCell == -1)
        {
            found.Add((int[])grid.Clone());
            return;
        }
        while (bestCand != 0)
        {
            int bit = bestCand & -bestCand;
            bestCand ^= bit;
            int mark = trail.Count;
            Place(bestCell, bit);
            if (Propagate(new[] { bestCell })) Search();
            UndoTo(mark);
            if (found.Count >= limit) return;
        }
    }

    public int[]? FirstFound() => found.Count == 0 ? null : found[0];
}

public static class Solution
{
    private static int CountSolutions(int[] board, int limit) => new SudokuCounter(board, limit).Run();

    private static int Uniqueness(int[] board) => new SudokuCounter(board, 2).Run();

    private static int[]? FirstSolution(int[] board)
    {
        var counter = new SudokuCounter(board, 1);
        counter.Run();
        return counter.FirstFound();
    }

    /// <summary>The 27 units (rows, columns, boxes) as arrays of nine values.</summary>
    private static IEnumerable<int[]> Units(int[] board)
    {
        for (int r = 0; r < 9; r++)
        {
            var unit = new int[9];
            for (int c = 0; c < 9; c++) unit[c] = board[r * 9 + c];
            yield return unit;
        }
        for (int c = 0; c < 9; c++)
        {
            var unit = new int[9];
            for (int r = 0; r < 9; r++) unit[r] = board[r * 9 + c];
            yield return unit;
        }
        for (int boxRow = 0; boxRow < 3; boxRow++)
        {
            for (int boxCol = 0; boxCol < 3; boxCol++)
            {
                var unit = new int[9];
                int k = 0;
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        unit[k++] = board[(boxRow * 3 + r) * 9 + boxCol * 3 + c];
                yield return unit;
            }
        }
    }

    private static bool IsSolvedSudoku(int[] board) =>
        Units(board).All(unit => unit.OrderBy(v => v).SequenceEqual(Enumerable.Range(1, 9)));

    private static bool UnitsAreDuplicateFree(int[] board) =>
        Units(board).All(unit => unit.Where(v => v != 0).Distinct().Count() == unit.Count(v => v != 0));

    private static readonly int[] Puzzle =
    {
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9,
    };

    private static readonly int[] KnownSolution =
    {
        5, 3, 4, 6, 7, 8, 9, 1, 2,
        6, 7, 2, 1, 9, 5, 3, 4, 8,
        1, 9, 8, 3, 4, 2, 5, 6, 7,
        8, 5, 9, 7, 6, 1, 4, 2, 3,
        4, 2, 6, 8, 5, 3, 7, 9, 1,
        7, 1, 3, 9, 2, 4, 8, 5, 6,
        9, 6, 1, 5, 3, 7, 2, 8, 4,
        2, 8, 7, 4, 1, 9, 6, 3, 5,
        3, 4, 5, 2, 8, 6, 1, 7, 9,
    };

    public static void Main()
    {
        var empty = new int[81];
        // empty board: plenty of solutions, so the capped counter reports "many" (2)
        Check(CountSolutions(empty, 2) == 2, "empty board has many solutions");
        Check(Uniqueness(empty) == 2, "empty board uniqueness is many");
        Check(CountSolutions(empty, 4) == 4, "empty board limit 4");

        // proper puzzle: exactly one completion; with limit 2 the whole tree is explored
        Check(CountSolutions(Puzzle, 2) == 1, "proper puzzle is unique");
        Check(Uniqueness(Puzzle) == 1, "proper puzzle uniqueness is 1");
        int[]? solution = FirstSolution(Puzzle);
        Check(solution != null && solution.SequenceEqual(KnownSolution), "first solution matches the published grid");
        Check(IsSolvedSudoku(solution!), "published grid is a valid sudoku");

        // unsatisfiable board: no completion, and it is not a trivially conflicting one
        var unsat = (int[])Puzzle.Clone();
        unsat[1] = 1;
        Check(UnitsAreDuplicateFree(unsat), "unsatisfiable board is unit-consistent");
        Check(CountSolutions(unsat, 2) == 0, "unsatisfiable board has no solution");
        Check(Uniqueness(unsat) == 0, "unsatisfiable uniqueness is 0");
        Check(FirstSolution(unsat) == null, "unsatisfiable has no first solution");

        // a plainly conflicting board is rejected while loading the givens
        var conflicting = (int[])Puzzle.Clone();
        conflicting[0] = 6; // two 6s in the top-left box
        Check(CountSolutions(conflicting, 2) == 0, "conflicting board has no solution");

        // sparse valid board (a few clues of the known solution) has multiple completions
        var sparse = new int[81];
        for (int index = 0; index < 81; index += 10) sparse[index] = KnownSolution[index];
        Check(Uniqueness(sparse) == 2, "sparse board has many completions");

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
