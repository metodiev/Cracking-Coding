// Task 286 - Sudoku solver with backtracking and constraint propagation. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private const int Full = 0x1FF;           // bits 0..8 for digits 1..9

    private static readonly int[][] Puzzle = {
        new[] { 5, 3, 0, 0, 7, 0, 0, 0, 0 },
        new[] { 6, 0, 0, 1, 9, 5, 0, 0, 0 },
        new[] { 0, 9, 8, 0, 0, 0, 0, 6, 0 },
        new[] { 8, 0, 0, 0, 6, 0, 0, 0, 3 },
        new[] { 4, 0, 0, 8, 0, 3, 0, 0, 1 },
        new[] { 7, 0, 0, 0, 2, 0, 0, 0, 6 },
        new[] { 0, 6, 0, 0, 0, 0, 2, 8, 0 },
        new[] { 0, 0, 0, 4, 1, 9, 0, 0, 5 },
        new[] { 0, 0, 0, 0, 8, 0, 0, 7, 9 },
    };

    private static readonly int[][] KnownSolution = {
        new[] { 5, 3, 4, 6, 7, 8, 9, 1, 2 },
        new[] { 6, 7, 2, 1, 9, 5, 3, 4, 8 },
        new[] { 1, 9, 8, 3, 4, 2, 5, 6, 7 },
        new[] { 8, 5, 9, 7, 6, 1, 4, 2, 3 },
        new[] { 4, 2, 6, 8, 5, 3, 7, 9, 1 },
        new[] { 7, 1, 3, 9, 2, 4, 8, 5, 6 },
        new[] { 9, 6, 1, 5, 3, 7, 2, 8, 4 },
        new[] { 2, 8, 7, 4, 1, 9, 6, 3, 5 },
        new[] { 3, 4, 5, 2, 8, 6, 1, 7, 9 },
    };

    private static int[][] grid = new int[9][];
    private static int[] rowsMask = new int[9];
    private static int[] colsMask = new int[9];
    private static int[] boxesMask = new int[9];

    /// <summary>
    /// Candidates come from row/column/box bitmasks and the search always branches on the
    /// most-constrained empty cell (MRV). Returns the solved board, or null if unsolvable.
    /// </summary>
    public static int[][]? SolveSudoku(int[][] board)
    {
        grid = new int[9][];
        for (int r = 0; r < 9; r++) grid[r] = (int[])board[r].Clone();
        rowsMask = new int[9];
        colsMask = new int[9];
        boxesMask = new int[9];
        for (int r = 0; r < 9; r++)
        {
            for (int c = 0; c < 9; c++)
            {
                int value = grid[r][c];
                if (value < 0 || value > 9) return null;
                if (value == 0) continue;
                int bit = 1 << (value - 1);
                int box = (r / 3) * 3 + c / 3;
                if ((rowsMask[r] & bit) != 0 || (colsMask[c] & bit) != 0 || (boxesMask[box] & bit) != 0)
                {
                    return null;                  // duplicate clue
                }
                rowsMask[r] |= bit;
                colsMask[c] |= bit;
                boxesMask[box] |= bit;
            }
        }
        return Search() ? grid : null;
    }

    private static bool Search()
    {
        int bestR = -1, bestC = -1, bestMask = 0, bestCount = 10;
        for (int r = 0; r < 9; r++)
        {
            for (int c = 0; c < 9; c++)
            {
                if (grid[r][c] != 0) continue;
                int box = (r / 3) * 3 + c / 3;
                int mask = ~(rowsMask[r] | colsMask[c] | boxesMask[box]) & Full;
                int count = Popcount(mask);
                if (count == 0) return false;     // dead end found by propagation
                if (count < bestCount)            // most-constrained cell
                {
                    bestCount = count;
                    bestR = r;
                    bestC = c;
                    bestMask = mask;
                }
            }
        }
        if (bestR == -1) return true;             // no empty cell left
        int boxIndex = (bestR / 3) * 3 + bestC / 3;
        for (int value = 1; value <= 9; value++)
        {
            int bit = 1 << (value - 1);
            if ((bestMask & bit) == 0) continue;
            grid[bestR][bestC] = value;
            rowsMask[bestR] |= bit;
            colsMask[bestC] |= bit;
            boxesMask[boxIndex] |= bit;
            if (Search()) return true;
            grid[bestR][bestC] = 0;
            rowsMask[bestR] &= ~bit;
            colsMask[bestC] &= ~bit;
            boxesMask[boxIndex] &= ~bit;
        }
        return false;
    }

    private static int Popcount(int mask)
    {
        int count = 0;
        while (mask != 0)
        {
            mask &= mask - 1;
            count++;
        }
        return count;
    }

    /// <summary>Every row, column and 3x3 box must contain the digits 1..9 exactly once.</summary>
    public static bool IsValidSolution(int[][] board)
    {
        var seen = new bool[10];
        for (int r = 0; r < 9; r++)
        {
            Array.Clear(seen, 0, seen.Length);
            for (int c = 0; c < 9; c++)
            {
                int value = board[r][c];
                if (value < 1 || value > 9 || seen[value]) return false;
                seen[value] = true;
            }
        }
        for (int c = 0; c < 9; c++)
        {
            Array.Clear(seen, 0, seen.Length);
            for (int r = 0; r < 9; r++)
            {
                int value = board[r][c];
                if (seen[value]) return false;
                seen[value] = true;
            }
        }
        for (int boxR = 0; boxR < 9; boxR += 3)
        {
            for (int boxC = 0; boxC < 9; boxC += 3)
            {
                Array.Clear(seen, 0, seen.Length);
                for (int dr = 0; dr < 3; dr++)
                {
                    for (int dc = 0; dc < 3; dc++)
                    {
                        int value = board[boxR + dr][boxC + dc];
                        if (seen[value]) return false;
                        seen[value] = true;
                    }
                }
            }
        }
        return true;
    }

    private static int seed = 13579;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static int[] RandomPermutation(int size)
    {
        var values = Enumerable.Range(0, size).ToArray();
        for (int i = size - 1; i > 0; i--)
        {
            int j = NextSeed() % (i + 1);
            int swap = values[i];
            values[i] = values[j];
            values[j] = swap;
        }
        return values;
    }

    private static int[][] Transform(int[][] source, int[] digitMap, int[] rowOrder, int[] colOrder)
    {
        var result = new int[9][];
        for (int r = 0; r < 9; r++)
        {
            result[r] = new int[9];
            for (int c = 0; c < 9; c++) result[r][c] = digitMap[source[rowOrder[r]][colOrder[c]]];
        }
        return result;
    }

    private static bool DeepEquals(int[][]? a, int[][]? b)
    {
        if (a == null || b == null) return ReferenceEquals(a, b);
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                if (a[r][c] != b[r][c]) return false;
        return true;
    }

    private static int[][] Copy(int[][] source)
    {
        var result = new int[9][];
        for (int r = 0; r < 9; r++) result[r] = (int[])source[r].Clone();
        return result;
    }

    public static void Main()
    {
        var solved = SolveSudoku(Puzzle);
        Check(DeepEquals(solved, KnownSolution), "classic puzzle");
        Check(IsValidSolution(solved!), "solution valid");
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++)
                Check(Puzzle[r][c] == 0 || solved![r][c] == Puzzle[r][c], "clue changed " + r + "," + c);

        Check(DeepEquals(SolveSudoku(KnownSolution), KnownSolution), "already solved input");

        var duplicateInColumn = Copy(KnownSolution);
        duplicateInColumn[0][0] = 6;              // column 0 already has a 6
        Check(SolveSudoku(duplicateInColumn) == null, "duplicate in column");

        var duplicateInRow = Copy(KnownSolution);
        duplicateInRow[0][1] = 5;                 // row 0 already has a 5
        Check(SolveSudoku(duplicateInRow) == null, "duplicate in row");

        var duplicateInBox = Copy(KnownSolution);
        duplicateInBox[1][1] = 5;                 // box 0 already has a 5
        Check(SolveSudoku(duplicateInBox) == null, "duplicate in box");

        var forcedConflict = new int[9][];
        for (int r = 0; r < 9; r++) forcedConflict[r] = new int[9];
        forcedConflict[0] = new[] { 1, 2, 3, 4, 5, 6, 0, 8, 9 };   // (0,6) can only be 7
        forcedConflict[3][6] = 7;                                  // but column 6 already has a 7
        Check(SolveSudoku(forcedConflict) == null, "empty candidate set without guessing");

        var empty = new int[9][];
        for (int r = 0; r < 9; r++) empty[r] = new int[9];
        var filled = SolveSudoku(empty);
        Check(filled != null && IsValidSolution(filled), "empty board");

        for (int trial = 0; trial < 25; trial++)
        {
            var permutation = RandomPermutation(9);
            var digitMap = new int[10];
            for (int i = 0; i < 9; i++) digitMap[i + 1] = permutation[i] + 1;
            var bands = RandomPermutation(3);
            var stacks = RandomPermutation(3);
            var rowOrder = new int[9];
            var colOrder = new int[9];
            for (int group = 0; group < 3; group++)
            {
                var inner = RandomPermutation(3);
                for (int i = 0; i < 3; i++) rowOrder[group * 3 + i] = bands[group] * 3 + inner[i];
                var inner2 = RandomPermutation(3);
                for (int i = 0; i < 3; i++) colOrder[group * 3 + i] = stacks[group] * 3 + inner2[i];
            }
            var puzzleVariant = Transform(Puzzle, digitMap, rowOrder, colOrder);
            var solutionVariant = Transform(KnownSolution, digitMap, rowOrder, colOrder);
            Check(IsValidSolution(solutionVariant), "variant valid " + trial);
            Check(DeepEquals(SolveSudoku(puzzleVariant), solutionVariant), "variant solved " + trial);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
