// Task 284 - Word search in a board with DFS backtracking. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static readonly int[][] Directions = { new[] { 1, 0 }, new[] { -1, 0 }, new[] { 0, 1 }, new[] { 0, -1 } };
    private static bool[,] visited = new bool[0, 0];
    private static string[] currentBoard = Array.Empty<string>();
    private static string currentWord = "";

    /// <summary>Backtracking DFS; cells cannot be reused. Returns true iff such a path exists.</summary>
    public static bool Exist(string[] board, string word)
    {
        if (word.Length == 0) return false;
        currentBoard = board;
        currentWord = word;
        visited = new bool[board.Length, board[0].Length];
        for (int r = 0; r < board.Length; r++)
            for (int c = 0; c < board[0].Length; c++)
                if (Search(r, c, 0)) return true;
        return false;
    }

    private static bool Search(int r, int c, int i)
    {
        if (i == currentWord.Length) return true;
        if (r < 0 || r >= currentBoard.Length || c < 0 || c >= currentBoard[0].Length) return false;
        if (visited[r, c] || currentBoard[r][c] != currentWord[i]) return false;
        visited[r, c] = true;
        bool found = false;
        foreach (var d in Directions)
        {
            if (Search(r + d[0], c + d[1], i + 1)) { found = true; break; }
        }
        visited[r, c] = false;                     // unmark: later branches may need this cell
        return found;
    }

    private static bool[,] buggyVisited = new bool[0, 0];
    private static string[] buggyBoard = Array.Empty<string>();
    private static string buggyWord = "";

    /// <summary>Deliberately never unmarks cells; only used to prove the trap case needs backtracking.</summary>
    public static bool BuggyExistWithoutUnmarking(string[] board, string word)
    {
        buggyBoard = board;
        buggyWord = word;
        buggyVisited = new bool[board.Length, board[0].Length];
        for (int r = 0; r < board.Length; r++)
            for (int c = 0; c < board[0].Length; c++)
                if (BuggySearch(r, c, 0)) return true;
        return false;
    }

    private static bool BuggySearch(int r, int c, int i)
    {
        if (i == buggyWord.Length) return true;
        if (r < 0 || r >= buggyBoard.Length || c < 0 || c >= buggyBoard[0].Length) return false;
        if (buggyVisited[r, c] || buggyBoard[r][c] != buggyWord[i]) return false;
        buggyVisited[r, c] = true;                 // never unmarked
        foreach (var d in Directions)
        {
            if (BuggySearch(r + d[0], c + d[1], i + 1)) return true;
        }
        return false;
    }

    /// <summary>Independent BFS over states (row, col, visited-cell mask).</summary>
    public static bool OracleExist(string[] board, string word)
    {
        int rows = board.Length, cols = board[0].Length;
        int shift = rows * cols;
        var queue = new Queue<int[]>();
        var seen = new HashSet<int>();
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                if (board[r][c] == word[0])
                {
                    int state = ((r * cols + c) << shift) | (1 << (r * cols + c));
                    if (seen.Add(state)) queue.Enqueue(new[] { r, c, 1 << (r * cols + c) });
                }
            }
        }
        while (queue.Count > 0)
        {
            var state = queue.Dequeue();
            int r = state[0], c = state[1], mask = state[2];
            int i = Popcount(mask);
            if (i == word.Length) return true;
            foreach (var d in Directions)
            {
                int nr = r + d[0], nc = c + d[1];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                if (((mask >> (nr * cols + nc)) & 1) != 0) continue;
                if (board[nr][nc] != word[i]) continue;
                int nextMask = mask | (1 << (nr * cols + nc));
                int encoded = ((nr * cols + nc) << shift) | nextMask;
                if (seen.Add(encoded)) queue.Enqueue(new[] { nr, nc, nextMask });
            }
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

    private static int seed = 13579;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static string[] RandomBoard(int rows, int cols)
    {
        var board = new string[rows];
        for (int r = 0; r < rows; r++)
        {
            var row = new char[cols];
            for (int c = 0; c < cols; c++) row[c] = "ABC"[NextSeed() % 3];
            board[r] = new string(row);
        }
        return board;
    }

    private static (string[] board, string word) RandomCase(int rows, int cols, int maxLen)
    {
        var board = RandomBoard(rows, cols);
        int length = 1 + NextSeed() % maxLen;
        var word = new char[length];
        for (int i = 0; i < length; i++) word[i] = "ABC"[NextSeed() % 3];
        return (board, new string(word));
    }

    /// <summary>Reads a random simple path, so the word is guaranteed to exist.</summary>
    private static (string[] board, string word) PlantedCase(int rows, int cols, int maxLen)
    {
        var board = RandomBoard(rows, cols);
        int r = NextSeed() % rows, c = NextSeed() % cols;
        var used = new bool[rows, cols];
        used[r, c] = true;
        var word = new List<char> { board[r][c] };
        int length = 1 + NextSeed() % maxLen;
        while (word.Count < length)
        {
            var candidates = new List<int[]>();
            foreach (var d in Directions)
            {
                int nr = r + d[0], nc = c + d[1];
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !used[nr, nc]) candidates.Add(new[] { nr, nc });
            }
            if (candidates.Count == 0) break;
            var pick = candidates[NextSeed() % candidates.Count];
            r = pick[0];
            c = pick[1];
            used[r, c] = true;
            word.Add(board[r][c]);
        }
        return (board, new string(word.ToArray()));
    }

    public static void Main()
    {
        var classic = new[] { "ABCE", "SFCS", "ADEE" };
        Check(Exist(classic, "ABCCED"), "ABCCED found");
        Check(Exist(classic, "SEE"), "SEE found");
        Check(!Exist(classic, "ABCB"), "ABCB not found");

        var trap = new[] { "ABCE", "SFES", "ADEE" };  // needs the visited-unmarking
        Check(Exist(trap, "ABCESEEEFS"), "trap found with backtracking");
        Check(!BuggyExistWithoutUnmarking(trap, "ABCESEEEFS"), "trap needs unmarking");

        Check(Exist(new[] { "A" }, "A"), "single cell");
        Check(!Exist(new[] { "A" }, "B"), "single cell wrong letter");
        Check(!Exist(new[] { "A" }, "AA"), "cell cannot be reused");

        Check(Exist(new[] { "AB", "CD" }, "ABDC"), "2x2 path");
        Check(!Exist(new[] { "AB", "CD" }, "ABCD"), "2x2 not adjacent");

        for (int trial = 0; trial < 150; trial++)
        {
            var (board, word) = RandomCase(3, 3, 4);
            Check(Exist(board, word) == OracleExist(board, word), "random " + trial + " " + word);
        }

        for (int trial = 0; trial < 50; trial++)
        {
            var (board, word) = PlantedCase(3, 3, 5);
            Check(Exist(board, word), "planted " + trial);
            Check(OracleExist(board, word), "planted oracle " + trial);
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
