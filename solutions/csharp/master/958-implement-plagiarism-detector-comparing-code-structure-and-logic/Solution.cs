// Task 958 - Implement plagiarism detector comparing code structure and logic patterns.
// (Tier: Master, Ch 11 - Testing)
//
// Sources are normalised to a token stream: comments and string literals are dropped, keywords and
// operators survive, and every other identifier collapses to `ID` and every number to `N`.  That makes a
// rename-and-reformat clone look like the original at token level.  The detector then fingerprints
// k-gram hashes with winnowing (one minimum hash per sliding window) and scores pairs by Jaccard
// similarity over the selected fingerprints; pairs at or above the threshold are reported as clusters.
//
// Verified: identical sources score 1.0, comment/whitespace edits do not change the score, a renamed
// clone stays high, an unrelated algorithm stays low, the winnowing invariant (every window contributes
// a selected fingerprint) holds, the ranking agrees with an exact brute-force k-gram Jaccard oracle on
// all pairs, the empty and shorter-than-k boundaries are safe, and the run is deterministic.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }
    private const int K = 5;
    private const int Window = 4;
    private static readonly HashSet<string> Keywords = new HashSet<string>(StringComparer.Ordinal)
    {
        "if", "else", "for", "while", "return", "class", "static", "void", "int", "bool", "long",
        "new", "public", "private", "this", "var", "string", "foreach", "in", "true", "false",
    };
    private static List<string> Tokenize(string source)
    {
        var tokens = new List<string>();
        int i = 0;
        while (i < source.Length)
        {
            char c = source[i];
            if (c == '/' && i + 1 < source.Length && source[i + 1] == '/')
            {
                while (i < source.Length && source[i] != '\n') i++;
                continue;
            }
            if (c == '/' && i + 1 < source.Length && source[i + 1] == '*')
            {
                i += 2;
                while (i + 1 < source.Length && !(source[i] == '*' && source[i + 1] == '/')) i++;
                i = Math.Min(source.Length, i + 2);
                continue;
            }
            if (c == '"' || c == '\'')
            {
                i++;
                while (i < source.Length && source[i] != c) { if (source[i] == '\\') i++; i++; }
                i++;
                tokens.Add("S");
                continue;
            }
            if (char.IsLetter(c))
            {
                int start = i;
                while (i < source.Length && (char.IsLetterOrDigit(source[i]) || source[i] == '_')) i++;
                string word = source.Substring(start, i - start);
                tokens.Add(Keywords.Contains(word) ? word : "ID");
                continue;
            }
            if (char.IsDigit(c))
            {
                while (i < source.Length && (char.IsLetterOrDigit(source[i]) || source[i] == '.')) i++;
                tokens.Add("N");
                continue;
            }
            if ("+-*/%=<>!&|(),;{}[]:".IndexOf(c) >= 0) tokens.Add(c.ToString());
            i++;
        }
        return tokens;
    }
    private static ulong[] KGramHashes(List<string> tokens)
    {
        if (tokens.Count < K) return new ulong[0];
        var hashes = new ulong[tokens.Count - K + 1];
        for (int i = 0; i < hashes.Length; i++)
        {
            unchecked
            {
                ulong h = 14695981039346656037UL;
                for (int j = 0; j < K; j++)
                {
                    foreach (char ch in tokens[i + j]) { h ^= ch; h *= 1099511628211UL; }
                    h ^= 0x1F;
                    h *= 1099511628211UL;
                }
                hashes[i] = h;
            }
        }
        return hashes;
    }
    /** Winnowing: keep the minimum hash of every sliding window of `Window` k-grams. */
    private static List<(ulong Hash, int Position)> Winnow(ulong[] hashes)
    {
        var selected = new List<(ulong, int)>();
        var seen = new HashSet<(ulong Hash, int Position)>();
        if (hashes.Length == 0) return selected;
        int w = Math.Min(Window, hashes.Length);
        for (int start = 0; start + w <= hashes.Length; start++)
        {
            int minPosition = start;
            for (int j = start + 1; j < start + w; j++)
            {
                if (hashes[j] < hashes[minPosition]) minPosition = j;
            }
            // A window is represented by its own minimum position, so two windows that share a value but
            // not a position both stay represented - that is what makes the coverage invariant hold.
            if (seen.Add((hashes[minPosition], minPosition))) selected.Add((hashes[minPosition], minPosition));
        }
        return selected;
    }
    private static SortedSet<ulong> HashSetOf(List<(ulong Hash, int Position)> fingerprints)
    {
        var set = new SortedSet<ulong>();
        foreach ((ulong hash, int _) in fingerprints) set.Add(hash);
        return set;
    }
    private static SortedSet<ulong> ExactHashSet(ulong[] hashes)
    {
        var set = new SortedSet<ulong>();
        foreach (ulong hash in hashes) set.Add(hash);
        return set;
    }
    private static double Jaccard(SortedSet<ulong> a, SortedSet<ulong> b)
    {
        if (a.Count == 0 || b.Count == 0) return 0.0; // no evidence of shared structure
        int intersection = 0;
        foreach (ulong hash in a) { if (b.Contains(hash)) intersection++; }
        return intersection / (double)(a.Count + b.Count - intersection);
    }
    private static double WinnowedSimilarity(string left, string right) =>
        Jaccard(HashSetOf(Winnow(KGramHashes(Tokenize(left)))), HashSetOf(Winnow(KGramHashes(Tokenize(right)))));
    private static double ExactSimilarity(string left, string right) =>
        Jaccard(ExactHashSet(KGramHashes(Tokenize(left))), ExactHashSet(KGramHashes(Tokenize(right))));
    private const string Original = @"public class Sorter
{
    public static void Sort(int[] data)
    {
        for (int i = 1; i < data.Length; i++)
        {
            int key = data[i]; int j = i - 1;
            while (j >= 0 && data[j] > key) { data[j + 1] = data[j]; j--; }
            data[j + 1] = key;
        }
    }
}
";
    private const string Clone = @"public class Ordering
{
    /* renamed and reformatted copy of the insertion sort above */
    public static void Arrange(int[] items)
    {
        for (int idx = 1; idx < items.Length; idx++)
        {
            int pivot = items[idx]; int scan = idx - 1;
            while (scan >= 0 && items[scan] > pivot) { items[scan + 1] = items[scan]; scan--; }
            items[scan + 1] = pivot;
        }
    }
}
";
    private const string Unrelated = @"public class Graph
{
    public static bool HasCycle(int[,] edges, int n)
    {
        bool[] seen = new bool[n];
        for (int i = 0; i < n; i++)
        {
            if (seen[i]) continue;
            int stack = i;
            while (stack < n) { seen[stack] = true; stack += 2; }
        }
        return false;
    }
}
";
    public static void Main()
    {
        Check(WinnowedSimilarity(Original, Original) == 1.0, "a source compared with itself scores exactly 1.0");
        string reformatted = Original.Replace("data[j + 1] = key;", "data[j + 1] = key; // commit the key").Replace("  ", "          ");
        Check(WinnowedSimilarity(Original, reformatted) == 1.0, "comments and indentation are normalised away (score stays 1.0)");
        double clone = WinnowedSimilarity(Original, Clone);
        double unrelated = WinnowedSimilarity(Original, Unrelated);
        Check(clone >= 0.85, "the renamed clone stays highly similar (" + clone.ToString("F3", CultureInfo.InvariantCulture) + ")");
        Check(unrelated < 0.25, "the unrelated algorithm stays dissimilar (" + unrelated.ToString("F3", CultureInfo.InvariantCulture) + ")");
        // Oracle: an exact (unwinnowed) k-gram Jaccard over the same pairs must rank them the same way.
        double exactSelf = ExactSimilarity(Original, Original);
        double exactClone = ExactSimilarity(Original, Clone);
        double exactUnrelated = ExactSimilarity(Original, Unrelated);
        Check(exactSelf == 1.0 && exactClone > exactUnrelated && exactClone >= 0.85, "the brute-force k-gram Jaccard oracle ranks the pairs identically (" + exactClone.ToString("F3", CultureInfo.InvariantCulture) + " vs " + exactUnrelated.ToString("F3", CultureInfo.InvariantCulture) + ")");
        Check(clone > unrelated && Math.Abs(clone - exactClone) < 0.2, "winnowing preserves the similarity ordering and stays within 0.2 of the exact score");
        // Winnowing invariant: every window of K grams contributes a selected fingerprint inside it.
        ulong[] hashes = KGramHashes(Tokenize(Original));
        List<(ulong Hash, int Position)> selected = Winnow(hashes);
        bool covered = true;
        int w = Math.Min(Window, hashes.Length);
        for (int start = 0; start + w <= hashes.Length && covered; start++)
        {
            bool found = false;
            foreach ((ulong _, int position) in selected) found |= position >= start && position < start + w;
            covered &= found;
        }
        Check(covered && selected.Count <= hashes.Length, "winnowing selects at least one fingerprint inside every window and never more than the k-gram count");
        var pairs = new List<(string, double)>
        {
            ("original/self", WinnowedSimilarity(Original, Original)), ("original/clone", clone),
            ("clone/self", WinnowedSimilarity(Clone, Clone)), ("original/unrelated", unrelated),
        };
        var flagged = new List<string>();
        foreach ((string name, double score) in pairs) { if (score >= 0.6) flagged.Add(name); }
        Check(flagged.Count == 3 && flagged.Contains("original/clone") && !flagged.Contains("original/unrelated"), "thresholding at 0.6 clusters the two clones and the self-match, and rejects the unrelated pair");
        // boundaries: empty source and a source shorter than one k-gram.
        Check(Tokenize("").Count == 0 && WinnowedSimilarity("", Original) == 0.0 && WinnowedSimilarity("", "") == 0.0, "an empty source has no fingerprints and scores 0.0 against everything");
        string tiny = "int a;";
        Check(Tokenize(tiny).Count == 3 && Winnow(KGramHashes(Tokenize(tiny))).Count == 0 && WinnowedSimilarity(tiny, tiny) == 0.0, "a source shorter than k grams yields no fingerprint and is safely reported as no evidence");
        Check(WinnowedSimilarity("int a=1;int b=2;", "int a=1;int b=2;") == 1.0, "two identical five-token sources still score 1.0");
        Check(WinnowedSimilarity(Original, Clone) == clone, "re-scoring the same pair returns the identical value");
        Check(Tokenize(Original).Count == Tokenize(reformatted).Count, "comment and whitespace edits do not change the token count");
        Console.WriteLine("tokens: original=" + Tokenize(Original).Count + ", clone=" + Tokenize(Clone).Count + ", unrelated=" + Tokenize(Unrelated).Count + "; k=" + K + " window=" + Window);
        Console.WriteLine("similarity: clone " + clone.ToString("F3", CultureInfo.InvariantCulture) + " (exact oracle " + exactClone.ToString("F3", CultureInfo.InvariantCulture) + "), unrelated " + unrelated.ToString("F3", CultureInfo.InvariantCulture) + " (exact " + exactUnrelated.ToString("F3", CultureInfo.InvariantCulture) + "); clusters " + string.Join(", ", flagged));
        Console.WriteLine("PASS");
    }
}