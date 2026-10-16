// Task 530 - Implement Lyndon array and uses. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// The Lyndon array lambda[i] is the length of the longest Lyndon word starting at position i. Three
// independent computations are implemented and cross-checked:
//   * the definitional one - try every length and keep the longest substring that is strictly smaller
//     than all of its proper suffixes (quadratic Lyndon tests, so only for short texts);
//   * from the suffix array ranks - lambda[i] is the distance to the next strictly smaller suffix,
//     i.e. the first j > i with rank[j] < rank[i] (n - i when there is none), which is the classical
//     "next smaller suffix" characterisation and needs only rank comparisons;
//   * the same next-smaller-suffix search done in linear time with a monotonic stack over the ranks.
// The suffix array itself is built by the doubling method (rank of the 2^k prefixes, re-sorted each
// round), and the ranks are the inverse permutation of it.
//
// Documented uses, all checked: (1) the Lyndon factorization of the text is recovered from the array -
// its factor lengths are lambda[0], lambda[L1], lambda[L1 + L2], ... and the resulting factors must
// equal Duval's factorization; (2) the text is a Lyndon word exactly when lambda[0] = n; (3) suffix i
// is a minimal suffix (the smallest of all suffixes starting at i or later) exactly when
// lambda[i] = n - i. Additionally the maximality of lambda is asserted from the definition: the
// substring of that length is a Lyndon word, while extending it by one character is not.
//
// Documented checks. (1) Hand-checked arrays: "banana" -> 1 2 1 2 1 1, "aab" -> 3 2 1, "cba" -> 1 1 1,
// "abbbbb" -> 6 1 1 1 1 1, "abc" -> 3 2 1, "aaa" -> 1 1 1, "zzzabc" -> 1 1 1 3 2 1.
// (2) All three methods agree element-wise on twelve texts (including the Fibonacci word, the
// Thue-Morse word and three LCG strings). (3) 1 <= lambda[i] <= n - i, and the next-smaller-suffix
// property is verified straight from the ranks. (4) The three documented uses hold on every text, and
// the recovered factorization is compared factor by factor with Duval's. (5) Errors: null inputs throw
// ArgumentNullException; the empty text yields an empty array.
using System;
using System.Collections.Generic;

internal static class LyndonArray
{
    /** Strict definition of a Lyndon word: smaller than every proper non-empty suffix. */
    internal static bool IsLyndonWord(string text, int start, int length)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        if (length == 0) return false;
        for (int shift = 1; shift < length; shift++)
        {
            if (string.CompareOrdinal(text, start, text, start + shift, length - shift) >= 0) return false;
        }
        return true;
    }

    internal static bool IsLyndonWord(string word)
    {
        return IsLyndonWord(word, 0, word.Length);
    }

    /** Definitional computation: the longest Lyndon prefix of every suffix. */
    internal static int[] Naive(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        int n = text.Length;
        var result = new int[n];
        for (int i = 0; i < n; i++)
        {
            int best = 1;
            for (int length = 1; i + length <= n; length++)
            {
                if (IsLyndonWord(text, i, length)) best = length;
            }
            result[i] = best;
        }
        return result;
    }

    /** Suffix array by the doubling method: sort positions by their 2^k-prefix ranks. */
    internal static int[] SuffixArray(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        int n = text.Length;
        if (n == 0) return new int[0];
        var order = new int[n];
        var rank = new int[n];
        var next = new int[n];
        for (int i = 0; i < n; i++)
        {
            order[i] = i;
            rank[i] = text[i];
        }
        for (int k = 1; ; k *= 2)
        {
            int step = k;
            Array.Sort(order, (a, b) =>
            {
                if (rank[a] != rank[b]) return rank[a].CompareTo(rank[b]);
                int ra = a + step < n ? rank[a + step] : -1;
                int rb = b + step < n ? rank[b + step] : -1;
                return ra.CompareTo(rb);
            });
            next[order[0]] = 0;
            for (int i = 1; i < n; i++)
            {
                int a = order[i - 1];
                int b = order[i];
                int ra = a + step < n ? rank[a + step] : -1;
                int rb = b + step < n ? rank[b + step] : -1;
                next[b] = next[a] + (rank[a] != rank[b] || ra != rb ? 1 : 0);
            }
            Array.Copy(next, rank, n);
            if (rank[order[n - 1]] == n - 1 || k > n) break;
        }
        return order;
    }

    /** Inverse permutation of the suffix array: rank[i] is the position of the suffix i. */
    internal static int[] Ranks(string text)
    {
        int[] order = SuffixArray(text);
        var ranks = new int[order.Length];
        for (int position = 0; position < order.Length; position++) ranks[order[position]] = position;
        return ranks;
    }

    /** Lyndon array by scanning for the next strictly smaller suffix (quadratic). */
    internal static int[] ByRanks(int[] rank)
    {
        if (rank == null) throw new ArgumentNullException(nameof(rank));
        int n = rank.Length;
        var result = new int[n];
        for (int i = 0; i < n; i++)
        {
            int j = i + 1;
            while (j < n && rank[j] > rank[i]) j++;
            result[i] = j - i;
        }
        return result;
    }

    /** The same next-smaller-suffix search in linear time with a monotonic stack. */
    internal static int[] ByStack(int[] rank)
    {
        if (rank == null) throw new ArgumentNullException(nameof(rank));
        int n = rank.Length;
        var result = new int[n];
        var stack = new List<int>();
        for (int i = n - 1; i >= 0; i--)
        {
            while (stack.Count > 0 && rank[stack[stack.Count - 1]] > rank[i]) stack.RemoveAt(stack.Count - 1);
            result[i] = stack.Count > 0 ? stack[stack.Count - 1] - i : n - i;
            stack.Add(i);
        }
        return result;
    }

    /** Duval's linear-time Lyndon factorization, used as the reference for the recovery check. */
    internal static List<string> Duval(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        var factors = new List<string>();
        int n = text.Length;
        int i = 0;
        while (i < n)
        {
            int j = i + 1;
            int k = i;
            while (j < n && text[k].CompareTo(text[j]) <= 0)
            {
                if (text[k].CompareTo(text[j]) < 0) k = i;
                else k++;
                j++;
            }
            while (i <= k)
            {
                factors.Add(text.Substring(i, j - k));
                i += j - k;
            }
        }
        return factors;
    }

    /** Use 1: read the Lyndon factorization off the array by jumping lambda[position] at a time. */
    internal static List<string> FactorizationFromArray(string text, int[] lambda)
    {
        var factors = new List<string>();
        int position = 0;
        while (position < text.Length)
        {
            factors.Add(text.Substring(position, lambda[position]));
            position += lambda[position];
        }
        return factors;
    }

    /** Use 3: suffix i is minimal when its rank is the smallest among suffixes i..n-1. */
    internal static bool IsMinimalSuffix(int[] rank, int i)
    {
        for (int j = i + 1; j < rank.Length; j++)
        {
            if (rank[j] < rank[i]) return false;
        }
        return true;
    }
}

public static class Solution
{
    public static void Main()
    {
        string[] texts = {
            string.Empty, "a", "ab", "aa", "banana", "aab", "cba", "abbbbb", "abc", "aaa", "zzzabc",
            "aabaab", FibonacciWord(20), ThueMorse(16), LcgText(22, 4242, 2), LcgText(24, 77, 3),
            LcgText(18, 31415, 5),
        };
        foreach (string text in texts)
        {
            int n = text.Length;
            int[] ranks = LyndonArray.Ranks(text);
            int[] lambda = LyndonArray.ByStack(ranks);
            string tag = "\"" + text + "\"";
            Check(lambda.Length == n, tag + ": one entry per position");
            CheckSame(LyndonArray.Naive(text), lambda, tag + ": the definitional computation agrees");
            CheckSame(LyndonArray.ByRanks(ranks), lambda, tag + ": the quadratic rank scan agrees");
            for (int i = 0; i < n; i++)
            {
                Check(lambda[i] >= 1 && lambda[i] <= n - i, tag + ": 1 <= lambda[" + i + "] <= n - i");
                Check(LyndonArray.IsLyndonWord(text, i, lambda[i]),
                        tag + ": the reported substring at " + i + " is Lyndon");
                if (i + lambda[i] < n)
                {
                    Check(!LyndonArray.IsLyndonWord(text, i, lambda[i] + 1),
                            tag + ": lambda[" + i + "] is maximal");
                }
                if (lambda[i] < n - i)
                {
                    Check(ranks[i + lambda[i]] < ranks[i],
                            tag + ": the next smaller suffix of " + i + " sits at " + (i + lambda[i]));
                }
                Check((lambda[i] == n - i) == LyndonArray.IsMinimalSuffix(ranks, i),
                        tag + ": the minimal-suffix use holds at " + i);
            }
            List<string> duval = LyndonArray.Duval(text);
            List<string> recovered = LyndonArray.FactorizationFromArray(text, lambda);
            Check(duval.Count == recovered.Count, tag + ": the recovery finds the same factor count");
            for (int i = 0; i < duval.Count; i++)
            {
                Check(duval[i] == recovered[i], tag + ": recovered factor " + i + " is " + duval[i]);
            }
            Check(LyndonArray.IsLyndonWord(text) == (n > 0 && lambda[0] == n),
                    tag + ": the text is Lyndon exactly when lambda[0] = n");
        }

        CheckSame(new[] { 1, 2, 1, 2, 1, 1 }, LyndonArray.ByStack(LyndonArray.Ranks("banana")),
                "hand-checked banana");
        CheckSame(new[] { 3, 2, 1 }, LyndonArray.ByStack(LyndonArray.Ranks("aab")), "hand-checked aab");
        CheckSame(new[] { 1, 1, 1 }, LyndonArray.ByStack(LyndonArray.Ranks("cba")), "hand-checked cba");
        CheckSame(new[] { 6, 1, 1, 1, 1, 1 }, LyndonArray.ByStack(LyndonArray.Ranks("abbbbb")),
                "hand-checked abbbbb");
        CheckSame(new[] { 3, 2, 1 }, LyndonArray.ByStack(LyndonArray.Ranks("abc")), "hand-checked abc");
        CheckSame(new[] { 1, 1, 1 }, LyndonArray.ByStack(LyndonArray.Ranks("aaa")), "hand-checked aaa");
        CheckSame(new[] { 1, 1, 1, 3, 2, 1 }, LyndonArray.ByStack(LyndonArray.Ranks("zzzabc")),
                "hand-checked zzzabc");
        Check(LyndonArray.Naive(string.Empty).Length == 0, "the empty text has an empty Lyndon array");

        int[] ranksOfBanana = LyndonArray.Ranks("banana");
        Check(ranksOfBanana.Length == 6 && LyndonArray.IsMinimalSuffix(ranksOfBanana, 5),
                "the last suffix is always minimal");
        Check(!LyndonArray.IsMinimalSuffix(ranksOfBanana, 0), "the whole string is not a minimal suffix");
        int[] suffixArrayOfBanana = LyndonArray.SuffixArray("banana");
        var sortedSuffixes = new List<string>();
        foreach (int position in suffixArrayOfBanana) sortedSuffixes.Add("banana".Substring(position));
        for (int i = 1; i < sortedSuffixes.Count; i++)
        {
            Check(string.CompareOrdinal(sortedSuffixes[i - 1], sortedSuffixes[i]) < 0,
                    "the suffix array of banana is sorted at " + i);
        }

        CheckThrows(() => LyndonArray.Naive(null!), "a null text throws");
        CheckThrows(() => LyndonArray.SuffixArray(null!), "a null suffix array input throws");
        CheckThrows(() => LyndonArray.ByStack(null!), "a null rank array throws");
        CheckThrows(() => LyndonArray.Duval(null!), "a null Duval input throws");

        Console.WriteLine("PASS");
    }

    private static void CheckSame(int[] expected, int[] actual, string label)
    {
        Check(expected.Length == actual.Length, label + ": length " + expected.Length + " vs " + actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Check(expected[i] == actual[i], label + ": entry " + i + " should be " + expected[i]);
        }
    }

    private static string LcgText(int n, int seed, int alphabet)
    {
        var chars = new char[n];
        for (int i = 0; i < n; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            chars[i] = (char)('a' + seed % alphabet);
        }
        return new string(chars);
    }

    private static string FibonacciWord(int length)
    {
        string a = "a";
        string b = "ab";
        while (b.Length < length)
        {
            string next = b + a;
            a = b;
            b = next;
        }
        return b.Substring(0, length);
    }

    private static string ThueMorse(int length)
    {
        string word = "a";
        while (word.Length < length)
        {
            var flipped = new char[word.Length];
            for (int i = 0; i < word.Length; i++) flipped[i] = word[i] == 'a' ? 'b' : 'a';
            word = word + new string(flipped);
        }
        return word.Substring(0, length);
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
