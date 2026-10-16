// Task 529 - Implement advanced string factoring (Lyndon factorization Duval). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Duval's algorithm factors a string into non-increasing Lyndon words - the Chen-Fox-Lyndon
// factorization - in linear time. A single scan keeps three indices (i = the start of the current
// period, j = the look-ahead, k = the period length); while s[k] <= s[j] the scan either restarts the
// period at i (when s[k] < s[j]) or advances k, and when the run breaks, the factor of length j - k is
// cut off repeatedly until the candidate start passes k. A Lyndon word here is the standard strict
// definition: the word must be strictly smaller than every one of its proper non-empty suffixes,
// which is implemented directly by ordinal comparisons and cross-checked against the equivalent
// characterization "the word is its own smallest rotation and is primitive".
//
// Documented checks. (1) The concatenation of the factors equals the input, every factor is a
// non-empty Lyndon word, and the sequence is non-increasing - the three defining properties - for
// twelve texts including the empty string, "a", "aa", "abab", "banana", "aab", "cba", "zzzabc",
// "aabaab", an LCG string, the Fibonacci word and the Thue-Morse word. (2) Hand-checked factorizations:
// "banana" -> b | an | an | a, "abab" -> ab | ab, "aab" -> aab, "aaa" -> a | a | a,
// "zzzabc" -> z | z | z | abc, "cba" -> c | b | a, "aabaab" -> aab | aab. (3) Duval's factors equal the
// greedy "take the longest Lyndon prefix, repeat" factorization, and the first factor is exactly the
// longest Lyndon prefix of the text. (4) Uniqueness: for all texts of length at most 12 an exhaustive
// search over non-increasing Lyndon decompositions finds exactly one, and it is Duval's.
// (5) Characterization: IsLyndon agrees with "smallest rotation is at offset 0 and the word is
// primitive" for every tested word, and the whole text is a Lyndon word exactly when the factorization
// consists of that single factor. (6) Errors: a null input throws ArgumentNullException.
using System;
using System.Collections.Generic;

internal static class Lyndon
{
    /** Duval's linear-time factorization into non-increasing Lyndon words. */
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

    /** Greedy reference: repeatedly cut off the longest Lyndon prefix. */
    internal static List<string> GreedyFactorization(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        var factors = new List<string>();
        int start = 0;
        while (start < text.Length)
        {
            int length = LongestLyndonPrefix(text, start);
            factors.Add(text.Substring(start, length));
            start += length;
        }
        return factors;
    }

    /** Strict definition: strictly smaller than every proper non-empty suffix. */
    internal static bool IsLyndon(string word)
    {
        if (word == null) throw new ArgumentNullException(nameof(word));
        if (word.Length == 0) return false;
        for (int i = 1; i < word.Length; i++)
        {
            if (string.CompareOrdinal(word, 0, word, i, word.Length - i) >= 0) return false;
        }
        return true;
    }

    internal static bool IsNonIncreasing(List<string> factors)
    {
        for (int i = 1; i < factors.Count; i++)
        {
            if (string.CompareOrdinal(factors[i - 1], factors[i]) < 0) return false;
        }
        return true;
    }

    /** Length of the longest Lyndon word starting at the given offset. */
    internal static int LongestLyndonPrefix(string text, int start)
    {
        int best = 0;
        for (int length = 1; start + length <= text.Length; length++)
        {
            if (IsLyndon(text.Substring(start, length))) best = length;
        }
        return best;
    }

    /** Exhaustive count of non-increasing Lyndon decompositions (only usable for short texts). */
    internal static int CountDecompositions(string text, int start, string previous)
    {
        if (start == text.Length) return 1;
        int total = 0;
        for (int length = 1; start + length <= text.Length; length++)
        {
            string factor = text.Substring(start, length);
            if (!IsLyndon(factor)) continue;
            if (previous.Length > 0 && string.CompareOrdinal(factor, previous) > 0) continue;
            total += CountDecompositions(text, start + length, factor);
        }
        return total;
    }

    /** Word is primitive when no proper divisor of its length is a period. */
    internal static bool IsPrimitive(string word)
    {
        int n = word.Length;
        if (n == 0) return false;
        for (int period = 1; period < n; period++)
        {
            if (n % period != 0) continue;
            bool ok = true;
            for (int i = period; i < n; i++)
            {
                if (word[i] != word[i - period])
                {
                    ok = false;
                    break;
                }
            }
            if (ok) return false;
        }
        return true;
    }

    /** Independent characterization: the smallest rotation sits at offset 0 (smallest index on ties). */
    internal static bool MinimumRotationIsZero(string word)
    {
        int n = word.Length;
        if (n == 0) return false;
        string doubled = word + word;
        for (int i = 1; i < n; i++)
        {
            if (string.CompareOrdinal(doubled, i, word, 0, n) < 0) return false;
        }
        return true;
    }

    internal static bool EndsWith(string text, string suffix)
    {
        return suffix.Length <= text.Length && string.CompareOrdinal(text, text.Length - suffix.Length, suffix, 0,
                suffix.Length) == 0;
    }
}

public static class Solution
{
    public static void Main()
    {
        string[] texts = {
            string.Empty, "a", "aa", "ab", "abab", "banana", "aab", "aaa", "zzzabc", "aabaab",
            LcgText(24, 4242, 2), FibonacciWord(21), ThueMorse(15),
        };
        foreach (string text in texts)
        {
            List<string> factors = Lyndon.Duval(text);
            string tag = "\"" + text + "\"";
            string joined = string.Empty;
            foreach (string factor in factors) joined += factor;
            Check(joined == text, tag + ": the factors concatenate to the input");
            foreach (string factor in factors)
            {
                Check(factor.Length > 0, tag + ": no factor is empty");
                Check(Lyndon.IsLyndon(factor), tag + ": factor " + factor + " is a Lyndon word");
            }
            Check(Lyndon.IsNonIncreasing(factors), tag + ": the factors are non-increasing");
            List<string> greedy = Lyndon.GreedyFactorization(text);
            Check(factors.Count == greedy.Count, tag + ": the greedy factorization has as many factors");
            for (int i = 0; i < factors.Count; i++)
            {
                Check(factors[i] == greedy[i], tag + ": factor " + i + " matches the greedy one");
            }
            if (text.Length > 0)
            {
                Check(factors[0] == text.Substring(0, Lyndon.LongestLyndonPrefix(text, 0)),
                        tag + ": the first factor is the longest Lyndon prefix");
            }
            Check(Lyndon.IsLyndon(text) == (factors.Count == 1), tag + ": a Lyndon text has one factor");
            if (text.Length <= 12)
            {
                Check(Lyndon.CountDecompositions(text, 0, string.Empty) == 1,
                        tag + ": the non-increasing Lyndon decomposition is unique");
            }
        }

        CheckFactors(Lyndon.Duval("banana"), new[] { "b", "an", "an", "a" }, "banana");
        CheckFactors(Lyndon.Duval("abab"), new[] { "ab", "ab" }, "abab");
        CheckFactors(Lyndon.Duval("aab"), new[] { "aab" }, "aab");
        CheckFactors(Lyndon.Duval("aaa"), new[] { "a", "a", "a" }, "aaa");
        CheckFactors(Lyndon.Duval("zzzabc"), new[] { "z", "z", "z", "abc" }, "zzzabc");
        CheckFactors(Lyndon.Duval("cba"), new[] { "c", "b", "a" }, "cba");
        CheckFactors(Lyndon.Duval("aabaab"), new[] { "aab", "aab" }, "aabaab");
        CheckFactors(Lyndon.Duval("abbbbb"), new[] { "abbbbb" }, "abbbbb");
        Check(Lyndon.Duval(string.Empty).Count == 0, "the empty string has no factors");
        Check(Lyndon.Duval("a").Count == 1 && Lyndon.Duval("a")[0] == "a", "a single character is one factor");

        Check(Lyndon.IsLyndon("a"), "a is a Lyndon word");
        Check(Lyndon.IsLyndon("ab"), "ab is a Lyndon word");
        Check(Lyndon.IsLyndon("aab"), "aab is a Lyndon word");
        Check(Lyndon.IsLyndon("abbbbb"), "abbbbb is a Lyndon word");
        Check(!Lyndon.IsLyndon("aa"), "aa is not a Lyndon word");
        Check(!Lyndon.IsLyndon("ba"), "ba is not a Lyndon word");
        Check(!Lyndon.IsLyndon("abab"), "abab is not a Lyndon word");
        Check(!Lyndon.IsLyndon("abba"), "abba is not a Lyndon word");
        Check(!Lyndon.IsLyndon(string.Empty), "the empty word is not a Lyndon word");

        int checkedWords = 0;
        for (int seed = 1; seed <= 120; seed++)
        {
            string word = LcgText(1 + seed % 9, seed * 17 + 3, 1 + seed % 3);
            bool lyndon = Lyndon.IsLyndon(word);
            Check(lyndon == (Lyndon.MinimumRotationIsZero(word) && Lyndon.IsPrimitive(word)),
                    "the characterization agrees for " + word);
            List<string> factors = Lyndon.Duval(word);
            Check(lyndon == (factors.Count == 1), "single-factor equivalence for " + word);
            if (factors.Count > 0)
            {
                Check(Lyndon.EndsWith(word, factors[factors.Count - 1]),
                        "the last factor of " + word + " is a suffix of the word");
            }
            if (word.Length <= 12)
            {
                Check(Lyndon.CountDecompositions(word, 0, string.Empty) == 1, "unique decomposition of " + word);
            }
            checkedWords++;
        }
        Check(checkedWords == 120, "all 120 LCG words were checked");

        Check(Lyndon.IsPrimitive("ab"), "ab is primitive");
        Check(!Lyndon.IsPrimitive("abab"), "abab is not primitive");
        Check(!Lyndon.IsPrimitive("aaa"), "aaa is not primitive");
        Check(Lyndon.IsPrimitive("aab"), "aab is primitive");
        Check(Lyndon.MinimumRotationIsZero("aab"), "aab is its own smallest rotation");
        Check(!Lyndon.MinimumRotationIsZero("baa"), "baa is not its own smallest rotation");
        Check(Lyndon.MinimumRotationIsZero("aabaab"), "aabaab has its smallest rotation at offset 0");
        Check(!Lyndon.IsLyndon("aabaab"), "aabaab is not primitive, hence not Lyndon");

        CheckThrows(() => Lyndon.Duval(null!), "a null text throws");
        CheckThrows(() => Lyndon.IsLyndon(null!), "a null word throws");
        CheckThrows(() => Lyndon.GreedyFactorization(null!), "a null greedy input throws");

        Console.WriteLine("PASS");
    }

    private static void CheckFactors(List<string> actual, string[] expected, string label)
    {
        Check(actual.Count == expected.Length, label + ": expected " + expected.Length + " factors");
        for (int i = 0; i < expected.Length; i++)
        {
            Check(actual[i] == expected[i], label + ": factor " + i + " should be \"" + expected[i] + "\"");
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
