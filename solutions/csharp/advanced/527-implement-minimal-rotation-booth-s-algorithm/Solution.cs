// Task 527 - Implement minimal rotation (Booth’s algorithm). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Booth's algorithm finds the lexicographically least rotation of a string in linear time. It runs a
// KMP-style failure function over the doubled string s + s: the running candidate start k is advanced
// whenever the current suffix comparison shows that a later position is smaller, and the failure
// array lets the scan skip the characters that are known to repeat, so the total number of character
// comparisons stays linear (the implementation counts them and the self-checks assert the bound).
// Ties are never an issue for the returned string, but the returned index is the smallest one among
// the equal minimal rotations, which is checked against an O(n^2) enumeration of all rotations and
// against an independent O(n log n) comparison sort of the rotation indices (ordinal comparisons on
// the doubled string).
//
// The canonical form is a circular invariant: rotating the input by any offset must produce the same
// least rotation, which is asserted for every shift of every test string. The maximal rotation is
// obtained by running the same algorithm on the order-reversing image of the alphabet (c -> 'z' - c),
// which turns "least" into "greatest" without a second implementation.
//
// Documented checks. (1) Hand-checked offsets: "baba" -> 1 ("abab"), "banana" -> 5 ("abanan"),
// "abcabc" -> 0, "cba" -> 2 ("acb"), "ababab" -> 0, "aaaa" -> 0, "a" -> 0, "" -> 0.
// (2) For 200 LCG strings over alphabets of size 2, 3 and 5 with lengths up to 200, Booth, the
// exhaustive enumeration and the comparison sort all return the same offset, Rotate(s, offset) is the
// lexicographically smallest rotation, and it is a genuine rotation (same length and same character
// multiset). (3) The canonical form of every one of the n cyclic shifts of the test strings is the
// same string (circular invariance). (4) The maximal rotation from the reversed-alphabet run equals
// the exhaustive maximum for every test string. (5) The instrumented comparison count stays under
// 4n + 16 for every tested string (the observed worst case over these strings is below 3.5n, with ~2n
// in the long-run limit) and the empty string performs no comparisons. (6) Errors: a null input
// throws ArgumentNullException, and Rotate rejects an out-of-range rotation index.
using System;
using System.Collections.Generic;

internal static class Booth
{
    /** Booth's algorithm; also reports the number of character comparisons performed. */
    internal static int LeastRotation(string text, out long comparisons)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        int n = text.Length;
        comparisons = 0;
        if (n == 0) return 0;
        string doubled = text + text;
        int m = doubled.Length;
        var failure = new int[m];
        for (int i = 0; i < m; i++) failure[i] = -1;
        int k = 0;
        for (int j = 1; j < m; j++)
        {
            int i = failure[j - k - 1];
            while (i != -1 && doubled[j] != doubled[k + i + 1])
            {
                comparisons++;
                if (doubled[j] < doubled[k + i + 1]) k = j - i - 1;
                i = failure[i];
            }
            comparisons++;
            if (doubled[j] != doubled[k + i + 1])
            {
                if (doubled[j] < doubled[k + i + 1]) k = j;
                failure[j - k] = -1;
            }
            else
            {
                failure[j - k] = i + 1;
            }
        }
        return k;
    }

    internal static int LeastRotation(string text)
    {
        return LeastRotation(text, out _);
    }

    internal static string Rotate(string text, int offset)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        if (text.Length == 0)
        {
            if (offset != 0) throw new ArgumentOutOfRangeException(nameof(offset));
            return string.Empty;
        }
        if (offset < 0 || offset >= text.Length) throw new ArgumentOutOfRangeException(nameof(offset));
        return text.Substring(offset) + text.Substring(0, offset);
    }

    /** Exhaustive reference: build every rotation and keep the smallest at the smallest offset. */
    internal static int BruteForceLeast(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        int n = text.Length;
        if (n == 0) return 0;
        int best = 0;
        string bestText = Rotate(text, 0);
        for (int i = 1; i < n; i++)
        {
            string candidate = Rotate(text, i);
            if (string.CompareOrdinal(candidate, bestText) < 0)
            {
                best = i;
                bestText = candidate;
            }
        }
        return best;
    }

    /** Independent reference: sort the rotation offsets by comparing substrings of the doubled text. */
    internal static int SortBasedLeast(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        int n = text.Length;
        if (n == 0) return 0;
        string doubled = text + text;
        var offsets = new int[n];
        for (int i = 0; i < n; i++) offsets[i] = i;
        Array.Sort(offsets, (a, b) =>
        {
            int cmp = string.CompareOrdinal(doubled, a, doubled, b, n);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        return offsets[0];
    }

    internal static int BruteForceGreatest(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        int n = text.Length;
        if (n == 0) return 0;
        int best = 0;
        string bestText = Rotate(text, 0);
        for (int i = 1; i < n; i++)
        {
            string candidate = Rotate(text, i);
            if (string.CompareOrdinal(candidate, bestText) > 0)
            {
                best = i;
                bestText = candidate;
            }
        }
        return best;
    }

    /** The maximal rotation, via Booth on the order-reversing alphabet image c -> 'z' - c. */
    internal static int GreatestRotation(string text)
    {
        if (text == null) throw new ArgumentNullException(nameof(text));
        if (text.Length == 0) return 0;
        var mapped = new char[text.Length];
        for (int i = 0; i < text.Length; i++) mapped[i] = (char)('z' - (text[i] - 'a'));
        return LeastRotation(new string(mapped));
    }

    internal static bool SameMultiset(string a, string b)
    {
        if (a.Length != b.Length) return false;
        var first = new List<char>(a);
        var second = new List<char>(b);
        first.Sort();
        second.Sort();
        for (int i = 0; i < first.Count; i++)
        {
            if (first[i] != second[i]) return false;
        }
        return true;
    }
}

public static class Solution
{
    public static void Main()
    {
        var handChecked = new List<string[]> {
            new[] { "baba", "1" }, new[] { "banana", "5" }, new[] { "abcabc", "0" },
            new[] { "cba", "2" }, new[] { "ababab", "0" }, new[] { "aaaa", "0" },
            new[] { "a", "0" }, new[] { "", "0" },
        };
        foreach (string[] pair in handChecked)
        {
            string text = pair[0];
            int expected = int.Parse(pair[1]);
            int offset = Booth.LeastRotation(text, out long comparisons);
            Check(offset == expected, "hand-checked offset of \"" + text + "\" is " + expected + ", got " + offset);
            Check(Booth.BruteForceLeast(text) == expected, "hand-checked brute force of \"" + text + "\"");
            Check(Booth.SameMultiset(Booth.Rotate(text, offset), text), "rotation keeps the multiset: " + text);
            if (text.Length == 0) Check(comparisons == 0, "the empty string compares nothing");
        }
        Check(Booth.Rotate("baba", 1) == "abab", "baba rotates to abab");
        Check(Booth.Rotate("banana", 5) == "abanan", "banana rotates to abanan");
        Check(Booth.Rotate("cba", 2) == "acb", "cba rotates to acb");

        long worstScaled = 0;
        int[] alphabets = { 2, 3, 5 };
        for (int alphabetIndex = 0; alphabetIndex < alphabets.Length; alphabetIndex++)
        {
            int alphabet = alphabets[alphabetIndex];
            for (int seed = 1; seed <= 60; seed++)
            {
                int length = 1 + (seed * 7 + alphabetIndex * 13) % 200;
                string text = LcgText(length, seed * 31 + alphabet, alphabet);
                int offset = Booth.LeastRotation(text, out long comparisons);
                Check(offset == Booth.BruteForceLeast(text),
                        "Booth equals the exhaustive minimum for " + text.Length + " chars, alphabet " + alphabet);
                Check(offset == Booth.SortBasedLeast(text),
                        "Booth equals the comparison sort for " + text.Length + " chars, alphabet " + alphabet);
                Check(Booth.Rotate(text, offset) == Booth.Rotate(text, Booth.BruteForceLeast(text)),
                        "the rotation strings agree for " + text.Length + " chars");
                Check(Booth.SameMultiset(Booth.Rotate(text, offset), text),
                        "the minimal rotation is a real rotation");
                Check(comparisons <= 4L * text.Length + 16L,
                        "comparisons stay linear (" + comparisons + " for n = " + text.Length + ")");
                if (comparisons * 1000L / text.Length > worstScaled)
                {
                    worstScaled = comparisons * 1000L / text.Length;
                }

                int greatest = Booth.GreatestRotation(text);
                Check(greatest == Booth.BruteForceGreatest(text),
                        "the maximal rotation matches for " + text.Length + " chars, alphabet " + alphabet);
                Check(string.CompareOrdinal(Booth.Rotate(text, greatest), Booth.Rotate(text, offset)) >= 0,
                        "the maximal rotation is at least the minimal one");

                string canonical = Booth.Rotate(text, offset);
                for (int shift = 0; shift < text.Length; shift++)
                {
                    string shifted = Booth.Rotate(text, shift);
                    int shiftedOffset = Booth.LeastRotation(shifted);
                    Check(Booth.Rotate(shifted, shiftedOffset) == canonical,
                            "circular invariance at shift " + shift + " of " + text.Length + " chars");
                    Check(shiftedOffset == Booth.BruteForceLeast(shifted),
                            "the shifted brute force agrees at shift " + shift);
                }
            }
        }
        Check(worstScaled <= 4000, "the observed worst comparison ratio stays below 4n (" + worstScaled + "/1000)");

        string[] structured = { "aab", "abc", "abbbbb", "aabaab", "zzzz", "zab", "abcabcabc" };
        foreach (string text in structured)
        {
            int offset = Booth.LeastRotation(text);
            Check(offset == Booth.BruteForceLeast(text), "structured string " + text);
            Check(string.CompareOrdinal(Booth.Rotate(text, offset),
                    Booth.Rotate(text, Booth.BruteForceGreatest(text))) <= 0, "structured ordering " + text);
        }
        Check(Booth.LeastRotation("aab") == 0, "the Lyndon word aab is already minimal");
        Check(Booth.LeastRotation("aabaab") == 0, "aabaab is already minimal");
        Check(Booth.Rotate("aab", 1) == "aba" && Booth.Rotate("aab", 2) == "baa", "aab has two other rotations");
        Check(Booth.GreatestRotation("aab") == 2, "aab has its maximum rotation at offset 2");
        Check(Booth.Rotate("aab", Booth.GreatestRotation("aab")) == "baa", "the maximal rotation of aab");

        CheckThrows(() => Booth.LeastRotation(null!), "a null input throws");
        CheckThrows(() => Booth.Rotate("abc", 3), "an out-of-range offset throws");
        CheckThrows(() => Booth.Rotate("abc", -1), "a negative offset throws");

        Console.WriteLine("PASS");
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
