// Task 388 - Implement string compression (run-length encoding + decode). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Text;

public static class Solution
{
    private static bool IsLetter(char ch)
    {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
    }

    private static bool IsDigitChar(char ch)
    {
        return ch >= '0' && ch <= '9';
    }

    /// <summary>Runs of length 1 stay bare; runs of length >= 2 become character + decimal count.</summary>
    private static string RleEncode(string text)
    {
        var outBuilder = new StringBuilder();
        int i = 0;
        while (i < text.Length)
        {
            int j = i;
            while (j < text.Length && text[j] == text[i]) j++;
            int run = j - i;
            outBuilder.Append(text[i]);
            if (run >= 2) outBuilder.Append(run);
            i = j;
        }
        return outBuilder.ToString();
    }

    /// <summary>Rejects non-canonical counts 0 and 1, and non-letter characters.</summary>
    private static string RleDecode(string encoded)
    {
        var outBuilder = new StringBuilder();
        int i = 0;
        while (i < encoded.Length)
        {
            char ch = encoded[i];
            if (!IsLetter(ch)) throw new InvalidOperationException("expected a letter at position " + i);
            i++;
            int count = 0;
            bool hasDigits = false;
            while (i < encoded.Length && IsDigitChar(encoded[i]))
            {
                count = count * 10 + (encoded[i] - '0');
                hasDigits = true;
                i++;
            }
            if (!hasDigits)
            {
                outBuilder.Append(ch);
                continue;
            }
            if (count < 2) throw new InvalidOperationException("count " + count + " is not canonical");
            outBuilder.Append(ch, count);
        }
        return outBuilder.ToString();
    }

    /// <summary>First character of every maximal run, in order (the letters of the encoding).</summary>
    private static string RunCharacters(string text)
    {
        var outBuilder = new StringBuilder();
        for (int i = 0; i < text.Length; i++)
            if (i == 0 || text[i] != text[i - 1]) outBuilder.Append(text[i]);
        return outBuilder.ToString();
    }

    private static int seed = 5150;

    private static int NextSeed()
    {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void Main()
    {
        Check(RleEncode("aaabbc") == "a3b2c", "encode aaabbc");
        Check(RleDecode("a3b2c") == "aaabbc", "decode a3b2c");

        // empty string, single characters, no runs
        Check(RleEncode("") == "" && RleDecode("") == "", "empty string");
        Check(RleEncode("a") == "a" && RleDecode("a") == "a", "single character");
        Check(RleEncode("abcd") == "abcd" && RleDecode("abcd") == "abcd", "no runs");

        // runs of exactly two keep their length; runs of three or more shrink
        Check(RleEncode("aab") == "a2b", "run of two");
        Check(RleEncode("abb") == "ab2", "run of two at the end");
        Check(RleEncode("aabbaa") == "a2b2a2", "alternating runs");
        Check(RleDecode("a2b2a2") == "aabbaa", "decode alternating runs");
        Check(RleEncode("aaabccccd") == "a3bc4d", "mixed runs");
        Check(RleDecode("a3bc4d") == "aaabccccd", "decode mixed runs");
        Check(RleDecode("z2") == "zz", "decode z2");

        // multi-digit counts (the 9 -> 10 boundary)
        Check(RleEncode(new string('a', 9) + "b") == "a9b", "run of nine");
        Check(RleEncode(new string('a', 10)) == "a10" && RleDecode("a10") == new string('a', 10),
                "run of ten");
        Check(RleEncode(new string('a', 12)) == "a12" && RleDecode("a12") == new string('a', 12),
                "run of twelve");
        Check(RleDecode("a99") == new string('a', 99), "run of ninety-nine");

        // canonical outputs round trip through decode then encode
        string[] canonical = { "a3b2c", "a2b2a2", "a9b", "a10", "a12", "a3bc4d", "abcd", "a", "a99" };
        foreach (string encoded in canonical)
            Check(RleEncode(RleDecode(encoded)) == encoded, "canonical " + encoded);

        // malformed input is rejected (documented)
        foreach (string bad in new[] { "3", "a0", "a1", "ab1" })
        {
            bool threw = false;
            try { RleDecode(bad); }
            catch (InvalidOperationException) { threw = true; }
            Check(threw, "malformed rejected " + bad);
        }

        // LCG strings: round trips, canonical idempotence, and the letters of the encoding
        for (int trial = 0; trial < 200; trial++)
        {
            int length = NextSeed() % 60;
            int alphabetSize = 2 + NextSeed() % 4;
            var builder = new StringBuilder();
            for (int i = 0; i < length; i++)
                builder.Append((char)('a' + NextSeed() % alphabetSize));
            string text = builder.ToString();
            string encoded = RleEncode(text);
            Check(encoded.Length <= text.Length, "encoding never grows " + trial);
            var letters = new StringBuilder();
            foreach (char ch in encoded)
                if (IsLetter(ch)) letters.Append(ch);
            Check(RunCharacters(text) == letters.ToString(), "run letters " + trial);
            Check(RleDecode(encoded) == text, "round trip " + trial);
            Check(RleEncode(RleDecode(encoded)) == encoded, "canonical idempotence " + trial);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
