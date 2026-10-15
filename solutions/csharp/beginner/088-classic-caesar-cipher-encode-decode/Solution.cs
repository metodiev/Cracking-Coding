// Task 088 - Classic Caesar cipher encode/decode. (Tier: Beginner, CTCI Ch 1)
using System;

public static class Solution
{
    public static string CaesarEncode(string text, int shift)
    {
        int normalized = ((shift % 26) + 26) % 26;
        char[] outChars = new char[text.Length];
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            if (ch >= 'a' && ch <= 'z')
            {
                outChars[i] = (char)('a' + (ch - 'a' + normalized) % 26);
            }
            else
            {
                outChars[i] = ch;
            }
        }
        return new string(outChars);
    }

    public static string CaesarDecode(string text, int shift)
    {
        return CaesarEncode(text, -shift);
    }

    public static void Main()
    {
        Check(CaesarEncode("abc", 3) == "def", "abc +3");
        Check(CaesarEncode("xyz", 3) == "abc", "wrap around");
        Check(CaesarEncode("hello, world!", 3) == "khoor, zruog!", "phrase");
        Check(CaesarEncode("abc", -3) == "xyz", "negative shift");
        Check(CaesarEncode("abc", 29) == "def", "shift > 26");
        Check(CaesarEncode("abc", 26) == "abc", "full rotation");
        Check(CaesarEncode("abc", -29) == "xyz", "large negative shift");
        Check(CaesarEncode("", 5) == "", "empty");

        Check(CaesarDecode("khoor", 3) == "hello", "decode");
        Check(CaesarDecode("def", 3) == "abc", "decode back");
        Check(CaesarDecode(CaesarEncode("attack at dawn!", 7), 7) == "attack at dawn!", "round trip 7");
        Check(CaesarDecode(CaesarEncode("the quick brown fox 123", 13), 13) == "the quick brown fox 123", "round trip 13");

        // non-letters are unchanged; uppercase is not lowercase, so it is left alone
        Check(CaesarEncode("a1 B2 c!", 1) == "b1 B2 d!", "non-letters unchanged");
        Check(CaesarEncode("Hello", 1) == "Hfmmp", "uppercase unchanged");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
