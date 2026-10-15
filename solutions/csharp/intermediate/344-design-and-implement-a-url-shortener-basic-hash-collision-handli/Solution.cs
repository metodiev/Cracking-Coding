// Task 344 - URL shortener: deterministic hash, base62 codes, linear probing. (Tier: Intermediate, CTCI Ch 7)
using System;
using System.Collections.Generic;
using System.Text;

public class UrlShortener
{
    public const string Alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    public const long Base = 62;
    public const int CodeWidth = 6;
    public const long Space = 62L * 62 * 62 * 62 * 62 * 62;   // 62^6 = 56_800_235_584

    private readonly Dictionary<string, string> urlToCode = new Dictionary<string, string>();
    private readonly Dictionary<string, string> codeToUrl = new Dictionary<string, string>();
    public long Probes { get; private set; }

    /// <summary>Deliberately simple additive hash so tests can engineer collisions.</summary>
    public static long HashUrl(string url)
    {
        long total = 0;
        foreach (char ch in url) total += ch;
        return total % Space;
    }

    public static string EncodeBase62(long value)
    {
        value %= Space;
        var digits = new char[CodeWidth];
        for (int i = CodeWidth - 1; i >= 0; i--)
        {
            digits[i] = Alphabet[(int)(value % Base)];
            value /= Base;
        }
        return new string(digits);
    }

    /// <summary>Independent inverse; used by the tests to verify the probing order.</summary>
    public static long DecodeBase62(string code)
    {
        long value = 0;
        foreach (char ch in code)
        {
            int index = Alphabet.IndexOf(ch);
            if (index < 0) throw new ArgumentException("invalid base62 character '" + ch + "'");
            value = value * Base + index;
        }
        return value;
    }

    public string Shorten(string url)
    {
        if (urlToCode.TryGetValue(url, out string? existing)) return existing;   // idempotent
        long index = HashUrl(url);
        while (true)
        {
            string code = EncodeBase62(index);
            if (!codeToUrl.ContainsKey(code))
            {
                urlToCode[url] = code;
                codeToUrl[code] = url;
                return code;
            }
            index = (index + 1) % Space;                                         // linear probing
            Probes++;
        }
    }

    public string? Expand(string code) => codeToUrl.TryGetValue(code, out string? url) ? url : null;

    public int Size => urlToCode.Count;
}

public static class Solution
{
    public static void Main()
    {
        var shortener = new UrlShortener();
        string url = "http://example.com/articles/42";
        string code = shortener.Shorten(url);
        Check(code.Length == UrlShortener.CodeWidth, "code width");
        foreach (char ch in code) Check(UrlShortener.Alphabet.IndexOf(ch) >= 0, "code alphabet");
        Check(code == UrlShortener.EncodeBase62(UrlShortener.HashUrl(url)), "initial slot free, no probing");
        Check(shortener.Expand(code) == url, "expand round trip");
        Check(shortener.Shorten(url) == code, "idempotent");
        Check(shortener.Size == 1, "size 1");
        Check(shortener.Probes == 0, "no probes yet");
        Check(shortener.Expand("zzzzzz") == null, "unknown code returns null");
        Check(shortener.Expand("") == null, "empty code unknown");
        Check(shortener.Expand("!!!!!!") == null, "invalid code unknown");

        Check(UrlShortener.EncodeBase62(0) == "000000", "encode 0");
        Check(UrlShortener.EncodeBase62(UrlShortener.Space - 1) == "ZZZZZZ", "encode max");
        int seed = 9137;
        for (int i = 0; i < 200; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            long value = seed * 12345L % UrlShortener.Space;
            Check(UrlShortener.DecodeBase62(UrlShortener.EncodeBase62(value)) == value, "base62 round trip");
        }

        // engineered collision: anagram paths hash to the same slot
        string u1 = "http://x.com/pages/ab";
        string u2 = "http://x.com/pages/ba";
        Check(UrlShortener.HashUrl(u1) == UrlShortener.HashUrl(u2), "engineered collision");
        var collision = new UrlShortener();
        string c1 = collision.Shorten(u1);
        string c2 = collision.Shorten(u2);
        Check(c1 != c2, "distinct codes despite collision");
        Check(UrlShortener.DecodeBase62(c1) == UrlShortener.HashUrl(u1), "c1 sits at the initial slot");
        Check(UrlShortener.DecodeBase62(c2) == (UrlShortener.DecodeBase62(c1) + 1) % UrlShortener.Space, "one probe step");
        Check(collision.Probes == 1, "exactly one probe");
        Check(collision.Expand(c1) == u1 && collision.Expand(c2) == u2, "both expand correctly");

        // many URLs, every one in a known-colliding pair
        var codes = new List<string>();
        var batch = new UrlShortener();
        for (int k = 0; k < 250; k++)
        {
            string ab = "http://bench/" + k + "/ab";
            string ba = "http://bench/" + k + "/ba";
            Check(UrlShortener.HashUrl(ab) == UrlShortener.HashUrl(ba), "pair collision " + k);
            foreach (string u in new[] { ab, ba })
            {
                string c = batch.Shorten(u);
                Check(batch.Shorten(u) == c, "idempotent batch " + u);
                Check(batch.Expand(c) == u, "round trip batch " + u);
                Check(c.Length == UrlShortener.CodeWidth, "batch width");
                codes.Add(c);
            }
        }
        Check(codes.Count == 500, "500 codes");
        Check(new HashSet<string>(codes).Count == 500, "all distinct despite collisions");
        Check(batch.Probes > 0, "collisions were exercised");
        Check(batch.Size == 500, "batch size");

        var again = new UrlShortener();
        var reproduced = new List<string>();
        for (int k = 0; k < 250; k++)
        {
            reproduced.Add(again.Shorten("http://bench/" + k + "/ab"));
            reproduced.Add(again.Shorten("http://bench/" + k + "/ba"));
        }
        Check(string.Join(",", reproduced) == string.Join(",", codes), "deterministic replay");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
