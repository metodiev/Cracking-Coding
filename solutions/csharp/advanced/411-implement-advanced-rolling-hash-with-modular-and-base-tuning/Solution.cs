// Task 411 - Implement advanced rolling hash with modular and base tuning. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Polynomial hashing: hash(s) = sum(s[i] * base^(n-1-i)) mod modulus. Tuning notes:
//   * double hashing uses bases 131/137 with the large primes 1_000_000_007 / 1_000_000_009;
//     with those sizes every cross product fits comfortably in a signed 64-bit long.
//   * the demo pair modulus 65537 with base 2 is deliberately bad: 2^16 = -1 and 2^32 = 1
//     (mod 65537), so same-length strings that only shift a character by 32 positions
//     collide — that is the constructed single-hash collision asserted below.
//   * base 1 degenerates the hash to the plain character sum, so anagrams collide.
//   * prefix hashes make every substring fingerprint O(1); MultiHash combines several
//     (base, modulus) pairs so an accidental equality must beat all of them at once.
using System;

internal class RollingHash
{
    public readonly long Base;
    public readonly long Mod;
    private readonly long[] pref;
    private readonly long[] pow;

    public RollingHash(string s, long baseValue, long mod)
    {
        Base = baseValue;
        Mod = mod;
        pref = new long[s.Length + 1];
        for (int i = 0; i < s.Length; i++)
        {
            pref[i + 1] = (pref[i] * baseValue + s[i]) % mod;
        }
        pow = new long[s.Length + 1];
        pow[0] = 1;
        for (int i = 1; i <= s.Length; i++)
        {
            pow[i] = pow[i - 1] * baseValue % mod;
        }
    }

    /// <summary>Fingerprint of s[l:r].</summary>
    public long Get(int l, int r)
    {
        return ((pref[r] - pref[l] * pow[r - l]) % Mod + Mod) % Mod;
    }
}

internal class MultiHash
{
    private readonly RollingHash[] hashes;

    public MultiHash(string s, long[] bases, long[] mods)
    {
        hashes = new RollingHash[bases.Length];
        for (int i = 0; i < bases.Length; i++) hashes[i] = new RollingHash(s, bases[i], mods[i]);
    }

    public long[] Fingerprint(int l, int r)
    {
        var outp = new long[hashes.Length];
        for (int i = 0; i < hashes.Length; i++) outp[i] = hashes[i].Get(l, r);
        return outp;
    }

    public bool Equal(int l1, int r1, int l2, int r2)
    {
        if (r1 - l1 != r2 - l2) return false;
        var a = Fingerprint(l1, r1);
        var b = Fingerprint(l2, r2);
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }
}

public static class Solution
{
    private const long Mod1 = 1_000_000_007L;
    private const long Mod2 = 1_000_000_009L;
    private const long Base1 = 131;
    private const long Base2 = 137;
    private const long DemoMod = 65537;
    private const long DemoBase = 2;

    public static long PolyHash(string s, long baseValue, long mod)
    {
        long h = 0;
        for (int i = 0; i < s.Length; i++)
        {
            h = (h * baseValue + s[i]) % mod;
        }
        return h;
    }

    private static int seed = 2024;

    private static string LcgString(int length, string alphabet)
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < length; i++)
        {
            seed = (seed * 75 + 74) % 65537;
            sb.Append(alphabet[seed % alphabet.Length]);
        }
        return sb.ToString();
    }

    public static void Main()
    {
        // hand values: "ab" = 97*131 + 98, "abc" = 97*131^2 + 98*131 + 99
        Check(PolyHash("ab", Base1, Mod1) == 12805, "hand ab");
        Check(PolyHash("abc", Base1, Mod1) == 1677554, "hand abc");

        var h = new RollingHash("abc", Base1, Mod1);
        Check(h.Get(0, 3) == 1677554, "prefix whole");
        Check(h.Get(0, 1) == 97, "prefix a");
        Check(h.Get(1, 2) == 98, "prefix b");
        Check(h.Get(2, 3) == 99, "prefix c");
        Check(h.Get(1, 3) == PolyHash("bc", Base1, Mod1), "prefix bc");
        Check(h.Get(0, 0) == 0 && h.Get(3, 3) == 0, "empty substring");

        // constructed single-hash collision: base 2, modulus 65537, length 33,
        // characters 32 positions apart swap places — weights repeat because 2^32 = 1.
        string s = "a" + new string('c', 31) + "b";
        string t = "b" + new string('c', 31) + "a";
        Check(s != t && s.Length == 33 && t.Length == 33, "collision strings differ");
        Check(PolyHash(s, DemoBase, DemoMod) == PolyHash(t, DemoBase, DemoMod), "single-hash collision");
        Check(new RollingHash(s, DemoBase, DemoMod).Get(0, 33)
              == new RollingHash(t, DemoBase, DemoMod).Get(0, 33), "rolling collision");

        // double hashing with tuned large primes separates them again
        long[] bases = { Base1, Base2 };
        long[] mods = { Mod1, Mod2 };
        var dblS = new MultiHash(s, bases, mods);
        var dblT = new MultiHash(t, bases, mods);
        var fs = dblS.Fingerprint(0, 33);
        var ft = dblT.Fingerprint(0, 33);
        bool same = true;
        for (int i = 0; i < fs.Length; i++) if (fs[i] != ft[i]) same = false;
        Check(!same, "double hash distinguishes");

        // base 1 degenerates to the character sum
        Check(PolyHash("ab", 1, Mod1) == PolyHash("ba", 1, Mod1), "base 1 anagram collision");
        Check(PolyHash("ab", Base1, Mod1) != PolyHash("ba", Base1, Mod1), "base 131 separates");
        Check(PolyHash("A", 131, 256) == 'A' % 256, "small modulus trims");

        for (int round = 0; round < 6; round++)
        {
            string text = LcgString(61, "abcde");
            var multi = new MultiHash(text, bases, mods);
            for (int i = 0; i < text.Length; i++)
            {
                for (int j = i; j <= text.Length; j++)
                {
                    Check(multi.Equal(i, j, i, j), "reflexive " + i + " " + j);
                }
            }
            for (int q = 0; q < 60; q++)
            {
                seed = (seed * 75 + 74) % 65537;
                int l1 = seed % text.Length;
                seed = (seed * 75 + 74) % 65537;
                int l2 = seed % text.Length;
                seed = (seed * 75 + 74) % 65537;
                int maxSpan = Math.Min(text.Length - l1, text.Length - l2);
                int span = seed % (maxSpan + 1);
                bool oracle = text.Substring(l1, span) == text.Substring(l2, span);
                Check(multi.Equal(l1, l1 + span, l2, l2 + span) == oracle, "oracle " + q);
            }
        }

        // every substring fingerprint equals the direct polynomial hash
        const string miss = "mississippi";
        foreach (long baseValue in bases)
        {
            foreach (long mod in mods)
            {
                var rh = new RollingHash(miss, baseValue, mod);
                for (int l = 0; l <= miss.Length; l++)
                {
                    for (int r = l; r <= miss.Length; r++)
                    {
                        Check(rh.Get(l, r) == PolyHash(miss.Substring(l, r - l), baseValue, mod), "direct " + l + " " + r);
                    }
                }
            }
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
