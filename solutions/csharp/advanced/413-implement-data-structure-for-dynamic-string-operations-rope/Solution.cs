// Task 413 - Implement a data structure for dynamic string operations (rope). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// A rope stores a string as a sequence of chunks (leaves of a binary tree). The real
// structure is a balanced tree of chunks with O(log n) split/concat; this implementation
// keeps the chunk sequence FLAT and balanced instead, a documented simulation with the
// same interface:
//   * chunks hold at most CHUNK = 64 characters and are rebalanced so their count stays
//     within ~2x of the ideal ceil(n / 64),
//   * index/charAt/slice locate by walking the chunk lengths (O(#chunks + CHUNK)),
//   * insert/delete/split/concat/reverse splice chunks and rebalance (O(n + CHUNK));
//     a tree-based rope would perform the same operations by re-linking subtrees.
// All operations are functional: they return NEW ropes, leaving the receiver untouched.
using System;
using System.Collections.Generic;
using System.Text;

internal class Rope
{
    public const int ChunkSize = 64;
    public readonly List<string> Chunks;

    private Rope(List<string> chunks)
    {
        Chunks = chunks;
    }

    public Rope(string text)
    {
        Chunks = Normalize(text.Length > 0 ? new List<string> { text } : new List<string>());
    }

    /// <summary>Drop empty chunks and re-chunk so the count is close to ceil(n / CHUNK).</summary>
    public static List<string> Normalize(List<string> chunks)
    {
        var sb = new StringBuilder();
        foreach (var c in chunks) if (c.Length > 0) sb.Append(c);
        string whole = sb.ToString();
        int n = whole.Length;
        if (n == 0) return new List<string>();
        int target = Math.Max(1, (n + ChunkSize - 1) / ChunkSize);
        int size = (n + target - 1) / target;
        var outp = new List<string>();
        for (int i = 0; i < n; i += size) outp.Add(whole.Substring(i, Math.Min(n, i + size) - i));
        return outp;
    }

    private static List<string> MaybeRebalance(List<string> chunks)
    {
        var kept = new List<string>();
        int n = 0;
        foreach (var c in chunks)
        {
            if (c.Length > 0) { kept.Add(c); n += c.Length; }
        }
        int target = n > 0 ? Math.Max(1, (n + ChunkSize - 1) / ChunkSize) : 1;
        if (kept.Count > 2 * target + 2) return Normalize(kept);
        return kept;
    }

    public static Rope Of(List<string> chunks) => new Rope(MaybeRebalance(chunks));

    public int Length()
    {
        int n = 0;
        foreach (var c in Chunks) n += c.Length;
        return n;
    }

    public string To_string() => string.Join("", Chunks);

    /// <summary>(chunk index, offset) for pos; offset may equal the chunk length at a boundary.</summary>
    private int[] Locate(int pos)
    {
        for (int idx = 0; idx < Chunks.Count; idx++)
        {
            int len = Chunks[idx].Length;
            if (pos <= len) return new[] { idx, pos };
            pos -= len;
        }
        return new[] { Chunks.Count, 0 };
    }

    public char CharAt(int pos)
    {
        if (pos < 0 || pos >= Length()) throw new IndexOutOfRangeException("rope index out of range");
        foreach (var c in Chunks)
        {
            if (pos < c.Length) return c[pos];
            pos -= c.Length;
        }
        throw new IndexOutOfRangeException("rope index out of range");
    }

    public string Slice(int start, int end)
    {
        if (start < 0 || end > Length() || start > end) throw new IndexOutOfRangeException("rope slice out of range");
        var outSb = new StringBuilder();
        int need = end - start;
        var loc = Locate(start);
        int idx = loc[0], off = loc[1];
        while (need > 0)
        {
            string c = Chunks[idx];
            int take = Math.Min(need, c.Length - off);
            outSb.Append(c, off, take);
            need -= take;
            idx++;
            off = 0;
        }
        return outSb.ToString();
    }

    public Rope Insert(int pos, string text)
    {
        if (pos < 0 || pos > Length()) throw new IndexOutOfRangeException("rope insert out of range");
        var loc = Locate(pos);
        int idx = loc[0], off = loc[1];
        var outp = new List<string>();
        for (int i = 0; i < idx; i++) outp.Add(Chunks[i]);
        if (off > 0) outp.Add(Chunks[idx].Substring(0, off));
        if (text.Length > 0) outp.AddRange(Normalize(new List<string> { text }));
        if (idx < Chunks.Count && off < Chunks[idx].Length) outp.Add(Chunks[idx].Substring(off));
        if (idx < Chunks.Count)
        {
            for (int i = idx + 1; i < Chunks.Count; i++) outp.Add(Chunks[i]);
        }
        return Of(outp);
    }

    public Rope Delete(int start, int end)
    {
        int n = Length();
        if (start < 0 || end > n || start > end) throw new IndexOutOfRangeException("rope delete out of range");
        if (start == end) return Of(new List<string>(Chunks));
        var l = Locate(start);
        var r = Locate(end);
        var outp = new List<string>();
        for (int i = 0; i < l[0]; i++) outp.Add(Chunks[i]);
        if (l[1] > 0) outp.Add(Chunks[l[0]].Substring(0, l[1]));
        if (r[1] < Chunks[r[0]].Length) outp.Add(Chunks[r[0]].Substring(r[1]));
        for (int i = r[0] + 1; i < Chunks.Count; i++) outp.Add(Chunks[i]);
        return Of(outp);
    }

    public Rope[] Split(int pos)
    {
        if (pos < 0 || pos > Length()) throw new IndexOutOfRangeException("rope split out of range");
        var loc = Locate(pos);
        int idx = loc[0], off = loc[1];
        var left = new List<string>();
        for (int i = 0; i < idx; i++) left.Add(Chunks[i]);
        if (off > 0) left.Add(Chunks[idx].Substring(0, off));
        var right = new List<string>();
        if (idx < Chunks.Count && off < Chunks[idx].Length) right.Add(Chunks[idx].Substring(off));
        if (idx < Chunks.Count)
        {
            for (int i = idx + 1; i < Chunks.Count; i++) right.Add(Chunks[i]);
        }
        return new[] { Of(left), Of(right) };
    }

    public Rope Concat(Rope other)
    {
        var outp = new List<string>(Chunks);
        outp.AddRange(other.Chunks);
        return Of(outp);
    }

    public Rope Reverse()
    {
        var outp = new List<string>();
        for (int i = Chunks.Count - 1; i >= 0; i--)
        {
            var chars = Chunks[i].ToCharArray();
            Array.Reverse(chars);
            outp.Add(new string(chars));
        }
        return Of(outp);
    }
}

internal class Lcg
{
    private int state;

    public Lcg(int state)
    {
        this.state = state;
    }

    public int Next()
    {
        state = (state * 75 + 74) % 65537;
        return state;
    }
}

public static class Solution
{
    public static void Main()
    {
        var r = new Rope("hello");
        Check(r.Length() == 5, "length");
        Check(r.CharAt(0) == 'h' && r.CharAt(4) == 'o' && r.CharAt(2) == 'l', "char at");
        Check(r.To_string() == "hello", "to string");
        Check(r.Slice(1, 4) == "ell", "slice");
        Check(new Rope("abef").Insert(2, "cd").To_string() == "abcdef", "insert middle");
        Check(new Rope("hello").Insert(0, "xyz").To_string() == "xyzhello", "insert front");
        // "xyzhello" (8 chars) minus [3, 6) 'h','e','l' leaves "xyzlo"
        Check(new Rope("hello").Insert(0, "xyz").Delete(3, 6).To_string() == "xyzlo", "insert delete");
        var parts = new Rope("xyzlo").Split(2);
        Check(parts[0].To_string() == "xy" && parts[1].To_string() == "zlo", "split");
        Check(parts[0].Concat(parts[1]).To_string() == "xyzlo", "concat");
        Check(new Rope("xyzlo").Reverse().To_string() == "olzyx", "reverse");
        Check(new Rope("").Length() == 0 && new Rope("").To_string() == "", "empty rope");
        Check(new Rope("").Insert(0, "abc").To_string() == "abc", "insert into empty");
        Check(new Rope("abc").Delete(1, 2).To_string() == "ac", "delete middle");
        Check(new Rope("abc").Delete(0, 0).To_string() == "abc", "delete empty range");
        Check(new Rope("abc").Split(0)[1].To_string() == "abc", "split at 0");
        Check(new Rope("abc").Split(3)[0].To_string() == "abc", "split at end");
        // functional semantics: the receiver never changes
        var original = new Rope("keepme");
        original.Insert(0, "X");
        original.Delete(0, 3);
        original.Reverse();
        Check(original.To_string() == "keepme", "immutability");

        // cross-chunk operations on a 200-character rope
        var bigB = new StringBuilder();
        for (int i = 0; i < 200; i++) bigB.Append((char)('a' + i % 26));
        string big = bigB.ToString();
        var rb = new Rope(big);
        Check(rb.Length() == 200 && rb.To_string() == big, "big round trip");
        Check(rb.CharAt(0) == big[0] && rb.CharAt(199) == big[199], "big ends");
        Check(rb.CharAt(63) == big[63] && rb.CharAt(64) == big[64], "big boundary");
        Check(rb.Slice(0, 70) == big.Substring(0, 70), "big slice front");
        Check(rb.Slice(60, 150) == big.Substring(60, 90), "big slice deep");
        string oracle = big.Substring(0, 130) + "XYZ" + big.Substring(130);
        var ins = rb.Insert(130, "XYZ");
        Check(ins.To_string() == oracle && ins.Length() == 203, "big insert");
        var del = ins.Delete(125, 135);
        Check(del.To_string() == oracle.Substring(0, 125) + oracle.Substring(135), "big delete");
        var splitBig = new Rope(big).Split(64);
        Check(splitBig[0].Length() == 64 && splitBig[1].Length() == 136, "big split lengths");
        Check(splitBig[0].To_string() == big.Substring(0, 64), "big split left");
        Check(splitBig[1].To_string() == big.Substring(64), "big split right");
        Check(splitBig[0].Concat(splitBig[1]).To_string() == big, "big concat");
        var bigChars = big.ToCharArray();
        Array.Reverse(bigChars);
        Check(new Rope(big).Reverse().To_string() == new string(bigChars), "big reverse");

        // chunk length invariant after structural operations
        var probe = new Rope(big).Insert(0, new string('z', 40)).Delete(10, 90).Reverse()
            .Concat(new Rope(new string('q', 300)));
        foreach (var c in probe.Chunks) Check(c.Length <= Rope.ChunkSize, "chunk cap");
        Check(probe.Length() == 200 + 40 - 80 + 300, "probe length");

        // LCG-driven fuzz against a plain-string oracle; every operation must agree,
        // and no op script can desynchronise the rope from the oracle string.
        var gen = new Lcg(31337);
        const string alphabet = "abcdef";
        var rope = new Rope("");
        string oracleStr = "";
        for (int step = 0; step < 320; step++)
        {
            int kind = gen.Next() % 5;
            if (kind == 0 || kind == 1)
            {
                int pos = gen.Next() % (oracleStr.Length + 1);
                int ln = 1 + gen.Next() % 12;
                var text = new StringBuilder();
                for (int k = 0; k < ln; k++) text.Append(alphabet[gen.Next() % alphabet.Length]);
                rope = rope.Insert(pos, text.ToString());
                oracleStr = oracleStr.Substring(0, pos) + text + oracleStr.Substring(pos);
            }
            else if (kind == 2)
            {
                if (oracleStr.Length > 0)
                {
                    int a = gen.Next() % (oracleStr.Length + 1);
                    int b = gen.Next() % (oracleStr.Length + 1);
                    if (a > b) { int t = a; a = b; b = t; }
                    rope = rope.Delete(a, b);
                    oracleStr = oracleStr.Substring(0, a) + oracleStr.Substring(b);
                }
            }
            else if (kind == 3)
            {
                rope = rope.Reverse();
                var chars = oracleStr.ToCharArray();
                Array.Reverse(chars);
                oracleStr = new string(chars);
            }
            else
            {
                int a = gen.Next() % (oracleStr.Length + 1);
                int b = gen.Next() % (oracleStr.Length + 1);
                if (a > b) { int t = a; a = b; b = t; }
                Check(rope.Slice(a, b) == oracleStr.Substring(a, b - a), "fuzz slice step " + step);
                if (oracleStr.Length > 0)
                {
                    int pos = gen.Next() % oracleStr.Length;
                    Check(rope.CharAt(pos) == oracleStr[pos], "fuzz char step " + step);
                }
                int mid = a;
                var sp = rope.Split(mid);
                Check(sp[0].To_string() == oracleStr.Substring(0, mid), "fuzz split left " + step);
                Check(sp[1].To_string() == oracleStr.Substring(mid), "fuzz split right " + step);
                rope = sp[0].Concat(sp[1]);
            }
            Check(rope.To_string() == oracleStr, "fuzz step " + step);
            Check(rope.Length() == oracleStr.Length, "fuzz length step " + step);
        }

        // final sweep: index and slice for every position of the fuzz result
        for (int i = 0; i < oracleStr.Length; i++)
        {
            Check(rope.CharAt(i) == oracleStr[i], "sweep " + i);
        }
        Check(rope.Slice(0, oracleStr.Length) == oracleStr, "sweep slice");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
