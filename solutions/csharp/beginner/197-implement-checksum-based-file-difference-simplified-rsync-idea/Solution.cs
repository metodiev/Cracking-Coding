// Task 197 - Implement checksum-based file difference (simplified rsync idea). (Tier: Beginner, CTCI Beyond)
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

public static class Solution
{
    public static void Main()
    {
        byte[] a = new byte[40];
        for (int i = 0; i < a.Length; i++) a[i] = (byte)i;

        Check(SameMatches(FindBlockMatches(a, a, 8), (0, 0), (1, 8), (2, 16), (3, 24), (4, 32)),
                "identical files match at their own offsets");

        byte[] changed = (byte[])a.Clone();
        changed[10] ^= 0xFF;                                       // one byte inside block 1
        Check(SameMatches(FindBlockMatches(a, changed, 8), (0, 0), (2, 16), (3, 24), (4, 32)),
                "one changed byte drops that block only");

        byte[] other = new byte[64];
        Array.Fill(other, (byte)9);
        Check(FindBlockMatches(a, other, 8).Count == 0, "completely different content");

        byte[] header = Encoding.UTF8.GetBytes("HEADER");
        byte[] shifted = new byte[header.Length + a.Length];
        Array.Copy(header, 0, shifted, 0, header.Length);
        Array.Copy(a, 0, shifted, header.Length, a.Length);
        Check(SameMatches(FindBlockMatches(a, shifted, 8), (0, 6), (1, 14), (2, 22), (3, 30), (4, 38)),
                "shifted match positions");

        Check(SameMatches(FindBlockMatches(Encoding.UTF8.GetBytes("ABABABAB"), Encoding.UTF8.GetBytes("ZZABABABAB"), 4),
                (0, 2), (1, 2)), "duplicate blocks share the first occurrence");
        Check(SameMatches(FindBlockMatches(Encoding.UTF8.GetBytes("abc"), Encoding.UTF8.GetBytes("xxabcxx"), 8), (0, 2)),
                "partial tail block");
        Check(FindBlockMatches(new byte[0], a, 8).Count == 0, "empty file A");
        Check(WeakChecksum(Encoding.UTF8.GetBytes("AB"), 0, 2) != WeakChecksum(Encoding.UTF8.GetBytes("BA"), 0, 2),
                "weak checksum is order-sensitive");
        CheckThrows();
        Console.WriteLine("PASS");
    }

    private static Dictionary<int, int> FindBlockMatches(byte[] a, byte[] b, int blockSize)
    {
        if (blockSize <= 0) throw new ArgumentException("block_size must be positive");
        var blocks = new List<byte[]>();
        for (int i = 0; i < a.Length; i += blockSize)
        {
            int length = Math.Min(blockSize, a.Length - i);
            var block = new byte[length];
            Array.Copy(a, i, block, 0, length);
            blocks.Add(block);
        }

        var indexByWeak = new Dictionary<long, List<int>>();
        for (int index = 0; index < blocks.Count; index++)
        {
            long weak = WeakChecksum(blocks[index], 0, blocks[index].Length);
            if (!indexByWeak.TryGetValue(weak, out var list))
            {
                list = new List<int>();
                indexByWeak[weak] = list;
            }
            list.Add(index);
        }

        var matches = new Dictionary<int, int>();
        for (int start = 0; start + blockSize <= b.Length; start++)
        {
            if (!indexByWeak.TryGetValue(WeakChecksum(b, start, blockSize), out var candidates)) continue;
            foreach (int index in candidates)
            {
                if (matches.ContainsKey(index) || blocks[index].Length != blockSize) continue;
                if (StrongHash(b, start, blockSize) == StrongHash(blocks[index], 0, blockSize))
                {
                    matches[index] = start;                        // weak -> strong confirmation
                }
            }
        }

        if (blocks.Count > 0)
        {
            int tailIndex = blocks.Count - 1;
            byte[] tail = blocks[tailIndex];
            if (tail.Length < blockSize && !matches.ContainsKey(tailIndex))
            {
                int position = IndexOf(b, tail);                   // simplified tail lookup
                if (position >= 0) matches[tailIndex] = position;
            }
        }
        return matches;
    }

    private static long WeakChecksum(byte[] data, int offset, int length)
    {
        long a = 0;
        long b = 0;
        for (int i = 0; i < length; i++)
        {
            int value = data[offset + i];
            a = (a + value) & 0xFFFF;
            b = (b + (long)(length - i) * value) & 0xFFFF;
        }
        return (a << 16) | b;
    }

    private static string StrongHash(byte[] data, int offset, int length)
    {
        return Convert.ToHexString(SHA256.HashData(data.AsSpan(offset, length))).ToLowerInvariant();
    }

    private static int IndexOf(byte[] haystack, byte[] needle)
    {
        for (int i = 0; i + needle.Length <= haystack.Length; i++)
        {
            bool found = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (haystack[i + j] != needle[j])
                {
                    found = false;
                    break;
                }
            }
            if (found) return i;
        }
        return -1;
    }

    private static bool SameMatches(Dictionary<int, int> actual, params (int Index, int Position)[] expected)
    {
        if (actual.Count != expected.Length) return false;
        foreach (var (index, position) in expected)
        {
            if (!actual.TryGetValue(index, out int actualPosition) || actualPosition != position) return false;
        }
        return true;
    }

    private static void CheckThrows()
    {
        try
        {
            FindBlockMatches(new byte[] { 1 }, new byte[] { 1 }, 0);
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: expected error for block_size 0");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
