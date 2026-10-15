// Task 168 - Implement basic checksum (sum bytes). (Tier: Beginner, CTCI Ch 5)
using System;
using System.Text;

public static class Solution
{
    public static int Checksum8(byte[] data)
    {
        int sum = 0;
        foreach (var b in data) sum += b;
        return sum % 256;
    }

    public static int Checksum8(string text) => Checksum8(Encoding.UTF8.GetBytes(text));

    public static int Checksum16(byte[] data)
    {
        int sum = 0;
        foreach (var b in data) sum += b;
        return sum % 65536;
    }

    public static int Checksum16(string text) => Checksum16(Encoding.UTF8.GetBytes(text));

    public static void Main()
    {
        Check(Checksum8("") == 0, "empty");
        Check(Checksum8("abc") == 38, "abc mod 256");
        Check(Checksum16("abc") == 294, "abc 16-bit");
        Check(Checksum8(new byte[] { 0xFF, 0x01 }) == 0, "wraps");
        Check(Checksum16(new byte[] { 0xFF, 0xFF }) == 510, "16-bit sum");

        var original = new byte[50];
        for (int i = 0; i < 50; i++) original[i] = (byte)i;
        var corrupt = (byte[])original.Clone();
        corrupt[10] ^= 0x01;
        Check(Checksum8(original) != Checksum8(corrupt), "detects single-byte flip");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
