// Task 163 - Implement basic encryption/decryption with XOR. (Tier: Beginner, CTCI Ch 5)
using System;
using System.Text;

public static class Solution
{
    public static byte[] XorBytes(byte[] data, byte[] key)
    {
        if (key.Length == 0) throw new ArgumentException("key must not be empty");
        var outArr = new byte[data.Length];
        for (int i = 0; i < data.Length; i++) outArr[i] = (byte)(data[i] ^ key[i % key.Length]);
        return outArr;
    }

    public static string XorText(string text, string key)
    {
        var cipher = XorBytes(Encoding.UTF8.GetBytes(text), Encoding.UTF8.GetBytes(key));
        var sb = new StringBuilder();
        foreach (var b in cipher) sb.Append(b.ToString("x2"));
        return sb.ToString();
    }

    public static string DecryptText(string hexCipher, string key)
    {
        var cipher = new byte[hexCipher.Length / 2];
        for (int i = 0; i < cipher.Length; i++)
            cipher[i] = Convert.ToByte(hexCipher.Substring(2 * i, 2), 16);
        return Encoding.UTF8.GetString(XorBytes(cipher, Encoding.UTF8.GetBytes(key)));
    }

    public static void Main()
    {
        string message = "attack at dawn";
        string key = "s3cret";
        string cipher = XorText(message, key);
        Check(cipher != message, "cipher differs from plaintext");
        Check(DecryptText(cipher, key) == message, "round trip");
        Check(XorText("", key) == "" && DecryptText("", key) == "", "empty input");
        var raw = new byte[] { 0, 0xFF, 0x7F };
        var k = Encoding.UTF8.GetBytes("k");
        Check(Convert.ToBase64String(XorBytes(XorBytes(raw, k), k)) == Convert.ToBase64String(raw), "self-inverse");
        Check(DecryptText(cipher, "other") != message, "wrong key is garbage");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
