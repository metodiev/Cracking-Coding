// Task 175 - Multiply large integers as strings. (Tier: Beginner, CTCI Ch 6)
using System;
using System.Numerics;
using System.Text;

public static class Solution
{
    public static string MultiplyStrings(string a, string b)
    {
        if (a == "0" || b == "0") return "0";
        bool negative = (a[0] == '-') != (b[0] == '-');
        string da = a.TrimStart('-');
        string db = b.TrimStart('-');
        var result = new int[da.Length + db.Length];
        for (int i = da.Length - 1; i >= 0; i--)
            for (int j = db.Length - 1; j >= 0; j--)
                result[i + j + 1] += (da[i] - '0') * (db[j] - '0');
        for (int k = result.Length - 1; k > 0; k--)
        {
            result[k - 1] += result[k] / 10;
            result[k] %= 10;
        }
        var sb = new StringBuilder();
        foreach (var d in result)
            if (sb.Length > 0 || d != 0) sb.Append(d);
        if (sb.Length == 0) return "0";
        return negative ? "-" + sb : sb.ToString();
    }

    public static void Main()
    {
        Check(MultiplyStrings("123", "456") == "56088", "123*456");
        Check(MultiplyStrings("0", "999") == "0", "zero");
        Check(MultiplyStrings("-12", "12") == "-144", "negative");
        Check(MultiplyStrings("-12", "-12") == "144", "double negative");
        Check(MultiplyStrings("99999", "99999") == "9999800001", "carry chain");

        var rng = new Random(11);
        for (int t = 0; t < 30; t++)
        {
            var x = RandomBigInteger(rng, 1 + rng.Next(120));
            var y = RandomBigInteger(rng, 1 + rng.Next(120));
            Check(MultiplyStrings(x.ToString(), y.ToString()) == (x * y).ToString(), "random " + t);
        }

        var nines = new string('9', 100);
        Check(MultiplyStrings(nines, nines) == (BigInteger.Parse(nines) * BigInteger.Parse(nines)).ToString(),
            "100 nines");
        Console.WriteLine("PASS");
    }

    private static BigInteger RandomBigInteger(Random rng, int digits)
    {
        var sb = new StringBuilder();
        sb.Append((char)('1' + rng.Next(9)));
        for (int i = 1; i < digits; i++) sb.Append((char)('0' + rng.Next(10)));
        return BigInteger.Parse(sb.ToString());
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
