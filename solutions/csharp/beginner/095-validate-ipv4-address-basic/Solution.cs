// Task 095 - Validate IPv4 address (basic). (Tier: Beginner, CTCI Ch 1)
// Leading zeros are rejected ("1.2.3.04" invalid), except the value "0" itself.
using System;

public static class Solution
{
    public static bool IsValidIpv4(string s)
    {
        string[] parts = s.Split('.');
        if (parts.Length != 4) return false;
        foreach (string part in parts)
        {
            if (part.Length == 0 || part.Length > 3) return false;
            foreach (char c in part)
            {
                if (c < '0' || c > '9') return false;
            }
            if (part.Length > 1 && part[0] == '0') return false;
            if (int.Parse(part) > 255) return false;
        }
        return true;
    }

    public static void Main()
    {
        Check(IsValidIpv4("192.168.1.1"), "normal");
        Check(IsValidIpv4("0.0.0.0"), "zeros");
        Check(IsValidIpv4("255.255.255.255"), "max");
        Check(IsValidIpv4("10.0.0.255"), "mixed");
        Check(IsValidIpv4("1.2.3.0"), "trailing zero");
        Check(!IsValidIpv4("256.1.1.1"), "part > 255");
        Check(!IsValidIpv4("1.2.3.256"), "last > 255");
        Check(!IsValidIpv4("1.2.3"), "three parts");
        Check(!IsValidIpv4("1.2.3.4.5"), "five parts");
        Check(!IsValidIpv4(""), "empty");
        Check(!IsValidIpv4(".1.2.3"), "leading dot");
        Check(!IsValidIpv4("1.2.3."), "trailing dot");
        Check(!IsValidIpv4("1..2.3"), "empty part");
        Check(!IsValidIpv4("1.2.3.a"), "letters");
        Check(!IsValidIpv4("1.2.3.4x"), "trailing letter");
        Check(!IsValidIpv4("01.2.3.4"), "leading zero");
        Check(!IsValidIpv4("1.2.3.00"), "leading zeros last");
        Check(!IsValidIpv4("1.2.3.1234"), "part too long");
        Check(!IsValidIpv4("-1.2.3.4"), "negative");
        Check(!IsValidIpv4("1.2.3.4 "), "trailing space");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
