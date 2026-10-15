// Task 045 - Replace spaces with %20 (URLify). (Tier: Beginner, CTCI Ch 1)
using System;
using System.Collections.Generic;
using System.Text;



public static class Solution
{
    public static char[] Urlify(char[] chars, int length)
    {
        int write = chars.Length - 1;
        for (int read = length - 1; read >= 0; read--)
        {
            if (chars[read] == ' ')
            {
                chars[write] = '0';
                chars[write - 1] = '2';
                chars[write - 2] = '%';
                write -= 3;
            }
            else
            {
                chars[write--] = chars[read];
            }
        }
        return chars;
    }

    public static void Main()
    {
        Check(new string(Urlify(("a b c" + new string(' ', 4)).ToCharArray(), 5)) == "a%20b%20c", "urlify");
        Check(new string(Urlify("abc".ToCharArray(), 3)) == "abc", "no spaces");
        Console.WriteLine("PASS");
    }
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
