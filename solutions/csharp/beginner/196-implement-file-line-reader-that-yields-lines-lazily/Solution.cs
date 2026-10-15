// Task 196 - Implement file line reader that yields lines lazily. (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;
using System.IO;

public static class Solution
{
    private static IEnumerable<string> ReadLines(string path)
    {
        // The iterator body (and the file open) runs only when enumeration starts.
        using (var reader = new StreamReader(path))
        {
            string? line;
            while ((line = reader.ReadLine()) != null) yield return line;
        }
    }

    public static void Main()
    {
        string dir = Path.Combine(Path.GetTempPath(), "line-reader-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string file = Path.Combine(dir, "sample.txt");
        try
        {
            File.WriteAllText(file, "alpha\nbeta\n\ngamma\ndelta");   // 5 lines, last one unterminated

            // Laziness: the enumerable does not touch the file until it is enumerated.
            var lazy = ReadLines(file);
            File.Delete(file);
            Check(FirstEnumerationThrows(lazy), "file opened lazily");

            File.WriteAllText(file, "alpha\nbeta\n\ngamma\ndelta");

            var lines = new List<string>(ReadLines(file));
            Check(lines.Count == 5 && lines[0] == "alpha" && lines[1] == "beta" && lines[2] == ""
                  && lines[3] == "gamma" && lines[4] == "delta", "lines match");

            File.WriteAllText(file, "x\ny\n");
            Check(new List<string>(ReadLines(file)).Count == 2, "trailing newline yields no empty tail");
        }
        finally
        {
            if (File.Exists(file)) File.Delete(file);
            if (Directory.Exists(dir)) Directory.Delete(dir);
        }
        Check(!File.Exists(file), "temp file removed");
        Check(!Directory.Exists(dir), "temp dir removed");
        Console.WriteLine("PASS");
    }

    private static bool FirstEnumerationThrows(IEnumerable<string> lines)
    {
        try
        {
            foreach (string _ in lines) return false;
        }
        catch (FileNotFoundException)
        {
            return true;
        }
        return false;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
