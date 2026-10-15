// Task 189 - Implement simple CSV parser (line -> fields). (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;
using System.Text;

public static class Solution
{
    /// <summary>
    /// RFC-4180 basics for a single line:
    /// - ',' separates fields outside quotes;
    /// - a field starting with '"' is quoted, and "" inside a quoted field is one literal quote;
    /// - only ',' or the end of the line may follow a closing quote — anything else is an error;
    /// - an empty line yields one empty field and a trailing comma yields a final empty field.
    ///
    /// Throws FormatException for an unterminated quoted field, junk after a closing quote,
    /// or a quote inside an unquoted field.
    /// </summary>
    private static List<string> ParseCsvLine(string line)
    {
        var fields = new List<string>();
        int i = 0;
        while (true)
        {
            var chunk = new StringBuilder();
            if (i < line.Length && line[i] == '"')
            {
                i++;
                while (true)
                {
                    if (i >= line.Length) throw new FormatException("unterminated quoted field");
                    if (line[i] == '"')
                    {
                        if (i + 1 < line.Length && line[i + 1] == '"')
                        {
                            chunk.Append('"');
                            i += 2;
                            continue;
                        }
                        i++;                     // closing quote
                        break;
                    }
                    chunk.Append(line[i]);
                    i++;
                }
                if (i < line.Length && line[i] != ',') throw new FormatException("unexpected character after closing quote");
            }
            else
            {
                while (i < line.Length && line[i] != ',')
                {
                    if (line[i] == '"') throw new FormatException("quote inside unquoted field");
                    chunk.Append(line[i]);
                    i++;
                }
            }
            fields.Add(chunk.ToString());
            if (i >= line.Length) return fields;
            i++;                                 // skip the separating comma
        }
    }

    private static bool Same(List<string> actual, params string[] expected)
    {
        if (actual.Count != expected.Length) return false;
        for (int i = 0; i < actual.Count; i++)
        {
            if (actual[i] != expected[i]) return false;
        }
        return true;
    }

    public static void Main()
    {
        Check(Same(ParseCsvLine("a,b,c"), "a", "b", "c"), "plain fields");
        Check(Same(ParseCsvLine("a,\"b,c\",d"), "a", "b,c", "d"), "quoted comma");
        Check(Same(ParseCsvLine("\"a\"\"b\",c"), "a\"b", "c"), "escaped quote");
        Check(Same(ParseCsvLine(""), ""), "empty line");
        Check(Same(ParseCsvLine("a,,c"), "a", "", "c"), "empty middle field");
        Check(Same(ParseCsvLine("a,b,"), "a", "b", ""), "trailing empty field");
        Check(Same(ParseCsvLine(",,"), "", "", ""), "all empty fields");
        Check(Same(ParseCsvLine("\"\",x"), "", "x"), "quoted empty field");
        Check(Same(ParseCsvLine("\"line\"\"break\""), "line\"break"), "escaped quote only");
        Check(Same(ParseCsvLine("\"a,b\",\"c,d\""), "a,b", "c,d"), "two quoted fields");
        Check(Same(ParseCsvLine(" , "), " ", " "), "spaces are data");
        Check(Same(ParseCsvLine("\"  spaced  \",tail"), "  spaced  ", "tail"), "quotes keep spaces");
        Check(Same(ParseCsvLine("single"), "single"), "single field");

        CheckThrows(() => ParseCsvLine("a,\"b,c"), "unterminated quote");
        CheckThrows(() => ParseCsvLine("\""), "lone quote");
        CheckThrows(() => ParseCsvLine("a,\"b\"c,d"), "junk after closing quote");
        CheckThrows(() => ParseCsvLine("a\"b"), "quote inside unquoted field");
        CheckThrows(() => ParseCsvLine("\"unterminated"), "unterminated at end of line");
        Console.WriteLine("PASS");
    }

    private static void CheckThrows(Action action, string label)
    {
        try
        {
            action();
        }
        catch (FormatException)
        {
            return;
        }
        throw new Exception("FAILED: expected FormatException: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
