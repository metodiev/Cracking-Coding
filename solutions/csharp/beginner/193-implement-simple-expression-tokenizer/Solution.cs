// Task 193 - Implement simple expression tokenizer. (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;

public static class Solution
{
    private class Token
    {
        public string Type { get; }
        public object Value { get; }

        public Token(string type, object value)
        {
            Type = type;
            Value = value;
        }

        public override bool Equals(object? obj)
        {
            return obj is Token other && Type == other.Type && Value.Equals(other.Value);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Type, Value);
        }

        public override string ToString()
        {
            return Type + "(" + Value + ")";
        }
    }

    public static void Main()
    {
        var tokens = Tokenize("12 + (3*4)");
        Check(SameTokens(tokens, new List<Token>
        {
            new Token("INT", 12), new Token("OP", "+"), new Token("LPAREN", "("),
            new Token("INT", 3), new Token("OP", "*"), new Token("INT", 4), new Token("RPAREN", ")"),
        }), "exact token stream for '12 + (3*4)'");
        Check(tokens[0].Type == "INT" && (int)tokens[0].Value == 12, "INT token carries int value");

        Check(SameTokens(Tokenize("1+2"), new List<Token>
        {
            new Token("INT", 1), new Token("OP", "+"), new Token("INT", 2),
        }), "no whitespace");
        Check(SameTokens(Tokenize("  7  "), new List<Token> { new Token("INT", 7) }), "surrounding whitespace");
        Check(Tokenize("").Count == 0, "empty input");
        Check(Tokenize(" \t\n ").Count == 0, "blank input");
        Check(SameTokens(Tokenize("10/2-3"), new List<Token>
        {
            new Token("INT", 10), new Token("OP", "/"), new Token("INT", 2),
            new Token("OP", "-"), new Token("INT", 3),
        }), "multi-digit numbers");

        CheckThrows("12.5", "'\u002E'", "index 2");
        CheckThrows("a+b", "'a'", "index 0");
        CheckThrows("1 $ 2", "'$'", "index 2");
        CheckThrows("#", "'#'", "index 0");
        Console.WriteLine("PASS");
    }

    private static List<Token> Tokenize(string expr)
    {
        var tokens = new List<Token>();
        int index = 0;
        while (index < expr.Length)
        {
            char ch = expr[index];
            if (char.IsWhiteSpace(ch))
            {
                index++;
            }
            else if (char.IsDigit(ch))
            {
                int end = index;
                while (end < expr.Length && char.IsDigit(expr[end])) end++;
                tokens.Add(new Token("INT", int.Parse(expr.Substring(index, end - index))));
                index = end;
            }
            else if ("+-*/".IndexOf(ch) >= 0)
            {
                tokens.Add(new Token("OP", ch.ToString()));
                index++;
            }
            else if (ch == '(')
            {
                tokens.Add(new Token("LPAREN", "("));
                index++;
            }
            else if (ch == ')')
            {
                tokens.Add(new Token("RPAREN", ")"));
                index++;
            }
            else
            {
                throw new ArgumentException("unknown character '" + ch + "' at index " + index);
            }
        }
        return tokens;
    }

    private static bool SameTokens(List<Token> actual, List<Token> expected)
    {
        if (actual.Count != expected.Count) return false;
        for (int i = 0; i < actual.Count; i++)
        {
            if (!actual[i].Equals(expected[i])) return false;
        }
        return true;
    }

    private static void CheckThrows(string expr, string charText, string indexText)
    {
        try
        {
            Tokenize(expr);
        }
        catch (ArgumentException exc)
        {
            if (!exc.Message.Contains(charText) || !exc.Message.Contains(indexText))
            {
                throw new Exception("FAILED: bad message '" + exc.Message + "' for " + expr);
            }
            return;
        }
        throw new Exception("FAILED: expected error for " + expr);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
