// Task 190 - Parse and evaluate postfix (RPN) expression. (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;

public static class Solution
{
    private static readonly Regex IntegerToken = new Regex(@"^-?\d+$", RegexOptions.Compiled);

    /// <summary>Split a whitespace-separated RPN expression into tokens.</summary>
    private static List<string> TokenizeRpn(string expression)
    {
        string trimmed = expression.Trim();
        var tokens = new List<string>();
        if (trimmed.Length == 0) return tokens;
        foreach (var token in trimmed.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries))
        {
            tokens.Add(token);
        }
        return tokens;
    }

    /// <summary>
    /// Tokens are separated by whitespace: integer operands and the operators + - * /.
    /// Evaluation is a stack sweep — push operands, pop two for each operator. '/' is
    /// integer division truncating toward zero (native C# int behaviour), division by zero
    /// throws DivideByZeroException, and malformed input throws FormatException.
    /// </summary>
    private static int EvalRpn(string expression)
    {
        var stack = new List<int>();
        foreach (string token in TokenizeRpn(expression))
        {
            if (token == "+" || token == "-" || token == "*" || token == "/")
            {
                if (stack.Count < 2) throw new FormatException("not enough operands for " + token);
                int right = stack[stack.Count - 1];
                int left = stack[stack.Count - 2];
                stack.RemoveRange(stack.Count - 2, 2);
                switch (token)
                {
                    case "+": stack.Add(left + right); break;
                    case "-": stack.Add(left - right); break;
                    case "*": stack.Add(left * right); break;
                    default:
                        if (right == 0) throw new DivideByZeroException("division by zero");
                        stack.Add(left / right);
                        break;
                }
            }
            else
            {
                if (!IntegerToken.IsMatch(token)) throw new FormatException("invalid token: " + token);
                stack.Add(int.Parse(token));
            }
        }
        if (stack.Count != 1) throw new FormatException("expression leaves " + stack.Count + " values on the stack");
        return stack[0];
    }

    public static void Main()
    {
        Check(Same(TokenizeRpn("5 1 2 + 4 * + 3 -"), "5", "1", "2", "+", "4", "*", "+", "3", "-"), "tokenize");
        Check(TokenizeRpn("").Count == 0, "tokenize empty");
        Check(Same(TokenizeRpn("  7   2 / "), "7", "2", "/"), "tokenize extra whitespace");

        Check(EvalRpn("3 4 +") == 7, "3 4 +");
        Check(EvalRpn("5 1 2 + 4 * + 3 -") == 14, "long example");
        Check(EvalRpn("7 2 /") == 3, "7 2 /");
        Check(EvalRpn("42") == 42, "single value");
        Check(EvalRpn("  42  ") == 42, "padded single value");
        Check(EvalRpn("2 3 4 * +") == 14, "2 3 4 * +");
        Check(EvalRpn("1 2 + 3 *") == 9, "1 2 + 3 *");
        Check(EvalRpn("10 3 - 4 2 / *") == 14, "mixed");
        Check(EvalRpn("5 1 - 1 -") == 3, "chained subtraction");
        Check(EvalRpn("0 5 /") == 0, "zero numerator");

        // integer division truncates toward zero
        Check(EvalRpn("-7 2 /") == -3, "-7 2 /");
        Check(EvalRpn("7 -2 /") == -3, "7 -2 /");
        Check(EvalRpn("-7 -2 /") == 3, "-7 -2 /");
        Check(EvalRpn("-1 3 /") == 0, "-1 3 /");

        foreach (string bad in new[] { "1 0 /", "0 0 /", "3 4", "1 2 + 3", "+", "", "   ", "abc", "1 2 &", "1.5 2 +", "+5 1 +", "3 -" })
        {
            CheckThrows(() => EvalRpn(bad), "rejects " + bad);
        }
        Console.WriteLine("PASS");
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
        catch (DivideByZeroException)
        {
            return;
        }
        throw new Exception("FAILED: expected FormatException/DivideByZeroException: " + label);
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
