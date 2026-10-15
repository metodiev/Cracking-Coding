// Task 191 - Convert infix to postfix (shunting-yard simplified). (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static void Main()
    {
        Check(InfixToPostfix("3 + 4 * 2") == "3 4 2 * +", "precedence");
        Check(InfixToPostfix("( 1 + 2 ) * 3") == "1 2 + 3 *", "parentheses");
        Check(InfixToPostfix("1 + 2 - 3") == "1 2 + 3 -", "left-associative + -");
        Check(InfixToPostfix("8 / 4 / 2") == "8 4 / 2 /", "left-associative /");
        Check(InfixToPostfix("2 * 3 + 4 * 5") == "2 3 * 4 5 * +", "mixed precedence");
        Check(InfixToPostfix("( 1 + 2 ) * ( 3 - 4 )") == "1 2 + 3 4 - *", "two groups");
        Check(InfixToPostfix("( ( 12 ) )") == "12", "nested redundant parentheses");
        Check(InfixToPostfix("") == "", "empty input");
        CheckThrows("1 + 2 )", "unmatched ')'");
        CheckThrows("( 1 + 2", "unmatched '('");
        CheckThrows("( ( 1 ) + 2 ) )", "extra ')'");
        CheckThrows("1 + a", "unknown token");
        CheckThrows("1 ^ 2", "unsupported operator");
        Console.WriteLine("PASS");
    }

    private static string InfixToPostfix(string expr)
    {
        string[] tokens = expr.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
        var output = new List<string>();
        var stack = new Stack<string>();
        foreach (string token in tokens)
        {
            if (IsNumber(token))
            {
                output.Add(token);
            }
            else if (IsOperator(token))
            {
                while (stack.Count > 0 && IsOperator(stack.Peek())
                       && Precedence(stack.Peek()) >= Precedence(token))
                {
                    output.Add(stack.Pop());
                }
                stack.Push(token);
            }
            else if (token == "(")
            {
                stack.Push(token);
            }
            else if (token == ")")
            {
                while (stack.Count > 0 && stack.Peek() != "(") output.Add(stack.Pop());
                if (stack.Count == 0) throw new ArgumentException("unmatched ')' in expression");
                stack.Pop();
            }
            else
            {
                throw new ArgumentException("unknown token: " + token);
            }
        }
        while (stack.Count > 0)
        {
            if (stack.Peek() == "(") throw new ArgumentException("unmatched '(' in expression");
            output.Add(stack.Pop());
        }
        return string.Join(" ", output);
    }

    private static bool IsNumber(string token)
    {
        if (token.Length == 0) return false;
        foreach (char c in token)
        {
            if (c < '0' || c > '9') return false;
        }
        return true;
    }

    private static bool IsOperator(string token)
    {
        return token == "+" || token == "-" || token == "*" || token == "/";
    }

    private static int Precedence(string op)
    {
        return (op == "*" || op == "/") ? 2 : 1;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    private static void CheckThrows(string expr, string label)
    {
        try
        {
            InfixToPostfix(expr);
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: expected error - " + label);
    }
}
