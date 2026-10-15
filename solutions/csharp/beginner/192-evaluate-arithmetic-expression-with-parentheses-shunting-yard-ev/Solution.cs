// Task 192 - Evaluate arithmetic expression with +,-,*,/, parentheses (shunting-yard + eval). (Tier: Beginner, CTCI Ch 3)
using System;
using System.Collections.Generic;

public static class Solution
{
    public static void Main()
    {
        Check(Evaluate("3 + 4 * 2") == 11, "precedence");
        Check(Evaluate("( 1 + 2 ) * 3") == 9, "parentheses");
        Check(Evaluate("10 / 3") == 3, "integer division truncates");
        Check(Evaluate("2 * ( 3 + 4 ) - 5") == 9, "mixed expression");
        Check(Evaluate("( 0 - 7 ) / 2") == -3, "truncation toward zero");
        Check(Evaluate("100 / 7 / 2") == 7, "left-associative division");
        Check(Evaluate("7") == 7, "single number");
        Check(Evaluate("( ( 2 + 3 ) * ( 4 - 1 ) )") == 15, "nested groupings");
        CheckThrows("10 / 0", "division by zero");
        CheckThrows("( 1 + 2", "unmatched '('");
        CheckThrows("1 + 2 )", "unmatched ')'");
        CheckThrows("( 1 + 2 ) )", "extra ')'");
        CheckThrows("1 +", "operator without operand");
        CheckThrows("", "empty input");
        CheckThrows("3 @ 4", "unknown token");
        Console.WriteLine("PASS");
    }

    private static int Evaluate(string expr)
    {
        return EvaluatePostfix(InfixToPostfix(Tokenize(expr)));
    }

    private static List<string> Tokenize(string expr)
    {
        var tokens = new List<string>(expr.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries));
        foreach (string token in tokens)
        {
            if (!IsNumber(token) && !IsOperator(token) && token != "(" && token != ")")
            {
                throw new ArgumentException("unknown token: " + token);
            }
        }
        return tokens;
    }

    private static List<string> InfixToPostfix(List<string> tokens)
    {
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
        }
        while (stack.Count > 0)
        {
            if (stack.Peek() == "(") throw new ArgumentException("unmatched '(' in expression");
            output.Add(stack.Pop());
        }
        return output;
    }

    private static int EvaluatePostfix(List<string> tokens)
    {
        var stack = new Stack<int>();
        foreach (string token in tokens)
        {
            if (IsOperator(token))
            {
                if (stack.Count < 2) throw new ArgumentException("malformed expression: operator without operands");
                int right = stack.Pop();
                int left = stack.Pop();
                stack.Push(Apply(token, left, right));
            }
            else
            {
                stack.Push(int.Parse(token));
            }
        }
        if (stack.Count != 1) throw new ArgumentException("malformed expression: stray operands");
        return stack.Pop();
    }

    private static int Apply(string op, int left, int right)
    {
        switch (op)
        {
            case "+": return left + right;
            case "-": return left - right;
            case "*": return left * right;
            default:
                if (right == 0) throw new ArgumentException("division by zero");
                return left / right;                       // C# int division truncates toward zero
        }
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
            Evaluate(expr);
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: expected error - " + label);
    }
}
