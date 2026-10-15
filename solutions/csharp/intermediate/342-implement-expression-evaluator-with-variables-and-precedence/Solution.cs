// Task 342 - Expression evaluator with variables and precedence. (Tier: Intermediate, CTCI Ch 8)
using System;
using System.Collections.Generic;

public class Token
{
    public const int Num = 0, Var = 1, Op = 2;

    public readonly int Kind;
    public readonly double Number;
    public readonly string? Text;

    private Token(int kind, double number, string? text)
    {
        Kind = kind;
        Number = number;
        Text = text;
    }

    public static Token NewNum(double value) => new Token(Num, value, null);
    public static Token NewVar(string name) => new Token(Var, 0.0, name);
    public static Token NewOp(string symbol) => new Token(Op, 0.0, symbol);
}

public static class Solution
{
    /// <summary>
    /// Shunting-yard evaluator: + - * /, parentheses, unary minus ('~' in the RPN) and variables
    /// from a dictionary.  All arithmetic is double (no integer truncation), so 100/8 == 12.5;
    /// malformed input throws ArgumentException.  The brief's example 3+4*2/(1-5) is 1, not 2:
    /// 3 + 8/(-4) = 3 - 2.
    /// </summary>
    public static List<Token> Tokenize(string expr)
    {
        var tokens = new List<Token>();
        int i = 0;
        while (i < expr.Length)
        {
            char ch = expr[i];
            if (ch == ' ')
            {
                i++;
            }
            else if (char.IsDigit(ch))
            {
                int j = i;
                while (j < expr.Length && char.IsDigit(expr[j])) j++;
                tokens.Add(Token.NewNum(double.Parse(expr.Substring(i, j - i))));
                i = j;
            }
            else if (char.IsLetter(ch))
            {
                int j = i;
                while (j < expr.Length && (char.IsLetterOrDigit(expr[j]) || expr[j] == '_')) j++;
                tokens.Add(Token.NewVar(expr.Substring(i, j - i)));
                i = j;
            }
            else if ("+-*/()".IndexOf(ch) >= 0)
            {
                tokens.Add(Token.NewOp(ch.ToString()));
                i++;
            }
            else
            {
                throw new ArgumentException("unexpected character '" + ch + "'");
            }
        }
        return tokens;
    }

    private static int Precedence(string op)
    {
        if (op == "~") return 4;                           // unary minus, right-associative
        if (op == "*" || op == "/") return 2;
        return 1;
    }

    public static List<Token> ToRpn(List<Token> tokens)
    {
        var output = new List<Token>();
        var ops = new Stack<Token>();
        bool expectOperand = true;
        foreach (Token token in tokens)
        {
            if (token.Kind == Token.Num)
            {
                if (!expectOperand) throw new ArgumentException("expected an operator before " + token.Number);
                output.Add(token);
                expectOperand = false;
            }
            else if (token.Kind == Token.Var)
            {
                if (!expectOperand) throw new ArgumentException("expected an operator before " + token.Text);
                output.Add(token);
                expectOperand = false;
            }
            else if (token.Text == "(")
            {
                if (!expectOperand) throw new ArgumentException("implicit multiplication is not supported");
                ops.Push(token);
            }
            else if (token.Text == ")")
            {
                if (expectOperand) throw new ArgumentException("empty or dangling parenthesis group");
                while (ops.Count > 0 && ops.Peek().Text != "(") output.Add(ops.Pop());
                if (ops.Count == 0) throw new ArgumentException("unbalanced ')'");
                ops.Pop();
                expectOperand = false;
            }
            else if (expectOperand)
            {
                if (token.Text == "-")
                {
                    ops.Push(Token.NewOp("~"));
                }
                else if (token.Text != "+")                    // unary plus is a no-op
                {
                    throw new ArgumentException("operator '" + token.Text + "' has no left operand");
                }
            }
            else
            {
                int precedence = Precedence(token.Text!);
                while (ops.Count > 0)
                {
                    Token top = ops.Peek();
                    if (top.Text == "(") break;
                    int topPrecedence = Precedence(top.Text!);
                    if (topPrecedence > precedence || (topPrecedence == precedence && top.Text != "~")) output.Add(ops.Pop());
                    else break;
                }
                ops.Push(token);
                expectOperand = true;
            }
        }
        if (expectOperand && tokens.Count > 0) throw new ArgumentException("dangling operator at end of expression");
        while (ops.Count > 0)
        {
            Token top = ops.Pop();
            if (top.Text == "(") throw new ArgumentException("unbalanced '('");
            output.Add(top);
        }
        return output;
    }

    public static double EvaluateRpn(List<Token> rpn, Dictionary<string, double> variables)
    {
        var stack = new Stack<double>();
        foreach (Token token in rpn)
        {
            if (token.Kind == Token.Num)
            {
                stack.Push(token.Number);
            }
            else if (token.Kind == Token.Var)
            {
                if (!variables.TryGetValue(token.Text!, out double value))
                {
                    throw new ArgumentException("unknown variable '" + token.Text + "'");
                }
                stack.Push(value);
            }
            else if (token.Text == "~")
            {
                if (stack.Count == 0) throw new ArgumentException("malformed expression");
                stack.Push(-stack.Pop());
            }
            else
            {
                if (stack.Count < 2) throw new ArgumentException("malformed expression");
                double b = stack.Pop();
                double a = stack.Pop();
                if (token.Text == "+") stack.Push(a + b);
                else if (token.Text == "-") stack.Push(a - b);
                else if (token.Text == "*") stack.Push(a * b);
                else
                {
                    if (b == 0.0) throw new ArgumentException("division by zero");
                    stack.Push(a / b);
                }
            }
        }
        if (stack.Count != 1) throw new ArgumentException("malformed expression");
        return stack.Pop();
    }

    public static double Evaluate(string expr, Dictionary<string, double> variables)
    {
        var tokens = Tokenize(expr);
        if (tokens.Count == 0) throw new ArgumentException("empty expression");
        return EvaluateRpn(ToRpn(tokens), variables);
    }

    private static void ExpectArgumentException(string expr, Dictionary<string, double> variables)
    {
        try
        {
            Evaluate(expr, variables);
        }
        catch (ArgumentException)
        {
            return;
        }
        throw new Exception("FAILED: expected ArgumentException for " + expr);
    }

    public static void Main()
    {
        var none = new Dictionary<string, double>();

        // brief examples - the first one is 1, not 2
        Check(Evaluate("3+4*2/(1-5)", none) == 1.0, "3+4*2/(1-5) = 1");
        Check(Evaluate("x+y*2", new Dictionary<string, double> { { "x", 3.0 }, { "y", 4.0 } }) == 11.0, "x+y*2 = 11");
        Check(Evaluate("(1+2)*(3+4)", none) == 21.0, "(1+2)*(3+4) = 21");

        // values cross-checked against CPython's expression parser
        Check(Evaluate("1+2*3-4/8", none) == 6.5, "6.5");
        Check(Evaluate("2*(3+4)-5", none) == 9.0, "9");
        Check(Evaluate("100/8", none) == 12.5, "12.5");
        Check(Evaluate("-(3+4)*2", none) == -14.0, "-14");
        Check(Evaluate("3*-2", none) == -6.0, "-6");
        Check(Evaluate("2*-3+4", none) == -2.0, "-2");
        Check(Evaluate("10-4-3", none) == 3.0, "left associative");
        Check(Evaluate("16/4/2", none) == 2.0, "(16/4)/2");
        Check(Evaluate("2*(1+2*(3-1))", none) == 10.0, "10");
        Check(Evaluate("((7))", none) == 7.0, "7");
        Check(Evaluate("8/(2+2)", none) == 2.0, "2");
        Check(Evaluate("--5", none) == 5.0, "chained unary minus");
        Check(Evaluate("-(-5)", none) == 5.0, "5");
        Check(Evaluate("3--2", none) == 5.0, "binary minus then unary minus");
        Check(Evaluate("2+3*4-6/3", none) == 12.0, "12");
        Check(Evaluate("((1+2)*(3+4))/3", none) == 7.0, "7");
        Check(Evaluate("1+((2+3)*4)", none) == 21.0, "21");
        Check(Evaluate("7/2", none) == 3.5, "3.5");
        Check(Evaluate("(8-3)*(2+1)/5", none) == 3.0, "3");
        Check(Evaluate(" 1 + 2 ", none) == 3.0, "whitespace ignored");
        Check(Evaluate("2*-5+20/4", none) == -5.0, "-5");
        Check(Evaluate("x*y+z", new Dictionary<string, double> { { "x", 3.0 }, { "y", 4.0 }, { "z", 5.0 } }) == 17.0, "17");
        Check(Evaluate("-x*2", new Dictionary<string, double> { { "x", 3.0 } }) == -6.0, "-6");
        Check(Evaluate("+x", new Dictionary<string, double> { { "x", 7.0 } }) == 7.0, "unary plus");
        Check(Evaluate("price*quantity", new Dictionary<string, double> { { "price", 2.5 }, { "quantity", 4.0 } }) == 10.0, "10");

        string[] bad = { "", "1+", "*3", "1**2", "(1+2", "1+2)", "()", "( )", "1 2", "3x",
                         "1(2)", "&", "1&2", "4/0", "0/0", "(1+)" };
        foreach (string expr in bad) ExpectArgumentException(expr, none);
        ExpectArgumentException("x+1", none);              // unknown variable
        ExpectArgumentException("sum", new Dictionary<string, double> { { "total", 1.0 } });
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
