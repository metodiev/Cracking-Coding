// Task 964 - Implement interactive REPL/web UI for running and testing problems in-browser (sandboxed).
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The REPL hosts an arithmetic language (integers, + - * /, parentheses, unary minus) inside a sandbox:
// the evaluator counts every node visited and aborts once a fixed step budget is exhausted, after which
// the session freezes and refuses input until `:reset`.  Two independent evaluators exist - a recursive
// descent parser that evaluates while parsing, and a two-stack operator-precedence machine - and they must
// agree on every generated expression, which is the oracle.
//
// Verified: 60 random expressions agree between both evaluators, precedence/unary minus match hand-computed
// values, division by zero and syntax errors are reported without freezing the sandbox, the step budget
// freezes it exactly once and `:reset` restores it, `:submit` is gated on `:tests`, history ignores blank
// lines, and replaying a script is byte-identical.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static ulong _state;

    private static ulong NextUInt64()
    {
        unchecked
        {
            _state += 0x9E3779B97F4A7C15UL;
            ulong z = _state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
    }

    private static int NextRange(int low, int high) => low + (int)(NextUInt64() % (ulong)(high - low + 1));
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private sealed class Token
    {
        public readonly string Kind;
        public readonly string Text;
        public Token(string kind, string text) { Kind = kind; Text = text; }
    }

    private static List<Token> Lex(string source)
    {
        var tokens = new List<Token>();
        for (int i = 0; i < source.Length; )
        {
            char c = source[i];
            if (char.IsWhiteSpace(c)) { i++; continue; }
            if (char.IsDigit(c))
            {
                int start = i; while (i < source.Length && char.IsDigit(source[i])) i++;
                tokens.Add(new Token("num", source.Substring(start, i - start)));
                continue;
            }
            if ("+-*/()".IndexOf(c) >= 0) { tokens.Add(new Token("op", c.ToString())); i++; continue; }
            throw new Exception("unexpected character '" + c + "'");
        }
        return tokens;
    }    /** Evaluator 1: recursive descent that evaluates while parsing, counting every visited node. */
    private sealed class Parser
    {
        private readonly List<Token> _tokens;
        private readonly int _budget;
        private int _index;
        private int _steps;
        public Parser(List<Token> tokens, int budget) { _tokens = tokens; _budget = budget; }
        private Token Peek() => _index < _tokens.Count ? _tokens[_index] : new Token("end", "");
        private Token Take() { Token token = Peek(); _index++; return token; }
        private void Step() { if (++_steps > _budget) throw new Exception("step budget exceeded"); }
        public long Program()
        {
            long result = Expression();
            if (Peek().Kind != "end") throw new Exception("unexpected trailing token '" + Peek().Text + "'");
            return result;
        }
        private long Expression()
        {
            long value = Term();
            while (Peek().Text == "+" || Peek().Text == "-")
            {
                Step();
                string op = Take().Text;
                long right = Term();
                value = op == "+" ? value + right : value - right;
            }
            return value;
        }
        private long Term()
        {
            long value = Factor();
            while (Peek().Text == "*" || Peek().Text == "/")
            {
                Step();
                string op = Take().Text;
                long right = Factor();
                if (op == "*") { value *= right; continue; }
                if (right == 0) throw new Exception("division by zero");
                value /= right;
            }
            return value;
        }
        private long Factor()
        {
            Step();
            if (Peek().Text == "-") { Take(); return -Factor(); }
            Token token = Take();
            if (token.Kind == "num") return long.Parse(token.Text, CultureInfo.InvariantCulture);
            if (token.Text != "(") throw new Exception("unexpected token '" + token.Text + "'");
            long inner = Expression();
            if (Take().Text != ")") throw new Exception("expected ')'");
            return inner;
        }
    }

    /** Evaluator 2 (the oracle): Dijkstra's two-stack operator-precedence machine. */
    private static long EvaluateTwoStack(List<Token> tokens)
    {
        var values = new Stack<long>();
        var operators = new Stack<string>();
        bool expectValue = true;
        foreach (Token token in tokens)
        {
            if (token.Kind == "num") { values.Push(long.Parse(token.Text, CultureInfo.InvariantCulture)); expectValue = false; continue; }
            if (token.Text == "(") { operators.Push("("); expectValue = true; continue; }
            if (token.Text == ")")
            {
                while (operators.Peek() != "(") Apply(values, operators.Pop());
                operators.Pop(); expectValue = false;
                continue;
            }
            string op = expectValue && token.Text == "-" ? "neg" : token.Text;
            while (operators.Count > 0 && operators.Peek() != "(" && op != "neg" && Rank(operators.Peek()) >= Rank(op)) Apply(values, operators.Pop());
            operators.Push(op);
            expectValue = true;
        }
        while (operators.Count > 0) Apply(values, operators.Pop());
        if (values.Count != 1) throw new Exception("malformed expression");
        return values.Pop();
    }

    private static int Rank(string op) => op == "neg" ? 3 : op == "*" || op == "/" ? 2 : 1;
    private static void Apply(Stack<long> values, string op)
    {
        if (op == "neg") { values.Push(-values.Pop()); return; }
        if (values.Count < 2) throw new Exception("malformed expression");
        long right = values.Pop(), left = values.Pop();
        if (op == "+") values.Push(left + right);
        else if (op == "-") values.Push(left - right);
        else if (op == "*") values.Push(left * right);
        else if (right == 0) throw new Exception("division by zero");
        else values.Push(left / right);
    }

    private sealed class Session
    {
        public readonly List<string> History = new List<string>();
        public readonly List<string> Transcript = new List<string>();
        public bool Frozen, TestsPassed;
        public string Loaded = "none";
    }
    private const int Budget = 220;

    private static string Execute(Session session, string input)
    {
        string line = input.Trim();
        if (line.Length == 0) return ""; // a blank line is a no-op and is not historical
        session.History.Add(line);
        string reply;
        if (line.StartsWith(":", StringComparison.Ordinal)) reply = Command(session, line);
        else if (session.Frozen) reply = "error: sandbox frozen (use :reset)";
        else
        {
            try { reply = new Parser(Lex(line), Budget).Program().ToString(CultureInfo.InvariantCulture); }
            catch (Exception ex) { if (ex.Message == "step budget exceeded") session.Frozen = true; reply = "error: " + ex.Message; }
        }
        session.Transcript.Add(reply);
        return reply;
    }
    private static bool SelfTest() => EvaluateTwoStack(Lex("1+1")) == 2 && EvaluateTwoStack(Lex("2*3")) == 6 && EvaluateTwoStack(Lex("10/2")) == 5;

    private static string Command(Session session, string line)
    {
        string[] parts = line.Split(' ', 2);
        string argument = parts.Length > 1 ? parts[1].Trim() : "";
        switch (parts[0])
        {
            case ":load": session.Loaded = argument; return "loaded " + argument;
            case ":hint": return session.Loaded == "none" ? "error: :load a problem first" : "hint: reformulate the loop iteratively";
            case ":tests": session.TestsPassed = SelfTest(); return session.TestsPassed ? "tests: 3 passed" : "tests: failed";
            case ":submit": return session.TestsPassed ? "accepted" : "error: run :tests first";
            case ":history": return "history: " + session.History.Count.ToString(CultureInfo.InvariantCulture);
            case ":reset": session.Frozen = false; session.TestsPassed = false; return "reset";
            default: return "error: unknown command " + parts[0];
        }
    }

    private static string RandomExpression(int depth)
    {
        if (depth == 0) return NextRange(0, 9).ToString(CultureInfo.InvariantCulture);
        string[] operators = { "+", "-", "*" };
        return "(" + RandomExpression(depth - 1) + operators[NextRange(0, 2)] + RandomExpression(depth - 1) + ")";
    }

    public static void Main()
    {
        _state = 20240901UL;
        int agreements = 0;
        for (int i = 0; i < 60; i++)
        {
            List<Token> tokens = Lex(RandomExpression(NextRange(1, 4)));
            if (new Parser(tokens, 100_000).Program() == EvaluateTwoStack(tokens)) agreements++;
        }
        Check(agreements == 60, "the recursive-descent and two-stack evaluators agree on all 60 random expressions");
        Check(EvaluateTwoStack(Lex("2+3*4")) == 14 && EvaluateTwoStack(Lex("(2+3)*4")) == 20 && EvaluateTwoStack(Lex("-3+5")) == 2 && EvaluateTwoStack(Lex("2*-3")) == -6, "precedence, parentheses and unary minus match the hand-computed 14, 20, 2 and -6");
        Check(new Parser(Lex("((1+1)*(1+1))"), 100_000).Program() == 4, "nested parentheses evaluate correctly");

        var session = new Session();
        Check(Execute(session, "6*6") == "36", "the REPL evaluates an expression");
        Check(Execute(session, "") == "" && session.History.Count == 1, "a blank line is a no-op and is not recorded in history");
        Check(Execute(session, "7") == "7", "the single-literal boundary evaluates to itself");
        Check(Execute(session, "1 +") == "error: unexpected token ''" && Execute(session, "8/2") == "4", "a syntax error is reported and the session stays usable");
        Check(Execute(session, "1/0") == "error: division by zero" && !session.Frozen, "division by zero is reported without freezing the sandbox");
        Check(Execute(session, "aa") == "error: unexpected character 'a'", "non-arithmetic input is rejected by the lexer");

        var long1 = new System.Text.StringBuilder("1");
        for (int i = 0; i < 300; i++) long1.Append("+1");
        Check(Execute(session, long1.ToString()) == "error: step budget exceeded" && session.Frozen, "an expression past the step budget aborts and freezes the session");
        Check(Execute(session, "1+1") == "error: sandbox frozen (use :reset)" && Execute(session, ":reset") == "reset" && Execute(session, "1+1") == "2", "frozen sessions reject evaluation until :reset unfreezes them");
        Check(Execute(session, ":submit") == "error: run :tests first", ":submit is gated on passing tests");
        Check(Execute(session, ":tests") == "tests: 3 passed" && Execute(session, ":submit") == "accepted", ":tests then :submit is accepted");
        Check(Execute(session, ":frobnicate") == "error: unknown command :frobnicate", "an unknown command is reported by name");
        Check(Execute(session, ":load 951") == "loaded 951" && Execute(session, ":hint").StartsWith("hint:", StringComparison.Ordinal), ":load then :hint works in order");

        var script = new[] { "2+2", ":load 964", ":tests", ":submit", ":reset", "1/0", "5*5" };
        var first = new List<string>();
        var replay = new Session();
        foreach (string line in script) first.Add(Execute(replay, line));
        var second = new List<string>();
        var replayAgain = new Session();
        foreach (string line in script) second.Add(Execute(replayAgain, line));
        Check(string.Join("|", first) == string.Join("|", second) && replay.History.Count == script.Length, "replaying a script is identical and every non-blank command stays in history order");

        Console.WriteLine("evaluators: recursive descent and two-stack agree on " + agreements + "/60 random expressions (budget " + Budget + ")");
        Console.WriteLine("transcript: " + string.Join(", ", first));
        Console.WriteLine("PASS");
    }
}
