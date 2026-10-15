// Task 157 - Design a simple FSM for number validation. (Tier: Beginner, CTCI Ch 7)
using System;
using System.Collections.Generic;

public static class Solution
{
    private const string Start = "START", Sign = "SIGN", Int = "INT", Dot = "DOT", Frac = "FRAC";

    /// <summary>A valid number must end in one of these states.</summary>
    private static readonly HashSet<string> Accepting = new HashSet<string> { Int, Frac };

    /// <summary>
    /// Data-driven transition table: (state, character class) -> next state.
    /// Exponent notation (e.g. "1e5") is intentionally unsupported and rejected.
    /// </summary>
    private static readonly Dictionary<(string State, string Cls), string> Transitions = new()
    {
        [(Start, "digit")] = Int,
        [(Start, "sign")] = Sign,
        [(Sign, "digit")] = Int,
        [(Int, "digit")] = Int,
        [(Int, "dot")] = Dot,
        [(Dot, "digit")] = Frac,
        [(Frac, "digit")] = Frac,
    };

    /// <summary>Character class the FSM consumes; anything else (spaces, letters) is rejected.</summary>
    private static string? CharClass(char c)
    {
        if (c >= '0' && c <= '9') return "digit";
        if (c == '+' || c == '-') return "sign";
        if (c == '.') return "dot";
        return null;
    }

    /// <summary>True iff `s` is a signed decimal number accepted by the FSM above.</summary>
    public static bool Accepts(string s)
    {
        var state = Start;
        foreach (var ch in s)
        {
            var cls = CharClass(ch);
            if (cls == null) return false;
            if (!Transitions.TryGetValue((state, cls), out var next)) return false;
            state = next;
        }
        return Accepting.Contains(state);
    }

    public static void Main()
    {
        Check(Accepts("123"), "123");
        Check(Accepts("+1"), "+1");
        Check(Accepts("-1.5"), "-1.5");
        Check(Accepts("0"), "0");
        Check(Accepts("-0.0"), "-0.0");
        Check(Accepts("007"), "leading zeros");
        Check(!Accepts("1."), "1. invalid (DOT not accepting)");
        Check(!Accepts(".5"), ".5 invalid (needs leading digit)");
        Check(!Accepts(""), "empty invalid");
        Check(!Accepts("+"), "+ invalid");
        Check(!Accepts("-"), "- invalid");
        Check(!Accepts("."), ". invalid");
        Check(!Accepts("1.2.3"), "1.2.3 invalid");
        Check(!Accepts("12a"), "12a invalid");
        Check(!Accepts(" 12"), "leading whitespace invalid");
        Check(!Accepts("12 "), "trailing whitespace invalid");
        Check(!Accepts("+-1"), "+-1 invalid");
        Check(!Accepts("1e5"), "exponents intentionally unsupported");

        // the transition table is explicit and data-driven
        Check(!Transitions.ContainsKey((Start, "dot")), "no leading dot transition");
        Check(Transitions[(Int, "dot")] == Dot, "INT --dot--> DOT");
        Check(Transitions[(Dot, "digit")] == Frac, "DOT --digit--> FRAC");
        Check(Accepting.Count == 2 && Accepting.Contains(Int) && Accepting.Contains(Frac), "accepting states");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
