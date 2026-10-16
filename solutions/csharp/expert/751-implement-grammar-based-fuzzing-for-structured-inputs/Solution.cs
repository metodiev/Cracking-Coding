// Task 751 - Implement grammar-based fuzzing for structured inputs. (Tier: Expert, CTCI Ch 11 - Testing)
//
// Fuzzing a structured format means generating inputs that are *by construction* in the input language and
// mutating them without losing grammaticality. This file implements that pipeline for a tiny expression
// grammar (a scaled-down stand-in for a real config/JSON grammar):
//
//     E -> T | T '+' E | T '*' E          (right associative, no precedence - deliberately tiny)
//     T -> num | '(' E ')' | '-' T        (num is exactly one digit, so a digit run like "77" is illegal)
//
// Generator: depth-budgeted derivation. `budget` counts remaining expansions; at budget 0 every nonterminal
// must take production 0, and the base productions (E -> T, T -> num) form a terminating chain, so the
// derivation always stops. Every production use is counted for rule coverage.
// Mutator: subtree splice. Pick a node of derivation A expanding nonterminal N and a node of derivation B
// expanding the same N, and graft B's subtree into A. Context-freedom guarantees the result is still a
// derivation of the start symbol; the file *measures* the accept rate on the corpus instead of assuming it.
//
// Invariants verified: (a) every generated and every mutated string is accepted by the recursive-descent
// parser, the parser re-prints the string identically (so the derivation tree was recovered) and evaluating
// the parsed tree equals evaluating the derivation tree (semantic soundness); (b) language equality - for a
// small token budget the set of token strings the parser accepts is *exactly* the set obtained by
// exhaustively enumerating derivations, both directions, which catches both false accepts and false rejects;
// (c) coverage - every production fires; (d) the tree-depth bound 2*budget+3 holds. Simplifications: no
// coverage feedback into scheduling, no heap/stack probes, no corpus persistence between runs.
using System;
using System.Collections.Generic;
using System.Text;

internal static class SplitMix64
{
    private static ulong state = 0x7510F0ADD5EEDUL;

    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;                 // deterministic stand-in for Random
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static int NextInt(int bound) => (int)(Next() % (ulong)bound);
}

internal sealed class Node
{
    internal readonly string Symbol;
    internal readonly List<Node>? Children;            // null exactly when the node is a terminal

    internal Node(string terminal) { Symbol = terminal; Children = null; }

    internal Node(string nonterminal, List<Node> children) { Symbol = nonterminal; Children = children; }
}

internal sealed class Grammar
{
    private readonly List<string> order = new List<string>();
    private readonly Dictionary<string, List<string[]>> rules = new Dictionary<string, List<string[]>>();

    internal IReadOnlyList<string> Nonterminals => order;
    internal bool IsNonterminal(string symbol) => rules.ContainsKey(symbol);
    internal int Alternatives(string symbol) => rules[symbol].Count;
    internal string[] Production(string symbol, int index) => rules[symbol][index];

    internal void Define(string name, params string[][] alternatives)
    {
        if (!rules.ContainsKey(name)) order.Add(name);
        rules[name] = new List<string[]>(alternatives);
    }

    /** Nonterminal names are single capitals; anything else in a production is a terminal written verbatim. */
    internal static Grammar Expression()
    {
        var grammar = new Grammar();
        grammar.Define("E", new[] { "T" }, new[] { "T", "+", "E" }, new[] { "T", "*", "E" });
        grammar.Define("T", new[] { "num" }, new[] { "(", "E", ")" }, new[] { "-", "T" });
        return grammar;
    }
}

internal sealed class Generator
{
    private readonly Grammar grammar;
    private readonly string digits;
    private readonly Dictionary<string, int[]> hits = new Dictionary<string, int[]>();

    internal Generator(Grammar grammar, string digits)
    {
        this.grammar = grammar;
        this.digits = digits;
        foreach (string nonterminal in grammar.Nonterminals)
        {
            hits[nonterminal] = new int[grammar.Alternatives(nonterminal)];
        }
    }

    internal int Derivations;
    internal int[] Hits(string nonterminal) => hits[nonterminal];

    internal Node Generate(string start, int budget)
    {
        Derivations++;
        string[] rhs = Pick(start, budget);
        var children = new List<Node>();
        foreach (string symbol in rhs)
        {
            if (!grammar.IsNonterminal(symbol))
            {
                children.Add(new Node(symbol == "num" ? digits[SplitMix64.NextInt(digits.Length)].ToString() : symbol));
            }
            else
            {
                children.Add(Generate(symbol, budget - 1));
            }
        }
        return new Node(start, children);
    }

    private string[] Pick(string nonterminal, int budget)
    {
        int count = grammar.Alternatives(nonterminal);
        int chosen = budget <= 0 ? 0 : SplitMix64.NextInt(count);   // budget exhausted -> base production
        hits[nonterminal][chosen]++;
        return grammar.Production(nonterminal, chosen);
    }
}

internal static class Printer
{
    internal static string Text(Node node)
    {
        var builder = new StringBuilder();
        Write(node, builder);
        return builder.ToString();
    }

    internal static int Depth(Node node)
    {
        if (node.Children == null) return 1;
        int best = 0;
        foreach (Node child in node.Children) best = Math.Max(best, Depth(child));
        return 1 + best;
    }

    internal static int TokenCount(Node node)
    {
        if (node.Children == null) return 1;
        int total = 0;
        foreach (Node child in node.Children) total += TokenCount(child);
        return total;
    }

    private static void Write(Node node, StringBuilder builder)
    {
        if (node.Children == null) { builder.Append(node.Symbol); return; }
        for (int i = 0; i < node.Children.Count; i++)
        {
            if (i > 0) builder.Append(' ');
            Write(node.Children[i], builder);
        }
    }
}

internal static class Mutator
{
    /** Graft a subtree of `donor` over a node of `host` that expands the same nonterminal. */
    internal static Node? Splice(Node donor, Node host)
    {
        var donorNodes = new List<Node>();
        CollectNonterminals(donor, donorNodes);
        if (donorNodes.Count == 0) return null;
        Node graft = donorNodes[SplitMix64.NextInt(donorNodes.Count)];
        var hostNodes = new List<Node>();
        CollectNonterminals(host, hostNodes);
        var candidates = hostNodes.FindAll(n => n.Symbol == graft.Symbol);
        if (candidates.Count == 0) return null;
        return Replace(host, candidates[SplitMix64.NextInt(candidates.Count)], graft);
    }

    private static void CollectNonterminals(Node node, List<Node> sink)
    {
        if (node.Children == null) return;
        sink.Add(node);
        foreach (Node child in node.Children) CollectNonterminals(child, sink);
    }

    private static Node Replace(Node node, Node target, Node graft)
    {
        if (ReferenceEquals(node, target)) return graft;
        if (node.Children == null) return node;
        var children = new List<Node>(node.Children.Count);
        foreach (Node child in node.Children) children.Add(Replace(child, target, graft));
        return new Node(node.Symbol, children);
    }
}

internal static class Lexer
{
    /** Returns the token list, or null when a character outside the terminal alphabet appears. */
    internal static List<string>? Tokenize(string text)
    {
        var tokens = new List<string>();
        foreach (char c in text)
        {
            if (c == ' ' || c == '\t') continue;
            bool digit = c >= '0' && c <= '9';
            bool punctuation = c == '+' || c == '*' || c == '-' || c == '(' || c == ')';
            if (!digit && !punctuation) return null;
            tokens.Add(c.ToString());                     // one digit = one NUM token, so "77" is two tokens
        }
        return tokens;
    }
}

internal sealed class Parser
{
    private readonly List<string> tokens;
    private int position;

    private Parser(List<string> tokens) { this.tokens = tokens; }

    internal static Node? Parse(string text)
    {
        List<string>? tokens = Lexer.Tokenize(text);
        if (tokens == null) return null;
        var parser = new Parser(tokens);
        Node? tree = parser.ParseE();
        return tree != null && parser.position == tokens.Count ? tree : null;
    }

    private Node? ParseE()
    {
        Node? left = ParseT();
        if (left == null) return null;
        if (position < tokens.Count && (tokens[position] == "+" || tokens[position] == "*"))
        {
            string op = tokens[position++];
            Node? right = ParseE();                       // right associative, matching the grammar
            if (right == null) return null;
            return new Node("E", new List<Node> { left, new Node(op), right });
        }
        return new Node("E", new List<Node> { left });
    }

    private Node? ParseT()
    {
        if (position >= tokens.Count) return null;
        string token = tokens[position];
        if (token.Length == 1 && token[0] >= '0' && token[0] <= '9')
        {
            position++;
            return new Node("T", new List<Node> { new Node(token) });
        }
        if (token == "(")
        {
            position++;
            Node? inner = ParseE();
            if (inner == null || position >= tokens.Count || tokens[position] != ")") return null;
            position++;
            return new Node("T", new List<Node> { new Node("("), inner, new Node(")") });
        }
        if (token == "-")
        {
            position++;
            Node? operand = ParseT();
            return operand == null ? null : new Node("T", new List<Node> { new Node("-"), operand });
        }
        return null;
    }
}

internal static class Evaluator
{
    internal static int Value(Node node)
    {
        if (node.Symbol == "E")
        {
            int left = Value(node.Children![0]);
            if (node.Children.Count == 1) return left;
            int right = Value(node.Children[2]);
            return node.Children[1].Symbol == "+" ? left + right : left * right;
        }
        Node first = node.Children![0];
        if (first.Symbol == "(") return Value(node.Children[1]);
        if (first.Symbol == "-") return -Value(node.Children[1]);
        return first.Symbol[0] - '0';
    }
}

internal static class DerivationEnumerator
{
    /** Every token sequence of at most `budget` tokens derivable from `start`, digits canonicalised to '7'. */
    internal static List<List<string>> Sequences(Grammar grammar, string start, int budget)
    {
        var seen = new HashSet<string>();
        var unique = new List<List<string>>();
        foreach (List<string> sequence in Expand(grammar, start, budget))
        {
            if (seen.Add(string.Join(" ", sequence))) unique.Add(sequence);
        }
        return unique;
    }

    private static List<List<string>> Expand(Grammar grammar, string symbol, int budget)
    {
        var result = new List<List<string>>();
        if (budget <= 0) return result;
        if (!grammar.IsNonterminal(symbol))
        {
            result.Add(new List<string> { symbol == "num" ? "7" : symbol });
            return result;
        }
        for (int production = 0; production < grammar.Alternatives(symbol); production++)
        {
            result.AddRange(Chain(grammar, grammar.Production(symbol, production), 0, budget));
        }
        return result;
    }

    private static List<List<string>> Chain(Grammar grammar, string[] rhs, int index, int budget)
    {
        if (index == rhs.Length) return new List<List<string>> { new List<string>() };
        var result = new List<List<string>>();
        for (int give = 0; give <= budget; give++)
        {
            foreach (List<string> head in Expand(grammar, rhs[index], give))
            {
                foreach (List<string> tail in Chain(grammar, rhs, index + 1, budget - give))
                {
                    var combined = new List<string>(head);
                    combined.AddRange(tail);
                    result.Add(combined);
                }
            }
        }
        return result;
    }
}

internal static class Candidates
{
    /** All strings of length 1..maxLength over the token alphabet (used to test parser rejection too). */
    internal static List<string> AllOver(string[] alphabet, int maxLength)
    {
        var all = new List<string>();
        Recurse(alphabet, string.Empty, maxLength, all);
        return all;
    }

    private static void Recurse(string[] alphabet, string prefix, int remaining, List<string> acc)
    {
        if (prefix.Length > 0) acc.Add(prefix);
        if (remaining == 0) return;
        foreach (string symbol in alphabet) Recurse(alphabet, prefix + symbol, remaining - 1, acc);
    }
}

public static class Solution
{
    private static readonly string[] Alphabet = { "7", "+", "*", "-", "(", ")" };

    public static void Main()
    {
        var grammar = Grammar.Expression();
        var generator = new Generator(grammar, "0123456789");

        // Budget 0 must collapse to the base chain E -> T -> num, i.e. exactly one digit.
        for (int i = 0; i < 25; i++)
        {
            Node root = generator.Generate("E", 0);
            string text = Printer.Text(root);
            Check(text.Length == 1 && text[0] >= '0' && text[0] <= '9', "budget 0 yields one digit: '" + text + "'");
            Check(Printer.Depth(root) == 3 && Printer.TokenCount(root) == 1, "budget 0 shape");
        }

        // The generated corpus: parseable, recovered exactly, and semantically identical to the derivation.
        var corpus = new List<string>();
        for (int i = 0; i < 500; i++)
        {
            int budget = i % 6;
            Node root = generator.Generate("E", budget);
            string text = Printer.Text(root);
            Node? parsed = Parser.Parse(text);
            Check(parsed != null, "generated string is in the language: '" + text + "'");
            Check(Printer.Text(parsed!) == text, "parse recovers the derivation: '" + text + "'");
            Check(Evaluator.Value(parsed!) == Evaluator.Value(root), "semantics agree: '" + text + "'");
            Check(Printer.Depth(root) <= 2 * budget + 3, "depth bound at budget " + budget);
            corpus.Add(text);
        }
        foreach (string nonterminal in grammar.Nonterminals)
        {
            int[] hits = generator.Hits(nonterminal);
            for (int production = 0; production < hits.Length; production++)
            {
                Check(hits[production] > 0, "production covered: " + nonterminal + " -> " + production);
            }
        }

        // Semantics are right-associative with no precedence: 1 + 2 * 3 would be 9 if left associative.
        Check(Evaluator.Value(Parser.Parse("1 + 2 * 3")!) == 7, "right associative plus then times");
        Check(Evaluator.Value(Parser.Parse("2 * 3 + 4")!) == 14, "right associative times then plus");
        Check(Evaluator.Value(Parser.Parse("- 3 * 2")!) == -6, "unary minus binds to the term");

        // Whitespace is not part of the token language: re-spacing must not change the decision or the tree.
        for (int i = 0; i < corpus.Count; i += 7)
        {
            string spaced = "  " + corpus[i].Replace(" ", "   ") + " ";
            Node? reparsed = Parser.Parse(spaced);
            Check(reparsed != null && Printer.Text(reparsed!) == corpus[i], "spacing invariance: '" + corpus[i] + "'");
        }

        // Malformed inputs must be rejected: this is the negative half of the language equality below.
        string[] malformed = { "", " ", "1+", "(1", "1 2", ")", "+1", "1 +", "((1)", "1)", "1 * * 2", "77", "1(2)", "@" };
        foreach (string bad in malformed)
        {
            Check(Parser.Parse(bad) == null, "rejected: '" + bad + "'");
        }

        // Grammar-directed mutation: subtree splice must stay inside the language.
        int mutants = 0, accepted = 0;
        var mutantTexts = new HashSet<string>();
        for (int i = 0; i < 400; i++)
        {
            Node host = generator.Generate("E", 1 + i % 4);
            Node donor = generator.Generate("E", 1 + (i * 7) % 4);
            Node? mutant = Mutator.Splice(donor, host);
            if (mutant == null) continue;
            mutants++;
            string text = Printer.Text(mutant);
            Node? parsed = Parser.Parse(text);
            Check(parsed != null, "mutant stayed grammatical: '" + text + "'");
            Check(Evaluator.Value(parsed!) == Evaluator.Value(mutant), "mutant semantics preserved");
            mutantTexts.Add(text);
            accepted++;
        }
        Check(mutants >= 300 && accepted == mutants, "splice accept rate is 100% (" + accepted + "/" + mutants + ")");
        Check(mutantTexts.Count > 40, "mutants are diverse: " + mutantTexts.Count + " distinct strings");
        Check(new HashSet<string>(corpus).Count > 20, "corpus diversity");

        // Oracle: exhaustive derivation enumeration vs parser acceptance, both directions, <= 5 tokens.
        var derived = new HashSet<string>();
        foreach (List<string> sequence in DerivationEnumerator.Sequences(grammar, "E", 5))
        {
            derived.Add(string.Join(string.Empty, sequence));   // same spacing the alphabet enumeration uses
        }
        var parserAccepted = new HashSet<string>();
        int tried = 0;
        foreach (string candidate in Candidates.AllOver(Alphabet, 5))
        {
            tried++;
            if (Parser.Parse(candidate) != null) parserAccepted.Add(candidate);
        }
        Check(derived.Count > 10 && derived.Count < 20000, "enumeration stayed bounded: " + derived.Count);
        Check(parserAccepted.Count == derived.Count && parserAccepted.SetEquals(derived),
            "language equality on <=5 tokens (parser " + parserAccepted.Count + " vs grammar " + derived.Count + ")");
        Check(tried == 9330, "candidate space size: " + tried);

        Console.WriteLine("  derivations: " + generator.Derivations + ", mutants accepted: " + accepted + "/" + mutants);
        Console.WriteLine("  derivations with <=5 tokens: " + derived.Count + ", candidates tested: " + tried);
        Console.WriteLine("  corpus: " + corpus.Count + " strings, longest " + Longest(corpus) + " tokens");
        Console.WriteLine("PASS");
    }

    private static int Longest(List<string> texts)
    {
        int best = 0;
        foreach (string text in texts)
        {
            best = Math.Max(best, text.Split(' ').Length);
        }
        return best;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
