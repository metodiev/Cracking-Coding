// Task 405 - Implement a minimal DFA for a fixed string set (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
//
// Scope: the DFA recognizes exactly the finite set of words it was built from. Only the
// *prefixes* of the set's words matter: the builder's states are exactly the distinct
// non-empty prefixes plus the empty root, and any character no prefix uses leads to an
// explicit dead state. The alphabet is the sorted union of the words' characters; characters
// outside it are documented to reject.
//
// Minimization uses the classic table-filling (Moore) algorithm: mark every pair that differs
// in acceptance, then repeatedly mark a pair when some character leads to an already marked
// pair; inseparable pairs are merged with union-find. For {"he", "she", "his"} the three
// accepting states merge into one class, so the complete minimal DFA has 7 states: {""},
// {"h"}, {"s"}, {"sh"}, {"hi"}, {"he","she","his"} and one dead state. Note "h" does NOT
// merge with "sh": "h" still accepts the continuation "is" ("his") while "sh" only accepts
// "e". Dropping the dead state leaves the 6-state partial DFA that tracks exactly the
// prefixes.
//
// Checks: the enumerated language up to length 4 equals the key set exactly, near-miss
// strings are rejected, every state is reachable, and the minimized automaton has no pair of
// equivalent states (product-BFS equivalence test), i.e. it is minimal.
using System;
using System.Collections.Generic;
using System.Linq;

internal class ExactMatchDfa
{
    public readonly List<string> States = new List<string>();
    public readonly Dictionary<string, int> Index = new Dictionary<string, int>();
    public readonly List<int[]> Trans = new List<int[]>();
    public readonly List<bool> Accept = new List<bool>();
    public readonly string Alphabet;
    public int Dead;

    public ExactMatchDfa(IEnumerable<string> words, string alphabet)
    {
        Alphabet = alphabet;
        States.Add("");
        Index[""] = 0;
        Trans.Add(Filled(-1));
        Accept.Add(false);
        foreach (string word in words)
        {
            int node = 0;
            string prefix = "";
            foreach (char ch in word)
            {
                string nxt = prefix + ch;
                if (!Index.TryGetValue(nxt, out int id))
                {
                    id = States.Count;
                    Index[nxt] = id;
                    States.Add(nxt);
                    Trans.Add(Filled(-1));
                    Accept.Add(false);
                }
                int col = alphabet.IndexOf(ch);
                if (Trans[node][col] == -1) Trans[node][col] = id;
                node = id;
                prefix = nxt;
            }
            Accept[node] = true;
        }
        Dead = States.Count;
        States.Add("(dead)");
        Accept.Add(false);
        Trans.Add(Filled(Dead));
        foreach (int[] row in Trans)
        {
            for (int c = 0; c < alphabet.Length; c++)
            {
                if (row[c] == -1) row[c] = Dead;
            }
        }
    }

    private int[] Filled(int value)
    {
        var row = new int[Alphabet.Length];
        for (int i = 0; i < row.Length; i++) row[i] = value;
        return row;
    }
}

internal class Minimized
{
    public readonly int[] ClassOf;
    public readonly int[][] Trans;
    public readonly bool[] Accept;

    public Minimized(int[] classOf, int[][] trans, bool[] accept)
    {
        ClassOf = classOf;
        Trans = trans;
        Accept = accept;
    }
}

internal static class DfaTools
{
    public static Minimized Minimize(int[][] trans, bool[] accept, int alphabetSize)
    {
        int n = trans.Length;
        var marked = new bool[n, n];
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
                if (accept[p] != accept[q]) marked[p, q] = true;
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int p = 0; p < n; p++)
            {
                for (int q = p + 1; q < n; q++)
                {
                    if (marked[p, q]) continue;
                    for (int c = 0; c < alphabetSize; c++)
                    {
                        int a = trans[p][c], b = trans[q][c];
                        if (a == b) continue;
                        int lo = Math.Min(a, b), hi = Math.Max(a, b);
                        if (marked[lo, hi])
                        {
                            marked[p, q] = true;
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
        var parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        for (int p = 0; p < n; p++)
        {
            for (int q = p + 1; q < n; q++)
            {
                if (marked[p, q]) continue;
                int rp = Find(parent, p), rq = Find(parent, q);
                if (rp != rq) parent[Math.Max(rp, rq)] = Math.Min(rp, rq);
            }
        }
        var reps = new SortedSet<int>();
        for (int i = 0; i < n; i++) reps.Add(Find(parent, i));
        var repList = reps.ToList();
        var remap = new Dictionary<int, int>();
        for (int i = 0; i < repList.Count; i++) remap[repList[i]] = i;
        var classOf = new int[n];
        for (int i = 0; i < n; i++) classOf[i] = remap[Find(parent, i)];
        var newTrans = new int[repList.Count][];
        var newAccept = new bool[repList.Count];
        for (int r = 0; r < repList.Count; r++)
        {
            newAccept[r] = accept[repList[r]];
            newTrans[r] = new int[alphabetSize];
            for (int c = 0; c < alphabetSize; c++) newTrans[r][c] = classOf[trans[repList[r]][c]];
        }
        return new Minimized(classOf, newTrans, newAccept);
    }

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    public static bool Accepts(int[][] trans, bool[] accept, string word, string alphabet)
    {
        int state = 0;
        foreach (char ch in word)
        {
            int col = alphabet.IndexOf(ch);
            if (col == -1) return false;           // documented: outside the alphabet, reject
            state = trans[state][col];
        }
        return accept[state];
    }

    public static HashSet<int> Reachable(int[][] trans)
    {
        var seen = new HashSet<int> { 0 };
        var stack = new Stack<int>();
        stack.Push(0);
        while (stack.Count > 0)
        {
            int state = stack.Pop();
            foreach (int nxt in trans[state])
            {
                if (seen.Add(nxt)) stack.Push(nxt);
            }
        }
        return seen;
    }

    public static bool AllPairsDistinguishable(int[][] trans, bool[] accept, int alphabetSize)
    {
        int n = trans.Length;
        for (int p = 0; p < n; p++)
        {
            for (int q = p + 1; q < n; q++)
            {
                var seen = new HashSet<(int, int)> { (p, q) };
                var stack = new Stack<(int, int)>();
                stack.Push((p, q));
                bool equivalent = true;
                while (stack.Count > 0 && equivalent)
                {
                    var (a, b) = stack.Pop();
                    if (accept[a] != accept[b])
                    {
                        equivalent = false;
                        break;
                    }
                    for (int c = 0; c < alphabetSize; c++)
                    {
                        int na = trans[a][c], nb = trans[b][c];
                        var key = (Math.Min(na, nb), Math.Max(na, nb));
                        if (seen.Add(key)) stack.Push((na, nb));
                    }
                }
                if (equivalent) return false;
            }
        }
        return true;
    }

    public static List<string> Language(int[][] trans, bool[] accept, string alphabet, int maxLen)
    {
        var accepted = new List<string>();
        var frontier = new List<string> { "" };
        for (int depth = 0; depth <= maxLen; depth++)
        {
            var nextFrontier = new List<string>();
            foreach (string word in frontier)
            {
                if (Accepts(trans, accept, word, alphabet)) accepted.Add(word);
                if (word.Length < maxLen)
                {
                    foreach (char ch in alphabet) nextFrontier.Add(word + ch);
                }
            }
            frontier = nextFrontier;
        }
        accepted.Sort(StringComparer.Ordinal);
        return accepted;
    }

    public static List<int> DeadStates(int[][] trans, bool[] accept)
    {
        var dead = new List<int>();
        for (int state = 0; state < trans.Length; state++)
        {
            if (accept[state]) continue;
            bool allSelf = true;
            foreach (int nxt in trans[state])
            {
                if (nxt != state)
                {
                    allSelf = false;
                    break;
                }
            }
            if (allSelf) dead.Add(state);
        }
        return dead;
    }
}

public static class Solution
{
    public static void Main()
    {
        const string alphabet = "ehis";
        var words = new[] { "he", "she", "his" };
        var builder = new ExactMatchDfa(words, alphabet);
        var expectReject = new[]
        {
            "", "h", "s", "e", "i", "heh", "hehe", "sh", "shi", "shee", "shes",
            "hiss", "hishe", "hhi", "hh", "sheh", "hise",
        };
        int[][] trans = builder.Trans.ToArray();
        bool[] accept = builder.Accept.ToArray();

        var prefixes = new SortedSet<string> { "" };
        foreach (string word in words)
        {
            for (int k = 1; k <= word.Length; k++) prefixes.Add(word.Substring(0, k));
        }
        Check(new SortedSet<string>(builder.States.Take(builder.States.Count - 1)).SetEquals(prefixes),
            "prefix states");
        Check(builder.States.Count == prefixes.Count + 1 && builder.Dead == builder.States.Count - 1, "dead state");
        Check(DfaTools.Reachable(trans).Count == trans.Length, "all reachable");

        foreach (string word in words) Check(DfaTools.Accepts(trans, accept, word, alphabet), "accept " + word);
        foreach (string word in expectReject) Check(!DfaTools.Accepts(trans, accept, word, alphabet), "reject " + word);
        foreach (string word in new[] { "hey", "his!", "she?" })
        {
            Check(!DfaTools.Accepts(trans, accept, word, alphabet), "outside alphabet " + word);
        }
        Check(DfaTools.Language(trans, accept, alphabet, 4).SequenceEqual(words.OrderBy(w => w, StringComparer.Ordinal)),
            "trie language");

        var minimal = DfaTools.Minimize(trans, accept, alphabet.Length);
        Check(minimal.Trans.Length == 7, "minimal states " + minimal.Trans.Length);
        Check(minimal.Accept.Count(v => v) == 1, "one accepting class");
        Check(DfaTools.Reachable(minimal.Trans).Count == minimal.Trans.Length, "minimal reachable");
        Check(DfaTools.AllPairsDistinguishable(minimal.Trans, minimal.Accept, alphabet.Length), "minimal");

        Check(DfaTools.Language(minimal.Trans, minimal.Accept, alphabet, 4)
            .SequenceEqual(words.OrderBy(w => w, StringComparer.Ordinal)), "minimal language");
        foreach (string word in words)
            Check(DfaTools.Accepts(minimal.Trans, minimal.Accept, word, alphabet), "min accept " + word);
        foreach (string word in expectReject)
            Check(!DfaTools.Accepts(minimal.Trans, minimal.Accept, word, alphabet), "min reject " + word);

        Check(minimal.ClassOf[builder.Index["he"]] == minimal.ClassOf[builder.Index["she"]]
            && minimal.ClassOf[builder.Index["she"]] == minimal.ClassOf[builder.Index["his"]], "accepting merge");
        Check(minimal.ClassOf[builder.Index["h"]] != minimal.ClassOf[builder.Index["sh"]], "h vs sh");
        Check(minimal.ClassOf[builder.Index["sh"]] != minimal.ClassOf[builder.Index["hi"]], "sh vs hi");
        Check(minimal.ClassOf[builder.Index["h"]] != minimal.ClassOf[builder.Index["s"]], "h vs s");
        Check(minimal.ClassOf[builder.Index[""]] != minimal.ClassOf[builder.Dead], "root vs dead");
        Check(new HashSet<int>(minimal.ClassOf).Count == 7, "distinct classes");
        Check(DfaTools.DeadStates(minimal.Trans, minimal.Accept).Count == 1, "single dead class");
        Check(minimal.Trans.Length - 1 == 6, "partial size");

        const string alphabet2 = "ab";
        var words2 = new[] { "ab", "b", "ba" };
        var builder2 = new ExactMatchDfa(words2, alphabet2);
        int[][] trans2 = builder2.Trans.ToArray();
        bool[] accept2 = builder2.Accept.ToArray();
        foreach (string word in words2) Check(DfaTools.Accepts(trans2, accept2, word, alphabet2), "set2 accept " + word);
        foreach (string word in new[] { "", "a", "bb", "aba", "abb", "bab", "baa" })
        {
            Check(!DfaTools.Accepts(trans2, accept2, word, alphabet2), "set2 reject " + word);
        }
        Check(DfaTools.Language(trans2, accept2, alphabet2, 4)
            .SequenceEqual(words2.OrderBy(w => w, StringComparer.Ordinal)), "set2 language");
        var minimal2 = DfaTools.Minimize(trans2, accept2, alphabet2.Length);
        Check(minimal2.Trans.Length == 5, "set2 minimal states " + minimal2.Trans.Length);
        Check(DfaTools.AllPairsDistinguishable(minimal2.Trans, minimal2.Accept, alphabet2.Length), "set2 minimal");
        Check(DfaTools.Language(minimal2.Trans, minimal2.Accept, alphabet2, 4)
            .SequenceEqual(words2.OrderBy(w => w, StringComparer.Ordinal)), "set2 minimal language");
        Check(minimal2.ClassOf[builder2.Index["ab"]] == minimal2.ClassOf[builder2.Index["ba"]], "set2 merge");
        Check(minimal2.ClassOf[builder2.Index["ab"]] != minimal2.ClassOf[builder2.Index["b"]], "set2 b differs");
        Check(minimal2.ClassOf[builder2.Index["a"]] != minimal2.ClassOf[builder2.Dead], "set2 a viable");
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
