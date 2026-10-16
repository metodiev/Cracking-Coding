// Task 910 - Implement neural-symbolic hybrid pipeline for structured reasoning tasks.
// (Tier: Master, Beyond - Machine Learning & Reasoning)
//
// A symbolic core and a learned-looking scorer cooperate on a small Horn-clause knowledge base (an
// ancestry/relatedness domain).  The symbolic part supplies the guarantee: a forward-chaining fixpoint over
// the *grounded* rule set, producing exactly the same entailments as a naive round-by-round saturation that
// is used as the oracle.  The neural part supplies the search order: a linear scorer over features of a
// candidate rule application
//
//     features = [1, head predicate == query predicate, head shares the query's first constant,
//                 1 / (1 + body size)]
//
// ranks the applicable rules, so the chaining is goal directed and a query is reached after far fewer
// applications than the unguided FIFO search needs - while both searches are complete and both derive the
// query, and both agree with the oracle on every entailment.
//
// Every derivation is checked: each applied rule's body atoms must already be facts or heads of earlier
// steps, and the final step must be the query itself.
//
// Simplifications: rules are grounded over a fixed constant set before search (no lifted unification), no
// negation or function symbols, and the "neural" module is a linear scorer with fixed weights rather than a
// trained graph.  Verified: oracle agreement on the closure and on six queries, the proof checker, the
// pruning effect of the scorer, the self-loop rule that can never fire, and reproducibility.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAIL: " + label);
    }

    internal sealed class GroundRule
    {
        internal string Head = string.Empty;
        internal string[] Body = Array.Empty<string>();
        internal int Index;
    }

    private static string Atom(string predicate, string first, string second) =>
        predicate + "(" + first + "," + second + ")";

    private static string PredicateOf(string atom) => atom.Substring(0, atom.IndexOf('('));

    private static string[] ArgsOf(string atom)
    {
        string inner = atom.Substring(atom.IndexOf('(') + 1);
        inner = inner.Substring(0, inner.Length - 1);
        return inner.Split(',');
    }

    /** All rules grounded over the constant set - the domain the symbolic engine reasons over. */
    private static List<GroundRule> Ground(string[] constants, string[] colors)
    {
        List<GroundRule> rules = new List<GroundRule>();
        int index = 0;
        void Add(string head, params string[] body)
        {
            rules.Add(new GroundRule { Head = head, Body = body, Index = index++ });
        }

        // An unrelated domain placed first on purpose: an unguided engine has no idea it is irrelevant.
        foreach (string x in constants)
        {
            foreach (string c in colors) Add(Atom("hue", x, c), Atom("hasHue", x, c));
        }
        foreach (string x in constants)
        {
            foreach (string y in constants)
            {
                foreach (string z in constants) Add(Atom("shares", x, y), Atom("owns", x, z), Atom("owns", y, z));
                Add(Atom("likes", x, y), Atom("shares", x, y));
                Add(Atom("likes", x, y), Atom("likes", y, x));
            }
        }

        // The reasoning domain the queries live in.
        foreach (string x in constants)
        {
            foreach (string y in constants) Add(Atom("anc", x, y), Atom("parent", x, y));
        }
        foreach (string x in constants)
        {
            foreach (string y in constants)
            {
                foreach (string z in constants) Add(Atom("anc", x, y), Atom("parent", x, z), Atom("anc", z, y));
            }
        }
        foreach (string x in constants)
        {
            foreach (string y in constants)
            {
                Add(Atom("related", x, y), Atom("anc", x, y));
                Add(Atom("related", x, y), Atom("anc", y, x));
                Add(Atom("reach", x, y), Atom("related", x, y));
                Add(Atom("related", x, y), Atom("reach", x, y));
            }
            Add(Atom("anc", x, x), Atom("anc", x, x)); // a self-loop: its head is known whenever its body is
        }
        return rules;
    }

    private static HashSet<string> Facts()
    {
        HashSet<string> facts = new HashSet<string>();
        foreach ((string from, string to) in new[] { ("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "f"), ("f", "c") })
        {
            facts.Add(Atom("parent", from, to));
        }
        foreach ((string who, string colour) in new[]
                 {
                     ("a", "red"), ("b", "red"), ("c", "green"), ("d", "green"), ("e", "blue"), ("f", "blue"),
                 })
        {
            facts.Add(Atom("hasHue", who, colour));
        }
        foreach ((string left, string right) in new[] { ("a", "b"), ("b", "c"), ("a", "c"), ("d", "e"), ("e", "f") })
        {
            facts.Add(Atom("owns", left, right));
        }
        return facts;
    }

    /** The symbolic oracle: naive round-by-round saturation; returns the closure and the attempt count. */
    private static (HashSet<string> Closure, int Attempts, int Rounds) SaturateNaive(HashSet<string> facts, List<GroundRule> rules)
    {
        HashSet<string> known = new HashSet<string>(facts);
        int attempts = 0;
        int rounds = 0;
        bool changed = true;
        while (changed)
        {
            changed = false;
            rounds++;
            foreach (GroundRule rule in rules)
            {
                attempts++;
                if (known.Contains(rule.Head)) continue;
                bool applicable = true;
                foreach (string atom in rule.Body)
                {
                    if (!known.Contains(atom))
                    {
                        applicable = false;
                        break;
                    }
                }
                if (applicable)
                {
                    known.Add(rule.Head);
                    changed = true;
                }
            }
        }
        return (known, attempts, rounds);
    }

    // ---- the neural half: a linear scorer over application features ----

    private static readonly double[] Weights = { 0.3, 1.5, 1.0, 0.4 };

    private static double[] Features(GroundRule rule, string query)
    {
        string headPredicate = PredicateOf(rule.Head);
        string queryPredicate = PredicateOf(query);
        string headFirst = ArgsOf(rule.Head)[0];
        string queryFirst = ArgsOf(query)[0];
        return new double[]
        {
            1.0,
            headPredicate == queryPredicate ? 1.0 : 0.0,
            headFirst == queryFirst ? 1.0 : 0.0,
            1.0 / (1.0 + rule.Body.Length),
        };
    }

    private static double Score(GroundRule rule, string query)
    {
        double[] features = Features(rule, query);
        double score = 0.0;
        for (int i = 0; i < features.Length; i++) score += Weights[i] * features[i];
        return score;
    }

    /** Guided forward chaining: complete, but it always applies the highest scoring applicable rule. */
    private static (bool Found, int Attempts, List<GroundRule> Proof, HashSet<string> Closure) Chain(
        HashSet<string> facts, List<GroundRule> rules, string query, bool neural, int cap, bool saturateAll)
    {
        HashSet<string> known = new HashSet<string>(facts);
        HashSet<int> applied = new HashSet<int>();
        List<GroundRule> proof = new List<GroundRule>();
        int attempts = 0;
        bool found = known.Contains(query);
        while ((saturateAll || !found) && attempts < cap)
        {
            int bestIndex = -1;
            double bestScore = double.NegativeInfinity;
            foreach (GroundRule rule in rules)
            {
                if (applied.Contains(rule.Index) || known.Contains(rule.Head)) continue;
                bool applicable = true;
                foreach (string atom in rule.Body)
                {
                    if (!known.Contains(atom))
                    {
                        applicable = false;
                        break;
                    }
                }
                if (!applicable) continue;
                double score = neural ? Score(rule, query) : -rule.Index; // unguided: fixed rule order
                if (bestIndex < 0 || score > bestScore)
                {
                    bestIndex = rule.Index;
                    bestScore = score;
                }
            }
            if (bestIndex < 0) break;
            GroundRule chosen = rules[bestIndex]; // Index is the position in the grounded rule list
            applied.Add(bestIndex);
            known.Add(chosen.Head);
            proof.Add(chosen);
            attempts++;
            found = known.Contains(query);
        }
        return (found, attempts, proof, known);
    }

    /** Independent proof checker: every step's premises must be facts or earlier heads, ending at the query. */
    private static bool VerifyProof(HashSet<string> facts, List<GroundRule> proof, string query)
    {
        HashSet<string> available = new HashSet<string>(facts);
        foreach (GroundRule step in proof)
        {
            foreach (string atom in step.Body)
            {
                if (!available.Contains(atom)) return false;
            }
            available.Add(step.Head);
        }
        return proof.Count > 0 ? proof[proof.Count - 1].Head == query : facts.Contains(query);
    }

    private static string[] Queries() =>
        new[]
        {
            Atom("anc", "a", "e"), Atom("related", "d", "a"), Atom("reach", "f", "e"), Atom("anc", "e", "a"),
            Atom("shares", "a", "b"), Atom("anc", "a", "a"),
        };

    public static void Main()
    {
        string[] constants = { "a", "b", "c", "d", "e", "f" };
        string[] colors = { "red", "green", "blue" };
        List<GroundRule> rules = Ground(constants, colors);
        HashSet<string> facts = Facts();

        // ---- 1. the guided search reproduces the oracle's closure ------------------------------------
        (HashSet<string> oracle, int oracleAttempts, int oracleRounds) = SaturateNaive(facts, rules);
        (bool _, int guidedAttempts, List<GroundRule> _, HashSet<string> guided) =
            Chain(facts, rules, Atom("anc", "a", "e"), neural: true, cap: 5000, saturateAll: true);
        bool sameClosure = guided.Count == oracle.Count;
        foreach (string atom in oracle) sameClosure &= guided.Contains(atom);
        Check(sameClosure, "the guided fixpoint equals the naive closure (" + guided.Count + " atoms)");
        Check(guided.Count > facts.Count, "reasoning added " + (guided.Count - facts.Count) + " atoms beyond the facts");
        Check(guidedAttempts < oracleAttempts, "one guided sweep costs " + guidedAttempts + " applications versus the oracle's " +
            oracleAttempts + " attempts over " + oracleRounds + " rounds");
        Check(guided.Contains(Atom("anc", "a", "e")) && guided.Contains(Atom("related", "d", "a")),
            "the closure contains both a long chain and a reversed relation");

        // ---- 2. the derivation is a checked proof ------------------------------------------------------
        (bool queryFound, int queryAttempts, List<GroundRule> proof, HashSet<string> _) =
            Chain(facts, rules, Atom("anc", "a", "e"), neural: true, cap: 5000, saturateAll: false);
        Check(queryFound, "the query anc(a,e) is entailed");
        Check(VerifyProof(facts, proof, Atom("anc", "a", "e")), "the recovered derivation passes the independent proof checker");
        List<GroundRule> truncated = new List<GroundRule>(proof);
        truncated.RemoveAt(truncated.Count - 1);
        Check(!VerifyProof(facts, truncated, Atom("anc", "a", "e")),
            "dropping the final step invalidates the derivation of the query");
        List<GroundRule> forged = new List<GroundRule>(proof);
        forged.Add(new GroundRule
        {
            Head = Atom("anc", "a", "e"),
            Body = new[] { Atom("parent", "z", "e") },
            Index = 9999,
        });
        Check(!VerifyProof(facts, forged, Atom("anc", "a", "e")),
            "a step whose premises are not available is rejected by the proof checker");

        // ---- 3. entailment agreement on a battery of queries --------------------------------------------
        int agreements = 0;
        foreach (string query in Queries())
        {
            (bool found, _, List<GroundRule> steps, HashSet<string> _) = Chain(facts, rules, query, neural: true, cap: 5000, saturateAll: false);
            bool entailed = oracle.Contains(query);
            Check(found == entailed, "hybrid and oracle agree on " + query + " (" + found + ")");
            Check(!found || VerifyProof(facts, steps, query), "the proof for " + query + " verifies");
            agreements++;
        }
        Check(agreements == 6, "six queries were cross-checked against the oracle");

        // ---- 4. the scorer prunes the search --------------------------------------------------------------
        (bool fifoFound, int fifoAttempts, List<GroundRule> _, HashSet<string> _) =
            Chain(facts, rules, Atom("anc", "a", "e"), neural: false, cap: 5000, saturateAll: false);
        Check(fifoFound, "the unguided search is complete too");
        Check(queryAttempts < fifoAttempts, "neural guidance reaches the query in " + queryAttempts +
            " applications versus " + fifoAttempts + " unguided");

        // ---- 5. the scorer against an independent dot product ---------------------------------------------
        GroundRule ancRule = rules.Find(rule => PredicateOf(rule.Head) == "anc")!;
        GroundRule unrelatedRule = rules.Find(rule => PredicateOf(rule.Head) == "likes")!;
        string targetQuery = Atom("anc", "a", "e");
        string[] headArgs = ArgsOf(ancRule.Head);
        double manual = 0.3 * 1.0
            + 1.5 * (PredicateOf(ancRule.Head) == "anc" ? 1.0 : 0.0)
            + 1.0 * (headArgs[0] == "a" ? 1.0 : 0.0)
            + 0.4 * (1.0 / (1.0 + ancRule.Body.Length));
        Check(Math.Abs(Score(ancRule, targetQuery) - manual) < 1e-12,
            "the scorer equals an independently computed dot product of the documented features");
        Check(Score(ancRule, targetQuery) > Score(unrelatedRule, targetQuery),
            "an ancestor-headed rule outranks a likes-headed one for an ancestor query");
        Check(Score(unrelatedRule, Atom("likes", "b", "a")) > Score(ancRule, Atom("likes", "b", "a")),
            "and the ranking flips when the query is about the other domain");

        // ---- 6. boundaries ---------------------------------------------------------------------------------
        HashSet<string> empty = new HashSet<string>();
        (bool emptyFound, int emptyAttempts, List<GroundRule> emptyProof, HashSet<string> emptyClosure) =
            Chain(empty, rules, Atom("anc", "a", "e"), neural: true, cap: 200, saturateAll: false);
        Check(!emptyFound && emptyAttempts == 0 && emptyProof.Count == 0 && emptyClosure.Count == 0,
            "an empty knowledge base entails nothing");
        (bool trivial, int trivialAttempts, List<GroundRule> trivialProof, HashSet<string> _) =
            Chain(facts, rules, Atom("parent", "a", "b"), neural: true, cap: 200, saturateAll: false);
        Check(trivial && trivialAttempts == 0 && trivialProof.Count == 0, "a queried fact needs no derivation steps");
        (bool unknown, _, _, HashSet<string> _) = Chain(facts, rules, Atom("anc", "a", "z"), neural: true, cap: 5000, saturateAll: false);
        Check(!unknown, "a query over an unknown constant is not entailed");
        (bool never, _, _, HashSet<string> _) = Chain(facts, rules, Atom("anc", "e", "e"), neural: true, cap: 5000, saturateAll: false);
        Check(!never, "the self-loop rule cannot invent facts about unrelated constants");
        Check(guided.Contains(Atom("related", "b", "a")) && guided.Contains(Atom("reach", "b", "a")),
            "the related/reach cycle saturates without looping forever");
        Check(guided.Contains(Atom("shares", "a", "b")) && guided.Contains(Atom("likes", "b", "a")),
            "the unrelated domain is saturated too, so the closure is genuinely complete");

        // ---- 7. reproducibility ------------------------------------------------------------------------------
        (bool _, int secondGuided, List<GroundRule> _, HashSet<string> secondClosure) =
            Chain(facts, rules, Atom("anc", "a", "e"), neural: true, cap: 5000, saturateAll: true);
        Check(secondGuided == guidedAttempts && secondClosure.Count == guided.Count,
            "a second run takes the identical number of applications");

        Console.WriteLine("symbolic: " + rules.Count + " grounded rules, closure of " + guided.Count + " atoms from " +
            facts.Count + " facts, oracle needed " + oracleAttempts + " attempts over " + oracleRounds + " rounds");
        Console.WriteLine("hybrid: query " + Atom("anc", "a", "e") + " proved in " + queryAttempts +
            " guided applications versus " + fifoAttempts + " unguided, proof length " + proof.Count);
        Console.WriteLine("checks: " + agreements + " queries agreed with the oracle, proof checker verified every derivation");
        Console.WriteLine("PASS");
    }
}
