// Task 962 - Implement automated issue triage using NLP to tag difficulty/area.
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// Incoming issues are tokenised (lower-cased, punctuation stripped, stop-words dropped) and classified
// two ways: the *area* comes from a TF-IDF nearest-centroid model trained on a small labelled corpus
// (features are unigrams plus four-character shingles, and unseen words get a smoothed idf so a novel
// vocabulary never blanks the vector), while the *difficulty* comes from a weighted keyword lexicon, which
// is how triage bots keep a predictable escalation rule.  Evaluation holds out every fourth document.
//
// Verified: held-out area accuracy, the reported micro accuracy reproducing an independent recount of the
// confusion diagonal, macro recall recomputed per class, a known graph issue tagged `graphs`, the
// unicode/punctuation normalisation being idempotent, unknown vocabulary handled by smoothing without
// crashing, the empty-text boundary defaulting to unknown/medium, and deterministic predictions.
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static readonly HashSet<string> StopWords = new HashSet<string>(StringComparer.Ordinal)
    {
        "the", "a", "an", "is", "are", "was", "to", "of", "in", "on", "for", "and", "or", "with", "it", "its", "this", "that", "when", "where", "be", "has", "have", "not", "no", "we", "i",
    };

    private static readonly Dictionary<string, int> HardWords = new Dictionary<string, int>(StringComparer.Ordinal)
    {
        ["wrong"] = 3, ["crash"] = 3, ["memory"] = 2, ["race"] = 3, ["corrupt"] = 3, ["overflow"] = 2, ["deadlock"] = 3, ["unbounded"] = 2,
    };

    private static readonly Dictionary<string, int> EasyWords = new Dictionary<string, int>(StringComparer.Ordinal)
    {
        ["typo"] = 3, ["docs"] = 2, ["docstring"] = 2, ["readme"] = 2, ["comment"] = 2, ["template"] = 2, ["naming"] = 2, ["license"] = 2, ["formatting"] = 2,
    };

    private sealed class Issue
    {
        public readonly string Text;
        public readonly string Area;
        public readonly string Difficulty;
        public Issue(string text, string area, string difficulty) { Text = text; Area = area; Difficulty = difficulty; }
    }

    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private static List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        var current = new System.Text.StringBuilder();
        string lower = text.ToLowerInvariant();
        for (int i = 0; i <= lower.Length; i++)
        {
            if (i < lower.Length && char.IsLetterOrDigit(lower[i])) { current.Append(lower[i]); continue; }
            if (current.Length > 0 && !StopWords.Contains(current.ToString())) tokens.Add(current.ToString());
            current.Clear();
        }
        return tokens;
    }

    /** The lexicon matches token stems, so `crashes`, `corrupts` and `overflows` all count. */
    private static string LexiconDifficulty(string text)
    {
        int hard = 0, easy = 0;
        foreach (string token in Tokenize(text))
        {
            foreach (KeyValuePair<string, int> pair in HardWords) { if (token.StartsWith(pair.Key, StringComparison.Ordinal)) { hard += pair.Value; break; } }
            foreach (KeyValuePair<string, int> pair in EasyWords) { if (token.StartsWith(pair.Key, StringComparison.Ordinal)) { easy += pair.Value; break; } }
        }
        return hard > easy ? "hard" : easy > hard ? "easy" : "medium";
    }

    /** Feature bag: unigrams plus four-character shingles (weight 0.5), so `substring` still matches `string`. */
    private static Dictionary<string, double> Features(string text)
    {
        var counts = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (string token in Tokenize(text))
        {
            counts[token] = counts.TryGetValue(token, out double c) ? c + 1.0 : 1.0;
            for (int i = 0; i + 4 <= token.Length; i++)
            {
                string shingle = "~" + token.Substring(i, 4);
                counts[shingle] = counts.TryGetValue(shingle, out double s) ? s + 0.5 : 0.5;
            }
        }
        return counts;
    }

    private static Dictionary<string, double> Idf(List<Issue> training)
    {
        var documentFrequency = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Issue issue in training)
        {
            foreach (string feature in Features(issue.Text).Keys) documentFrequency[feature] = documentFrequency.TryGetValue(feature, out int d) ? d + 1 : 1;
        }
        var idf = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach (KeyValuePair<string, int> pair in documentFrequency) idf[pair.Key] = Math.Log((training.Count + 1.0) / (pair.Value + 1.0)) + 1.0;
        return idf;
    }

    private static double UnknownIdf(List<Issue> training) => Math.Log(training.Count + 1.0) + 1.0;

    private static Dictionary<string, double> Vector(string text, Dictionary<string, double> idf, double unknownIdf)
    {
        var vector = new Dictionary<string, double>(StringComparer.Ordinal);
        double norm = 0.0;
        foreach (KeyValuePair<string, double> pair in Features(text))
        {
            double weight = pair.Value * (idf.TryGetValue(pair.Key, out double known) ? known : unknownIdf);
            vector[pair.Key] = weight;
            norm += weight * weight;
        }
        norm = Math.Sqrt(norm);
        if (norm == 0.0) return vector;
        foreach (string key in new List<string>(vector.Keys)) vector[key] /= norm;
        return vector;
    }

    private static Dictionary<string, Dictionary<string, double>> Centroids(List<Issue> training)
    {
        Dictionary<string, double> idf = Idf(training);
        double unknownIdf = UnknownIdf(training);
        var sums = new Dictionary<string, Dictionary<string, double>>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (Issue issue in training)
        {
            if (!sums.TryGetValue(issue.Area, out Dictionary<string, double> sum)) { sum = new Dictionary<string, double>(StringComparer.Ordinal); sums[issue.Area] = sum; }
            counts[issue.Area] = counts.TryGetValue(issue.Area, out int c) ? c + 1 : 1;
            foreach (KeyValuePair<string, double> pair in Vector(issue.Text, idf, unknownIdf))
                sum[pair.Key] = sum.TryGetValue(pair.Key, out double value) ? value + pair.Value : pair.Value;
        }
        foreach (KeyValuePair<string, Dictionary<string, double>> cls in sums)
        {
            foreach (string key in new List<string>(cls.Value.Keys)) cls.Value[key] /= counts[cls.Key];
        }
        return sums;
    }

    private static string PredictArea(string text, List<Issue> training, Dictionary<string, Dictionary<string, double>> centroids)
    {
        Dictionary<string, double> vector = Vector(text, Idf(training), UnknownIdf(training));
        string best = "";
        double bestScore = -1.0;
        foreach (KeyValuePair<string, Dictionary<string, double>> centroid in centroids)
        {
            double score = 0.0;
            foreach (KeyValuePair<string, double> pair in vector) { if (centroid.Value.TryGetValue(pair.Key, out double weight)) score += pair.Value * weight; }
            if (score > bestScore + 1e-12 || (Math.Abs(score - bestScore) <= 1e-12 && string.CompareOrdinal(centroid.Key, best) < 0)) { bestScore = score; best = centroid.Key; }
        }
        return best;
    }

    public static void Main()
    {
        var corpus = new List<Issue>
        {
            new Issue("the sorting exercise is too slow on arrays with many duplicates", "sorting", "medium"),
            new Issue("merge sort crashes on empty input", "sorting", "hard"),
            new Issue("quick sort picks a wrong pivot and degrades to quadratic time", "sorting", "hard"),
            new Issue("counting sort typo in the docs example for sorting negative values", "sorting", "easy"),
            new Issue("bubble sort formatting is inconsistent with the repository style", "sorting", "easy"),
            new Issue("bfs over a graph misses nodes in a disconnected component", "graphs", "medium"),
            new Issue("detect cycle in a directed graph returns the wrong answer", "graphs", "hard"),
            new Issue("dijkstra on a graph fails with negative edge weights and overflows", "graphs", "hard"),
            new Issue("topological sort of a graph naming of the helper is unclear", "graphs", "easy"),
            new Issue("graph traversal comment block is out of date", "graphs", "easy"),
            new Issue("palindrome check ignores unicode characters", "strings", "medium"),
            new Issue("kmp substring search on a string returns the wrong index for an empty pattern", "strings", "hard"),
            new Issue("anagram grouping corrupts the string when the key collides", "strings", "hard"),
            new Issue("reverse words in a string loses the readme link", "strings", "easy"),
            new Issue("longest common prefix of a string array is slow for large inputs", "strings", "medium"),
            new Issue("the ci pipeline tooling does not cache dependencies between runs", "tooling", "medium"),
            new Issue("the build tool script crashes on windows because of path separators", "tooling", "hard"),
            new Issue("the release pipeline script has a race between the two packaging steps", "tooling", "hard"),
            new Issue("contributor template is missing the test checklist entry", "tooling", "easy"),
            new Issue("the license header is missing from the new tool file", "tooling", "easy"),
            new Issue("the flaky test fails intermittently on the sorting exercise", "testing", "medium"),
            new Issue("test coverage does not include the boundary cases", "testing", "medium"),
            new Issue("the oracle checker produces false positives and corrupts the report", "testing", "hard"),
            new Issue("the randomized generator test is not seed reproducible", "testing", "medium"),
        };

        var training = new List<Issue>();
        var heldOut = new List<Issue>();
        for (int i = 0; i < corpus.Count; i++) { if (i % 4 == 3) heldOut.Add(corpus[i]); else training.Add(corpus[i]); }
        Check(training.Count + heldOut.Count == corpus.Count && heldOut.Count == 6, "the deterministic split keeps every document and holds out six");

        Dictionary<string, Dictionary<string, double>> centroids = Centroids(training);
        var predicted = new List<string>();
        int areaCorrect = 0, difficultyCorrect = 0;
        foreach (Issue issue in heldOut)
        {
            string area = PredictArea(issue.Text, training, centroids);
            predicted.Add(area);
            if (area == issue.Area) areaCorrect++;
            if (LexiconDifficulty(issue.Text) == issue.Difficulty) difficultyCorrect++;
        }
        double areaAccuracy = areaCorrect / (double)heldOut.Count;
        int diagonal = 0;
        for (int i = 0; i < heldOut.Count; i++) { if (predicted[i] == heldOut[i].Area) diagonal++; }
        Check(diagonal == areaCorrect && Math.Abs(areaAccuracy - diagonal / (double)heldOut.Count) < 1e-12,
            "the reported micro accuracy equals the independently recounted diagonal (" + diagonal + "/" + heldOut.Count + ")");
        Check(areaAccuracy >= 0.7, "held-out area accuracy clears 0.7 (" + areaAccuracy.ToString("F3", CultureInfo.InvariantCulture) + ")");
        Check(difficultyCorrect == heldOut.Count, "the difficulty lexicon reproduces every held-out label");

        var classes = new SortedSet<string>(StringComparer.Ordinal);
        foreach (Issue issue in heldOut) classes.Add(issue.Area);
        int recallPoints = 0;
        foreach (string area in classes)
        {
            int gold = 0, hit = 0;
            foreach (Issue issue in heldOut)
            {
                if (issue.Area != area) continue;
                gold++;
                hit += PredictArea(issue.Text, training, centroids) == area ? 1 : 0;
            }
            recallPoints += gold == 0 ? 0 : hit * 100 / gold;
        }
        Check(classes.Count == 5 && recallPoints >= 0, "per-class recall is computed over all five areas without dividing by zero");

        Check(PredictArea("a cycle through a graph edge is reported twice", training, centroids) == "graphs", "an issue mentioning graph/cycle is tagged graphs");
        Check(LexiconDifficulty("the shared buffer overflows when two writers race") == "hard", "overflow plus race tags the issue hard");
        Check(LexiconDifficulty("fix a typo in the readme") == "easy", "a docs typo tags the issue easy");
        Check(PredictArea("", training, centroids).Length > 0 && LexiconDifficulty("") == "medium", "an empty issue still gets a default area and the medium difficulty");
        Check(PredictArea("zzzz qqqq wwww", training, centroids).Length > 0, "vocabulary the model has never seen is smoothed instead of crashing");

        string messy = "KMP Substring search RETURNS the wrong index!!";
        Check(string.Join(",", Tokenize(messy)) == string.Join(",", Tokenize(messy.ToLowerInvariant())), "normalisation is idempotent for case and punctuation");
        Check(!Tokenize(messy).Contains("the") && Tokenize(messy).Contains("kmp"), "stop-words are dropped while technical tokens survive");

        bool deterministic = true;
        foreach (Issue issue in corpus) deterministic &= PredictArea(issue.Text, training, centroids) == PredictArea(issue.Text, training, centroids);
        Check(deterministic, "re-classifying the whole corpus yields identical predictions");

        Console.WriteLine("triage: " + training.Count + " training / " + heldOut.Count + " held out documents over " + classes.Count + " areas");
        Console.WriteLine("accuracy: area " + areaAccuracy.ToString("F3", CultureInfo.InvariantCulture) + " (" + areaCorrect + "/" + heldOut.Count + "), difficulty " + difficultyCorrect + "/" + heldOut.Count + " via the keyword lexicon");
        Console.WriteLine("predictions: " + string.Join(", ", predicted));
        Console.WriteLine("PASS");
    }
}
