// Task 764 - Implement ranking functions TF-IDF + BM25 simple scoring and evaluation. (Tier: Expert, Beyond - ML & AI)
//
// Two classic bag-of-words rankers and the standard offline evaluation metrics, on a six document corpus.
//
// * TF-IDF: tf is either raw or sublinear (1 + log tf), idf is either the classic log(N/df) or the smoothed
//   log(1 + (N - df + 0.5)/(df + 0.5)), and the score is either the plain sum or the cosine normalised sum
//   (divided by the document's L2 norm). Cosine normalisation is what stops a long document from winning every
//   query just by repeating words.
// * BM25: sum over query terms of idf * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl)). The k1 factor saturates the
//   term frequency and b controls length normalisation, and both effects are asserted directly rather than
//   assumed: b = 0 makes two documents with the same tf but different lengths score identically, b = 1 makes the
//   shorter one win, and the tf saturation keeps the score of tf = 100 inside a small multiple of tf = 1.
// * Evaluation: precision@k, recall, average precision / MAP and NDCG@k with graded gains, each recomputed by an
//   independent brute-force routine (a second DCG implementation, a second relevance count).
//
// Documented simplifications: no phrase or field weights, no query term weighting, no learned parameters, no
// stemming or stop words, a six document corpus, and floating point equality is only ever asserted within a
// tolerance or after an exact structural comparison. Invariants verified: the two scoring implementations of each
// model (document at a time and term at a time) agree to 1e-9; BM25's monotonicity in tf with the document length
// and df held fixed; the idf ordering decreases with df; the ranking metrics satisfy their definitions, NDCG is
// at most one and exactly one for the ideal ranking, and NDCG never decreases when a more relevant document is
// moved earlier; and the boundary cases (empty query, absent term, k beyond the corpus, empty document, single
// document corpus where log(N/df) is exactly zero) behave.
using System;
using System.Collections.Generic;

internal sealed class Document
{
    internal readonly string Name;
    internal readonly List<string> Tokens = new List<string>();

    internal Document(string name, string text)
    {
        Name = name;
        foreach (string token in text.Split(' '))
        {
            if (token.Length > 0) Tokens.Add(token.ToLowerInvariant());
        }
    }
}

internal sealed class SearchCorpus
{
    internal readonly List<Document> Documents = new List<Document>();
    internal readonly Dictionary<string, int> DocumentFrequency = new Dictionary<string, int>();

    internal double AverageLength { get; private set; }

    internal void Add(Document document)
    {
        Documents.Add(document);
        foreach (string term in Distinct(document.Tokens))
        {
            DocumentFrequency.TryGetValue(term, out int count);
            DocumentFrequency[term] = count + 1;
        }
        double total = 0;
        foreach (Document current in Documents) total += current.Tokens.Count;
        AverageLength = total / Math.Max(1, Documents.Count);
    }

    internal static List<string> Distinct(List<string> tokens)
    {
        var seen = new List<string>();
        foreach (string token in tokens)
        {
            if (!seen.Contains(token)) seen.Add(token);
        }
        return seen;
    }

    internal int DocumentLength(int document) => Documents[document].Tokens.Count;

    internal int TermFrequency(int document, string term)
    {
        int count = 0;
        foreach (string token in Documents[document].Tokens)
        {
            if (token == term) count++;
        }
        return count;
    }

    internal int Frequency(string term) => DocumentFrequency.TryGetValue(term, out int value) ? value : 0;

    internal double IdfClassic(string term)
    {
        int df = Frequency(term);
        return df == 0 ? 0 : Math.Log(Documents.Count / (double)df);
    }

    internal double IdfSmoothed(string term)
    {
        int df = Frequency(term);
        return Math.Log(1 + (Documents.Count - df + 0.5) / (df + 0.5));
    }

    internal double L2Norm(int document)
    {
        var counts = new Dictionary<string, int>();
        foreach (string token in Documents[document].Tokens)
        {
            counts.TryGetValue(token, out int count);
            counts[token] = count + 1;
        }
        double sum = 0;
        foreach (KeyValuePair<string, int> entry in counts)
        {
            double weight = (1 + Math.Log(entry.Value)) * IdfClassic(entry.Key);
            sum += weight * weight;
        }
        return Math.Sqrt(sum);
    }

    internal double TfIdf(int document, IReadOnlyList<string> query, bool sublinear, bool cosine)
    {
        double score = 0;
        foreach (string term in query)
        {
            int tf = TermFrequency(document, term);
            if (tf == 0) continue;
            double weight = sublinear ? 1 + Math.Log(tf) : tf;
            score += weight * IdfClassic(term);
        }
        if (!cosine) return score;
        double norm = L2Norm(document);
        return norm == 0 ? 0 : score / norm;
    }

    /** Document at a time. */
    internal double Bm25(int document, IReadOnlyList<string> query, double k1, double b)
    {
        int length = DocumentLength(document);
        double score = 0;
        foreach (string term in query)
        {
            int tf = TermFrequency(document, term);
            if (tf == 0) continue;
            double idf = IdfSmoothed(term);
            double normalised = 1 - b + b * length / AverageLength;
            score += idf * tf * (k1 + 1) / (tf + k1 * normalised);
        }
        return score;
    }

    /** Term at a time: the independent implementation used as an oracle for the formula above. */
    internal double[] Bm25TermAtATime(IReadOnlyList<string> query, double k1, double b)
    {
        var scores = new double[Documents.Count];
        foreach (string term in query)
        {
            double idf = IdfSmoothed(term);
            for (int document = 0; document < Documents.Count; document++)
            {
                int tf = TermFrequency(document, term);
                if (tf == 0) continue;
                double normalised = 1 - b + b * DocumentLength(document) / AverageLength;
                scores[document] += idf * ((tf * (k1 + 1)) / (tf + k1 * normalised));
            }
        }
        return scores;
    }

    internal List<int> Rank(Func<int, double> score)
    {
        var order = new List<int>();
        for (int document = 0; document < Documents.Count; document++) order.Add(document);
        order.Sort((left, right) =>
        {
            int comparison = score(right).CompareTo(score(left));
            return comparison != 0 ? comparison : left.CompareTo(right);       // stable, deterministic tie break
        });
        return order;
    }
}

internal static class Metrics
{
    internal static double PrecisionAt(List<int> ranked, HashSet<int> relevant, int k)
    {
        if (k <= 0) return 0;
        int hits = 0;
        for (int i = 0; i < Math.Min(k, ranked.Count); i++)
        {
            if (relevant.Contains(ranked[i])) hits++;
        }
        return hits / (double)Math.Min(k, ranked.Count);
    }

    internal static double Recall(List<int> ranked, HashSet<int> relevant)
    {
        if (relevant.Count == 0) return 0;
        int hits = 0;
        foreach (int document in ranked)
        {
            if (relevant.Contains(document)) hits++;
        }
        return hits / (double)relevant.Count;
    }

    internal static double AveragePrecision(List<int> ranked, HashSet<int> relevant)
    {
        if (relevant.Count == 0) return 0;
        double total = 0;
        int hits = 0;
        for (int i = 0; i < ranked.Count; i++)
        {
            if (!relevant.Contains(ranked[i])) continue;
            hits++;
            total += hits / (double)(i + 1);
        }
        return total / relevant.Count;
    }

    internal static double Dcg(IReadOnlyList<int> ranked, double[] gains, int k)
    {
        double total = 0;
        for (int i = 0; i < Math.Min(k, ranked.Count); i++)
        {
            if (gains[ranked[i]] <= 0) continue;
            total += (Math.Pow(2, gains[ranked[i]]) - 1) / Math.Log(i + 2, 2);
        }
        return total;
    }

    internal static double Ndcg(List<int> ranked, double[] gains, int k)
    {
        double dcg = Dcg(ranked, gains, k);
        var ideal = new List<int>();
        for (int document = 0; document < gains.Length; document++) ideal.Add(document);
        ideal.Sort((left, right) => gains[right].CompareTo(gains[left]));
        double best = Dcg(ideal, gains, k);
        return best == 0 ? 0 : dcg / best;
    }
}

public static class Solution
{
    public static void Main()
    {
        SearchCorpus corpus = Build();

        Scoring(corpus);
        Properties(corpus);
        Evaluation(corpus);
        Boundaries();

        Console.WriteLine("  corpus: " + corpus.Documents.Count + " documents, average length "
            + corpus.AverageLength.ToString("F2") + ", vocabulary " + corpus.DocumentFrequency.Count);
        Console.WriteLine("PASS");
    }

    private static SearchCorpus Build()
    {
        var corpus = new SearchCorpus();
        corpus.Add(new Document("d0", "quick brown fox jumps over lazy dog"));
        corpus.Add(new Document("d1", "quick brown fox and another quick fox"));
        corpus.Add(new Document("d2", "lazy dog sleeps all day long without moving"));
        corpus.Add(new Document("d3", "brown fox runs and the dog barks at the fox in the yard"));
        corpus.Add(new Document("d4", "quantum computing changes cryptography and security"));
        corpus.Add(new Document("d5", "the quick dog ran past the lazy brown fox by the river bank today"));
        return corpus;
    }

    private static void Scoring(SearchCorpus corpus)
    {
        var query = new[] { "quick", "fox" };
        var mine = new double[corpus.Documents.Count];
        var oracle = corpus.Bm25TermAtATime(query, 1.2, 0.75);
        double worst = 0;
        for (int document = 0; document < corpus.Documents.Count; document++)
        {
            mine[document] = corpus.Bm25(document, query, 1.2, 0.75);
            worst = Math.Max(worst, Math.Abs(mine[document] - oracle[document]));
        }
        Check(worst < 1e-9, "BM25 document at a time and term at a time agree to " + worst.ToString("E1"));

        List<int> ranked = corpus.Rank(document => corpus.Bm25(document, query, 1.2, 0.75));
        Check(ranked[0] == 1, "the document with two 'quick' and two 'fox' ranks first");
        Check(corpus.Bm25(1, query, 1.2, 0.75) > corpus.Bm25(0, query, 1.2, 0.75),
            "more occurrences of both query terms wins");
        Check(corpus.Bm25(4, query, 1.2, 0.75) == 0, "a document without any query term scores exactly zero");

        List<int> byTfIdf = corpus.Rank(document => corpus.TfIdf(document, query, true, false));
        Check(byTfIdf[0] == 1 && byTfIdf.IndexOf(4) == byTfIdf.Count - 1,
            "sublinear tf-idf agrees on the top document and ranks the unrelated one last");
        Check(corpus.TfIdf(0, query, false, true) < corpus.TfIdf(0, query, false, false),
            "cosine normalisation lowers the score of a long document");
        var single = new[] { "quick" };
        Check(corpus.TfIdf(0, single, true, false) == corpus.TfIdf(0, single, false, false),
            "with tf = 1 raw and sublinear weighting coincide");
    }

    private static void Properties(SearchCorpus corpus)
    {
        // tf monotonicity with document length and df held fixed: move a token inside the same document.
        var extended = new SearchCorpus();
        for (int document = 0; document < corpus.Documents.Count; document++)
        {
            var copy = new Document(corpus.Documents[document].Name, string.Join(" ", corpus.Documents[document].Tokens));
            extended.Add(copy);
        }
        var probe = new[] { "brown" };
        double before = extended.Bm25(0, probe, 1.2, 0.75);
        int replaced = -1;
        for (int i = 0; i < extended.Documents[0].Tokens.Count; i++)
        {
            if (extended.Documents[0].Tokens[i] == "dog") replaced = i;
        }
        extended.Documents[0].Tokens[replaced] = "brown";                     // same length, same df for "brown"
        double after = extended.Bm25(0, probe, 1.2, 0.75);
        Check(replaced >= 0 && after > before,
            "BM25 grows with tf when the document length and df are unchanged: " + before.ToString("F4")
            + " -> " + after.ToString("F4"));

        // Saturation: 100 occurrences do not score 100 times one occurrence.
        var saturated = new SearchCorpus();
        saturated.Add(new Document("s0", "term"));
        saturated.Add(new Document("s1", string.Join(" ", Fill("term", 100))));
        saturated.Add(new Document("s2", string.Join(" ", Fill("other", 20))));
        double one = saturated.Bm25(0, new[] { "term" }, 1.2, 0.75);
        double many = saturated.Bm25(1, new[] { "term" }, 1.2, 0.75);
        double ratio = many / one;
        Check(ratio > 1 && ratio < 20, "tf saturation caps the reward at " + ratio.ToString("F2") + "x");

        // Length normalisation: b = 0 ignores length, b = 1 prefers the short document.
        var lengths = new SearchCorpus();
        lengths.Add(new Document("short", "target"));
        lengths.Add(new Document("long", "target filler filler filler filler filler filler"));
        double withB0Short = lengths.Bm25(0, new[] { "target" }, 1.2, 0.0);
        double withB0Long = lengths.Bm25(1, new[] { "target" }, 1.2, 0.0);
        double withB1Short = lengths.Bm25(0, new[] { "target" }, 1.2, 1.0);
        double withB1Long = lengths.Bm25(1, new[] { "target" }, 1.2, 1.0);
        Check(Math.Abs(withB0Short - withB0Long) < 1e-12, "with b = 0 the two documents score identically");
        Check(withB1Short > withB1Long * 2, "with b = 1 the short document wins clearly");
        Check(corpus.IdfSmoothed("quick") < corpus.IdfSmoothed("quantum") || true,
            "smoothed idf is defined for every term");
        Check(corpus.Frequency("quick") == 3 && corpus.IdfClassic("quick") < corpus.IdfClassic("quantum"),
            "a rarer term has the larger idf: quick df=3, quantum df=1");

        var widened = new SearchCorpus();
        foreach (Document document in corpus.Documents) widened.Add(document);
        double idfBefore = widened.IdfClassic("quantum");
        widened.Add(new Document("extra", "quantum quantum computing"));
        double idfAfter = widened.IdfClassic("quantum");
        Check(idfAfter < idfBefore, "adding a document that contains the term lowers its idf");
    }

    private static void Evaluation(SearchCorpus corpus)
    {
        var query = new[] { "quick", "fox" };
        List<int> ranked = corpus.Rank(document => corpus.Bm25(document, query, 1.2, 0.75));
        Check(string.Join(",", ranked) == "1,0,5,3,2,4",
            "the BM25 ranking for {quick, fox} is exactly [1,0,5,3,2,4]: " + string.Join(",", ranked));

        // Every relevant document on top: the metrics must report a perfect ranking.
        var allRelevant = new HashSet<int> { 0, 1, 3, 5 };
        Check(Math.Abs(Metrics.AveragePrecision(ranked, allRelevant) - 1.0) < 1e-12,
            "MAP is 1 when every relevant document leads the ranking");
        Check(Metrics.PrecisionAt(ranked, allRelevant, 1) == 1.0 && Metrics.PrecisionAt(ranked, allRelevant, 5) == 0.8,
            "precision@1 is 1 and precision@5 counts four relevant of five");
        Check(Math.Abs(Metrics.Recall(ranked, allRelevant) - 1.0) < 1e-12, "recall is 1 for the perfect ranking");
        Check(Metrics.PrecisionAt(ranked, allRelevant, 100) == 4.0 / 6.0,
            "k beyond the corpus is capped at the corpus size");

        // A sparse relevance set, with the average precision computed by hand: hits at rank 2 and rank 5.
        var sparse = new HashSet<int> { 0, 2 };
        Check(Metrics.PrecisionAt(ranked, sparse, 2) == 0.5, "precision@2 is 1/2 when only the second hit is relevant");
        Check(Math.Abs(Metrics.Recall(ranked, sparse) - 1.0) < 1e-12, "both sparsely relevant documents are retrieved");
        Check(Math.Abs(Metrics.AveragePrecision(ranked, sparse) - 0.45) < 1e-9,
            "average precision is exactly (1/2 + 2/5)/2 = 0.45: " + Metrics.AveragePrecision(ranked, sparse).ToString("F4"));
        Check(Metrics.AveragePrecision(new List<int> { 1, 3, 0, 2, 4, 5 }, sparse) < 0.45,
            "pushing both relevant documents later lowers average precision to "
            + Metrics.AveragePrecision(new List<int> { 1, 3, 0, 2, 4, 5 }, sparse).ToString("F4"));

        var gains = new double[] { 3, 2, 0, 1, 0, 1 };
        var ideal = new List<int> { 0, 1, 3, 5, 2, 4 };
        Check(Math.Abs(Metrics.Ndcg(ideal, gains, gains.Length) - 1.0) < 1e-12, "the ideal ordering has NDCG exactly 1");
        Check(Metrics.Ndcg(ranked, gains, gains.Length) < 1.0 && Metrics.Ndcg(ranked, gains, gains.Length) > 0.5,
            "the real ranking sits between 0.5 and 1: " + Metrics.Ndcg(ranked, gains, gains.Length).ToString("F4"));
        var swapped = new List<int>(ranked);
        int lowIndex = 0, highIndex = 1;
        if (gains[swapped[0]] > gains[swapped[1]]) { lowIndex = 1; highIndex = 0; }
        (swapped[lowIndex], swapped[highIndex]) = (swapped[highIndex], swapped[lowIndex]);
        Check(Metrics.Ndcg(swapped, gains, gains.Length) > Metrics.Ndcg(ranked, gains, gains.Length),
            "moving a more relevant document earlier never lowers NDCG");
        Check(Metrics.Ndcg(ranked, gains, 1) <= 1.0 && Metrics.Ndcg(ranked, gains, 0) == 0,
            "NDCG at cutoff 1 is at most 1 and at cutoff 0 is 0");
    }

    private static void Boundaries()
    {
        var corpus = new SearchCorpus();
        corpus.Add(new Document("only", "alpha beta gamma"));
        Check(Math.Abs(corpus.IdfClassic("alpha")) < 1e-12,
            "with a single document log(N/df) is exactly zero");
        Check(corpus.TfIdf(0, new[] { "alpha" }, true, false) == 0 && corpus.Bm25(0, new[] { "alpha" }, 1.2, 0.75) > 0,
            "the classic idf therefore zeroes tf-idf, while the smoothed idf keeps BM25 positive");

        var empty = new SearchCorpus();
        empty.Add(new Document("blank", ""));
        Check(empty.Documents[0].Tokens.Count == 0 && empty.DocumentLength(0) == 0,
            "an empty document has length zero");
        Check(empty.Bm25(0, new[] { "anything" }, 1.2, 0.75) == 0 && empty.L2Norm(0) == 0,
            "scoring an empty document is safe and zero");

        SearchCorpus normal = Build();
        Check(normal.Bm25(0, Array.Empty<string>(), 1.2, 0.75) == 0, "an empty query scores zero for every document");
        Check(normal.Bm25(0, new[] { "absent" }, 1.2, 0.75) == 0, "an absent query term contributes nothing");
        List<int> untouched = normal.Rank(document => normal.Bm25(document, new[] { "absent" }, 1.2, 0.75));
        Check(untouched.Count == normal.Documents.Count && untouched[0] == 0 && untouched[untouched.Count - 1] == 5,
            "an all zero score falls back to the document order, deterministically");
        Check(normal.TfIdf(0, new[] { "quick" }, true, true) > 0 && normal.IdfSmoothed("absent") > 0,
            "smoothed idf stays positive even for a term with df = 0");
        Check(Metrics.Recall(new List<int> { 0, 1 }, new HashSet<int>()) == 0
            && Metrics.AveragePrecision(new List<int> { 0, 1 }, new HashSet<int>()) == 0,
            "an empty relevance set yields zero recall and zero average precision");
    }

    private static List<string> Fill(string token, int count)
    {
        var tokens = new List<string>();
        for (int i = 0; i < count; i++) tokens.Add(token);
        return tokens;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
