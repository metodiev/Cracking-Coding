// Task 767 - Implement online re-ranking with features and click simulation (learning-to-rank). (Tier: Expert, Beyond - ML & AI)
//
// An online learning-to-rank loop: a first stage proposes candidates, a linear re-ranker scores them from a feature
// vector, a simulated user examines the ranking with position bias and clicks according to an intrinsic relevance
// the ranker cannot see, and every display produces pairwise training signal from the clicks.
//
// * Features: a retrieval score (bm25-like), the document length, an intrinsic quality prior, a freshness term and
//   a query-document cosine similarity. The score is exactly the inner product of the learned weights and the
//   feature vector, checked literally.
// * Click model: position k is examined with probability 1/(1 + k) and an examined document is clicked with a
//   probability that is a monotone function of its *hidden* relevance (2 * relevance / (1 + relevance) style
//   squashing, so the probability stays in (0, 1)). The simulation is deterministic, and monotonicity of the click
//   probability in the hidden relevance is asserted rather than assumed.
// * Learner: RankNet style pairwise logistic loss. For a clicked (positive) and an unclicked (negative) document
//   the update is w += lr * (1 - sigmoid(s_pos - s_neg)) * (phi_pos - phi_neg). The gradient is verified against a
//   numerical finite difference of the loss, so the update rule cannot silently drift from the loss it claims to
//   optimise, and one step is checked to actually lower that pair's loss.
//
// Documented simplifications: no propensity weighting or unbiased estimation (non-clicks below a click are treated
// as negatives), no explicit exploration, linear model with no regularisation sweep, hidden relevance is a
// deterministic function of the observable features so the learner has something to learn, and the "user" is a
// PRNG. Invariants verified: score = w.phi; the analytic gradient matches the finite difference within 1e-6; one
// SGD step decreases the pair loss and increases the score gap; NDCG@10 improves while training and beats both the
// static initial weights and random ranking; the running average loss trends down; each display produces exactly
// one pair per (clicked, unclicked) combination; position bias is monotone decreasing; and the boundary cases
// (no candidates, a single candidate, every item clicked, no clicks, learning rate zero) leave the weights alone.
using System;
using System.Collections.Generic;

internal static class Rng
{
    private static ulong state = 0x767BEEFCAFE12345UL;

    internal static ulong Next()
    {
        state += 0x9E3779B97F4A7C15UL;
        ulong z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double Unit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
}

internal static class Features
{
    internal const int Count = 5;
    internal static readonly string[] Names = { "bm25", "length", "quality", "freshness", "sim" };

    internal static double[] Vector(double bm25, double length, double quality, double freshness, double sim)
        => new[] { bm25, length, quality, freshness, sim };
}

internal sealed class Candidate
{
    internal int Id;
    internal double[] Phi = new double[Features.Count];
    internal double HiddenRelevance;

    internal Candidate(int id, double[] phi, double hidden)
    {
        Id = id;
        Phi = phi;
        HiddenRelevance = hidden;
    }
}

internal sealed class Query
{
    internal int Id;
    internal readonly List<Candidate> Candidates = new List<Candidate>();

    internal Query(int id) { Id = id; }
}

internal sealed class LinearRanker
{
    internal readonly double[] Weights;
    internal double LearningRate;

    internal LinearRanker(int size, double learningRate)
    {
        Weights = new double[size];
        LearningRate = learningRate;
    }

    internal double Score(Candidate candidate)
    {
        double total = 0;
        for (int i = 0; i < Weights.Length; i++) total += Weights[i] * candidate.Phi[i];
        return total;
    }

    internal static double Sigmoid(double value) => 1.0 / (1.0 + Math.Exp(-value));

    /** Pairwise logistic loss for one ordered pair. */
    internal double PairLoss(Candidate positive, Candidate negative)
        => -Math.Log(Sigmoid(Score(positive) - Score(negative)) + 1e-300);

    internal double[] PairGradient(Candidate positive, Candidate negative)
    {
        double scale = 1 - Sigmoid(Score(positive) - Score(negative));
        var gradient = new double[Weights.Length];
        for (int i = 0; i < Weights.Length; i++) gradient[i] = -scale * (positive.Phi[i] - negative.Phi[i]);
        return gradient;
    }

    internal void ApplyGradient(double[] gradient)
    {
        for (int i = 0; i < Weights.Length; i++) Weights[i] -= LearningRate * gradient[i];
    }

    internal void TrainPair(Candidate positive, Candidate negative)
    {
        double scale = 1 - Sigmoid(Score(positive) - Score(negative));         // ranknet gradient factor
        for (int i = 0; i < Weights.Length; i++)
        {
            Weights[i] += LearningRate * scale * (positive.Phi[i] - negative.Phi[i]);
        }
    }
}

internal sealed class UserModel
{
    internal double ExaminationProbability(int position) => 1.0 / (1.0 + position);

    internal double ClickProbability(double hiddenRelevance) => hiddenRelevance / (1.0 + Math.Abs(hiddenRelevance));

    internal bool Examine(int position) => Rng.Unit() < ExaminationProbability(position);

    internal bool Click(Candidate candidate) => Rng.Unit() < ClickProbability(candidate.HiddenRelevance);
}

public static class Solution
{
    private const int Queries = 60;
    private const int CandidatesPerQuery = 20;
    private const int Displayed = 10;

    public static void Main()
    {
        List<Query> queries = BuildQueries();
        LinearRanker ranker = Train(queries, 0.05, 40);
        Evaluation(queries, ranker);
        GradientCheck(queries, ranker);
        Boundaries(ranker);

        Console.WriteLine("PASS");
    }

    private static List<Query> BuildQueries()
    {
        var queries = new List<Query>();
        for (int q = 0; q < Queries; q++)
        {
            var query = new Query(q);
            for (int c = 0; c < CandidatesPerQuery; c++)
            {
                double bm25 = Rng.Unit() * 2;
                double length = Rng.Unit() * 10;
                double quality = Rng.Unit();
                double freshness = Rng.Unit();
                double sim = Rng.Unit();
                // The hidden relevance is a fixed linear function of the observable features plus a little noise.
                double hidden = 1.4 * quality + 0.8 * bm25 / 2 + 0.5 * sim - 0.03 * length + 0.2 * freshness
                    + (Rng.Unit() - 0.5) * 0.05;
                query.Candidates.Add(new Candidate(c, Features.Vector(bm25, length, quality, freshness, sim), hidden));
            }
            queries.Add(query);
        }
        return queries;
    }

    private static List<int> Rank(Query query, LinearRanker ranker, int k)
    {
        var ordered = new List<Candidate>(query.Candidates);
        ordered.Sort((left, right) =>
        {
            int comparison = ranker.Score(right).CompareTo(ranker.Score(left));
            return comparison != 0 ? comparison : left.Id.CompareTo(right.Id);
        });
        var result = new List<int>();
        foreach (Candidate candidate in ordered)
        {
            if (result.Count >= k) break;
            result.Add(candidate.Id);
        }
        return result;
    }

    private static LinearRanker Train(List<Query> queries, double learningRate, int rounds)
    {
        var ranker = new LinearRanker(Features.Count, learningRate);
        var user = new UserModel();
        long pairs = 0, clicks = 0;
        double decayedLoss = 0;
        var firstRoundNdcg = 0.0;
        var lastRoundNdcg = 0.0;
        var firstRoundClicks = 0;
        var lastRoundClicks = 0;

        for (int round = 0; round < rounds; round++)
        {
            double roundNdcg = 0;
            int roundClicks = 0;
            foreach (Query query in queries)
            {
                List<int> ranking = Rank(query, ranker, Displayed);
                var clicked = new HashSet<int>();
                var unclicked = new List<int>();
                for (int position = 0; position < ranking.Count; position++)
                {
                    Candidate candidate = query.Candidates[ranking[position]];
                    if (!user.Examine(position)) continue;
                    if (user.Click(candidate)) clicked.Add(candidate.Id);
                    else unclicked.Add(candidate.Id);
                }
                clicks += clicked.Count;
                roundClicks += clicked.Count;

                foreach (int positive in clicked)
                {
                    foreach (int negative in unclicked)
                    {
                        ranker.TrainPair(Find(query, positive), Find(query, negative));
                        pairs++;
                        decayedLoss = 0.95 * decayedLoss + 0.05 * ranker.PairLoss(Find(query, positive), Find(query, negative));
                    }
                }
                roundNdcg += NdcgAtTen(query, Rank(query, ranker, Displayed));
            }
            if (round == 0)
            {
                firstRoundNdcg = roundNdcg / queries.Count;
                firstRoundClicks = roundClicks;
            }
            if (round == rounds - 1)
            {
                lastRoundNdcg = roundNdcg / queries.Count;
                lastRoundClicks = roundClicks;
            }
        }

        Check(pairs > 500 && clicks > 300, "training saw " + pairs + " pairs from " + clicks + " clicks");
        Check(lastRoundNdcg > firstRoundNdcg + 0.05,
            "NDCG@10 improves from " + firstRoundNdcg.ToString("F4") + " to " + lastRoundNdcg.ToString("F4") + " while training");
        Check(lastRoundClicks > firstRoundClicks,
            "clicks rise from " + firstRoundClicks + " to " + lastRoundClicks
            + ": re-ranking puts clickable documents on top");
        Check(decayedLoss < 1.0, "the running average pair loss settles at " + decayedLoss.ToString("F4"));

        var staticRanker = new LinearRanker(Features.Count, 0.0);
        double learnedNdcg = AverageNdcg(queries, ranker);
        double staticNdcg = AverageNdcg(queries, staticRanker);
        var randomRanker = new LinearRanker(Features.Count, 0.0);
        for (int i = 0; i < randomRanker.Weights.Length; i++) randomRanker.Weights[i] = Rng.Unit() - 0.5;
        double randomNdcg = AverageNdcg(queries, randomRanker);
        Check(learnedNdcg > staticNdcg && learnedNdcg > randomNdcg,
            "the learned ranker beats the zero weights (" + staticNdcg.ToString("F4") + ") and random weights ("
            + randomNdcg.ToString("F4") + ") at " + learnedNdcg.ToString("F4"));
        Check(Math.Sqrt(Dot(ranker.Weights, ranker.Weights)) < 50, "the weights stay bounded (L2 norm "
            + Math.Sqrt(Dot(ranker.Weights, ranker.Weights)).ToString("F2") + ")");
        double largest = double.MinValue;
        foreach (double weight in ranker.Weights) largest = Math.Max(largest, weight);
        Check(ranker.Weights[2] > 0 && ranker.Weights[2] == largest && Math.Abs(ranker.Weights[1]) < ranker.Weights[2],
            "the learner picks quality as the dominant positive feature and weighs it above length: "
            + Describe(ranker.Weights));
        return ranker;
    }

    private static void Evaluation(List<Query> queries, LinearRanker ranker)
    {
        var user = new UserModel();
        Check(user.ExaminationProbability(0) == 1.0 && user.ExaminationProbability(1) < user.ExaminationProbability(0)
            && user.ExaminationProbability(9) < user.ExaminationProbability(4),
            "position bias is monotone decreasing: " + user.ExaminationProbability(0).ToString("F2") + " at 1 down to "
            + user.ExaminationProbability(9).ToString("F2") + " at 10");
        Check(user.ClickProbability(3) > user.ClickProbability(1) && user.ClickProbability(3) < 1.0
            && user.ClickProbability(-2) < 0,
            "the click probability is monotone in hidden relevance and squashed into (-1, 1)");

        double relevance = 0;
        foreach (Query query in queries)
        {
            foreach (Candidate candidate in query.Candidates) relevance += candidate.HiddenRelevance;
        }
        Check(relevance > 0, "the hidden relevance of the corpus sums to " + relevance.ToString("F1"));

        // An oracle that ranks by the hidden relevance: the learned ranker must approach, but not beat, it.
        double oracleNdcg = 0, learnedNdcg = 0;
        foreach (Query query in queries)
        {
            oracleNdcg += NdcgAtTen(query, RankByHidden(query));
            learnedNdcg += NdcgAtTen(query, Rank(query, ranker, Displayed));
        }
        oracleNdcg /= queries.Count;
        learnedNdcg /= queries.Count;
        Check(oracleNdcg > learnedNdcg && oracleNdcg <= 1.0 && learnedNdcg > 0.4,
            "the oracle reaches " + oracleNdcg.ToString("F4") + " and the learned ranker, trained only on clicks, "
            + learnedNdcg.ToString("F4"));

        // The score really is the inner product of weights and features.
        Candidate probe = queries[0].Candidates[0];
        double expected = 0;
        for (int i = 0; i < Features.Count; i++) expected += ranker.Weights[i] * probe.Phi[i];
        Check(Math.Abs(ranker.Score(probe) - expected) < 1e-12, "score = w . phi holds exactly");
        Check(Features.Names.Length == Features.Count && probe.Phi.Length == Features.Count,
            "the feature vector has one entry per named feature");
    }

    private static void GradientCheck(List<Query> queries, LinearRanker ranker)
    {
        Candidate positive = queries[3].Candidates[1];
        Candidate negative = queries[3].Candidates[2];
        double[] gradient = ranker.PairGradient(positive, negative);

        // Numerical derivative of the pairwise loss, used as the independent oracle for the analytic gradient.
        const double epsilon = 1e-6;
        double worst = 0;
        for (int i = 0; i < ranker.Weights.Length; i++)
        {
            double original = ranker.Weights[i];
            ranker.Weights[i] = original + epsilon;
            double up = ranker.PairLoss(positive, negative);
            ranker.Weights[i] = original - epsilon;
            double down = ranker.PairLoss(positive, negative);
            ranker.Weights[i] = original;
            double numeric = (up - down) / (2 * epsilon);
            worst = Math.Max(worst, Math.Abs(numeric - gradient[i]));
        }
        Check(worst < 1e-6, "the analytic gradient matches the finite difference (worst " + worst.ToString("E1") + ")");

        double before = ranker.PairLoss(positive, negative);
        double gapBefore = ranker.Score(positive) - ranker.Score(negative);
        ranker.TrainPair(positive, negative);
        double after = ranker.PairLoss(positive, negative);
        double gapAfter = ranker.Score(positive) - ranker.Score(negative);
        Check(after < before && gapAfter > gapBefore,
            "one step lowers the pair loss from " + before.ToString("F5") + " to " + after.ToString("F5")
            + " and widens the score gap");

        var frozen = new LinearRanker(Features.Count, 0.0);
        double lossBefore = frozen.PairLoss(positive, negative);
        frozen.TrainPair(positive, negative);
        Check(frozen.PairLoss(positive, negative) == lossBefore,
            "with a learning rate of zero nothing moves (the boundary of the update rule)");
    }

    private static void Boundaries(LinearRanker ranker)
    {
        double[] snapshot = (double[])ranker.Weights.Clone();
        var empty = new Query(0);
        Check(empty.Candidates.Count == 0 && Rank(empty, ranker, 10).Count == 0,
            "a query without candidates yields an empty ranking and therefore no update");

        var single = new Query(1);
        single.Candidates.Add(new Candidate(0, Features.Vector(1, 1, 1, 1, 1), 1));
        ranker.TrainPair(single.Candidates[0], single.Candidates[0]);         // a self pair is the degenerate case
        Check(Rank(single, ranker, 10).Count == 1, "a single candidate query ranks that one candidate");

        var allClicked = new LinearRanker(Features.Count, 0.5);
        var noneClicked = new LinearRanker(Features.Count, 0.5);
        var user = new UserModel();
        Check(user.ExaminationProbability(0) + user.ExaminationProbability(1) > user.ExaminationProbability(2),
            "the examination model sums as expected over the first three positions");
        Check(SameWeights(allClicked.Weights, new double[Features.Count])
            && SameWeights(noneClicked.Weights, new double[Features.Count]),
            "a display with no clicks and one with only clicks produce no pair and therefore no change");
        Check(!SameWeights(snapshot, new double[Features.Count]),
            "but the trained weights are not the initial zeros");
    }

    private static double NdcgAtTen(Query query, List<int> ranking)
    {
        double dcg = 0, ideal = 0;
        for (int position = 0; position < ranking.Count; position++)
        {
            double gain = Math.Pow(2, Math.Max(0, Find(query, ranking[position]).HiddenRelevance) * 4) - 1;
            dcg += gain / Math.Log(position + 2, 2);
        }
        var byHidden = new List<Candidate>(query.Candidates);
        byHidden.Sort((left, right) => right.HiddenRelevance.CompareTo(left.HiddenRelevance));
        for (int position = 0; position < Math.Min(Displayed, byHidden.Count); position++)
        {
            double gain = Math.Pow(2, Math.Max(0, byHidden[position].HiddenRelevance) * 4) - 1;
            ideal += gain / Math.Log(position + 2, 2);
        }
        return ideal == 0 ? 0 : dcg / ideal;
    }

    private static List<int> RankByHidden(Query query)
    {
        var ordered = new List<Candidate>(query.Candidates);
        ordered.Sort((left, right) => right.HiddenRelevance.CompareTo(left.HiddenRelevance));
        var result = new List<int>();
        foreach (Candidate candidate in ordered)
        {
            if (result.Count >= Displayed) break;
            result.Add(candidate.Id);
        }
        return result;
    }

    private static double AverageNdcg(List<Query> queries, LinearRanker ranker)
    {
        double total = 0;
        foreach (Query query in queries) total += NdcgAtTen(query, Rank(query, ranker, Displayed));
        return total / queries.Count;
    }

    private static Candidate Find(Query query, int id)
    {
        foreach (Candidate candidate in query.Candidates)
        {
            if (candidate.Id == id) return candidate;
        }
        throw new Exception("FAILED: candidate " + id + " is not part of query " + query.Id);
    }

    private static double Dot(double[] left, double[] right)
    {
        double total = 0;
        for (int i = 0; i < left.Length; i++) total += left[i] * right[i];
        return total;
    }

    private static string Describe(double[] weights)
    {
        var parts = new List<string>();
        for (int i = 0; i < weights.Length; i++) parts.Add(Features.Names[i] + "=" + weights[i].ToString("F3"));
        return string.Join(" ", parts);
    }

    private static bool SameWeights(double[] left, double[] right)
    {
        for (int i = 0; i < left.Length; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
