// Task 785 - Implement collapsed Gibbs for LDA and perplexity evaluation. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
//
// Collapsed (Rao-Blackwellised) Gibbs sampling for LDA: only the token-to-topic assignment z is sampled, while
// theta (document mixtures) and phi (topic-word distributions) are integrated out analytically, so a sweep
// costs one sample per token. With the current token removed from the counters its conditional is
//   p(z = k | .) ∝ (ndk + alpha) * (nkw + beta) / (nk + V * beta).
// Invariants asserted after sweeps: row sums of nkw equal nk, sum(nk) equals the token count, and the topic
// stored for token i is the topic counted in ndk / nkw at that position.
//
// Estimates: phi[k][w] = (nkw + beta) / (nk + V beta), theta[d][k] = (ndk + alpha) / (N_d + K alpha), predictive
// mixture p(w|d) = sum_k theta_dk phi_kw, perplexity exp(-sum c_dw log p(w|d) / N). Held-out documents are
// folded in by the standard fixed-point iteration on theta with phi frozen.
//
// Independent oracles: (a) K = 1 collapses to a smoothed unigram model with a closed-form perplexity from the
// raw word counts; (b) a two-token document admits exact enumeration of the collapsed joint
// p(w,z) = Γ(Kα)/Γ(N+Kα) Π_k Γ(n_dk+α) Π_k [Γ(Vβ)/Γ(n_k+Vβ) Π_w Γ(n_kw+β)], so the chain's long-run state
// histogram is compared with the true posterior over all K^N assignments; (c) the synthetic corpus is built
// from three disjoint word groups, so the document partition read off the samples is scored by exhaustive
// best-permutation accuracy against the generator's ground truth (topic labels are exchangeable -> K! search).
//
// Documented simplification: the corpus is synthetic (V = 24, 3 ground-truth topics, 841 tokens) so convergence
// takes a fraction of a second; the sampler, the estimates and the fold-in perplexity are the full algorithm,
// not a sketch. All randomness comes from a hand-written splitmix64 stream, so the run is reproducible.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    /** Uniform double in [0,1) with 53 bits of resolution; NextInt is uniform on [0,bound). */
    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class CollapsedGibbsLda
{
    private readonly int _topics, _vocab;
    private readonly double _alpha, _beta;
    private readonly int[] _tokens, _docStart, _topic, _nk;
    private readonly int[,] _nkw, _ndk;
    private ulong _rng;

    internal CollapsedGibbsLda(int[][] documents, int vocab, int topics, double alpha, double beta, ulong seed)
    {
        if (documents.Length == 0) throw new ArgumentException("corpus must not be empty", nameof(documents));
        if (topics < 1) throw new ArgumentException("topics must be >= 1", nameof(topics));
        if (vocab < 1) throw new ArgumentException("vocab must be >= 1", nameof(vocab));
        _topics = topics; _vocab = vocab; _alpha = alpha; _beta = beta; _rng = seed;
        _docStart = new int[documents.Length + 1];
        var flat = new List<int>();
        for (int d = 0; d < documents.Length; d++)
        {
            _docStart[d] = flat.Count;
            flat.AddRange(documents[d]);
        }
        _docStart[documents.Length] = flat.Count;
        _tokens = flat.ToArray();
        _topic = new int[_tokens.Length];
        _nkw = new int[topics, vocab];
        _ndk = new int[documents.Length, topics];
        _nk = new int[topics];
        for (int i = 0; i < _tokens.Length; i++)
        {
            int k = SplitMix64.NextInt(ref _rng, _topics);       // random initial assignment
            _topic[i] = k;
            _nkw[k, _tokens[i]]++;
            _ndk[DocOf(i), k]++;
            _nk[k]++;
        }
    }

    internal int TokenCount => _tokens.Length;
    internal int DocumentCount => _docStart.Length - 1;

    private int DocOf(int token)
    {
        int lo = 0, hi = _docStart.Length - 1;
        while (lo + 1 < hi)
        {
            int mid = (lo + hi) / 2;
            if (_docStart[mid] <= token) lo = mid; else hi = mid;
        }
        return lo;
    }

    /** One pass over the token stream, resampling every z from its collapsed conditional. */
    internal void Sweep()
    {
        var p = new double[_topics];
        double betaTerm = _vocab * _beta;
        for (int i = 0; i < _tokens.Length; i++)
        {
            int w = _tokens[i], d = DocOf(i), old = _topic[i];
            _nkw[old, w]--; _ndk[d, old]--; _nk[old]--;                 // remove the token before sampling
            double total = 0.0;
            for (int k = 0; k < _topics; k++)
            {
                p[k] = (_ndk[d, k] + _alpha) * (_nkw[k, w] + _beta) / (_nk[k] + betaTerm);
                total += p[k];
            }
            double u = SplitMix64.NextDouble(ref _rng) * total;
            int pick = _topics - 1;
            for (int k = 0; k < _topics; k++) { u -= p[k]; if (u <= 0.0) { pick = k; break; } }
            _topic[i] = pick;
            _nkw[pick, w]++; _ndk[d, pick]++; _nk[pick]++;
        }
    }

    internal void Fit(int sweeps) { for (int s = 0; s < sweeps; s++) Sweep(); }

    /** phi[k][w] = (nkw + beta) / (nk + V beta); every row sums to one. */
    internal double[][] Phi()
    {
        var phi = new double[_topics][];
        for (int k = 0; k < _topics; k++)
        {
            phi[k] = new double[_vocab];
            double denom = _nk[k] + _vocab * _beta;
            for (int w = 0; w < _vocab; w++) phi[k][w] = (_nkw[k, w] + _beta) / denom;
        }
        return phi;
    }

    internal double[] Theta(int doc)
    {
        var theta = new double[_topics];
        double denom = _docStart[doc + 1] - _docStart[doc] + _topics * _alpha;
        for (int k = 0; k < _topics; k++) theta[k] = (_ndk[doc, k] + _alpha) / denom;
        return theta;
    }

    private int[] DocCounts(int doc)
    {
        var counts = new int[_vocab];
        for (int i = _docStart[doc]; i < _docStart[doc + 1]; i++) counts[_tokens[i]]++;
        return counts;
    }

    internal int DominantTopic(int doc)
    {
        int best = 0;
        for (int k = 1; k < _topics; k++) if (_ndk[doc, k] > _ndk[doc, best]) best = k;
        return best;
    }

    internal int[] DominantTopicPerDoc()
    {
        var reps = new int[DocumentCount];
        for (int d = 0; d < reps.Length; d++) reps[d] = DominantTopic(d);
        return reps;
    }

    /** exp(-sum_d sum_w c_dw log sum_k theta_dk phi_kw / N) with theta and phi taken from the sampler. */
    internal double TrainingPerplexity()
    {
        double[][] phi = Phi();
        double logProb = 0.0;
        int tokens = 0;
        for (int d = 0; d < DocumentCount; d++)
        {
            double[] theta = Theta(d);
            int[] counts = DocCounts(d);
            for (int w = 0; w < _vocab; w++)
            {
                if (counts[w] == 0) continue;
                double pw = 0.0;
                for (int k = 0; k < _topics; k++) pw += theta[k] * phi[k][w];
                logProb += counts[w] * Math.Log(pw);
                tokens += counts[w];
            }
        }
        return Math.Exp(-logProb / tokens);
    }

    /** Fold-in: theta fixed point with phi frozen (30 iterations), then the predictive mixture likelihood. */
    internal double HeldOutLogProb(int[] doc, double[][] phi)
    {
        if (doc.Length == 0) return 0.0;
        var counts = new int[_vocab];
        foreach (int w in doc) counts[w]++;
        var theta = new double[_topics];
        var next = new double[_topics];
        for (int k = 0; k < _topics; k++) theta[k] = 1.0 / _topics;
        for (int it = 0; it < 30; it++)
        {
            for (int k = 0; k < _topics; k++) next[k] = _alpha;
            for (int w = 0; w < _vocab; w++)
            {
                if (counts[w] == 0) continue;
                double denom = 0.0;
                for (int k = 0; k < _topics; k++) denom += phi[k][w] * theta[k];
                for (int k = 0; k < _topics; k++) next[k] += counts[w] * phi[k][w] * theta[k] / denom;
            }
            double sum = 0.0;
            for (int k = 0; k < _topics; k++) sum += next[k];
            for (int k = 0; k < _topics; k++) theta[k] = next[k] / sum;
        }
        double logProb = 0.0;
        foreach (int w in doc)
        {
            double pw = 0.0;
            for (int k = 0; k < _topics; k++) pw += theta[k] * phi[k][w];
            logProb += Math.Log(pw);
        }
        return logProb;
    }

    internal double HeldOutPerplexity(int[][] documents)
    {
        double[][] phi = Phi();
        double logProb = 0.0;
        int tokens = 0;
        foreach (int[] doc in documents)
        {
            logProb += HeldOutLogProb(doc, phi);
            tokens += doc.Length;
        }
        return Math.Exp(-logProb / tokens);
    }

    /** Total mixture mass of document d, sum_w p(w|d); one for any admissible theta and phi. */
    internal double MixtureMass(int doc, double[][] phi)
    {
        double[] theta = Theta(doc);
        double mass = 0.0;
        for (int w = 0; w < _vocab; w++)
        {
            double pw = 0.0;
            for (int k = 0; k < _topics; k++) pw += theta[k] * phi[k][w];
            mass += pw;
        }
        return mass;
    }

    internal void AssertCountersCoherent()
    {
        int total = 0;
        for (int k = 0; k < _topics; k++)
        {
            int row = 0;
            for (int w = 0; w < _vocab; w++) row += _nkw[k, w];
            if (row != _nk[k]) throw new Exception("FAILED: nkw row sum != nk for topic " + k);
            total += _nk[k];
        }
        if (total != _tokens.Length) throw new Exception("FAILED: sum of nk != token count");
        for (int d = 0; d < DocumentCount; d++)
        {
            for (int i = _docStart[d]; i < _docStart[d + 1]; i++)
            {
                if (_ndk[d, _topic[i]] <= 0) throw new Exception("FAILED: token counted in the wrong cell");
            }
        }
    }

    /** Current assignment packed into an integer: sum_i z_i * K^i. */
    internal int StateIndex()
    {
        int code = 0, mul = 1;
        for (int i = 0; i < _topic.Length; i++) { code += _topic[i] * mul; mul *= _topics; }
        return code;
    }

    /** Collapsed joint p(w, z) for one document, theta and phi integrated out analytically. */
    internal static double LogCollapsedJoint(int[] words, int[] assignment, int topics, int vocab, double alpha, double beta)
    {
        var nkw = new int[topics, vocab];
        var nk = new int[topics];
        for (int i = 0; i < words.Length; i++) { nkw[assignment[i], words[i]]++; nk[assignment[i]]++; }
        double logp = LogGamma(topics * alpha) - LogGamma(words.Length + topics * alpha);
        foreach (int n in nk) logp += LogGamma(n + alpha);
        for (int k = 0; k < topics; k++)
        {
            logp += LogGamma(vocab * beta) - LogGamma(nk[k] + vocab * beta);
            for (int w = 0; w < vocab; w++) logp += LogGamma(nkw[k, w] + beta);
        }
        return logp;
    }

    /** Brute-force posterior over all K^N assignments of a tiny document - the exact Gibbs target. */
    internal static double[] ExactStateDistribution(int[] words, int topics, int vocab, double alpha, double beta)
    {
        int states = 1;
        for (int i = 0; i < words.Length; i++) states *= topics;
        var weights = new double[states];
        var assignment = new int[words.Length];
        for (int code = 0; code < states; code++)
        {
            int c = code;
            for (int i = 0; i < words.Length; i++) { assignment[i] = c % topics; c /= topics; }
            weights[code] = LogCollapsedJoint(words, assignment, topics, vocab, alpha, beta);
        }
        double max = weights[0];
        foreach (double v in weights) max = Math.Max(max, v);
        double total = 0.0;
        for (int i = 0; i < states; i++) { weights[i] = Math.Exp(weights[i] - max); total += weights[i]; }
        for (int i = 0; i < states; i++) weights[i] /= total;
        return weights;
    }

    /** Lanczos log-Gamma (g = 7, n = 9), kept only for the exact enumeration oracle. */
    private static double LogGamma(double x)
    {
        double[] c =
        {
            0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
            12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
        };
        if (x < 0.5) return Math.Log(Math.PI / Math.Sin(Math.PI * x)) - LogGamma(1.0 - x);
        double z = x - 1.0, acc = c[0];
        for (int i = 1; i < 9; i++) acc += c[i] / (z + i);
        double t = z + 7.5;
        return 0.5 * Math.Log(2.0 * Math.PI) + (z + 0.5) * Math.Log(t) - t + Math.Log(acc);
    }
}

public static class Solution
{
    private const int Vocab = 24, TrueTopics = 3, GroupSize = 8;

    private static int[][] Synthetic(ulong seed, int docs, int minLen, int maxLen, double dominantWeight,
        bool uniformWords, out int[] truth)
    {
        var rng = seed;
        var corpus = new int[docs][];
        truth = new int[docs];
        for (int d = 0; d < docs; d++)
        {
            int t = d % TrueTopics;
            truth[d] = t;
            int len = minLen + SplitMix64.NextInt(ref rng, maxLen - minLen + 1);
            var doc = new int[len];
            for (int i = 0; i < len; i++)
            {
                doc[i] = uniformWords || SplitMix64.NextDouble(ref rng) >= dominantWeight
                    ? SplitMix64.NextInt(ref rng, Vocab)
                    : t * GroupSize + SplitMix64.NextInt(ref rng, GroupSize);
            }
            corpus[d] = doc;
        }
        return corpus;
    }

    /** Exhaustive best agreement over all K! label permutations - the clustering oracle. */
    private static double BestPermutationAccuracy(int[] truth, int[] found, int topics)
    {
        var perm = new int[topics];
        for (int i = 0; i < topics; i++) perm[i] = i;
        int best = 0;
        void Walk(int at)
        {
            if (at == perm.Length)
            {
                int agree = 0;
                for (int i = 0; i < truth.Length; i++) if (perm[found[i]] == truth[i]) agree++;
                best = Math.Max(best, agree);
                return;
            }
            for (int i = at; i < perm.Length; i++)
            {
                (perm[at], perm[i]) = (perm[i], perm[at]);
                Walk(at + 1);
                (perm[at], perm[i]) = (perm[i], perm[at]);
            }
        }
        Walk(0);
        return best / (double)truth.Length;
    }

    private static int[] WordCounts(int[][] documents)
    {
        var counts = new int[Vocab];
        foreach (int[] d in documents) foreach (int w in d) counts[w]++;
        return counts;
    }

    public static void Main()
    {
        const double alpha = 0.1, beta = 0.01;
        int[][] train = Synthetic(0x5EED01UL, 30, 20, 40, 0.85, false, out int[] trainTruth);
        int[][] heldIn = Synthetic(0x5EED02UL, 10, 20, 40, 0.85, false, out _);
        int[][] heldOut = Synthetic(0x5EED03UL, 10, 20, 40, 0.85, true, out _);

        var model = new CollapsedGibbsLda(train, Vocab, TrueTopics, alpha, beta, 0xABCDEF123UL);
        model.Fit(200);
        double trainPpl = model.TrainingPerplexity();
        model.AssertCountersCoherent();
        double pplIn = model.HeldOutPerplexity(heldIn), pplOut = model.HeldOutPerplexity(heldOut);
        double accuracy = BestPermutationAccuracy(trainTruth, model.DominantTopicPerDoc(), TrueTopics);
        Console.WriteLine("  tokens=" + model.TokenCount + " K=" + TrueTopics + " V=" + Vocab
            + " train ppl=" + trainPpl.ToString("F3") + " held-in=" + pplIn.ToString("F3")
            + " held-out(OOD)=" + pplOut.ToString("F3") + " partition accuracy=" + accuracy.ToString("F3"));

        // 1-3. Closed-form oracle at K = 1: the sampler degenerates to a smoothed unigram model.
        var unigram = new CollapsedGibbsLda(train, Vocab, 1, alpha, beta, 0x1357UL);
        unigram.Fit(50);
        int[] counts = WordCounts(train);
        double closedForm = Math.Exp(-LogUnigram(train, counts, beta) / model.TokenCount);
        Check(Math.Abs(unigram.TrainingPerplexity() - closedForm) < 1e-12, "K=1 perplexity equals the closed form");
        double[] phi1 = unigram.Phi()[0];
        double rowSum = 0.0;
        foreach (double p in phi1) rowSum += p;
        Check(Math.Abs(rowSum - 1.0) < 1e-12, "phi row sums to one");
        Check(Math.Abs(phi1[0] - (counts[0] + beta) / (model.TokenCount + Vocab * beta)) < 1e-12,
            "K=1 phi is the smoothed empirical unigram");

        // 4-5. Invariants survive further sweeps; mixture densities stay normalised.
        model.Sweep();
        model.AssertCountersCoherent();
        var phi = model.Phi();
        bool rowsNormalised = true;
        for (int k = 0; k < TrueTopics; k++)
        {
            double s = 0.0;
            foreach (double p in phi[k]) s += p;
            if (Math.Abs(s - 1.0) > 1e-12) rowsNormalised = false;
        }
        Check(rowsNormalised && Math.Abs(model.MixtureMass(0, phi) - 1.0) < 1e-12,
            "phi rows and per-document mixture mass are normalised");

        // 6-7. The generator's partition is recovered and in-domain text stays predictable.
        Check(accuracy >= 0.95, "ground-truth document partition recovered (accuracy " + accuracy.ToString("F3") + ")");
        Check(pplIn < pplOut && pplIn < trainPpl * 1.6, "in-domain held-out perplexity beats out-of-domain");

        // 8. Single-token document: the fold-in posterior is a convex combination, so the predictive is bracketed
        //    by min_k phi_kw and max_k phi_kw, and the exact Dirichlet-prior predictive is mean_k phi_kw.
        var single = new int[] { counts.Length - 1 };
        double lower = double.MaxValue, upper = 0.0, priorPredictive = 0.0;
        for (int k = 0; k < TrueTopics; k++)
        {
            lower = Math.Min(lower, phi[k][single[0]]);
            upper = Math.Max(upper, phi[k][single[0]]);
            priorPredictive += phi[k][single[0]] / TrueTopics;
        }
        double pSingle = Math.Exp(model.HeldOutLogProb(single, phi));
        double ratioSingle = pSingle / priorPredictive;
        Check(pSingle > lower && pSingle < upper, "single-token predictive inside the phi range");
        Check(ratioSingle > 0.25 && ratioSingle < 4.0,
            "fold-in within factor 4 of the exact prior predictive (ratio " + ratioSingle.ToString("F3") + ")");

        // 9. Exact-enumeration oracle: a two-token, two-topic document has 2^2 assignments whose collapsed
        //    posterior is written down in closed form, and the chain's state histogram must match it.
        var tinyDoc = new int[] { 0, 1 };
        double[] exact = CollapsedGibbsLda.ExactStateDistribution(tinyDoc, 2, 2, 0.3, 0.2);
        double exactSum = 0.0;
        foreach (double p in exact) exactSum += p;
        Check(Math.Abs(exactSum - 1.0) < 1e-12, "exact posterior over the 2^2 assignments is normalised");
        var chain = new CollapsedGibbsLda(new[] { tinyDoc }, 2, 2, 0.3, 0.2, 0x2468UL);
        var histogram = new int[exact.Length];
        int samples = 0;
        for (int s = 0; s < 60000; s++)
        {
            chain.Sweep();
            if (s >= 500) { histogram[chain.StateIndex()]++; samples++; }
        }
        double maxDeviation = 0.0;
        for (int i = 0; i < exact.Length; i++)
        {
            maxDeviation = Math.Max(maxDeviation, Math.Abs(histogram[i] / (double)samples - exact[i]));
        }
        Console.WriteLine("  Gibbs chain vs exact posterior: max deviation=" + maxDeviation.ToString("F4")
            + " over " + samples + " sweeps");
        Check(maxDeviation < 0.01, "sampled state histogram matches the exactly enumerated posterior");

        // 10. Sampling improves the predictive likelihood over the random initialisation.
        var fresh = new CollapsedGibbsLda(train, Vocab, TrueTopics, alpha, beta, 0xABCDEF123UL);
        double before = Math.Log(fresh.TrainingPerplexity());
        fresh.Fit(200);
        double after = Math.Log(fresh.TrainingPerplexity());
        Check(after < before, "200 sweeps lower log-perplexity " + before.ToString("F4") + " -> " + after.ToString("F4"));

        // 11-13. Boundary cases: single document, deterministic one-word vocabulary, degenerate topics.
        var tiny = new CollapsedGibbsLda(new[] { new[] { 0, 0, 1, 2, 3 } }, 4, 2, alpha, beta, 0x99UL);
        tiny.Fit(10);
        tiny.AssertCountersCoherent();
        Check(tiny.TrainingPerplexity() < 5.0, "single-document corpus yields a finite perplexity");
        var oneVocab = new CollapsedGibbsLda(new[] { new[] { 0, 0, 0 } }, 1, 3, alpha, beta, 0x77UL);
        oneVocab.Fit(5);
        Check(Math.Abs(oneVocab.TrainingPerplexity() - 1.0) < 1e-12, "V=1 deterministic word gives perplexity 1");

        // 14. Determinism: identical seed, identical run.
        var again = new CollapsedGibbsLda(train, Vocab, TrueTopics, alpha, beta, 0xABCDEF123UL);
        again.Fit(200);
        var repsA = model.DominantTopicPerDoc();
        var repsB = again.DominantTopicPerDoc();
        bool sameTopics = true;
        for (int d = 0; d < repsA.Length; d++) if (repsA[d] != repsB[d]) sameTopics = false;
        Check(Math.Abs(again.TrainingPerplexity() - trainPpl) < 1e-12 && sameTopics,
            "same seed reproduces the perplexity and the document-topic assignment");

        // 15-16. Rejected input.
        Check(Throws(() => new CollapsedGibbsLda(train, Vocab, 0, alpha, beta, 1UL)), "K = 0 rejected");
        Check(Throws(() => new CollapsedGibbsLda(Array.Empty<int[]>(), Vocab, 2, alpha, beta, 1UL)),
            "empty corpus rejected");

        Console.WriteLine("PASS");
    }

    private static double LogUnigram(int[][] documents, int[] counts, double beta)
    {
        int n = 0;
        foreach (int[] d in documents) n += d.Length;
        double denom = n + Vocab * beta, logProb = 0.0;
        for (int w = 0; w < counts.Length; w++)
        {
            if (counts[w] > 0) logProb += counts[w] * Math.Log((counts[w] + beta) / denom);
        }
        return logProb;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; } catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
