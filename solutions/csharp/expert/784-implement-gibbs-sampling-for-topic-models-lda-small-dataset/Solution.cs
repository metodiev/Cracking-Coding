// Task 784 - Implement Gibbs sampling for topic models (LDA) small dataset. (Tier: Expert, CTCI Beyond - Machine Learning & AI)
//
// Latent Dirichlet allocation on a small synthetic corpus, inferred with collapsed Gibbs sampling: the
// topic assignments z are the only sampled variables, and the continuous parameters are integrated out,
// so a token is resampled from
//
//   p(z = k | rest)  proportional to  (n_dk + alpha) * (n_kw + beta) / (n_k + V * beta)
//
// where n_dk counts the topic's tokens in the document, n_kw the topic's occurrences of the word and
// n_k the topic's total. The count tables are maintained incrementally - a token is taken out before
// its new topic is drawn and put back afterwards - and the file verifies that bookkeeping against a
// table rebuilt from scratch after every sweep.
//
// Verified: the count tables always agree with a recount (every token in exactly one topic, the per
// document and per topic totals adding up), the sampler is reproducible for a fixed seed, topics are
// recovered on a corpus whose ground-truth topics use disjoint word sets (the token assignment
// accuracy against the truth reaches the expected level, and the top words of each inferred topic are
// exactly the ground-truth words of some topic), the training perplexity falls as the chains run and
// ends well below the uniform baseline, a held-out document is scored against the trained topics
// without touching them, the per-token log likelihood of the recovered topics beats a random
// assignment, and the degenerate cases (one topic, one document, a one-word vocabulary, invalid
// hyperparameters) behave correctly.
//
// Documented simplification: a single chain with a deterministic splitmix64 stream, a fixed number of
// sweeps, symmetric hyperparameters, and perplexity measured with the standard plug-in estimator on
// the training corpus instead of a fully held-out document completion.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;            // deterministic stand-in for Random
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextUnit(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    internal static int NextInt(ref ulong state, int bound) => (int)(Next(ref state) % (ulong)bound);
}

internal sealed class Corpus
{
    internal readonly int[] Words;          // word of every token
    internal readonly int[] Documents;      // document of every token
    internal readonly int[] Truth;          // ground-truth topic of every token
    internal readonly int DocumentCount;
    internal readonly int Vocabulary;

    internal Corpus(int[] words, int[] documents, int[] truth, int documentCount, int vocabulary)
    {
        Words = words;
        Documents = documents;
        Truth = truth;
        DocumentCount = documentCount;
        Vocabulary = vocabulary;
    }

    internal int Tokens => Words.Length;
}

internal sealed class LdaSampler
{
    internal readonly Corpus Data;
    internal readonly int Topics;
    internal readonly double Alpha;
    internal readonly double Beta;

    private readonly int[] _topic;
    private readonly int[,] _docTopic;
    private readonly int[,] _topicWord;
    private readonly int[] _topicTotal;
    private readonly int[] _docLength;
    private ulong _state;

    internal long Sweeps;
    internal double LastPerplexity = double.MaxValue;

    internal LdaSampler(Corpus data, int topics, double alpha, double beta, ulong seed)
    {
        if (topics < 1) throw new ArgumentException("a topic model needs at least one topic");
        if (!(alpha > 0)) throw new ArgumentException("alpha must be positive");
        if (!(beta > 0)) throw new ArgumentException("beta must be positive");
        if (data.Tokens == 0) throw new ArgumentException("the corpus is empty");
        Data = data;
        Topics = topics;
        Alpha = alpha;
        Beta = beta;
        _state = seed;
        _topic = new int[data.Tokens];
        _docTopic = new int[data.DocumentCount, topics];
        _topicWord = new int[topics, data.Vocabulary];
        _topicTotal = new int[topics];
        _docLength = new int[data.DocumentCount];
        for (int t = 0; t < data.Tokens; t++)
        {
            _topic[t] = SplitMix64.NextInt(ref _state, topics);     // deterministic initialisation
            _docTopic[data.Documents[t], _topic[t]]++;
            _topicWord[_topic[t], data.Words[t]]++;
            _topicTotal[_topic[t]]++;
            _docLength[data.Documents[t]]++;
        }
    }

    internal int TopicOf(int token) => _topic[token];
    internal int DocTopic(int document, int topic) => _docTopic[document, topic];
    internal int TopicWord(int topic, int word) => _topicWord[topic, word];
    internal int TopicTotal(int topic) => _topicTotal[topic];

    /// <summary>One sweep of collapsed Gibbs sampling; returns how many tokens changed topic.</summary>
    internal int Sweep()
    {
        int changed = 0;
        for (int token = 0; token < Data.Tokens; token++)
        {
            int document = Data.Documents[token];
            int word = Data.Words[token];
            int oldTopic = _topic[token];

            _docTopic[document, oldTopic]--;
            _topicWord[oldTopic, word]--;
            _topicTotal[oldTopic]--;

            double total = 0;
            var weights = new double[Topics];
            for (int k = 0; k < Topics; k++)
            {
                weights[k] = (_docTopic[document, k] + Alpha) * (_topicWord[k, word] + Beta) / (_topicTotal[k] + Data.Vocabulary * Beta);
                total += weights[k];
            }
            double draw = SplitMix64.NextUnit(ref _state) * total;
            int chosen = Topics - 1;
            double running = 0;
            for (int k = 0; k < Topics; k++)
            {
                running += weights[k];
                if (draw <= running) { chosen = k; break; }
            }

            _topic[token] = chosen;
            _docTopic[document, chosen]++;
            _topicWord[chosen, word]++;
            _topicTotal[chosen]++;
            if (chosen != oldTopic) changed++;
        }
        Sweeps++;
        LastPerplexity = Perplexity();
        return changed;
    }

    /// <summary>Count tables rebuilt from the token assignments, used to audit the incremental ones.</summary>
    internal string BookkeepingViolation()
    {
        var docTopic = new int[Data.DocumentCount, Topics];
        var topicWord = new int[Topics, Data.Vocabulary];
        var topicTotal = new int[Topics];
        var docLength = new int[Data.DocumentCount];
        for (int token = 0; token < Data.Tokens; token++)
        {
            int k = _topic[token];
            if (k < 0 || k >= Topics) return "token " + token + " has topic " + k;
            docTopic[Data.Documents[token], k]++;
            topicWord[k, Data.Words[token]]++;
            topicTotal[k]++;
            docLength[Data.Documents[token]]++;
        }
        for (int d = 0; d < Data.DocumentCount; d++)
        {
            for (int k = 0; k < Topics; k++) if (docTopic[d, k] != _docTopic[d, k]) return "doc-topic table differs at (" + d + "," + k + ")";
            if (docLength[d] != _docLength[d]) return "document " + d + " length bookkeeping differs";
        }
        for (int k = 0; k < Topics; k++)
        {
            for (int w = 0; w < Data.Vocabulary; w++) if (topicWord[k, w] != _topicWord[k, w]) return "topic-word table differs at (" + k + "," + w + ")";
            if (topicTotal[k] != _topicTotal[k]) return "topic " + k + " total differs";
        }
        int sum = 0;
        foreach (int value in _topicTotal) sum += value;
        if (sum != Data.Tokens) return "the topic totals sum to " + sum + " instead of " + Data.Tokens;
        return null;
    }

    internal double[] TopicWordDistribution(int topic, double smoothing)
    {
        var distribution = new double[Data.Vocabulary];
        double total = 0;
        for (int w = 0; w < Data.Vocabulary; w++)
        {
            distribution[w] = _topicWord[topic, w] + smoothing;
            total += distribution[w];
        }
        for (int w = 0; w < Data.Vocabulary; w++) distribution[w] /= total;
        return distribution;
    }

    internal double[] DocumentTopicDistribution(int document)
    {
        var distribution = new double[Topics];
        double total = 0;
        for (int k = 0; k < Topics; k++)
        {
            distribution[k] = _docTopic[document, k] + Alpha;
            total += distribution[k];
        }
        for (int k = 0; k < Topics; k++) distribution[k] /= total;
        return distribution;
    }

    /// <summary>Standard plug-in perplexity of a token list, using the current topic-word distributions.</summary>
    internal double PerplexityOf(IReadOnlyList<int> words, double[] documentTopic)
    {
        double logLikelihood = 0;
        var topicWord = new double[Topics][];
        for (int k = 0; k < Topics; k++) topicWord[k] = TopicWordDistribution(k, Beta);
        foreach (int word in words)
        {
            double probability = 0;
            for (int k = 0; k < Topics; k++) probability += documentTopic[k] * topicWord[k][word];
            logLikelihood += Math.Log(Math.Max(probability, 1e-300));
        }
        return Math.Exp(-logLikelihood / Math.Max(1, words.Count));
    }

    internal double Perplexity()
    {
        double logLikelihood = 0;
        var topicWord = new double[Topics][];
        for (int k = 0; k < Topics; k++) topicWord[k] = TopicWordDistribution(k, Beta);
        for (int token = 0; token < Data.Tokens; token++)
        {
            double[] documentTopic = DocumentTopicDistribution(Data.Documents[token]);
            double probability = 0;
            for (int k = 0; k < Topics; k++) probability += documentTopic[k] * topicWord[k][Data.Words[token]];
            logLikelihood += Math.Log(Math.Max(probability, 1e-300));
        }
        return Math.Exp(-logLikelihood / Data.Tokens);
    }

    /// <summary>Per-token log likelihood of a fixed topic assignment, used for the sanity baseline.</summary>
    internal double LogLikelihoodOf(int[] assignment)
    {
        var docTopic = new int[Data.DocumentCount, Topics];
        var topicWord = new int[Topics, Data.Vocabulary];
        var topicTotal = new int[Topics];
        for (int token = 0; token < Data.Tokens; token++)
        {
            int k = assignment[token];
            docTopic[Data.Documents[token], k]++;
            topicWord[k, Data.Words[token]]++;
            topicTotal[k]++;
        }
        double logLikelihood = 0;
        for (int token = 0; token < Data.Tokens; token++)
        {
            int d = Data.Documents[token], w = Data.Words[token];
            double probability = 0;
            for (int k = 0; k < Topics; k++)
            {
                probability += (docTopic[d, k] + Alpha) / (_docLength[d] + Topics * Alpha)
                    * (topicWord[k, w] + Beta) / (topicTotal[k] + Data.Vocabulary * Beta);
            }
            logLikelihood += Math.Log(Math.Max(probability, 1e-300));
        }
        return logLikelihood;
    }
}

public static class Solution
{
    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    /// <summary>Corpus whose topics own disjoint word sets, so recovery is unambiguous.</summary>
    private static Corpus BuildCorpus(ulong seed, int topics, int documents, int tokensPerDocument, int wordsPerTopic)
    {
        int vocabulary = topics * wordsPerTopic;
        var words = new List<int>();
        var docs = new List<int>();
        var truth = new List<int>();
        ulong state = seed;
        for (int d = 0; d < documents; d++)
        {
            int primary = d % topics;                       // each document leans on one topic
            for (int t = 0; t < tokensPerDocument; t++)
            {
                int topic = SplitMix64.NextUnit(ref state) < 0.8 ? primary : SplitMix64.NextInt(ref state, topics);
                int word = topic * wordsPerTopic + SplitMix64.NextInt(ref state, wordsPerTopic);
                words.Add(word);
                docs.Add(d);
                truth.Add(topic);
            }
        }
        return new Corpus(words.ToArray(), docs.ToArray(), truth.ToArray(), documents, vocabulary);
    }

    /// <summary>Best accuracy of an inferred labelling against the truth over all permutations.</summary>
    private static double Accuracy(int[] inferred, int[] truth, int topics)
    {
        var used = new bool[topics];
        double best = 0;
        void Search(int index, double score)
        {
            if (index == topics)
            {
                if (score > best) best = score;
                return;
            }
            for (int c = 0; c < topics; c++)
            {
                if (used[c]) continue;
                used[c] = true;
                double gained = 0;
                for (int i = 0; i < truth.Length; i++) if (truth[i] == index && inferred[i] == c) gained++;
                Search(index + 1, score + gained);
                used[c] = false;
            }
        }
        Search(0, 0);
        return best / truth.Length;
    }

    /// <summary>Do the top words of each inferred topic equal the ground-truth word set of some topic?</summary>
    private static bool WordSetsRecovered(LdaSampler sampler, int topics, int wordsPerTopic)
    {
        var truthSets = new List<HashSet<int>>();
        for (int k = 0; k < topics; k++)
        {
            var set = new HashSet<int>();
            for (int w = 0; w < wordsPerTopic; w++) set.Add(k * wordsPerTopic + w);
            truthSets.Add(set);
        }
        var matched = new bool[topics];
        for (int k = 0; k < topics; k++)
        {
            var ranked = new List<KeyValuePair<int, int>>();
            for (int w = 0; w < sampler.Data.Vocabulary; w++) ranked.Add(new KeyValuePair<int, int>(sampler.TopicWord(k, w), w));
            ranked.Sort((a, b) => b.Key != a.Key ? b.Key.CompareTo(a.Key) : a.Value.CompareTo(b.Value));
            var top = new HashSet<int>();
            for (int i = 0; i < wordsPerTopic; i++) top.Add(ranked[i].Value);
            for (int candidate = 0; candidate < topics; candidate++)
            {
                if (matched[candidate]) continue;
                if (top.SetEquals(truthSets[candidate])) { matched[candidate] = true; break; }
            }
        }
        foreach (bool value in matched) if (!value) return false;
        return true;
    }

    public static void Main()
    {
        // ---- sampling, bookkeeping and recovery ------------------------------------------------------
        {
            const int Topics = 3, Documents = 8, TokensPerDocument = 12, WordsPerTopic = 4;
            Corpus corpus = BuildCorpus(0x5EED784000000001UL, Topics, Documents, TokensPerDocument, WordsPerTopic);
            var sampler = new LdaSampler(corpus, Topics, 0.1, 0.01, 0x5EED784000000002UL);
            Check(corpus.Tokens == Documents * TokensPerDocument, "the corpus has the expected number of tokens");
            Check(sampler.BookkeepingViolation() == null, "the initial tables are consistent");

            double initialPerplexity = sampler.Perplexity();
            bool bookkeepingOks = true;
            string problem = "ok";
            for (int sweep = 0; sweep < 120; sweep++)
            {
                sampler.Sweep();
                string violation = sampler.BookkeepingViolation();
                if (violation != null) { bookkeepingOks = false; problem = "sweep " + sweep + ": " + violation; break; }
            }
            Check(bookkeepingOks, "the incremental count tables match a full recount after every sweep: " + problem);
            double finalPerplexity = sampler.Perplexity();
            Check(finalPerplexity < initialPerplexity, "training perplexity falls over the sweeps ("
                + initialPerplexity.ToString("F4") + " -> " + finalPerplexity.ToString("F4") + ")");
            Check(finalPerplexity < corpus.Vocabulary / 2.0, "and ends well below the uniform baseline (" + corpus.Vocabulary + ")");

            var inferred = new int[corpus.Tokens];
            for (int t = 0; t < corpus.Tokens; t++) inferred[t] = sampler.TopicOf(t);
            double accuracy = Accuracy(inferred, corpus.Truth, Topics);
            Check(accuracy > 0.85, "the topic assignments match the ground truth (" + accuracy.ToString("F4") + ")");
            Check(WordSetsRecovered(sampler, Topics, WordsPerTopic), "the top words of the inferred topics are the ground-truth word sets");

            var random = new int[corpus.Tokens];
            ulong state = 0x5EED784000000003UL;
            for (int t = 0; t < corpus.Tokens; t++) random[t] = SplitMix64.NextInt(ref state, Topics);
            Check(sampler.LogLikelihoodOf(inferred) > sampler.LogLikelihoodOf(random),
                "the recovered assignment explains the corpus better than a random one ("
                + sampler.LogLikelihoodOf(inferred).ToString("F2") + " > " + sampler.LogLikelihoodOf(random).ToString("F2") + ")");

            var distributions = sampler.TopicWordDistribution(0, 0.01);
            double sum = 0;
            foreach (double value in distributions) sum += value;
            Check(Math.Abs(sum - 1) < 1e-12, "the topic-word distribution is normalised (" + sum.ToString("F12") + ")");
            var documentDistribution = sampler.DocumentTopicDistribution(0);
            sum = 0;
            foreach (double value in documentDistribution) sum += value;
            Check(Math.Abs(sum - 1) < 1e-12, "the document-topic distribution is normalised");
            Console.WriteLine("recovery: perplexity " + initialPerplexity.ToString("F4") + " -> " + finalPerplexity.ToString("F4")
                + " accuracy=" + accuracy.ToString("F4") + " topics=" + Topics + " vocabulary=" + corpus.Vocabulary);
        }

        // ---- the chain settles down and scores a held-out document ------------------------------------
        {
            const int Topics = 2, Documents = 6, TokensPerDocument = 10, WordsPerTopic = 3;
            Corpus corpus = BuildCorpus(0x5EED784000000004UL, Topics, Documents, TokensPerDocument, WordsPerTopic);
            var sampler = new LdaSampler(corpus, Topics, 0.2, 0.02, 0x5EED784000000005UL);
            var changes = new List<int>();
            for (int sweep = 0; sweep < 200; sweep++) changes.Add(sampler.Sweep());
            double earlyAverage = 0, lateAverage = 0;
            for (int i = 0; i < 20; i++) earlyAverage += changes[i];
            for (int i = changes.Count - 20; i < changes.Count; i++) lateAverage += changes[i];
            earlyAverage /= 20;
            lateAverage /= 20;
            Check(lateAverage < earlyAverage / 3, "the chain settles: the swap rate drops from " + earlyAverage.ToString("F1")
                + " to " + lateAverage.ToString("F1") + " tokens per sweep");
            Check(lateAverage <= corpus.Tokens / 3.0, "and the fluctuations stay small (" + lateAverage.ToString("F1") + ")");
            Check(changes[changes.Count - 1] <= corpus.Tokens / 2, "no sweep rewrites the whole corpus any more");
            Check(sampler.BookkeepingViolation() == null, "the settled chain keeps its tables consistent");

            var heldOut = new List<int>();
            for (int i = 0; i < 8; i++) heldOut.Add(i % WordsPerTopic);      // a fresh document about topic zero
            double heldOutPerplexity = sampler.PerplexityOf(heldOut, sampler.DocumentTopicDistribution(0));
            Check(heldOutPerplexity > 1, "the held-out perplexity is a proper perplexity (" + heldOutPerplexity.ToString("F4") + ")");
            Check(heldOutPerplexity < corpus.Vocabulary, "the trained topics beat a uniform distribution on held-out words ("
                + heldOutPerplexity.ToString("F4") + " < " + corpus.Vocabulary + ")");

            int before = sampler.TopicTotal(0) + sampler.TopicTotal(1);
            sampler.PerplexityOf(heldOut, sampler.DocumentTopicDistribution(0));
            Check(sampler.TopicTotal(0) + sampler.TopicTotal(1) == before, "scoring a held-out document does not change the model");
            Console.WriteLine("settling: lateChanges=" + lateAverage.ToString("F1") + " heldOutPerplexity=" + heldOutPerplexity.ToString("F4")
                + " vocabulary=" + corpus.Vocabulary);
        }

        // ---- determinism ------------------------------------------------------------------------------
        {
            Corpus corpus = BuildCorpus(0x5EED784000000006UL, 2, 4, 8, 3);
            var first = new LdaSampler(corpus, 2, 0.1, 0.01, 0x5EED784000000007UL);
            var second = new LdaSampler(corpus, 2, 0.1, 0.01, 0x5EED784000000007UL);
            for (int sweep = 0; sweep < 40; sweep++) { first.Sweep(); second.Sweep(); }
            bool identical = true;
            for (int t = 0; t < corpus.Tokens; t++) if (first.TopicOf(t) != second.TopicOf(t)) identical = false;
            Check(identical, "two chains with the same seed assign every token the same topic");
            Check(Math.Abs(first.LastPerplexity - second.LastPerplexity) < 1e-12, "and reach the same perplexity");
            Console.WriteLine("determinism: perplexity=" + first.LastPerplexity.ToString("F6") + " tokens=" + corpus.Tokens);
        }

        // ---- degenerate cases -------------------------------------------------------------------------
        {
            Corpus corpus = BuildCorpus(0x5EED784000000008UL, 3, 5, 9, 2);
            var single = new LdaSampler(corpus, 1, 0.1, 0.01, 0x5EED784000000009UL);
            for (int sweep = 0; sweep < 20; sweep++) single.Sweep();
            bool allOne = true;
            for (int t = 0; t < corpus.Tokens; t++) if (single.TopicOf(t) != 0) allOne = false;
            Check(allOne, "with one topic every token stays in topic zero");
            Check(single.TopicTotal(0) == corpus.Tokens, "and the topic holds every token");
            Check(single.BookkeepingViolation() == null, "the single-topic model is consistent");

            var tiny = new Corpus(new[] { 0 }, new[] { 0 }, new[] { 0 }, 1, 1);
            var tinySampler = new LdaSampler(tiny, 2, 0.1, 0.01, 0x5EED78400000000AUL);
            tinySampler.Sweep();
            Check(tinySampler.BookkeepingViolation() == null, "a one-token, one-word corpus works");
            Check(tinySampler.TopicTotal(0) + tinySampler.TopicTotal(1) == 1, "and its single token is in exactly one topic");

            bool threwTopics = false;
            try { new LdaSampler(corpus, 0, 0.1, 0.01, 1UL); } catch (ArgumentException) { threwTopics = true; }
            Check(threwTopics, "zero topics is rejected");
            bool threwAlpha = false;
            try { new LdaSampler(corpus, 2, 0, 0.01, 1UL); } catch (ArgumentException) { threwAlpha = true; }
            Check(threwAlpha, "a non-positive alpha is rejected");
            bool threwBeta = false;
            try { new LdaSampler(corpus, 2, 0.1, -1, 1UL); } catch (ArgumentException) { threwBeta = true; }
            Check(threwBeta, "a negative beta is rejected");
            bool threwEmpty = false;
            try { new LdaSampler(new Corpus(new int[0], new int[0], new int[0], 0, 3), 2, 0.1, 0.01, 1UL); } catch (ArgumentException) { threwEmpty = true; }
            Check(threwEmpty, "an empty corpus is rejected");

            Corpus flat = BuildCorpus(0x5EED78400000000BUL, 2, 3, 6, 1);
            var oneWord = new LdaSampler(flat, 2, 0.1, 0.01, 0x5EED78400000000CUL);
            for (int sweep = 0; sweep < 10; sweep++) oneWord.Sweep();
            Check(oneWord.BookkeepingViolation() == null, "a vocabulary that cannot separate topics still keeps the tables consistent");
            Console.WriteLine("degenerate: singleTopicTotal=" + single.TopicTotal(0) + " tinyTokens=" + tinySampler.TopicTotal(0)
                + "/" + tinySampler.TopicTotal(1));
        }

        Console.WriteLine("PASS");
    }
}
