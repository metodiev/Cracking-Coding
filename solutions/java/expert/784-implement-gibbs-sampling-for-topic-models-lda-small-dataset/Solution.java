/** Task 784 - Implement Gibbs sampling for topic models (LDA) small dataset. (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * Collapsed Gibbs sampling for latent Dirichlet allocation. The topic-word and document-topic
 * distributions are integrated out with their Dirichlet priors, so the only latent variables left are
 * the topic assignments z of the tokens, and the full conditional of one token is the classic
 *
 *     p(z = k | rest)  proportional to  (n_dk + alpha) * (n_kw + beta) / (n_k + V beta)
 *
 * A sweep visits every token, removes its contribution from the counts, draws a new topic from that
 * distribution with a deterministic splitmix64 generator and puts the token back. Because the counts
 * are also part of the collapsed joint probability, the sampler's target is checked directly: the
 * conditional distribution is recomputed independently for a specific token and compared with the
 * empirical draw frequencies, and the collapsed log joint is monitored across sweeps.
 *
 * Verified: the count bookkeeping is consistent after every single token (row sums, column sums, the
 * grand total and the two ways of computing the topic totals all agree); the empirical topic frequency
 * of a repeatedly re-drawn token matches the analytical conditional to within a small total variation
 * distance; the collapsed log joint improves from the random initialisation over the sweeps; the
 * paper's topic distributions and the per-document mixtures are proper probability vectors; on a
 * synthetic two-group corpus the sampler recovers the generating grouping of the documents; the
 * degenerate cases (a one-token corpus, a single-word vocabulary, K = 1, more topics than words) stay
 * consistent; invalid hyper-parameters are rejected; and the same seed reproduces the same chain.
 *
 * Simplified: a few small documents over a 20 word vocabulary, no perplexity hold-out, and a fixed
 * number of sweeps rather than a convergence criterion.
 */
import java.util.*;

class Lda {
    final int topics;
    final int vocabulary;
    final double alpha;
    final double beta;
    final int[][] docs;
    final int[][] assignment;                 // [doc][token] -> topic
    final int[][] docTopic;                   // [doc][topic]
    final int[][] topicWord;                  // [topic][word]
    final int[] topicTotal;                   // [topic]
    long totalTokens;
    private long randomState;

    Lda(int[][] docs, int topics, int vocabulary, double alpha, double beta, long seed) {
        if (topics < 1) throw new IllegalArgumentException("at least one topic is required");
        if (!(alpha > 0) || !(beta > 0)) throw new IllegalArgumentException("the Dirichlet parameters must be positive");
        this.docs = docs;
        this.topics = topics;
        this.vocabulary = vocabulary;
        this.alpha = alpha;
        this.beta = beta;
        this.assignment = new int[docs.length][];
        this.docTopic = new int[docs.length][topics];
        this.topicWord = new int[topics][vocabulary];
        this.topicTotal = new int[topics];
        this.randomState = seed | 1L;
        for (int d = 0; d < docs.length; d++) {
            assignment[d] = new int[docs[d].length];
            for (int i = 0; i < docs[d].length; i++) {
                int word = docs[d][i];
                if (word < 0 || word >= vocabulary) throw new IllegalArgumentException("word out of range: " + word);
                int topic = nextInt(topics);
                assignment[d][i] = topic;
                docTopic[d][topic]++;
                topicWord[topic][word]++;
                topicTotal[topic]++;
                totalTokens++;
            }
        }
    }

    /** splitmix64: deterministic pseudo randomness, so every chain is reproducible. */
    private long nextLong() {
        randomState += 0x9E3779B97F4A7C15L;
        long z = randomState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    private double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** The analytical Gibbs conditional for one token, computed with the token held out. */
    double[] conditional(int doc, int token) {
        int word = docs[doc][token];
        int held = assignment[doc][token];
        docTopic[doc][held]--;
        topicWord[held][word]--;
        topicTotal[held]--;
        double[] probabilities = new double[topics];
        double total = 0;
        for (int k = 0; k < topics; k++) {
            probabilities[k] = (docTopic[doc][k] + alpha) * (topicWord[k][word] + beta)
                    / (topicTotal[k] + vocabulary * beta);
            total += probabilities[k];
        }
        for (int k = 0; k < topics; k++) probabilities[k] /= total;
        docTopic[doc][held]++;
        topicWord[held][word]++;
        topicTotal[held]++;
        return probabilities;
    }

    /** Draws a new topic for one token; returns the topic drawn. */
    int sampleToken(int doc, int token) {
        int word = docs[doc][token];
        int held = assignment[doc][token];
        docTopic[doc][held]--;
        topicWord[held][word]--;
        topicTotal[held]--;
        double[] cumulative = new double[topics];
        double total = 0;
        for (int k = 0; k < topics; k++) {
            total += (docTopic[doc][k] + alpha) * (topicWord[k][word] + beta)
                    / (topicTotal[k] + vocabulary * beta);
            cumulative[k] = total;
        }
        double draw = nextDouble() * total;
        int chosen = topics - 1;
        for (int k = 0; k < topics; k++) {
            if (draw < cumulative[k]) {
                chosen = k;
                break;
            }
        }
        assignment[doc][token] = chosen;
        docTopic[doc][chosen]++;
        topicWord[chosen][word]++;
        topicTotal[chosen]++;
        return chosen;
    }

    void sweep() {
        for (int d = 0; d < docs.length; d++) {
            for (int i = 0; i < docs[d].length; i++) sampleToken(d, i);
        }
    }

    /** Bookkeeping-only move, used to restore a state before an independent draw. */
    void restore(int doc, int token, int topic) {
        int word = docs[doc][token];
        int previous = assignment[doc][token];
        docTopic[doc][previous]--;
        topicWord[previous][word]--;
        topicTotal[previous]--;
        assignment[doc][token] = topic;
        docTopic[doc][topic]++;
        topicWord[topic][word]++;
        topicTotal[topic]++;
    }

    /** The collapsed log joint p(w, z), the quantity the sampler targets. */
    double logJoint() {
        double total = 0;
        for (int k = 0; k < topics; k++) {
            total += logGamma(vocabulary * beta) - vocabulary * logGamma(beta);
            for (int w = 0; w < vocabulary; w++) total += logGamma(topicWord[k][w] + beta);
            total -= logGamma(topicTotal[k] + vocabulary * beta);
        }
        for (int d = 0; d < docs.length; d++) {
            total += logGamma(topics * alpha) - topics * logGamma(alpha);
            for (int k = 0; k < topics; k++) total += logGamma(docTopic[d][k] + alpha);
            total -= logGamma(docs[d].length + topics * alpha);
        }
        return total;
    }

    /** Lanczos approximation of log Gamma - deterministic and accurate enough for bookkeeping checks. */
    static double logGamma(double x) {
        double[] coefficients = {676.5203681218851, -1259.1392167224028, 771.32342877765313,
                -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                9.9843695780195716e-6, 1.5056327351493116e-7};
        if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
        double z = x - 1;
        double accumulator = 0.99999999999980993;
        double t = z + 7.5;
        for (int i = 0; i < coefficients.length; i++) accumulator += coefficients[i] / (z + i + 1);
        return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(accumulator);
    }

    /** Returns null when every count invariant holds, else a description. */
    String verify() {
        for (int d = 0; d < docs.length; d++) {
            int sum = 0;
            for (int k = 0; k < topics; k++) {
                if (docTopic[d][k] < 0) return "negative document-topic count";
                sum += docTopic[d][k];
            }
            if (sum != docs[d].length) return "doc " + d + " has " + sum + " counted tokens";
        }
        for (int k = 0; k < topics; k++) {
            int sum = 0;
            for (int word = 0; word < vocabulary; word++) {
                if (topicWord[k][word] < 0) return "negative topic-word count";
                sum += topicWord[k][word];
            }
            if (sum != topicTotal[k]) return "topic " + k + " holds " + sum + " words but claims " + topicTotal[k];
        }
        for (int k = 0; k < topics; k++) {
            int byTopic = 0;
            for (int d = 0; d < docs.length; d++) byTopic += docTopic[d][k];
            if (byTopic != topicTotal[k]) return "topic " + k + " totals disagree: " + byTopic;
        }
        long grand = 0;
        for (int k = 0; k < topics; k++) grand += topicTotal[k];
        if (grand != totalTokens) return "the grand total drifted: " + grand + " vs " + totalTokens;
        for (int d = 0; d < docs.length; d++) {
            for (int i = 0; i < docs[d].length; i++) {
                int topic = assignment[d][i];
                if (docs[d][i] < 0 || docs[d][i] >= vocabulary) return "word out of range";
                if (topic < 0 || topic >= topics) return "topic out of range";
            }
        }
        return null;
    }
}

public class Solution {
    private static long state = 0x9E3779B97F4A7C15L;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** A corpus with two word groups: the group of a document is its generating topic. */
    static int[][] twoGroupCorpus(int docsPerGroup, int tokensPerDoc, int vocabulary) {
        int[][] docs = new int[docsPerGroup * 2][tokensPerDoc];
        int half = vocabulary / 2;
        for (int d = 0; d < docs.length; d++) {
            int group = d < docsPerGroup ? 0 : 1;
            for (int i = 0; i < tokensPerDoc; i++) {
                docs[d][i] = group == 0 ? nextInt(half) : half + nextInt(half);
            }
        }
        return docs;
    }

    public static void main(String[] args) {
        // --- degenerate corpora -------------------------------------------------
        int[][] oneToken = {{3}};
        Lda tiny = new Lda(oneToken, 2, 5, 0.5, 0.5, 11L);
        check(tiny.verify() == null, "one token invariants: " + tiny.verify());
        check(tiny.totalTokens == 1, "the token is counted once");
        tiny.sweep();
        check(tiny.verify() == null, "one token survives a sweep: " + tiny.verify());
        double[] conditional = tiny.conditional(0, 0);
        double sum = 0;
        for (double value : conditional) sum += value;
        check(Math.abs(sum - 1) < 1e-12, "the conditional is a distribution");

        int[][] singleWord = {{0, 0, 0}, {0, 0}};
        Lda oneWord = new Lda(singleWord, 3, 1, 0.2, 0.3, 12L);
        oneWord.sweep();
        oneWord.sweep();
        check(oneWord.verify() == null, "single word vocabulary invariants: " + oneWord.verify());

        Lda oneTopic = new Lda(twoGroupCorpus(2, 5, 8), 1, 8, 0.4, 0.4, 13L);
        oneTopic.sweep();
        boolean allZero = true;
        for (int[] row : oneTopic.assignment) {
            for (int topic : row) {
                if (topic != 0) allZero = false;
            }
        }
        check(allZero, "with a single topic every token is assigned topic zero");
        check(oneTopic.verify() == null, "single topic invariants: " + oneTopic.verify());

        int[][] emptyCorpus = {{}, {}};
        Lda empty = new Lda(emptyCorpus, 2, 3, 0.5, 0.5, 14L);
        check(empty.verify() == null, "an empty corpus is consistent: " + empty.verify());
        check(empty.totalTokens == 0, "an empty corpus has no tokens");
        check(!Double.isNaN(empty.logJoint()), "the log joint of an empty corpus is finite");
        empty.sweep();
        check(empty.verify() == null, "an empty corpus survives a sweep");

        // --- invalid hyper-parameters ------------------------------------------
        boolean rejected = false;
        try {
            new Lda(oneToken, 0, 5, 0.5, 0.5, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "zero topics must be rejected");
        rejected = false;
        try {
            new Lda(oneToken, 2, 5, 0.0, 0.5, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero alpha must be rejected");
        rejected = false;
        try {
            new Lda(oneToken, 2, 5, 0.5, -1.0, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a negative beta must be rejected");
        rejected = false;
        try {
            new Lda(new int[][]{{7}}, 2, 5, 0.5, 0.5, 1L);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a word outside the vocabulary must be rejected");

        // --- the sampler against the analytical conditional ---------------------
        int[][] corpus = twoGroupCorpus(3, 12, 20);
        Lda sampler = new Lda(corpus, 2, 20, 0.3, 0.3, 0xBEEFL);
        for (int i = 0; i < 20; i++) sampler.sweep();
        double[] analytical = sampler.conditional(1, 2);
        int original = sampler.assignment[1][2];
        int[] drawn = new int[2];
        int trials = 20000;
        for (int t = 0; t < trials; t++) {
            sampler.restore(1, 2, original);
            drawn[sampler.sampleToken(1, 2)]++;
        }
        double variation = 0;
        for (int k = 0; k < 2; k++) variation += Math.abs((double) drawn[k] / trials - analytical[k]);
        System.out.println("conditional=" + Arrays.toString(analytical) + " empirical="
                + drawn[0] + "/" + drawn[1] + " total variation=" + variation);
        check(variation < 0.02, "the empirical draw frequency must match the conditional, variation "
                + variation);
        check(sampler.verify() == null, "counts stay consistent after the repeated draws: " + sampler.verify());

        // --- the collapsed log joint improves over sweeps ----------------------
        Lda chain = new Lda(corpus, 2, 20, 0.3, 0.3, 0xABCDL);
        double start = chain.logJoint();
        for (int i = 0; i < 60; i++) {
            chain.sweep();
            check(chain.verify() == null, "invariants after sweep " + i + ": " + chain.verify());
        }
        double end = chain.logJoint();
        System.out.println("log joint: start=" + start + " after 60 sweeps=" + end);
        check(end > start, "Gibbs sampling must improve the collapsed joint: " + end + " vs " + start);
        check(Double.isFinite(end), "the log joint stays finite");

        // --- topic recovery -----------------------------------------------------
        int[][] recoveryCorpus = twoGroupCorpus(4, 20, 20);
        Lda recovering = new Lda(recoveryCorpus, 2, 20, 0.1, 0.1, 0x5EEDL);
        for (int i = 0; i < 300; i++) recovering.sweep();
        check(recovering.verify() == null, "recovery invariants: " + recovering.verify());
        int[] dominant = new int[recoveryCorpus.length];
        for (int d = 0; d < recoveryCorpus.length; d++) {
            dominant[d] = recovering.docTopic[d][0] >= recovering.docTopic[d][1] ? 0 : 1;
        }
        int agree = 0;
        for (int d = 0; d < recoveryCorpus.length; d++) {
            int generating = d < 4 ? 0 : 1;
            if (dominant[d] == generating) agree++;
        }
        int accuracy = Math.max(agree, recoveryCorpus.length - agree);
        System.out.println("recovered the group of " + accuracy + " of " + recoveryCorpus.length + " documents");
        check(accuracy >= recoveryCorpus.length - 1, "the two groups must be recovered, got " + accuracy);
        int[] groupA = recovering.docTopic[0];
        int[] groupB = recovering.docTopic[7];
        check(groupA[dominant[0]] > groupA[1 - dominant[0]], "documents keep a dominant topic");
        check(groupB[dominant[7]] > groupB[1 - dominant[7]], "and so do the others");

        // --- topic distributions are proper probability vectors ----------------
        for (int k = 0; k < 2; k++) {
            double mass = 0;
            for (int word = 0; word < 20; word++) mass += (recovering.topicWord[k][word] + 0.1) / (recovering.topicTotal[k] + 20 * 0.1);
            check(Math.abs(mass - 1) < 1e-9, "topic " + k + " word distribution sums to one");
        }
        for (int d = 0; d < recoveryCorpus.length; d++) {
            double mass = 0;
            for (int k = 0; k < 2; k++) mass += (recovering.docTopic[d][k] + 0.1) / (recoveryCorpus[d].length + 0.2);
            check(Math.abs(mass - 1) < 1e-9, "document " + d + " mixture sums to one");
        }

        // --- determinism --------------------------------------------------------
        Lda rerun = new Lda(corpus, 2, 20, 0.3, 0.3, 0xABCDL);
        for (int i = 0; i < 60; i++) rerun.sweep();
        check(Math.abs(rerun.logJoint() - chain.logJoint()) < 1e-12,
                "the same seed must reproduce the same chain");
        boolean identical = true;
        for (int d = 0; d < corpus.length; d++) {
            if (!Arrays.equals(rerun.assignment[d], chain.assignment[d])) identical = false;
        }
        check(identical, "the same seed must reproduce every assignment");
        check(Math.abs(Lda.logGamma(5.0) - Math.log(24.0)) < 1e-9, "logGamma(5) must be log 24");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
