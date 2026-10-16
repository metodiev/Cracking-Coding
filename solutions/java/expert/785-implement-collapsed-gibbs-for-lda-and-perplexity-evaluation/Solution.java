/** Task 785 - Implement collapsed Gibbs for LDA and perplexity evaluation. (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * LDA draws for document d a mixture theta_d ~ Dir(alpha) and, per token, a topic z ~ Cat(theta_d) then a
 * word w ~ Cat(phi_z). Dirichlet-multinomial conjugacy integrates the mixtures out, so the topic
 * indicators are the only latent variables and their conditional posterior is closed form:
 *
 *     p(z_dn = k | z_-, w)  ∝  (n_dk + alpha) * (n_kw + beta) / (n_k + V beta)
 *
 * (n_dk = tokens of doc d in topic k, n_kw = times word w got topic k, n_k = sum_w n_kw). A sweep removes
 * each token's own contribution from those tables, draws from that categorical distribution and adds the
 * token back, so the chain's stationary distribution is exactly p(z | w); phi_kw ∝ n_kw + beta is the
 * posterior mean topic-word distribution.
 *
 * Held-out perplexity uses document completion: held-out tokens are Gibbs sampled against the frozen
 * training topic-word counts and scored with leave-one-out counts, i.e. a genuine predictive likelihood
 * rather than an in-sample one. alpha is deliberately large (5.0): with a peaked document prior the
 * collapsed posterior has a second, far worse mode in which every document stays topic-pure and the
 * minority words contaminate phi, and one short chain never escapes it.
 *
 * Verified: the table conservation identities hold after every sweep; the conditional equals an
 * independently written formula and 40k draws from it match within sampling error; the joint log
 * likelihood improves along the chain and ends far above the random start; both generator topics are
 * recovered (dominant-topic accuracy, permutation-matched L1 distance of phi); held-out perplexity is far
 * below V and clearly below that of a corpus with the same word support but no topical structure; a K = 1
 * model and a single-token held-out document work; and two runs with the same seed agree exactly.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL; return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** Collapsed Gibbs sampler for LDA over integer-coded documents. */
class CollapsedLda {
    final int vocab, topics;
    final int[][] docs;
    final double alpha, beta;
    final int[][] assignment;   // assignment[d][n] = topic of token n of document d
    final int[][] docTopic;     // docTopic[d][k] = n_dk
    final int[] docLength;      // running sum of docTopic[d][*], kept for the invariant check
    final int[][] topicWord;    // topicWord[k][w] = n_kw
    final int[] topicCount;     // topicCount[k] = n_k
    final SplitMix64 rng;

    CollapsedLda(int[][] docs, int vocab, int topics, double alpha, double beta, long seed) {
        this.docs = docs; this.vocab = vocab; this.topics = topics;
        this.alpha = alpha; this.beta = beta; this.rng = new SplitMix64(seed);
        assignment = new int[docs.length][]; docTopic = new int[docs.length][topics];
        docLength = new int[docs.length]; topicWord = new int[topics][vocab]; topicCount = new int[topics];
    }

    private void increment(int d, int w, int k) { docTopic[d][k]++; docLength[d]++; topicWord[k][w]++; topicCount[k]++; }

    private void decrement(int d, int w, int k) { docTopic[d][k]--; docLength[d]--; topicWord[k][w]--; topicCount[k]--; }

    void initializeRandomly() {
        for (int d = 0; d < docs.length; d++) {
            assignment[d] = new int[docs[d].length];
            for (int n = 0; n < docs[d].length; n++) {
                assignment[d][n] = rng.nextInt(topics); increment(d, docs[d][n], assignment[d][n]);
            }
        }
    }

    /** Analytic conditional p(z_dn = . | z_-dn, w); the token is temporarily removed from the tables. */
    double[] conditional(int d, int n) {
        int w = docs[d][n], own = assignment[d][n]; decrement(d, w, own);
        double[] p = new double[topics]; double total = 0;
        for (int k = 0; k < topics; k++) {
            p[k] = (docTopic[d][k] + alpha) * (topicWord[k][w] + beta) / (topicCount[k] + vocab * beta);
            total += p[k];
        }
        for (int k = 0; k < topics; k++) p[k] /= total; increment(d, w, own); return p;
    }

    void resample(int d, int n) {
        double[] p = conditional(d, n); double u = rng.nextDouble(), acc = 0; int chosen = topics - 1;
        for (int k = 0; k < topics; k++) { acc += p[k]; if (u < acc) { chosen = k; break; } }
        int w = docs[d][n]; decrement(d, w, assignment[d][n]); increment(d, w, chosen); assignment[d][n] = chosen;
    }

    void sweep() {
        for (int d = 0; d < docs.length; d++) for (int n = 0; n < docs[d].length; n++) resample(d, n);
    }

    /** log p(w, z) with the Dirichlet-multinomials integrated out: the LDA collapsed likelihood. */
    double logJoint() {
        double logp = 0;
        for (int d = 0; d < docs.length; d++) {
            logp += logGamma(topics * alpha) - topics * logGamma(alpha);
            for (int k = 0; k < topics; k++) logp += logGamma(docTopic[d][k] + alpha);
            logp -= logGamma(docLength[d] + topics * alpha);
        }
        for (int k = 0; k < topics; k++) {
            logp += logGamma(vocab * beta) - vocab * logGamma(beta);
            for (int w = 0; w < vocab; w++) logp += logGamma(topicWord[k][w] + beta);
            logp -= logGamma(topicCount[k] + vocab * beta);
        }
        return logp;
    }

    /**
     * Perplexity of unseen documents: their topics are Gibbs sampled with the training topic-word tables
     * frozen, then every token is scored with leave-one-out document counts (a predictive likelihood).
     */
    double heldOutPerplexity(int[][] heldOut, int burnIn) {
        double logp = 0; int tokens = 0;
        for (int[] doc : heldOut) {
            int len = doc.length; int[] local = new int[topics], z = new int[len];
            for (int n = 0; n < len; n++) { z[n] = rng.nextInt(topics); local[z[n]]++; }
            for (int s = 0; s < burnIn; s++) for (int n = 0; n < len; n++) {
                int w = doc[n]; local[z[n]]--; double[] p = new double[topics]; double total = 0;
                for (int k = 0; k < topics; k++) {
                    p[k] = (local[k] + alpha) * (topicWord[k][w] + beta) / (topicCount[k] + vocab * beta);
                    total += p[k];
                }
                double u = rng.nextDouble() * total; z[n] = topics - 1;
                for (int k = 0; k < topics; k++) { u -= p[k]; if (u <= 0) { z[n] = k; break; } }
                local[z[n]]++;
            }
            for (int n = 0; n < len; n++) {
                local[z[n]]--; double sum = 0;
                for (int k = 0; k < topics; k++)
                    sum += (local[k] + alpha) / (len - 1 + topics * alpha)
                            * (topicWord[k][doc[n]] + beta) / (topicCount[k] + vocab * beta);
                local[z[n]]++; logp += Math.log(sum); tokens++;
            }
        }
        return Math.exp(-logp / tokens);
    }

    /** Returns a human readable violation of the count conservation identities, or null when intact. */
    String invariantViolation() {
        int[] corpus = new int[vocab], wordTotals = new int[vocab];
        for (int[] doc : docs) for (int w : doc) corpus[w]++;
        for (int d = 0; d < docs.length; d++) {
            int[] observed = new int[topics];
            for (int n = 0; n < docs[d].length; n++) observed[assignment[d][n]]++; int sum = 0;
            for (int k = 0; k < topics; k++) {
                if (docTopic[d][k] < 0 || observed[k] != docTopic[d][k]) return "indicator/count mismatch in doc " + d;
                sum += docTopic[d][k];
            }
            if (sum != docLength[d] || docLength[d] != docs[d].length) return "length mismatch in doc " + d;
        }
        for (int k = 0; k < topics; k++) {
            int sum = 0;
            for (int w = 0; w < vocab; w++) { sum += topicWord[k][w]; wordTotals[w] += topicWord[k][w]; }
            if (sum != topicCount[k]) return "n_k mismatch in topic " + k;
        }
        for (int w = 0; w < vocab; w++) if (corpus[w] != wordTotals[w]) return "word " + w + " lost tokens";
        return null;
    }

    /** Lanczos log-Gamma, needed for the Dirichlet-multinomial normalisers in logJoint(). */
    static double logGamma(double x) {
        double[] c = {76.18009172947146, -86.50532032941677, 24.01409824083091,
                -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};
        double y = x, tmp = x + 5.5; tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015; for (int j = 0; j < 6; j++) ser += c[j] / ++y;
        return -tmp + Math.log(2.5066282746310005 * ser / x);
    }
}

public class Solution {
    static final int VOCAB = 8, TOPICS = 2;
    static final double ALPHA = 5.0, BETA = 0.05;

    static int drawFrom(double[] p, SplitMix64 rng) {
        double u = rng.nextDouble(), acc = 0;
        for (int i = 0; i < p.length; i++) { acc += p[i]; if (u < acc) return i; }
        return p.length - 1;
    }

    static int[][] generateCorpus(double[][] truePhi, int[] labels, SplitMix64 rng, int docs, int length) {
        int[][] corpus = new int[docs][];
        for (int d = 0; d < docs; d++) {
            double[] theta = new double[TOPICS];
            for (int k = 0; k < TOPICS; k++) theta[k] = (k == labels[d]) ? 0.85 : 0.15;
            corpus[d] = new int[length];
            for (int n = 0; n < length; n++) corpus[d][n] = drawFrom(truePhi[drawFrom(theta, rng)], rng);
        }
        return corpus;
    }

    static double l1(double[] a, double[] b) {
        double sum = 0; for (int i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]); return sum;
    }

    static double total(double[] a) {
        double sum = 0; for (double v : a) sum += v; return sum;
    }

    public static void main(String[] args) {
        double[][] truePhi = new double[TOPICS][VOCAB]; truePhi[0] = new double[]{0.55, 0.30, 0.15, 0, 0, 0, 0, 0};
        truePhi[1] = new double[]{0, 0, 0, 0, 0, 0.55, 0.30, 0.15}; int docs = 24, length = 30;
        int[] labels = new int[docs]; for (int d = 0; d < docs; d++) labels[d] = d % TOPICS;
        int[][] corpus = generateCorpus(truePhi, labels, new SplitMix64(20240785L), docs, length);

        CollapsedLda lda = new CollapsedLda(corpus, VOCAB, TOPICS, ALPHA, BETA, 987654321L);
        lda.initializeRandomly();
        check(lda.invariantViolation() == null, "invariant holds after the random initialisation");
        double randomLogJoint = lda.logJoint();

        // --- the sampler's conditional against an independently derived formula ---------
        check(Math.abs(total(lda.conditional(0, 0)) - 1.0) < 1e-12, "conditional is a normalised distribution");

        int probeDoc = 3, probeToken = 5, probeWord = corpus[probeDoc][probeToken];
        int own = lda.assignment[probeDoc][probeToken]; double[] analytic = new double[TOPICS];
        for (int k = 0; k < TOPICS; k++) {
            // Formula re-derived from the raw tables, removing the token's own contribution by hand.
            int self = (own == k ? 1 : 0);
            analytic[k] = (lda.docTopic[probeDoc][k] - self + ALPHA)
                    * (lda.topicWord[k][probeWord] - self + BETA) / (lda.topicCount[k] - self + VOCAB * BETA);
        }
        double analyticSum = total(analytic); for (int k = 0; k < TOPICS; k++) analytic[k] /= analyticSum;
        check(l1(analytic, lda.conditional(probeDoc, probeToken)) < 1e-12,
                "conditional matches the hand-derived formula");

        int draws = 40000, hits = 0;
        for (int i = 0; i < draws; i++) {
            lda.resample(probeDoc, probeToken); if (lda.assignment[probeDoc][probeToken] == 0) hits++;
        }
        check(lda.invariantViolation() == null, "invariant holds after 40k single-token resamples");
        double empirical = (double) hits / draws;
        check(Math.abs(empirical - analytic[0]) < 0.015,
                "empirical draw frequency " + empirical + " vs analytic " + analytic[0]);

        // --- the chain concentrates on high probability assignments ---------------------
        double firstHalf = 0, secondHalf = 0, best = randomLogJoint; int half = 60;
        for (int s = 0; s < 2 * half; s++) {
            lda.sweep(); check(lda.invariantViolation() == null, "invariant holds after sweep " + s);
            double logJoint = lda.logJoint(); best = Math.max(best, logJoint);
            if (s < half) firstHalf += logJoint; else secondHalf += logJoint;
        }
        check(secondHalf / half > firstHalf / half + 5.0,
                "mean log joint improves along the chain, " + firstHalf / half + " -> " + secondHalf / half);
        check(best > randomLogJoint + 40.0, "sampled best " + best + " dwarfs the random start " + randomLogJoint);

        // --- topic recovery --------------------------------------------------------------
        int correct = 0;
        for (int d = 0; d < docs; d++)
            if ((lda.docTopic[d][0] >= lda.docTopic[d][1] ? 0 : 1) == labels[d]) correct++;
        double accuracy = (double) correct / docs;
        check(accuracy >= 0.9, "dominant-topic accuracy " + accuracy + " should be at least 0.9");

        double[][] learned = new double[TOPICS][VOCAB];
        for (int k = 0; k < TOPICS; k++)
            for (int w = 0; w < VOCAB; w++)
                learned[k][w] = (lda.topicWord[k][w] + BETA) / (lda.topicCount[k] + VOCAB * BETA);
        double direct = l1(learned[0], truePhi[0]) + l1(learned[1], truePhi[1]);
        double swapped = l1(learned[0], truePhi[1]) + l1(learned[1], truePhi[0]);
        int matched = direct <= swapped ? 0 : 1; double mismatch = Math.min(direct, swapped);
        double blockMass = matched == 0 ? learned[0][0] + learned[0][1] + learned[0][2]
                : learned[1][5] + learned[1][6] + learned[1][7];
        check(mismatch < 0.25, "permutation-matched L1 distance of phi from the truth is " + mismatch);
        check(blockMass > 0.9, "recovered topic " + matched + " keeps " + blockMass + " on its word block");

        // --- perplexity -------------------------------------------------------------------
        int[][] heldOut = generateCorpus(truePhi, labels, new SplitMix64(555L), 8, length);
        double perplexity = lda.heldOutPerplexity(heldOut, 30);
        check(perplexity > 1.0 && perplexity < VOCAB, "held-out perplexity " + perplexity + " beats uniform");

        SplitMix64 noiseRng = new SplitMix64(777L); int[] support = {0, 1, 2, 5, 6, 7};
        int[][] noise = new int[8][length];
        for (int[] doc : noise) for (int n = 0; n < length; n++) doc[n] = support[noiseRng.nextInt(support.length)];
        double noisePerplexity = lda.heldOutPerplexity(noise, 30);
        check(noisePerplexity > perplexity * 1.05,
                "structured held-out perplexity " + perplexity + " beats unstructured " + noisePerplexity);

        // --- determinism, then degenerate corpora -------------------------------------------
        CollapsedLda first = new CollapsedLda(corpus, VOCAB, TOPICS, ALPHA, BETA, 4242L);
        CollapsedLda second = new CollapsedLda(corpus, VOCAB, TOPICS, ALPHA, BETA, 4242L);
        first.initializeRandomly(); second.initializeRandomly(); boolean identical = true;
        for (int s = 0; s < 10; s++) {
            first.sweep(); second.sweep();
            for (int d = 0; d < docs; d++) identical &= Arrays.equals(first.assignment[d], second.assignment[d]);
        }
        check(identical, "same seed reproduces the chain exactly");

        CollapsedLda degenerate = new CollapsedLda(new int[][]{{0, 1, 2}, {5, 6}}, VOCAB, 1, ALPHA, BETA, 9L);
        degenerate.initializeRandomly(); degenerate.sweep();
        check(degenerate.invariantViolation() == null && degenerate.topicCount[0] == 5,
                "K = 1 keeps every token in its only topic");
        double degeneratePerplexity = degenerate.heldOutPerplexity(new int[][]{{4}}, 10);
        check(Double.isFinite(degeneratePerplexity) && degeneratePerplexity > 1.0,
                "single-token held-out perplexity " + degeneratePerplexity);

        System.out.println("topic accuracy = " + accuracy + ", phi L1 mismatch = " + mismatch);
        System.out.println("log joint random = " + randomLogJoint + ", sampled best = " + best);
        System.out.println("held-out perplexity = " + perplexity + ", unstructured = " + noisePerplexity
                + ", vocabulary = " + VOCAB);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
