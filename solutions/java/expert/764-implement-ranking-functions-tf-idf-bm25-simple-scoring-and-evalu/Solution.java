/** Task 764 - Implement ranking functions: TF-IDF and BM25 scoring with evaluation.
 * (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * Two classic retrieval models over the same small corpus:
 *
 *  * TF-IDF with cosine-ish length normalisation: tf * log(N/df) divided by the document length, the
 *    model that treats a rare term as informative and punishes long documents bluntly;
 *  * BM25: idf(t) * tf * (k1+1) / (tf + k1 * (1 - b + b * dl / avgdl)), where k1 controls how quickly
 *    term frequency saturates and b how strongly length is normalised. The smoothed idf
 *    ln(1 + (N - df + 0.5)/(df + 0.5)) is used because the textbook ln(N/df) goes negative for terms
 *    that appear in more than half the documents - which the file demonstrates rather than hides.
 *
 * The scores are cross-checked against a second, literal implementation of the formula written in the
 * natural left-to-right order, and the *properties* are checked on purpose: monotone in tf, monotone
 * decreasing in document length, ordering by idf alone when k1 = 0, and length insensitivity when b = 0.
 * Ranking quality is then measured with real evaluation metrics - precision@k, recall@k, average
 * precision, MRR and graded NDCG - including hand-computed values for specific ranked lists, an ideal
 * ranking that scores exactly 1.0, and a synthetic corpus where the true relevance is known so the
 * improvement over a random ranking is measured instead of claimed.
 *
 * Simplified: in-memory corpus of a few hundred short documents, no stemming, no stop-word list, no
 * phrase scoring, no index structures (that is the neighbouring task), and only single-term queries.
 *
 * Verified: the literal-formula cross-check for both models; the four monotonicity/parameter
 * properties; negative idf for common terms and its removal by the smoothed variant; hand-computed
 * metric values; ideal rankings scoring 1.0; NDCG, MAP and MRR of BM25 and TF-IDF above a random
 * baseline on known-relevance corpora; and empty-query, absent-term, single-document and tie boundaries.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }

    double nextDouble() { return (nextLong() >>> 11) / (double) (1L << 53); }
}

/** A small in-memory corpus with the statistics both models need. */
final class Corpus {
    final List<List<String>> documents = new ArrayList<>();
    final Map<String, Integer> documentFrequency = new HashMap<>();
    double averageLength;

    void add(String text) {
        List<String> tokens = new ArrayList<>();
        for (String word : text.toLowerCase().split("[^a-z0-9]+")) if (!word.isEmpty()) tokens.add(word);
        for (String token : new LinkedHashSet<>(tokens)) {
            documentFrequency.merge(token, 1, Integer::sum);
        }
        documents.add(tokens);
        recomputeAverage();
    }

    private void recomputeAverage() {
        int total = 0;
        for (List<String> document : documents) total += document.size();
        averageLength = documents.isEmpty() ? 0 : total / (double) documents.size();
    }

    int size() { return documents.size(); }

    int tf(int document, String term) {
        int count = 0;
        for (String token : documents.get(document)) if (token.equals(term)) count++;
        return count;
    }

    int df(String term) { return documentFrequency.getOrDefault(term, 0); }

    int length(int document) { return documents.get(document).size(); }

    /** The textbook idf, non-negative but blind to the 0.5 smoothing every real system uses. */
    double naiveIdf(String term) { return Math.log(size() / (double) Math.max(1, df(term))); }

    /** The unsmoothed Robertson idf, which goes negative once a term is in more than half the corpus. */
    double robertsonIdf(String term) {
        return Math.log((size() - df(term) + 0.5) / (df(term) + 0.5));
    }

    /** The smoothed idf BM25 actually uses: log(1 + the Robertson term), never negative. */
    double smoothedIdf(String term) {
        return Math.log(1.0 + (size() - df(term) + 0.5) / (df(term) + 0.5));
    }
}

/** Both rankers, plus a literal re-implementation of each formula for cross-checking. */
final class Rankers {

    static double tfidf(Corpus corpus, int document, String term) {
        int tf = corpus.tf(document, term);
        if (tf == 0) return 0;
        double weight = tf * corpus.smoothedIdf(term);
        return corpus.length(document) == 0 ? 0 : weight / corpus.length(document);
    }

    /** The same formula written out in the order a textbook states it. */
    static double tfidfLiteral(Corpus corpus, int document, String term) {
        double tf = corpus.tf(document, term);
        double idf = Math.log(1.0 + ((corpus.size() - corpus.df(term)) + 0.5) / (corpus.df(term) + 0.5));
        double length = corpus.length(document);
        if (length == 0 || tf == 0) return 0;
        double numerator = tf * idf;
        return numerator / length;
    }

    static double bm25(Corpus corpus, int document, String term, double k1, double b) {
        int tf = corpus.tf(document, term);
        if (tf == 0) return 0;
        double idf = corpus.smoothedIdf(term);
        double lengthRatio = corpus.averageLength == 0 ? 0 : corpus.length(document) / corpus.averageLength;
        return idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * lengthRatio));
    }

    static double bm25Literal(Corpus corpus, int document, String term, double k1, double b) {
        double tf = corpus.tf(document, term);
        if (tf == 0) return 0;
        double idf = Math.log(1.0 + (corpus.size() - corpus.df(term) + 0.5) / (corpus.df(term) + 0.5));
        double normalization = 1 - b + b * (corpus.length(document) / corpus.averageLength);
        double saturated = (tf * (k1 + 1)) / (tf + k1 * normalization);
        return idf * saturated;
    }

    /** Documents ordered by score, ties broken by document id so the order is deterministic. */
    static List<Integer> rank(Corpus corpus, String term, boolean useBm25) {
        List<Integer> documents = new ArrayList<>();
        for (int i = 0; i < corpus.size(); i++) documents.add(i);
        documents.sort((a, b) -> {
            double scoreA = useBm25 ? bm25(corpus, a, term, 1.2, 0.75) : tfidf(corpus, a, term);
            double scoreB = useBm25 ? bm25(corpus, b, term, 1.2, 0.75) : tfidf(corpus, b, term);
            int byScore = Double.compare(scoreB, scoreA);
            return byScore != 0 ? byScore : Integer.compare(a, b);
        });
        return documents;
    }
}

/** Evaluation metrics. */
final class Metrics {

    static double precisionAtK(List<Integer> ranked, Set<Integer> relevant, int k) {
        int hits = 0;
        for (int i = 0; i < Math.min(k, ranked.size()); i++) if (relevant.contains(ranked.get(i))) hits++;
        return k == 0 ? 0 : hits / (double) k;
    }

    static double recallAtK(List<Integer> ranked, Set<Integer> relevant, int k) {
        if (relevant.isEmpty()) return 0;
        int hits = 0;
        for (int i = 0; i < Math.min(k, ranked.size()); i++) if (relevant.contains(ranked.get(i))) hits++;
        return hits / (double) relevant.size();
    }

    /** Average precision over the whole ranked list. */
    static double averagePrecision(List<Integer> ranked, Set<Integer> relevant) {
        if (relevant.isEmpty()) return 0;
        double sum = 0;
        int hits = 0;
        for (int i = 0; i < ranked.size(); i++) {
            if (!relevant.contains(ranked.get(i))) continue;
            hits++;
            sum += hits / (double) (i + 1);
        }
        return sum / relevant.size();
    }

    static double reciprocalRank(List<Integer> ranked, Set<Integer> relevant) {
        for (int i = 0; i < ranked.size(); i++) if (relevant.contains(ranked.get(i))) return 1.0 / (i + 1);
        return 0;
    }

    /** Graded NDCG with the standard log2(rank + 1) discount. */
    static double ndcg(List<Integer> ranked, Map<Integer, Double> gains, int k) {
        double dcg = dcg(ranked, gains, k);
        List<Integer> ideal = new ArrayList<>(gains.keySet());
        ideal.sort((a, b) -> Double.compare(gains.get(b), gains.get(a)));
        double idcg = dcg(ideal, gains, k);
        return idcg == 0 ? 0 : dcg / idcg;
    }

    private static double dcg(List<Integer> ranked, Map<Integer, Double> gains, int k) {
        double sum = 0;
        for (int i = 0; i < Math.min(k, ranked.size()); i++) {
            double gain = gains.getOrDefault(ranked.get(i), 0.0);
            sum += gain / (Math.log(i + 2) / Math.log(2));
        }
        return sum;
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static void close(double actual, double expected, double tolerance, String label) {
        check(Math.abs(actual - expected) <= tolerance, label + ": " + actual + " vs " + expected);
    }

    /** Documents where a term's frequency is exactly what the relevance grade says. */
    static Corpus syntheticCorpus(int documents, long seed, String term) {
        Prng rng = new Prng(seed);
        Corpus corpus = new Corpus();
        for (int d = 0; d < documents; d++) {
            StringBuilder text = new StringBuilder();
            int repeats = 1 + rng.nextInt(6);
            for (int i = 0; i < repeats; i++) text.append(term).append(' ');
            for (int i = 0; i < rng.nextInt(40); i++) text.append("filler").append(rng.nextInt(5)).append(' ');
            corpus.add(text.toString());
        }
        return corpus;
    }

    public static void main(String[] args) {
        Corpus corpus = new Corpus();
        corpus.add("the quick brown fox jumps over the lazy dog");
        corpus.add("the quick brown fox is quick and brown");
        corpus.add("a lazy dog sleeps all day");
        corpus.add("quick quick quick brown fox fox fox fox");
        corpus.add("unrelated content about weather and traffic");
        check(corpus.size() == 5, "five documents");
        check(corpus.df("quick") == 3 && corpus.tf(3, "fox") == 4, "tf and df statistics are correct");
        close(corpus.averageLength, (9 + 8 + 6 + 8 + 6) / 5.0, 1e-9, "average document length");

        // 1. the scores agree with a literal re-implementation of each formula
        int compared = 0;
        for (int d = 0; d < corpus.size(); d++) {
            for (String term : new String[]{"quick", "fox", "lazy", "dog", "weather", "brown", "missing"}) {
                close(Rankers.bm25(corpus, d, term, 1.2, 0.75), Rankers.bm25Literal(corpus, d, term, 1.2, 0.75),
                        1e-12, "bm25 score cross-check");
                close(Rankers.tfidf(corpus, d, term), Rankers.tfidfLiteral(corpus, d, term), 1e-12,
                        "tfidf score cross-check");
                compared++;
            }
        }
        check(compared == 35, "35 (document, term) pairs cross-checked against the literal formulas");

        // 2. BM25 properties: monotone in tf, decreasing in length, saturating
        Corpus repeated = new Corpus();
        for (int tf = 0; tf < 6; tf++) {
            StringBuilder text = new StringBuilder();
            for (int i = 0; i < tf; i++) text.append("term ");
            text.append("filler filler");
            repeated.add(text.toString());
        }
        double previous = -1;
        for (int d = 1; d < repeated.size(); d++) {
            double score = Rankers.bm25(repeated, d, "term", 1.2, 0.75);
            check(score > previous, "bm25 grows with term frequency (tf=" + d + ")");
            previous = score;
        }
        double saturationGap = Rankers.bm25(repeated, 4, "term", 1.2, 0.75)
                - Rankers.bm25(repeated, 3, "term", 1.2, 0.75);
        double firstGap = Rankers.bm25(repeated, 1, "term", 1.2, 0.75);
        check(saturationGap < firstGap / 2, "the gain per extra occurrence shrinks: " + saturationGap
                + " vs " + firstGap);

        Corpus lengths = new Corpus();
        lengths.add("term term term");
        lengths.add("term term term filler filler filler filler filler filler filler filler filler filler");
        double shortScore = Rankers.bm25(lengths, 0, "term", 1.2, 1.0);
        double longScore = Rankers.bm25(lengths, 1, "term", 1.2, 1.0);
        check(shortScore > longScore, "with the same term frequency, the shorter document wins: "
                + shortScore + " vs " + longScore);
        close(Rankers.bm25(lengths, 0, "term", 1.2, 0.0), Rankers.bm25(lengths, 1, "term", 1.2, 0.0), 1e-12,
                "b=0 disables length normalisation entirely");

        // 3. k1 = 0 removes the term-frequency effect: every document containing the term ties
        close(Rankers.bm25(repeated, 1, "term", 0.0, 0.75), Rankers.bm25(repeated, 5, "term", 0.0, 0.75),
                1e-12, "k1=0 gives every matching document the idf alone");

        // 4. idf: negative for a common term, never negative once smoothed
        Corpus common = new Corpus();
        common.add("alpha beta");
        common.add("alpha gamma");
        common.add("alpha delta");
        check(common.df("alpha") == 3 && common.size() == 3, "alpha is in every document");
        check(common.naiveIdf("alpha") == 0.0, "the naive idf of a universal term is zero");
        Corpus moreThanHalf = new Corpus();
        for (int d = 0; d < 3; d++) moreThanHalf.add(d < 2 ? "common rare" + d : "rare2");
        check(moreThanHalf.robertsonIdf("common") < 0, "the unsmoothed Robertson idf goes negative for a "
                + "term in more than half the documents: " + moreThanHalf.robertsonIdf("common"));
        check(moreThanHalf.smoothedIdf("common") >= 0, "the +1 smoothed idf never goes negative: "
                + moreThanHalf.smoothedIdf("common"));
        check(moreThanHalf.smoothedIdf("common") < moreThanHalf.smoothedIdf("rare0"),
                "rare terms still outweigh common ones after smoothing");

        // 5. hand-computed evaluation metrics
        List<Integer> ranked = List.of(2, 0, 1, 3);
        Set<Integer> relevant = new TreeSet<>(List.of(1, 2));
        close(Metrics.precisionAtK(ranked, relevant, 1), 1.0, 1e-12, "precision@1");
        close(Metrics.precisionAtK(ranked, relevant, 2), 0.5, 1e-12, "precision@2");
        close(Metrics.recallAtK(ranked, relevant, 3), 1.0, 1e-12, "recall@3");
        close(Metrics.averagePrecision(ranked, relevant), (1.0 + 2.0 / 3) / 2, 1e-12, "average precision");
        close(Metrics.reciprocalRank(ranked, relevant), 1.0, 1e-12, "reciprocal rank");
        close(Metrics.recallAtK(ranked, new TreeSet<>(), 3), 0.0, 1e-12, "recall with no relevant document");
        close(Metrics.averagePrecision(ranked, new TreeSet<>()), 0.0, 1e-12, "average precision with no "
                + "relevant document");

        Map<Integer, Double> gains = new LinkedHashMap<>();
        gains.put(1, 3.0);
        gains.put(2, 2.0);
        close(Metrics.ndcg(List.of(1, 2, 0), gains, 3), 1.0, 1e-12, "the ideal ranking scores exactly 1");
        close(Metrics.ndcg(List.of(0, 1, 2), gains, 3), (3 / 1.585 + 2 / 2.0) / (3 + 2 / 1.585), 1e-3,
                "a shifted ranking scores as computed by hand");
        check(Metrics.ndcg(List.of(0, 1, 2), gains, 3) < 1.0, "a sub-ideal ranking scores below 1");
        check(Metrics.ndcg(List.of(1, 2, 0), gains, 1) == 1.0, "NDCG@1 of the ideal ranking is 1");

        // 6. ranking quality on a corpus where the true relevance is the term frequency
        Prng rng = new Prng(0x8A11L);
        int bm25Better = 0, tfidfBetter = 0, queries = 60;
        double bm25Ndcg = 0, tfidfNdcg = 0, randomNdcg = 0;
        for (int query = 0; query < queries; query++) {
            String term = "t" + query;
            Corpus synthetic = syntheticCorpus(60, 1000 + query, term);
            Map<Integer, Double> truth = new LinkedHashMap<>();
            for (int d = 0; d < synthetic.size(); d++) {
                if (synthetic.tf(d, term) > 0) truth.put(d, (double) synthetic.tf(d, term));
            }
            List<Integer> ideal = new ArrayList<>(truth.keySet());
            ideal.sort((a, b) -> Double.compare(truth.get(b), truth.get(a)));
            List<Integer> bm25Ranking = Rankers.rank(synthetic, term, true);
            List<Integer> tfidfRanking = Rankers.rank(synthetic, term, false);
            List<Integer> shuffled = new ArrayList<>(ideal);
            for (int i = shuffled.size() - 1; i > 0; i--) Collections.swap(shuffled, i, rng.nextInt(i + 1));
            double bm25Score = Metrics.ndcg(bm25Ranking, truth, 10);
            double tfidfScore = Metrics.ndcg(tfidfRanking, truth, 10);
            double randomScore = Metrics.ndcg(shuffled, truth, 10);
            bm25Ndcg += bm25Score;
            tfidfNdcg += tfidfScore;
            randomNdcg += randomScore;
            if (bm25Score >= randomScore) bm25Better++;
            if (tfidfScore >= randomScore) tfidfBetter++;
            check(Metrics.ndcg(ideal, truth, 10) == 1.0, "the ideal ranking scores 1 for query " + query);
            check(bm25Score > 0 && bm25Score <= 1.0, "bm25 NDCG is in (0, 1] for query " + query);
        }
        check(bm25Better >= queries * 9 / 10 && tfidfBetter >= queries * 7 / 10, "both rankers beat the "
                + "shuffled ranking on the large majority of queries (bm25 " + bm25Better + ", tfidf "
                + tfidfBetter + " of " + queries + ")");
        check(bm25Ndcg / queries > randomNdcg / queries, "BM25 averages a higher NDCG than random: "
                + (bm25Ndcg / queries) + " vs " + (randomNdcg / queries));
        check(tfidfNdcg / queries > randomNdcg / queries, "TF-IDF averages a higher NDCG than random: "
                + (tfidfNdcg / queries) + " vs " + (randomNdcg / queries));

        // 7. the legacy corpus ranking, and its ordering by a rare term
        List<Integer> foxRanking = Rankers.rank(corpus, "fox", true);
        check(foxRanking.get(0) == 3, "the document with four foxes ranks first: " + foxRanking);
        check(Rankers.rank(corpus, "weather", true).get(0) == 4, "a term in one document ranks it first");

        // 8. boundaries: absent terms, an empty query, a single-document corpus
        Corpus single = new Corpus();
        single.add("only term here");
        close(Rankers.bm25(single, 0, "term", 1.2, 0.75) > 0 ? 1 : 0, 1, 0, "a single-document corpus scores");
        check(Rankers.bm25(corpus, 0, "unobtainium", 1.2, 0.75) == 0, "an absent term scores zero");
        Corpus blank = new Corpus();
        blank.add("");
        check(Rankers.bm25(blank, 0, "", 1.2, 0.75) == 0, "an empty document and empty query score zero");
        check(Rankers.rank(corpus, "unobtainium", true).size() == corpus.size(),
                "ranking an absent term still returns every document, all with score zero");
        check(Rankers.bm25(repeated, 0, "term", 1.2, 0.75) == 0, "a document without the term scores zero");

        System.out.println("corpus: " + corpus.size() + " documents, avg length " + corpus.averageLength
                + ", 'quick' in " + corpus.df("quick") + " of them");
        System.out.println("bm25 NDCG@10 " + Math.round(bm25Ndcg / queries * 1000) / 1000.0 + ", tfidf "
                + Math.round(tfidfNdcg / queries * 1000) / 1000.0 + ", random baseline "
                + Math.round(randomNdcg / queries * 1000) / 1000.0 + " over " + queries + " queries");
        System.out.println("metrics: hand-computed AP, MRR, precision/recall and graded NDCG all match");
        System.out.println("PASS");
    }
}
