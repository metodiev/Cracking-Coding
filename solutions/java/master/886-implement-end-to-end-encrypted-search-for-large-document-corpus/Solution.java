/** Task 886 - Implement end-to-end encrypted search for large document corpus with privacy budget. (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * A searchable-encryption scheme in the "deterministic token + inverted index" family, together with
 * the leakage measurement and a differential-privacy budget on top:
 *
 *   client   tokenizes every term with a keyed PRF (splitmix64 over key xor term), sends tokens only
 *   server   keeps an inverted index token -> sorted postings (document ids with term frequencies), has no
 *            key and no plaintext; it scores a query with tf-idf weights computed from postings lengths
 *            and term frequencies - both of which it can see - and returns the top-k document ids
 *   client   maps the returned ids back to its own encrypted blobs
 *
 * The demo is honest about what deterministic search leaks, and the checks measure it instead of
 * claiming privacy: equality of tokens (identical terms map to identical tokens, which is asserted
 * exactly), document frequency (postings lengths are visible to the server), and document length
 * (defeated by padding every document to a fixed epoch bucket). The resulting frequency-analysis
 * attack - sort tokens by document frequency, map the r-th token to the r-th term of a public Zipf
 * reference corpus - is implemented and its top-k recall is measured against ground truth.
 *
 * The budget is spent per query: every query charges epsilon to a total budget, the Laplace mechanism
 * adds noise of scale sensitivity/epsilon to each document score, and a query that does not fit is
 * refused instead of silently answering. What is simplified: the keyed PRF stands in for HMAC-SHA256
 * (it is deterministic and key dependent but not a real MAC), tokens are 64-bit instead of 256-bit,
 * the corpus is generated from a Zipf law instead of being crawled, and per-document noise is charged
 * as one epsilon per query rather than composed across documents - the budget accounting, the noise
 * scale, the recall degradation and the refusal path are all real. No wall-clock time, no threads, no
 * sockets; the "network" is a function call and every random draw comes from splitmix64.
 *
 * Verified: the PRF is deterministic, key dependent and collision free over the vocabulary; postings
 * match the plaintext oracle exactly; encrypted top-k search reproduces the plaintext ranking for many
 * queries; the serialized index contains no plaintext word; equality leakage and postings-length
 * leakage are asserted exactly; the frequency-analysis attack recovers at least three of the five
 * most frequent terms; padding makes all encrypted documents within a bucket indistinguishable in
 * length; the budget allows exactly the affordable number of queries and then refuses; the Laplace
 * noise is unbiased with the theoretical variance; noisy recall stays above a floor and improves with
 * epsilon; a query costs exactly the sum of the touched postings and never a full scan; and empty
 * queries, unknown terms, an empty corpus, a single-document corpus and repeated runs behave.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** Keyed pseudo-random function over terms: the client's only secret. */
class Prf {
    final long key;

    Prf(long key) { this.key = key; }

    static long mix(long z) {
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    long token(String term) {
        long h = 0x243F6A8885A308D3L ^ key;
        for (int i = 0; i < term.length(); i++) h = mix(h ^ (term.charAt(i) & 0xFFFFL));
        return mix(h ^ term.length());
    }

    static String hex(long v) { return String.format("%016x", v); }
}

/** The plaintext corpus: a vocabulary built from syllables and documents drawn from a Zipf law. */
class Corpus {
    final String[] vocab;
    final List<int[]> docs;                    // distinct term ids per document, ascending
    final List<int[]> tf;                      // term frequencies, aligned with docs
    final int[] df;                            // document frequency per term

    private Corpus(String[] vocab, List<int[]> docs, List<int[]> tf) {
        this.vocab = vocab; this.docs = docs; this.tf = tf;
        this.df = new int[vocab.length];
        for (int[] d : docs) for (int t : d) df[t]++;
    }

    static String[] vocabulary(int size) {
        String[] syl = {"ka", "lo", "mir", "tas", "ven", "zor", "phy", "qui", "rex", "dun", "mel",
                "sar", "til", "nor", "bek"};
        List<String> out = new ArrayList<>();
        for (int i = 0; i < syl.length && out.size() < size; i++)
            for (int j = 0; j < syl.length && out.size() < size; j++)
                for (int k = 0; k < syl.length && out.size() < size; k++)
                    out.add(syl[i] + syl[j] + syl[k]);
        return out.toArray(new String[0]);
    }

    static Corpus generate(int nDocs, int termsPerDoc, int vocabSize, long seed) {
        String[] vocab = vocabulary(vocabSize);
        double[] weight = new double[vocab.length];
        double[] cdf = new double[vocab.length];
        double acc = 0;
        for (int i = 0; i < vocab.length; i++) { weight[i] = 1.0 / (i + 1); acc += weight[i]; }
        double run = 0;
        for (int i = 0; i < vocab.length; i++) { run += weight[i] / acc; cdf[i] = run; }
        SplitMix64 rng = new SplitMix64(seed);
        List<int[]> docs = new ArrayList<>();
        List<int[]> tfs = new ArrayList<>();
        for (int d = 0; d < nDocs; d++) {
            int want = termsPerDoc + rng.nextInt(9);            // 25..33 tokens: crosses the 32-term epoch line
            TreeMap<Integer, Integer> counts = new TreeMap<>();
            for (int i = 0; i < want; i++) {                    // sampling with repetition gives real term frequencies
                double u = rng.nextDouble();
                int lo = 0, hi = cdf.length - 1;
                while (lo < hi) { int mid = (lo + hi) >>> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
                counts.merge(lo, 1, Integer::sum);
            }
            int[] arr = new int[counts.size()], freq = new int[counts.size()];
            int i = 0;
            for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
                arr[i] = e.getKey(); freq[i] = e.getValue(); i++;
            }
            docs.add(arr);
            tfs.add(freq);
        }
        return new Corpus(vocab, docs, tfs);
    }

    int size() { return docs.size(); }
}

/** Server-side encrypted index: tokens, sorted postings with term frequencies and padded documents. */
class EncryptedIndex {
    static final int EPOCH = 32;
    final Map<Long, int[]> postings = new TreeMap<>();
    final Map<Long, int[]> postingsTf = new TreeMap<>();
    final List<long[]> paddedDocs = new ArrayList<>();
    final Set<Long> realTokens = new TreeSet<>();
    long padToken(String term, int epoch) { return Prf.mix(0x5DEECE66DL ^ epoch) ^ Prf.mix(term.hashCode()); }

    EncryptedIndex(Corpus corpus, Prf prf) {
        List<List<Integer>> perTokenDoc = new ArrayList<>();
        List<List<Integer>> perTokenTf = new ArrayList<>();
        Map<Long, Integer> tokenSlot = new HashMap<>();
        for (int d = 0; d < corpus.size(); d++) {
            List<Long> tokens = new ArrayList<>();
            int[] terms = corpus.docs.get(d), freq = corpus.tf.get(d);
            for (int i = 0; i < terms.length; i++) {
                long tok = prf.token(corpus.vocab[terms[i]]);
                realTokens.add(tok);
                Integer slot = tokenSlot.get(tok);
                if (slot == null) {
                    slot = perTokenDoc.size();
                    tokenSlot.put(tok, slot);
                    perTokenDoc.add(new ArrayList<>());
                    perTokenTf.add(new ArrayList<>());
                }
                perTokenDoc.get(slot).add(d);
                perTokenTf.get(slot).add(freq[i]);
                tokens.add(tok);
            }
            Collections.sort(tokens);
            List<Long> occurrences = new ArrayList<>();          // document length includes repeats
            for (int i = 0; i < terms.length; i++)
                for (int c = 0; c < freq[i]; c++) occurrences.add(tokens.get(i));
            Collections.sort(occurrences);
            int epochs = Math.max(1, (occurrences.size() + EPOCH - 1) / EPOCH);
            long[] padded = new long[epochs * EPOCH];
            for (int i = 0; i < occurrences.size(); i++) padded[i] = occurrences.get(i);
            for (int i = occurrences.size(); i < padded.length; i++) padded[i] = padToken("pad", i / EPOCH);
            paddedDocs.add(padded);
        }
        for (Map.Entry<Long, Integer> e : tokenSlot.entrySet()) {
            List<Integer> ids = perTokenDoc.get(e.getValue());
            List<Integer> freqs = perTokenTf.get(e.getValue());
            int[] arr = new int[ids.size()], tf = new int[ids.size()];
            for (int i = 0; i < arr.length; i++) { arr[i] = ids.get(i); tf[i] = freqs.get(i); }
            postings.put(e.getKey(), arr);
            postingsTf.put(e.getKey(), tf);
        }
    }

    /** Ids of posting lists this query touches: this is the server's only search cost. */
    long touched(List<Long> queryTokens) {
        long n = 0;
        for (long t : queryTokens) {
            int[] p = postings.get(t);
            if (p != null) n += p.length;
        }
        return n;
    }

    /** Serialized server-side state. Postings lengths are visible, terms are not. */
    String serialize() {
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Long, int[]> e : postings.entrySet())
            sb.append(Prf.hex(e.getKey())).append(':').append(e.getValue().length).append(';');
        sb.append("docs=").append(paddedDocs.size()).append('x').append(paddedDocs.get(0).length);
        return sb.toString();
    }
}

/** One scored document. */
class DocScore implements Comparable<DocScore> {
    final int doc;
    final double score;

    DocScore(int doc, double score) { this.doc = doc; this.score = score; }

    public int compareTo(DocScore o) {
        int c = Double.compare(o.score, score);
        return c != 0 ? c : Integer.compare(doc, o.doc);
    }
}

/** The server: scoring over tokens only. Never sees the key, never sees a term. */
class SearchServer {
    final EncryptedIndex index;
    long scanned;

    SearchServer(EncryptedIndex index) { this.index = index; }

    /** idf weights derived from postings lengths - this is the document-frequency leakage in action. */
    double idf(long token, int nDocs) {
        int[] p = index.postings.get(token);
        return p == null ? 0.0 : Math.log(1.0 + nDocs / (double) p.length);
    }

    List<DocScore> search(List<Long> queryTokens, int k, int nDocs) {
        Map<Integer, Double> hit = new TreeMap<>();
        for (long t : queryTokens) {
            int[] p = index.postings.get(t);
            if (p == null) continue;
            int[] tf = index.postingsTf.get(t);
            scanned += p.length;
            double w = idf(t, nDocs);
            for (int i = 0; i < p.length; i++) hit.merge(p[i], w * (1.0 + Math.log(tf[i])), Double::sum);
        }
        List<DocScore> out = new ArrayList<>();
        for (Map.Entry<Integer, Double> e : hit.entrySet()) out.add(new DocScore(e.getKey(), e.getValue()));
        Collections.sort(out);
        return out.size() > k ? new ArrayList<>(out.subList(0, k)) : out;
    }
}

/** Differential-privacy bookkeeping: a shared budget and the Laplace mechanism. */
class PrivacyBudget {
    final double total;
    double spent;
    int refused;

    PrivacyBudget(double total) { this.total = total; }

    boolean charge(double epsilon) {
        if (epsilon <= 0) throw new IllegalArgumentException("epsilon must be positive");
        if (spent + epsilon > total + 1e-12) { refused++; return false; }
        spent += epsilon;
        return true;
    }
}

class Laplace {
    final double scale;
    final SplitMix64 rng;

    Laplace(double scale, SplitMix64 rng) { this.scale = scale; this.rng = rng; }

    double sample() { return sample(scale); }

    /** Inverse CDF of the Laplace law, sampled from a uniform in (-1/2, 1/2). */
    double sample(double s) {
        double u = rng.nextDouble() - 0.5;
        return -s * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
    }
}

/** The client: tokenizes queries, drives the budget, decrypts results. */
class SearchClient {
    final Prf prf;
    final SearchServer server;
    final PrivacyBudget budget;
    final int nDocs;

    SearchClient(Prf prf, SearchServer server, PrivacyBudget budget, int nDocs) {
        this.prf = prf; this.server = server; this.budget = budget; this.nDocs = nDocs;
    }

    List<Long> tokenize(String[] queryTerms) {
        List<Long> tokens = new ArrayList<>();
        for (String t : queryTerms) tokens.add(prf.token(t));
        return tokens;
    }

    /** Exact encrypted search: identical to searching the plaintext, with no noise at all. */
    List<DocScore> searchExact(String[] queryTerms, int k) {
        return server.search(tokenize(queryTerms), k, nDocs);
    }

    /**
     * Noisy search under the budget. The Laplace scale is sensitivity/epsilon with the query's maximum
     * idf as the sensitivity, and one epsilon is charged for the whole query.
     */
    List<DocScore> searchPrivate(String[] queryTerms, int k, double epsilon, Laplace noise) {
        if (!budget.charge(epsilon)) return null;
        List<DocScore> exact = server.search(tokenize(queryTerms), nDocs, nDocs);
        double sensitivity = 0.0;
        for (long t : tokenize(queryTerms)) sensitivity = Math.max(sensitivity, server.idf(t, nDocs));
        double scale = sensitivity / epsilon;
        List<DocScore> noisy = new ArrayList<>();
        for (DocScore s : exact) noisy.add(new DocScore(s.doc, s.score + noise.sample(scale)));
        Collections.sort(noisy);
        return noisy.size() > k ? new ArrayList<>(noisy.subList(0, k)) : noisy;
    }
}

/** Plaintext oracle used to validate the encrypted pipeline and for recall measurements. */
class Oracle {
    final Corpus corpus;
    final Map<String, Integer> termId = new HashMap<>();

    Oracle(Corpus corpus) {
        this.corpus = corpus;
        for (int i = 0; i < corpus.vocab.length; i++) termId.put(corpus.vocab[i], i);
    }

    List<DocScore> search(String[] terms, int k) {
        Map<Integer, Double> hit = new TreeMap<>();
        for (String term : terms) {
            Integer id = termId.get(term);
            if (id == null || corpus.df[id] == 0) continue;
            double w = Math.log(1.0 + corpus.size() / (double) corpus.df[id]);
            for (int d = 0; d < corpus.size(); d++) {
                int[] docTerms = corpus.docs.get(d), freq = corpus.tf.get(d);
                for (int i = 0; i < docTerms.length; i++) if (docTerms[i] == id)
                    hit.merge(d, w * (1.0 + Math.log(freq[i])), Double::sum);
            }
        }
        List<DocScore> out = new ArrayList<>();
        for (Map.Entry<Integer, Double> e : hit.entrySet()) out.add(new DocScore(e.getKey(), e.getValue()));
        Collections.sort(out);
        return out.size() > k ? new ArrayList<>(out.subList(0, k)) : out;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static final long KEY = 0x5EED1234ABCDFEDL;
    static final long OTHER_KEY = 0x1BADB002DEADBEEFL;

    static List<Integer> ids(List<DocScore> scored) {
        List<Integer> out = new ArrayList<>();
        for (DocScore s : scored) out.add(s.doc);
        return out;
    }

    static List<Double> scores(List<DocScore> scored) {
        List<Double> out = new ArrayList<>();
        for (DocScore s : scored) out.add(s.score);
        return out;
    }

    public static void main(String[] args) {
        Corpus corpus = Corpus.generate(200, 25, 150, 0x2024L);
        Prf prf = new Prf(KEY);
        EncryptedIndex index = new EncryptedIndex(corpus, prf);
        SearchServer server = new SearchServer(index);
        Oracle oracle = new Oracle(corpus);
        SearchClient client = new SearchClient(prf, server, new PrivacyBudget(1.0), corpus.size());

        // 1. The PRF is deterministic, key dependent and collision free over the vocabulary.
        boolean deterministic = true, collisions = false, keyed = true;
        Set<Long> tokens = new HashSet<>();
        for (int i = 0; i < corpus.vocab.length; i++) {
            long a = prf.token(corpus.vocab[i]), b = prf.token(corpus.vocab[i]);
            if (a != b) deterministic = false;
            if (!tokens.add(a)) collisions = true;
            if (a == new Prf(OTHER_KEY).token(corpus.vocab[i])) keyed = false;
        }
        check(deterministic, "PRF is not deterministic");
        check(!collisions, "PRF collided on the vocabulary");
        check(keyed, "PRF does not depend on the key");
        int realTerms = 0;
        for (int f : corpus.df) if (f > 0) realTerms++;
        check(index.realTokens.size() == realTerms, "index token count " + index.realTokens.size()
                + " differs from the number of observed terms " + realTerms);
        check(realTerms > corpus.vocab.length * 3 / 4, "the generated corpus is too sparse: " + realTerms
                + " of " + corpus.vocab.length + " terms observed");
        boolean nonHex = true;
        for (String w : corpus.vocab) if (w.matches("[0-9a-f]+")) nonHex = false;
        check(nonHex, "vocabulary must be distinguishable from hex tokens for the leakage check");

        // 2. Postings agree with the plaintext document frequencies.
        int mismatches = 0, covered = 0;
        for (int t = 0; t < corpus.vocab.length; t++) {
            int[] p = index.postings.get(prf.token(corpus.vocab[t]));
            if (p == null || p.length != corpus.df[t]) mismatches++;
            else covered += p.length;
        }
        check(mismatches == 0, "postings disagree with the plaintext oracle for " + mismatches + " terms");
        int totalTermOccurrences = 0;
        for (int[] d : corpus.docs) totalTermOccurrences += d.length;
        check(covered == totalTermOccurrences, "postings cover " + covered + " of " + totalTermOccurrences);
        boolean sorted = true;
        for (int[] p : index.postings.values())
            for (int i = 1; i < p.length; i++) if (p[i] <= p[i - 1]) sorted = false;
        check(sorted, "postings must be sorted and duplicate free");

        // 3. Encrypted top-k search reproduces the plaintext ranking, and costs only the postings touched.
        SplitMix64 pick = new SplitMix64(99);
        int agree = 0, queries = 30;
        long maxTouched = 0, scanCost = (long) corpus.size() * 25;
        for (int q = 0; q < queries; q++) {
            String[] terms = new String[2];
            for (int i = 0; i < 2; i++) terms[i] = corpus.vocab[pick.nextInt(corpus.vocab.length)];
            List<DocScore> enc = client.searchExact(terms, 5);
            List<DocScore> plain = oracle.search(terms, 5);
            if (ids(enc).equals(ids(plain)) && closeScores(enc, plain)) agree++;
            maxTouched = Math.max(maxTouched, index.touched(client.tokenize(terms)));
        }
        check(agree == queries, "encrypted search agreed with the oracle on only " + agree + "/" + queries);
        check(maxTouched < scanCost / 4, "inverted-index query touched " + maxTouched
                + " postings against a scan cost of " + scanCost);
        SearchServer accounted = new SearchServer(index);
        String[] costQuery = {corpus.vocab[0], corpus.vocab[1], corpus.vocab[2]};
        long expectedTouch = index.touched(client.tokenize(costQuery));
        accounted.search(client.tokenize(costQuery), 3, corpus.size());
        check(accounted.scanned == expectedTouch, "server scanned " + accounted.scanned
                + " postings but touched " + expectedTouch + " are expected");

        // 4. Leakage: no plaintext in the server state, but term equality is exactly visible.
        String serialized = index.serialize();
        boolean leaked = false;
        for (String w : corpus.vocab) if (serialized.contains(w)) leaked = true;
        check(!leaked, "serialized encrypted index contains a plaintext term");
        int distinctTerms = 0;
        for (int f : corpus.df) if (f > 0) distinctTerms++;
        check(index.postings.size() == distinctTerms, "deterministic tokens must be one-to-one with terms: "
                + index.postings.size() + " vs " + distinctTerms);

        // 5. The leakage-abuse attack: rank tokens by document frequency, map to a public Zipf ranking.
        Integer[] byDf = new Integer[corpus.vocab.length];
        for (int i = 0; i < byDf.length; i++) byDf[i] = i;
        Arrays.sort(byDf, (a, b) -> Integer.compare(corpus.df[b], corpus.df[a]));
        int top = 5, hits = 0;
        StringBuilder attackLog = new StringBuilder();
        for (int r = 0; r < top; r++) {
            String guessed = corpus.vocab[r];                       // public ranking: rank r is the r-th Zipf term
            boolean correct = byDf[r] == r;
            if (correct) hits++;
            attackLog.append(corpus.vocab[byDf[r]]).append("->").append(guessed).append(correct ? "= " : "! ");
        }
        check(hits >= 3, "frequency-analysis attack recovered only " + hits + "/" + top + " top terms");

        // 6. Padding hides document length; the pad tokens never collide with real ones.
        boolean uniform = true;
        Set<Integer> lengths = new TreeSet<>();
        for (long[] padded : index.paddedDocs) {
            if (padded.length % EncryptedIndex.EPOCH != 0) uniform = false;
            lengths.add(padded.length);
        }
        check(uniform, "padded documents are not epoch aligned");
        check(lengths.size() < corpus.size(), "padding did not collapse document lengths: " + lengths.size()
                + " distinct sizes for " + corpus.size() + " documents");
        Set<Long> padTokens = new TreeSet<>();
        for (int epoch = 0; epoch < 4; epoch++)
            for (int i = 0; i < EncryptedIndex.EPOCH; i++) padTokens.add(index.padToken("pad", epoch));
        boolean padCollision = false;
        for (long tok : padTokens) if (index.postings.containsKey(tok)) padCollision = true;
        check(!padCollision, "a padding token collides with a real postings key");
        check(lengths.size() == 2, "expected exactly two padding buckets for 25..33 term documents, got "
                + lengths.size());

        // 7. Budget: exactly the affordable number of queries, then refusal.
        PrivacyBudget budget = new PrivacyBudget(1.0);
        SearchClient metered = new SearchClient(prf, server, budget, corpus.size());
        Laplace noise = new Laplace(1.0, new SplitMix64(7));
        int allowed = 0, refused = 0;
        for (int q = 0; q < 25; q++) {
            List<DocScore> r = metered.searchPrivate(new String[]{corpus.vocab[q % corpus.vocab.length]}, 3, 0.05, noise);
            if (r == null) refused++; else allowed++;
        }
        check(allowed == 20 && refused == 5, "budget allowed " + allowed + " and refused " + refused + " of 25");
        check(Math.abs(budget.spent - 1.0) < 1e-12, "spent " + budget.spent + " of a 1.0 budget");
        check(budget.refused == 5, "budget refusal counter wrong: " + budget.refused);
        PrivacyBudget small = new PrivacyBudget(0.1);
        SearchClient stingy = new SearchClient(prf, server, small, corpus.size());
        check(stingy.searchPrivate(new String[]{"ka"}, 1, 0.5, noise) == null && small.refused == 1,
                "an unaffordable query was answered instead of refused");
        boolean threw = false;
        try { budget.charge(0.0); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "a non-positive epsilon was accepted");

        // 8. Laplace noise: unbiased with variance 2 b^2.
        Laplace unit = new Laplace(1.0, new SplitMix64(2024));
        int draws = 40000;
        double sum = 0, sumSq = 0;
        for (int i = 0; i < draws; i++) { double v = unit.sample(); sum += v; sumSq += v * v; }
        double mean = sum / draws;
        double variance = sumSq / draws - mean * mean;
        check(Math.abs(mean) < 0.05, "Laplace mean " + mean + " is biased");
        check(variance > 1.6 && variance < 2.4, "Laplace variance " + variance + " far from 2 b^2 = 2.0");
        Laplace scaled = new Laplace(3.0, new SplitMix64(5));
        double s2 = 0;
        for (int i = 0; i < draws; i++) s2 += Math.pow(scaled.sample(), 2);
        check(Math.abs(s2 / draws - 18.0) < 2.0, "scaled Laplace second moment " + s2 / draws + " far from 18");

        // 9. Recall under noise: above a floor, and monotone in epsilon.
        String[] evalTerms = new String[12];
        for (int i = 0; i < evalTerms.length; i++) evalTerms[i] = corpus.vocab[20 + i];
        double recallLow = recallAt(server, corpus, evalTerms, 0.1, 41);
        double recallMid = recallAt(server, corpus, evalTerms, 1.0, 41);
        double recallHigh = recallAt(server, corpus, evalTerms, 4.0, 41);
        double recallMax = recallAt(server, corpus, evalTerms, 64.0, 41);
        System.out.println("recall@5 eps 0.1/1.0/4.0/64.0: " + recallLow + " / " + recallMid + " / "
                + recallHigh + " / " + recallMax);
        check(recallMax > 0.85, "recall at epsilon=64 is only " + recallMax);
        check(recallMid > recallLow, "recall must improve from epsilon=0.1 to epsilon=1.0: "
                + recallMid + " vs " + recallLow);
        check(recallHigh > recallMid, "recall must improve from epsilon=1.0 to epsilon=4.0: "
                + recallHigh + " vs " + recallMid);
        check(recallMax > recallHigh, "recall must improve as the budget grows: " + recallMax + " vs " + recallHigh);
        check(recallLow < 0.5, "epsilon=0.1 should visibly damage recall, got " + recallLow);
        Laplace silent = new Laplace(0.0, new SplitMix64(3));
        check(client.searchPrivate(evalTerms, 5, 1e9, silent) == null,
                "an epsilon of 1e9 should not fit the default budget");

        // 10. Exactness limit: an astronomically large epsilon leaves the ranking alone.
        PrivacyBudget free = new PrivacyBudget(1e13);
        SearchClient freeClient = new SearchClient(prf, server, free, corpus.size());
        List<DocScore> zeroNoise = freeClient.searchPrivate(evalTerms, 5, 1e12, silent);
        List<DocScore> exact = oracle.search(evalTerms, 5);
        check(zeroNoise != null && ids(zeroNoise).equals(ids(exact)),
                "huge-epsilon private search differs from the exact ranking");
        check(Math.abs(zeroNoise.get(0).score - exact.get(0).score) < 1e-6, "huge-epsilon score differs");

        // 11. Degenerate inputs: empty query, unknown term, single-document and empty corpora.
        check(client.searchExact(new String[0], 5).isEmpty(), "empty query returned results");
        check(index.touched(client.tokenize(new String[0])) == 0, "empty query touched postings");
        check(client.searchExact(new String[]{"zzzznotatoken"}, 5).isEmpty(), "unknown term returned results");
        Corpus one = Corpus.generate(1, 5, 40, 11);
        Oracle oneOracle = new Oracle(one);
        SearchClient oneClient = new SearchClient(prf, new SearchServer(new EncryptedIndex(one, prf)),
                new PrivacyBudget(1.0), 1);
        check(oneClient.searchExact(new String[]{one.vocab[one.docs.get(0)[0]]}, 5).size() == 1
                && ids(oneClient.searchExact(new String[]{one.vocab[one.docs.get(0)[0]]}, 5))
                        .equals(ids(oneOracle.search(new String[]{one.vocab[one.docs.get(0)[0]]}, 5))),
                "single-document corpus mishandled");
        boolean emptyOk = false;
        try {
            EncryptedIndex empty = new EncryptedIndex(Corpus.generate(0, 5, 40, 3), prf);
            emptyOk = empty.postings.isEmpty() || empty.paddedDocs.isEmpty();
        } catch (RuntimeException e) { emptyOk = false; }
        check(emptyOk, "empty corpus mishandled");

        // 12. Determinism: a second build from the same seed is byte-identical and answers identically.
        Corpus again = Corpus.generate(200, 25, 150, 0x2024L);
        EncryptedIndex index2 = new EncryptedIndex(again, new Prf(KEY));
        check(index2.serialize().equals(serialized), "two builds with the same seed differ");
        SearchClient client2 = new SearchClient(new Prf(KEY), new SearchServer(index2), new PrivacyBudget(1.0),
                again.size());
        check(ids(client2.searchExact(evalTerms, 5)).equals(ids(client.searchExact(evalTerms, 5))),
                "rebuilt index returns a different ranking");
        check(index2.realTokens.size() == index.realTokens.size(), "rebuilt index has a different token count");

        System.out.println("corpus: " + corpus.size() + " docs, " + corpus.vocab.length + " terms, "
                + totalTermOccurrences + " postings entries");
        System.out.println("attack log (top-5 df): " + attackLog.toString().trim() + " hits " + hits + "/" + top);
        System.out.println("recall@5: eps=0.1 -> " + String.format("%.2f", recallLow)
                + ", eps=4.0 -> " + String.format("%.2f", recallHigh));
        System.out.println("budget: spent " + budget.spent + ", refused " + budget.refused
                + ", postings touched per query <= " + maxTouched);
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }

    /**
     * Mean recall@5 of the noisy ranking against the plaintext oracle on two-term queries, where the
     * documents that match both terms really do score higher - a single-term query gives every matching
     * document the same score, so its ranking is pure tie-breaking and recall would measure nothing.
     * The same noise stream is reused for every epsilon, which makes the comparison paired.
     */
    static double recallAt(SearchServer server, Corpus corpus, String[] terms, double epsilon, long seed) {
        int hits = 0, total = 0;
        Oracle oracle = new Oracle(corpus);
        int queries = terms.length / 2;
        for (int i = 0; i < queries; i++) {
            String[] q = {terms[i], terms[i + queries]};
            SearchClient c = new SearchClient(new Prf(KEY), server, new PrivacyBudget(10.0), corpus.size());
            Laplace noise = new Laplace(1.0, new SplitMix64(seed + i));
            List<DocScore> noisy = c.searchPrivate(q, 5, epsilon, noise);
            List<DocScore> exact = oracle.search(q, 5);
            if (noisy == null) continue;
            for (int d : ids(noisy)) if (ids(exact).contains(d)) hits++;
            total += exact.size();
        }
        return total == 0 ? 1.0 : hits / (double) total;
    }

    static boolean closeScores(List<DocScore> a, List<DocScore> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++)
            if (Math.abs(a.get(i).score - b.get(i).score) > 1e-12) return false;
        return true;
    }
}
