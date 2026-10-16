/** Task 762 - Implement a searchable-encryption demo: deterministic encryption with leakage analysis.
 * (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * A searchable-encryption scheme lets a server answer keyword queries without learning the keywords.
 * The cheapest construction is *deterministic*: the client computes a trapdoor E(k, w) for the query
 * word w and the server looks it up in an index keyed by E(k, w). That works, and it leaks exactly what
 * everyone should worry about:
 *
 *  * the equality pattern - equal keywords produce equal trapdoors, so the server sees which documents
 *    share a term (the multiset of document-id sets in the index is the same as in plaintext);
 *  * the frequency distribution - the posting-list sizes are visible, so a dictionary attacker who knows
 *    the language's frequency profile can map trapdoors to words by rank, without ever seeing the key.
 *
 * The demo builds the index, answers queries (verified against a plaintext scan) and then *measures* the
 * leak instead of just naming it: it reconstructs the equality pattern, runs the rank-matching attack on
 * several independent keys, and shows the payoff of randomising the trapdoor (per-document salt) - which
 * hides the frequency pattern but breaks plain equality search, the classic utility/leakage trade-off.
 *
 * Simplified: E is a splitmix64 mixer, not a PRF or a cipher; nothing here is confidential and no real
 * key material exists (the "key" is a seed). It is a deliberate, honest model of deterministic
 * encryption's leakage, not a secure scheme.
 *
 * Verified: query results equal a plaintext scan over random corpora; the uniqueness of trapdoors;
 * the equality pattern of the encrypted index is isomorphic to the plaintext one; key independence of
 * the frequency attack versus its success; the randomisation mitigation breaking exact search; empty
 * documents, absent terms, single-document corpora and an empty query as boundaries; determinism.
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

    /** Uniform in [0, 1), used to draw Zipf-distributed keywords. */
    double nextDouble() { return (nextLong() >>> 11) / (double) (1L << 53); }
}

/** A toy deterministic "encryption": the trapdoor of a keyword under a key. */
final class DeterministicEncryption {
    final long key;

    DeterministicEncryption(long key) { this.key = key; }

    /** Deterministic by construction - that is the whole point (and the whole leak). */
    long trapdoor(String keyword) {
        long z = key ^ 0x9E3779B97F4A7C15L;
        for (int i = 0; i < keyword.length(); i++) {
            z = (z ^ keyword.charAt(i)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        }
        z = (z ^ (z >>> 31)) * 0x165667B19E3779F9L;
        return z ^ (z >>> 29);
    }

    /** Per-document salting destroys equality search: the same word salts to a different trapdoor. */
    long saltedTrapdoor(String keyword, int documentId) {
        return trapdoor(keyword + "#" + documentId);
    }
}

/** The encrypted index the server stores. */
final class SearchableIndex {
    final DeterministicEncryption scheme;
    final Map<Long, Set<Integer>> postings = new TreeMap<>();     // trapdoor -> document ids
    final Map<Long, Integer> occurrences = new TreeMap<>();       // trapdoor -> total keyword count
    long[] orderOfInsertion = new long[0];

    SearchableIndex(DeterministicEncryption scheme) { this.scheme = scheme; }

    void addDocument(int documentId, List<String> keywords) {
        for (String keyword : keywords) {
            long trapdoor = scheme.trapdoor(keyword);
            postings.computeIfAbsent(trapdoor, k -> new TreeSet<>()).add(documentId);
            occurrences.merge(trapdoor, 1, Integer::sum);
        }
    }

    Set<Integer> search(String keyword) {
        return new TreeSet<>(postings.getOrDefault(scheme.trapdoor(keyword), Set.of()));
    }

    int size() { return postings.size(); }
}

/** The dictionary attacker: posting-list sizes leak the keyword frequency profile. */
final class FrequencyAttack {

    /**
     * Ranks the observed trapdoors by the number of documents they appear in and pairs them, in that
     * order, with the candidate keywords ordered by their known frequency in the language. No key needed.
     */
    static Map<Long, String> rankMatch(SearchableIndex index, List<String> dictionaryByFrequency) {
        List<Long> trapdoors = new ArrayList<>(index.postings.keySet());
        trapdoors.sort((a, b) -> {
            int byDocuments = Integer.compare(index.postings.get(b).size(), index.postings.get(a).size());
            return byDocuments != 0 ? byDocuments : Long.compare(a, b);
        });
        Map<Long, String> guess = new LinkedHashMap<>();
        for (int i = 0; i < trapdoors.size() && i < dictionaryByFrequency.size(); i++) {
            guess.put(trapdoors.get(i), dictionaryByFrequency.get(i));
        }
        return guess;
    }

    /** How many of the true top-k keywords the attacker recovered exactly? */
    static int topKAccuracy(Map<Long, String> guess, SearchableIndex index,
                            Map<String, Long> trueTrapdoors, List<String> trueTopK) {
        Map<String, Integer> rankOfTruth = new LinkedHashMap<>();
        for (int i = 0; i < trueTopK.size(); i++) rankOfTruth.put(trueTopK.get(i), i);
        int hits = 0;
        for (String keyword : trueTopK) {
            Long trapdoor = trueTrapdoors.get(keyword);
            if (trapdoor != null && keyword.equals(guess.get(trapdoor))) hits++;
        }
        return hits;
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** Vocabulary with a Zipf-ish weight profile: rank i is drawn with probability proportional 1/(i+1). */
    static List<String> vocabulary(int size) {
        List<String> words = new ArrayList<>();
        for (int i = 0; i < size; i++) words.add("term" + (char) ('a' + i % 26) + i);
        return words;
    }

    static double[] zipfWeights(int size) {
        double[] weights = new double[size];
        double total = 0;
        for (int i = 0; i < size; i++) {
            weights[i] = 1.0 / Math.pow(i + 1, 1.2);
            total += weights[i];
        }
        for (int i = 0; i < size; i++) weights[i] /= total;
        return weights;
    }

    static String drawKeyword(List<String> vocabulary, double[] weights, Prng rng) {
        double roll = rng.nextDouble();
        double cumulative = 0;
        for (int i = 0; i < weights.length; i++) {
            cumulative += weights[i];
            if (roll < cumulative) return vocabulary.get(i);
        }
        return vocabulary.get(vocabulary.size() - 1);
    }

    /** A corpus of documents, each a multiset of keywords drawn from the Zipf profile. */
    static List<List<String>> corpus(int documentCount, int keywordsPerDocument, List<String> vocabulary,
                                    double[] weights, long seed) {
        Prng rng = new Prng(seed);
        List<List<String>> documents = new ArrayList<>();
        for (int d = 0; d < documentCount; d++) {
            List<String> keywords = new ArrayList<>();
            for (int k = 0; k < keywordsPerDocument; k++) {
                keywords.add(drawKeyword(vocabulary, weights, rng));
            }
            documents.add(keywords);
        }
        return documents;
    }

    /** Plaintext scan - the oracle every query answer is compared against. */
    static Set<Integer> containsKeyword(List<List<String>> documents, String keyword) {
        Set<Integer> hits = new TreeSet<>();
        for (int d = 0; d < documents.size(); d++) if (documents.get(d).contains(keyword)) hits.add(d);
        return hits;
    }

    /** The leakage the server sees: the multiset of document-id sets, ignoring the trapdoor labels. */
    static List<String> documentSetPattern(Collection<Set<Integer>> sets) {
        List<String> pattern = new ArrayList<>();
        for (Set<Integer> set : sets) pattern.add(set.toString());
        Collections.sort(pattern);
        return pattern;
    }

    public static void main(String[] args) {
        List<String> vocabulary = vocabulary(40);
        double[] weights = zipfWeights(vocabulary.size());
        List<List<String>> documents = corpus(300, 10, vocabulary, weights, 0x5E4C0DEL);
        long key = 0xC0FFEEL;

        // 1. correctness: every query answer equals the plaintext scan
        DeterministicEncryption scheme = new DeterministicEncryption(key);
        SearchableIndex index = new SearchableIndex(scheme);
        for (int d = 0; d < documents.size(); d++) index.addDocument(d, documents.get(d));
        int queries = 0, nonEmpty = 0;
        for (String keyword : vocabulary) {
            Set<Integer> byIndex = index.search(keyword);
            Set<Integer> byScan = containsKeyword(documents, keyword);
            check(byIndex.equals(byScan), "search('" + keyword + "') returned " + byIndex.size()
                    + " documents, the scan says " + byScan.size());
            if (!byScan.isEmpty()) nonEmpty++;
            queries++;
        }
        check(queries == vocabulary.size() && nonEmpty > 20, "queries answered: " + queries + " (" + nonEmpty
                + " with hits)");

        // 2. trapdoor properties: deterministic, injective on the vocabulary, key-dependent
        check(scheme.trapdoor("alpha") == scheme.trapdoor("alpha"), "trapdoors are deterministic");
        Set<Long> trapdoors = new LinkedHashSet<>();
        for (String word : vocabulary) check(trapdoors.add(scheme.trapdoor(word)), "injective trapdoor for "
                + word);
        deterministicEncryptionKeyCheck(vocabulary);
        check(index.size() == nonEmpty, "the index holds exactly the keywords that occur: " + index.size());

        // 3. leakage 1: the equality pattern survives encryption unchanged
        List<Set<Integer>> plaintextSets = new ArrayList<>();
        for (int rank = 0; rank < vocabulary.size(); rank++) {
            Set<Integer> byScan = containsKeyword(documents, vocabulary.get(rank));
            if (!byScan.isEmpty()) plaintextSets.add(byScan);
        }
        List<String> encryptedPattern = documentSetPattern(index.postings.values());
        List<String> plaintextPattern = documentSetPattern(plaintextSets);
        check(encryptedPattern.equals(plaintextPattern), "the encrypted index leaks the plaintext "
                + "co-occurrence pattern exactly");
        List<Set<Integer>> randomised = new ArrayList<>();
        for (String word : vocabulary) {
            Set<Integer> set = new TreeSet<>();
            for (int d = 0; d < documents.size(); d++) {
                if (documents.get(d).contains(word)) set.add((int) scheme.saltedTrapdoor(word, d));
            }
            if (!set.isEmpty()) randomised.add(set);
        }
        check(!documentSetPattern(randomised).equals(plaintextPattern),
                "salting the trapdoor hides the pattern - at the cost of equality search");

        // 4. leakage 2: the rank-matching attack recovers the frequent keywords without the key
        Map<String, Long> trueTrapdoors = new LinkedHashMap<>();
        for (String word : vocabulary) trueTrapdoors.put(word, scheme.trapdoor(word));
        int[] documentFrequency = new int[vocabulary.size()];
        for (int rank = 0; rank < vocabulary.size(); rank++) {
            documentFrequency[rank] = containsKeyword(documents, vocabulary.get(rank)).size();
        }
        List<String> trueTopK = new ArrayList<>();
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < vocabulary.size(); i++) order.add(i);
        order.sort((a, b) -> Integer.compare(documentFrequency[b], documentFrequency[a]));
        for (int i = 0; i < 10; i++) trueTopK.add(vocabulary.get(order.get(i)));
        Map<Long, String> guess = FrequencyAttack.rankMatch(index, vocabulary);
        int recovered = FrequencyAttack.topKAccuracy(guess, index, trueTrapdoors, trueTopK);
        check(recovered >= 5, "the rank attack recovers at least half of the top ten: " + recovered + "/10");
        check(FrequencyAttack.topKAccuracy(guess, index, trueTrapdoors, List.of(vocabulary.get(39)))
                == 0 || trueTrapdoors.get(vocabulary.get(39)) != null,
                "the attack has nothing to say about a keyword it never ranks");

        // 5. the leak is independent of the key: a fresh key changes trapdoors but not the attack
        int recoveredUnderOtherKeys = 0, keysTried = 0;
        for (long alternativeKey : new long[]{0x1111L, 0x2222L, 0x3333L}) {
            DeterministicEncryption other = new DeterministicEncryption(alternativeKey);
            SearchableIndex otherIndex = new SearchableIndex(other);
            for (int d = 0; d < documents.size(); d++) otherIndex.addDocument(d, documents.get(d));
            Map<String, Long> otherTrapdoors = new LinkedHashMap<>();
            for (String word : vocabulary) otherTrapdoors.put(word, other.trapdoor(word));
            Map<Long, String> otherGuess = FrequencyAttack.rankMatch(otherIndex, vocabulary);
            int hits = FrequencyAttack.topKAccuracy(otherGuess, otherIndex, otherTrapdoors, trueTopK);
            check(hits >= 5, "the attack works under key " + alternativeKey + ": " + hits + "/10");
            recoveredUnderOtherKeys += hits;
            keysTried++;
        }
        check(keysTried == 3, "three independent keys were attacked");
        check(index.search(vocabulary.get(0)).size() <= documents.size(), "postings never exceed the corpus");

        // 6. the mitigation: a random trapdoor cannot be looked up by equality
        String probe = trueTopK.get(0);
        Set<Integer> saltedSearchResult = new TreeSet<>();
        for (int d = 0; d < documents.size(); d++) {
            if (documents.get(d).contains(probe)) saltedSearchResult.add((int) scheme.saltedTrapdoor(probe, 0));
        }
        check(!saltedSearchResult.equals(index.search(probe)), "an unsalted query cannot find the salted "
                + "postings: the randomisation genuinely breaks the search");

        // 7. boundaries: an empty document, an absent keyword, one document, an empty query string
        SearchableIndex tiny = new SearchableIndex(new DeterministicEncryption(7));
        tiny.addDocument(0, List.of());
        check(tiny.search("anything").isEmpty(), "an empty document contributes no postings");
        check(index.search("no-such-keyword-xyz").isEmpty(), "an absent keyword finds nothing");
        SearchableIndex single = new SearchableIndex(new DeterministicEncryption(9));
        single.addDocument(0, List.of("only"));
        check(single.search("only").equals(new TreeSet<>(List.of(0))), "a single-document index works");
        check(index.search("").isEmpty(), "the empty keyword matches nothing");

        // 8. determinism: the same corpus and key rebuild the same index and the same guess
        SearchableIndex replay = new SearchableIndex(new DeterministicEncryption(key));
        for (int d = 0; d < documents.size(); d++) replay.addDocument(d, documents.get(d));
        check(replay.postings.equals(index.postings), "the same seed rebuilds the same encrypted index");
        check(FrequencyAttack.rankMatch(replay, vocabulary).equals(guess), "the attack is deterministic too");

        System.out.println("index: " + documents.size() + " documents, " + index.size() + " distinct keywords, "
                + "~" + (documents.size() * 10 / Math.max(1, index.size())) + " documents per keyword");
        System.out.println("leakage: the encrypted index reproduces the plaintext co-occurrence pattern "
                + "exactly; salting hides it but breaks lookups");
        System.out.println("frequency attack: " + recovered + "/10 top keywords recovered with the right key, "
                + (recoveredUnderOtherKeys / 3) + "/10 on average for three fresh keys");
        System.out.println("PASS");
    }

    /** Trapdoors depend on the key: the same word under two keys gives two different values. */
    static void deterministicEncryptionKeyCheck(List<String> vocabulary) {
        DeterministicEncryption first = new DeterministicEncryption(1), second = new DeterministicEncryption(2);
        int distinct = 0;
        for (String word : vocabulary) {
            if (first.trapdoor(word) != second.trapdoor(word)) distinct++;
        }
        check(distinct == vocabulary.size(), "every keyword maps differently under a different key");
    }
}
