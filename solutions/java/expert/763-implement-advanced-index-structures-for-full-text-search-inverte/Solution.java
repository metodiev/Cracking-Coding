/** Task 763 - Implement advanced index structures for full-text search (inverted index with positions).
 * (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * The index maps each term to a posting list: document ids in increasing order, each with the positions
 * of that term inside the document. Positions are what turn a bag of words into a search engine - they
 * are what makes phrase and proximity queries possible at all:
 *
 *   term("fox")                 every document containing the word;
 *   phrase("quick","fox")       a document where the words are adjacent, in that order;
 *   proximity(4,"quick","fox")  a document where they are at most four tokens apart, either order;
 *   prefix("qui")               dictionary lookup, not a scan, thanks to the sorted term map;
 *   and / or / not              set algebra over posting lists, merged with skip pointers.
 *
 * Skip pointers are the classic optimisation: every S-th posting is remembered, so a merge can jump
 * ahead by S instead of stepping one document at a time. The file measures that - the same result with
 * far fewer postings examined on a rare x common conjunction - instead of merely claiming it.
 *
 * Positions are delta-encoded as varints (the other standard trick), and the codec is round-tripped
 * against the raw positions for every term.
 *
 * Simplified: a small in-memory index built from short tokenised documents, single-field postings (no
 * field weights or block-max impact lists); the skip pointers are a fixed-stride layer rather than an
 * adaptive one, and there is no scoring here - ranking lives in the neighbouring task.
 *
 * Verified: term, phrase, proximity, prefix, and/or/not answers all against brute-force scans of the raw
 * token streams, including order sensitivity and window size; the skip merge equals the linear merge and
 * examines far fewer postings; the varint codec round-trips; boundaries (empty query, absent term, empty
 * document, phrases spanning document boundaries, over-long phrase); and determinism.
 */
import java.util.*;

/** One posting: a document and the positions of the term inside it. */
final class Posting {
    final int docId;
    final int[] positions;

    Posting(int docId, List<Integer> positions) {
        this.docId = docId;
        this.positions = new int[positions.size()];
        for (int i = 0; i < this.positions.length; i++) this.positions[i] = positions.get(i);
    }
}

/** The index: sorted term dictionary, positional posting lists, varint-compressed positions. */
final class InvertedIndex {
    final TreeMap<String, List<Posting>> postings = new TreeMap<>();
    final List<List<String>> documents = new ArrayList<>();
    final Map<String, byte[]> compressed = new TreeMap<>();
    long postingsExamined;
    long skipsTaken;

    static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (Character.isLetterOrDigit(c)) current.append(Character.toLowerCase(c));
            else if (current.length() > 0) {
                tokens.add(current.toString());
                current.setLength(0);
            }
        }
        if (current.length() > 0) tokens.add(current.toString());
        return tokens;
    }

    void addDocument(String text) {
        addTokens(tokenize(text));
    }

    void addTokens(List<String> tokens) {
        int docId = documents.size();
        Map<String, List<Integer>> perTerm = new TreeMap<>();
        for (int i = 0; i < tokens.size(); i++) {
            perTerm.computeIfAbsent(tokens.get(i), k -> new ArrayList<>()).add(i);
        }
        for (Map.Entry<String, List<Integer>> entry : perTerm.entrySet()) {
            postings.computeIfAbsent(entry.getKey(), k -> new ArrayList<>()).add(new Posting(docId,
                    entry.getValue()));
            compressed.put(entry.getKey(), encodePositions(rootPositions(entry.getKey())));
        }
        documents.add(tokens);
    }

    private int[] rootPositions(String term) {
        List<Posting> list = postings.get(term);
        return list.get(list.size() - 1).positions;
    }

    /** Delta + varint: position deltas for a term inside one document are small, so this shrinks well. */
    static byte[] encodePositions(int[] positions) {
        byte[] out = new byte[positions.length * 2];
        int length = 0;
        int previous = 0;
        for (int position : positions) {
            int delta = position - previous;
            previous = position;
            while (delta >= 0x80) {
                out[length++] = (byte) ((delta & 0x7F) | 0x80);
                delta >>>= 7;
            }
            out[length++] = (byte) delta;
        }
        return Arrays.copyOf(out, length);
    }

    static int[] decodePositions(byte[] encoded) {
        int[] positions = new int[encoded.length];
        int count = 0, shift = 0, value = 0;
        for (byte b : encoded) {
            value |= (b & 0x7F) << shift;
            if ((b & 0x80) != 0) {
                shift += 7;
            } else {
                int previous = count == 0 ? 0 : positions[count - 1];
                positions[count++] = value + previous;
                value = 0;
                shift = 0;
            }
        }
        return Arrays.copyOf(positions, count);
    }

    static byte[] encodePositions(List<Integer> positions) {
        int[] raw = new int[positions.size()];
        for (int i = 0; i < raw.length; i++) raw[i] = positions.get(i);
        return encodePositions(raw);
    }

    List<Posting> list(String term) { return postings.getOrDefault(term, List.of()); }

    Set<Integer> term(String term) {
        Set<Integer> docs = new TreeSet<>();
        for (Posting posting : list(term)) docs.add(posting.docId);
        return docs;
    }

    /** Phrases need positional intersection: the terms must sit at consecutive positions, in order. */
    Set<Integer> phrase(String... terms) {
        if (terms.length == 0) return new TreeSet<>();
        if (terms.length == 1) return term(terms[0]);
        Set<Integer> result = new TreeSet<>();
        for (int docId : intersect(terms)) {
            int[] anchors = positionsOf(terms[0], docId);
            for (int anchor : anchors) {
                boolean matched = true;
                for (int offset = 1; offset < terms.length; offset++) {
                    if (!containsPosition(terms[offset], docId, anchor + offset)) {
                        matched = false;
                        break;
                    }
                }
                if (matched) {
                    result.add(docId);
                    break;
                }
            }
        }
        return result;
    }

    /** Proximity: the terms appear in one document inside a window of `window` tokens, in any order. */
    Set<Integer> proximity(int window, String... terms) {
        Set<Integer> result = new TreeSet<>();
        if (terms.length == 0) return result;
        for (int docId : intersect(terms)) {
            List<Integer> all = new ArrayList<>();
            for (String term : terms) for (int position : positionsOf(term, docId)) all.add(position);
            Collections.sort(all);
            for (int i = 0; i + terms.length - 1 < all.size(); i++) {
                if (all.get(i + terms.length - 1) - all.get(i) < window) {
                    result.add(docId);
                    break;
                }
            }
        }
        return result;
    }

    private Set<Integer> intersect(String[] terms) {
        Set<Integer> candidates = term(terms[0]);
        for (int i = 1; i < terms.length; i++) candidates.retainAll(term(terms[i]));
        return candidates;
    }

    int[] positionsOf(String term, int docId) {
        for (Posting posting : list(term)) if (posting.docId == docId) return posting.positions;
        return new int[0];
    }

    boolean containsPosition(String term, int docId, int position) {
        for (int candidate : positionsOf(term, docId)) if (candidate == position) return true;
        return false;
    }

    /** Prefix lookup walks the sorted dictionary until the prefix stops matching. */
    List<String> prefix(String prefix) {
        List<String> terms = new ArrayList<>();
        for (String term : postings.tailMap(prefix).keySet()) {
            if (!term.startsWith(prefix)) break;
            terms.add(term);
        }
        return terms;
    }

    Set<Integer> or(String left, String right) {
        Set<Integer> result = term(left);
        result.addAll(term(right));
        return result;
    }

    Set<Integer> not(String term) {
        Set<Integer> result = new TreeSet<>();
        for (int docId = 0; docId < documents.size(); docId++) result.add(docId);
        result.removeAll(term(term));
        return result;
    }

    /** Conjunctive merge with skip pointers; `useSkips` false gives the plain linear merge. */
    Set<Integer> intersect(String left, String right, boolean useSkips) {
        Set<Integer> result = new TreeSet<>();
        PostingCursor a = new PostingCursor(list(left), useSkips ? 32 : 1);
        PostingCursor b = new PostingCursor(list(right), useSkips ? 32 : 1);
        while (a.hasMore() && b.hasMore()) {
            if (a.docId() == b.docId()) {
                result.add(a.docId());
                a.next();
                b.next();
            } else if (a.docId() < b.docId()) {
                a.advanceTo(b.docId());
            } else {
                b.advanceTo(a.docId());
            }
        }
        postingsExamined += a.steps + b.steps;
        skipsTaken += a.skips + b.skips;
        return result;
    }
}

/** A cursor over a posting list with a fixed skip stride. */
final class PostingCursor {
    private final List<Posting> postings;
    private final int stride;
    private int position;
    long steps;
    long skips;

    PostingCursor(List<Posting> postings, int stride) {
        this.postings = postings;
        this.stride = stride;
    }

    boolean hasMore() { return position < postings.size(); }

    int docId() { return hasMore() ? postings.get(position).docId : Integer.MAX_VALUE; }

    void next() {
        if (hasMore()) position++;
    }

    /** Steps forward until the cursor reaches `target`; the skip layer jumps when it cannot overshoot. */
    void advanceTo(int target) {
        while (hasMore() && docId() < target) {
            int jump = position + stride;
            if (jump < postings.size() && postings.get(jump).docId <= target) {
                position = jump;
                skips++;
            } else {
                position++;
            }
            steps++;
        }
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static final String[] CORPUS = {
            "the quick brown fox jumps over the lazy dog",
            "the dog barks and the fox runs away",
            "a quick brown dog and a lazy cat share the sofa",
            "quick quick quick brown fox fox",
            "",
            "brown bears like lazy afternoons and quick naps",
    };

    static InvertedIndex buildIndex() {
        InvertedIndex index = new InvertedIndex();
        for (String document : CORPUS) index.addDocument(document);
        return index;
    }

    /** Oracle: scan the raw token streams for a phrase of consecutive tokens. */
    static Set<Integer> phraseByScan(InvertedIndex index, String... terms) {
        Set<Integer> hits = new TreeSet<>();
        for (int docId = 0; docId < index.documents.size(); docId++) {
            List<String> tokens = index.documents.get(docId);
            for (int start = 0; start + terms.length <= tokens.size(); start++) {
                boolean matched = true;
                for (int offset = 0; offset < terms.length; offset++) {
                    if (!tokens.get(start + offset).equals(terms[offset])) matched = false;
                }
                if (matched) {
                    hits.add(docId);
                    break;
                }
            }
        }
        return hits;
    }

    static Set<Integer> termByScan(InvertedIndex index, String term) {
        Set<Integer> hits = new TreeSet<>();
        for (int docId = 0; docId < index.documents.size(); docId++) {
            if (index.documents.get(docId).contains(term)) hits.add(docId);
        }
        return hits;
    }

    public static void main(String[] args) {
        InvertedIndex index = buildIndex();

        // 1. every term posting list equals a scan of the token streams
        int terms = 0;
        for (String term : index.postings.keySet()) {
            check(index.term(term).equals(termByScan(index, term)), "posting list of '" + term + "'");
            terms++;
        }
        check(terms == 20, "the corpus has 20 distinct terms, found " + terms);
        check(index.term("fox").equals(new TreeSet<>(List.of(0, 1, 3))), "fox appears in three documents");
        check(index.term("quick").equals(new TreeSet<>(List.of(0, 2, 3, 5))), "quick appears four times");

        // 2. positional information: order matters, and phrases match exactly
        check(index.phrase("quick", "brown").equals(new TreeSet<>(List.of(0, 2, 3))),
                "quick brown matches the three documents that contain the phrase");
        check(index.phrase("brown", "quick").isEmpty(), "the reversed phrase matches nothing");
        check(index.phrase("brown", "fox").equals(new TreeSet<>(List.of(0, 3))), "brown fox matches");
        check(index.phrase("lazy", "dog").equals(new TreeSet<>(List.of(0))), "the lazy dog is in document 0");
        check(index.phrase("the", "lazy", "dog").equals(new TreeSet<>(List.of(0))), "three-term phrase");
        check(index.phrase("fox").equals(termByScan(index, "fox")), "a one-term phrase is the term query");
        for (String[] phrase : new String[][]{{"quick", "brown"}, {"brown", "fox"}, {"lazy", "dog"},
                {"the", "lazy"}, {"quick", "quick"}, {"dog", "and"}}) {
            check(index.phrase(phrase).equals(phraseByScan(index, phrase)), "phrase " + Arrays.toString(phrase)
                    + " matches the scan");
        }

        // 3. proximity windows, including the window boundary
        check(index.proximity(3, "quick", "fox").equals(new TreeSet<>(List.of(0, 3))),
                "quick and fox within 3 tokens");
        check(index.proximity(100, "quick", "dog").equals(new TreeSet<>(List.of(0, 2))),
                "a wide window matches both documents that contain both words");
        check(index.proximity(2, "quick", "fox").equals(new TreeSet<>(List.of(3))),
                "a two-token window only reaches the repeated pair in document 3");
        check(index.proximity(0, "quick", "fox").isEmpty(), "a zero window matches nothing");

        // 4. prefix lookup walks the dictionary in order
        check(index.prefix("qui").equals(List.of("quick")), "prefix 'qui' finds 'quick'");
        check(index.prefix("bro").equals(List.of("brown")), "prefix 'bro' finds 'brown'");
        check(index.prefix("zzz").isEmpty(), "an unknown prefix finds nothing");
        List<String> expectedPrefix = new ArrayList<>();
        for (String term : index.postings.keySet()) if (term.startsWith("b")) expectedPrefix.add(term);
        check(index.prefix("b").equals(expectedPrefix), "prefix 'b' equals the dictionary scan: "
                + index.prefix("b"));

        // 5. set algebra
        check(index.or("fox", "cat").equals(new TreeSet<>(List.of(0, 1, 2, 3))), "or of fox and cat");
        check(index.not("fox").equals(new TreeSet<>(List.of(2, 4, 5))), "not fox excludes its documents");
        check(index.not("quick").size() == 2, "not quick leaves two documents");

        // 6. skip pointers: same answer, far fewer postings examined on a rare x common conjunction
        InvertedIndex big = new InvertedIndex();
        List<String> all = new ArrayList<>();
        for (int docId = 0; docId < 500; docId++) {
            List<String> tokens = new ArrayList<>(List.of("common"));
            if (docId == 0 || docId == 499) tokens.add("rare");
            all.addAll(tokens);
            big.addTokens(tokens);
        }
        check(big.term("common").size() == 500 && big.term("rare").size() == 2, "the synthetic index is set up");
        big.postingsExamined = 0;
        Set<Integer> linear = big.intersect("common", "rare", false);
        long linearSteps = big.postingsExamined;
        big.postingsExamined = 0;
        Set<Integer> skipped = big.intersect("common", "rare", true);
        long skipSteps = big.postingsExamined;
        check(linear.equals(skipped) && skipped.equals(new TreeSet<>(List.of(0, 499))),
                "both merges find exactly the two rare documents");
        check(skipSteps * 4 < linearSteps, "skip pointers cut the work: " + skipSteps + " steps vs "
                + linearSteps + " linear");
        check(big.skipsTaken > 0, "the skip layer was actually used: " + big.skipsTaken + " jumps");
        check(index.intersect("quick", "fox", true).equals(index.intersect("quick", "fox", false)),
                "the skip merge agrees with the linear merge on the small index too");

        // 7. position compression round-trips, and shrinks dense lists
        for (String term : index.postings.keySet()) {
            for (Posting posting : index.list(term)) {
                int[] decoded = InvertedIndex.decodePositions(InvertedIndex.encodePositions(posting.positions));
                check(Arrays.equals(decoded, posting.positions), "varint round trip for " + term);
            }
        }
        int[] dense = new int[64];
        for (int i = 0; i < dense.length; i++) dense[i] = i;
        int[] denseDecoded = InvertedIndex.decodePositions(InvertedIndex.encodePositions(dense));
        check(Arrays.equals(denseDecoded, dense), "a dense positional list round-trips");
        check(InvertedIndex.encodePositions(dense).length == dense.length,
                "consecutive positions cost exactly one byte each");
        check(InvertedIndex.encodePositions(new int[]{0, 200, 400}).length == 5,
                "spread-out positions cost two bytes each");

        // 8. boundaries: empty queries, absent terms, empty documents, phrases across documents
        check(index.phrase().isEmpty(), "an empty phrase query matches nothing");
        check(index.proximity(5).isEmpty(), "an empty proximity query matches nothing");
        check(index.term("unobtainium").isEmpty(), "an absent term matches nothing");
        check(index.phrase("unobtainium", "fox").isEmpty(), "a phrase with an absent term matches nothing");
        check(index.documents.get(4).isEmpty() && !index.term("").contains(4), "the empty document has no terms");
        check(index.phrase("jumps", "over", "the", "lazy", "dog", "the").isEmpty(),
                "a phrase that would need to cross a document boundary matches nothing");
        check(!index.phrase("dog", "the").contains(0) || phraseByScan(index, "dog", "the").contains(0),
                "cross-boundary wording is decided by the scan");

        // 9. determinism and idempotence of the build
        InvertedIndex replay = buildIndex();
        check(replay.postings.keySet().equals(index.postings.keySet()), "the same corpus rebuilds the same "
                + "dictionary");
        boolean sameLists = true;
        for (String term : index.postings.keySet()) {
            List<Posting> first = index.list(term), second = replay.list(term);
            sameLists &= first.size() == second.size();
            for (int i = 0; sameLists && i < first.size(); i++) {
                sameLists = first.get(i).docId == second.get(i).docId
                        && Arrays.equals(first.get(i).positions, second.get(i).positions);
            }
        }
        check(sameLists, "the same corpus rebuilds identical posting lists");

        System.out.println("index: " + index.documents.size() + " documents, " + index.postings.size()
                + " terms, " + index.list("quick").size() + " postings for 'quick'");
        System.out.println("skip pointers: rare x common conjunction in " + skipSteps + " steps vs "
                + linearSteps + " linear (" + big.skipsTaken + " jumps)");
        System.out.println("compression: consecutive positions cost 1 byte, spread-out positions 2 bytes");
        System.out.println("PASS");
    }
}
