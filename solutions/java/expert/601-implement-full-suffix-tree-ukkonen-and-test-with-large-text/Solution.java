/** Task 601 - Implement full suffix tree (Ukkonen) and test with large text. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Ukkonen's online construction: one left-to-right scan, O(n) amortised, carrying the active point
 * (activeNode, activeEdge, activeLength), the pending suffix count (remaining) and one shared open
 * leaf end. A terminator char absent from the text makes every suffix end at a leaf.
 *
 * Checked invariants: one leaf per suffix, leaf depths exactly 1..N; non-root internal nodes have
 * >= 2 leaves below and nodes <= 2N - 1; each internal node's suffix link spells that node's path
 * minus its first character (string comparison); sum over nodes of (depth(v) - depth(parent(v)))
 * minus N equals the distinct substring count, verified against an independent suffix-array + LCP
 * oracle on a 4000 character text and against HashSet / naive brute force on small texts.
 */
import java.util.*;

class UkkonenTree {
    static final char TERMINATOR = '\u0001';
    final int[] edgeStart, edgeEnd, suffixLink;   // label = s[edgeStart..edgeEnd]; end == -1 marks
    final int[] parent, depth, subtreeLeaves;     // a leaf whose end is the current leafEnd
    final Map<Character, Integer>[] kids;
    final String s;
    final int n;
    int nodes, leafEnd;
    boolean analysed;

    @SuppressWarnings("unchecked")
    UkkonenTree(String raw) {
        s = raw + TERMINATOR;
        n = s.length();
        int capacity = Math.max(2, 2 * n + 2);
        edgeStart = new int[capacity]; edgeEnd = new int[capacity]; suffixLink = new int[capacity];
        parent = new int[capacity]; depth = new int[capacity]; subtreeLeaves = new int[capacity];
        kids = new HashMap[capacity];
        int root = makeNode(-1, -1);
        suffixLink[root] = -1;
        int activeNode = root, activeEdge = -1, activeLength = 0, remaining = 0, lastNew = -1;
        for (int i = 0; i < n; i++) {
            leafEnd = i;
            remaining++;
            lastNew = -1;
            while (remaining > 0) {
                if (activeLength == 0) activeEdge = i;
                char c = s.charAt(activeEdge);
                Integer target = kids[activeNode].get(c);
                if (target == null) {                       // new leaf hanging off activeNode
                    kids[activeNode].put(c, makeNode(i, -1));
                    if (lastNew != -1) { suffixLink[lastNew] = activeNode; lastNew = -1; }
                } else {
                    int w = target, edge = edgeLen(w);
                    if (activeLength >= edge) {             // active point below w: walk down
                        activeEdge += edge;
                        activeLength -= edge;
                        activeNode = w;
                        continue;
                    }
                    if (s.charAt(edgeStart[w] + activeLength) == s.charAt(i)) {
                        if (lastNew != -1) { suffixLink[lastNew] = activeNode; lastNew = -1; }
                        activeLength++;                     // rule 3: implicit extension only
                        break;
                    }
                    int mid = makeNode(edgeStart[w], edgeStart[w] + activeLength - 1);
                    kids[activeNode].put(c, mid);           // split w's edge at the active point
                    kids[mid].put(s.charAt(i), makeNode(i, -1));
                    edgeStart[w] += activeLength;
                    kids[mid].put(s.charAt(edgeStart[w]), w);
                    if (lastNew != -1) suffixLink[lastNew] = mid;
                    lastNew = mid;
                }
                remaining--;
                if (activeNode == root && activeLength > 0) {
                    activeLength--;                         // no suffix link leaves the root
                    activeEdge = i - remaining + 1;
                } else if (activeNode != root) {
                    activeNode = suffixLink[activeNode];
                }
            }
        }
        if (remaining != 0 || activeLength != 0) throw new IllegalStateException("incomplete scan");
    }

    int makeNode(int start, int end) {
        int id = nodes++;
        edgeStart[id] = start; edgeEnd[id] = end; suffixLink[id] = -1; kids[id] = new HashMap<>();
        return id;
    }

    int edgeLen(int v) { return (edgeEnd[v] == -1 ? leafEnd : edgeEnd[v]) - edgeStart[v] + 1; }

    /** The root carries the open-edge marker as well, so it must be excluded explicitly. */
    boolean isLeaf(int v) { return v != 0 && edgeEnd[v] == -1; }

    /** Breadth-first pass filling parent/depth, then a reverse pass accumulating leaf counts. */
    void analyse() {
        if (analysed) return;
        int[] order = new int[nodes];
        int head = 0, tail = 1;
        parent[0] = -1;
        while (head < tail) {
            int v = order[head++];
            for (int u : kids[v].values()) {
                parent[u] = v;
                depth[u] = depth[v] + edgeLen(u);
                order[tail++] = u;
            }
        }
        for (int i = nodes - 1; i >= 0; i--) {
            int v = order[i];
            subtreeLeaves[v] += isLeaf(v) ? 1 : 0;          // children already pushed their counts up
            if (parent[v] != -1) subtreeLeaves[parent[v]] += subtreeLeaves[v];
        }
        analysed = true;
    }

    int leafCount() { int c = 0; for (int v = 0; v < nodes; v++) if (isLeaf(v)) c++; return c; }

    /** Node where a match of p ends (possibly inside an edge), or -1 when p is absent. */
    int locus(String p) {
        int v = 0, i = 0;
        while (i < p.length()) {
            Integer t = kids[v].get(p.charAt(i));
            if (t == null) return -1;
            int w = t, edge = edgeLen(w);
            for (int k = 0; k < edge && i < p.length(); k++, i++) {
                if (s.charAt(edgeStart[w] + k) != p.charAt(i)) return -1;
            }
            v = w;
        }
        return v;
    }

    boolean contains(String p) { return locus(p) != -1; }

    int occurrences(String p) {
        analyse();
        int v = locus(p);
        return v == -1 ? 0 : subtreeLeaves[v];
    }

    /** Every node and every position strictly inside an edge spells one distinct substring: the
     *  geometric sum minus the N substrings that contain the terminator. */
    long distinctSubstrings() {
        analyse();
        long sum = 0;
        for (int v = 1; v < nodes; v++) sum += depth[v] - depth[parent[v]];
        return sum - n;
    }

    int longestRepeatedLength() {
        analyse();
        int best = 0;
        for (int v = 1; v < nodes; v++) if (!isLeaf(v) && subtreeLeaves[v] >= 2) best = Math.max(best, depth[v]);
        return best;
    }

    String pathOf(int v) {
        analyse();
        char[] buf = new char[depth[v]];
        int at = depth[v];
        for (int cur = v; parent[cur] != -1; cur = parent[cur]) {
            int len = edgeLen(cur);
            at -= len;
            for (int k = 0; k < len; k++) buf[at + k] = s.charAt(edgeStart[cur] + k);
        }
        return new String(buf);
    }

    /** Every non-root internal node links to the node spelling its path minus the first character:
     *  the depth relation always, the full string comparison only when strict (small texts). */
    boolean suffixLinksValid(boolean strict) {
        analyse();
        for (int v = 1; v < nodes; v++) {
            if (isLeaf(v)) continue;
            int l = suffixLink[v];
            if (l == -1 || isLeaf(l) || depth[l] != depth[v] - 1) return false;
            if (strict && !pathOf(l).equals(pathOf(v).substring(1))) return false;
        }
        return true;
    }
}

/** Independent oracle: suffix array by prefix doubling; distinct substrings from Kasai's LCP array
 *  (each suffix contributes its length minus the LCP with its sorted predecessor). */
class SuffixArrayOracle {
    static int[] doubling(String s) {
        int n = s.length();
        if (n == 0) return new int[0];
        Integer[] sa = new Integer[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) { sa[i] = i; rank[i] = s.charAt(i); }
        for (int k = 1; ; k <<= 1) {
            final int step = k;
            Arrays.sort(sa, (a, b) -> {
                if (rank[a] != rank[b]) return Integer.compare(rank[a], rank[b]);
                int ra = a + step < n ? rank[a + step] : -1;
                int rb = b + step < n ? rank[b + step] : -1;
                return Integer.compare(ra, rb);
            });
            int[] next = new int[n];
            for (int i = 1; i < n; i++) {
                int prev = sa[i - 1], cur = sa[i];
                int pa = prev + step < n ? rank[prev + step] : -1;
                int pb = cur + step < n ? rank[cur + step] : -1;
                next[cur] = next[prev] + (rank[prev] != rank[cur] || pa != pb ? 1 : 0);
            }
            System.arraycopy(next, 0, rank, 0, n);
            if (rank[sa[n - 1]] == n - 1) break;
        }
        return Arrays.stream(sa).mapToInt(Integer::intValue).toArray();
    }

    static long distinctSubstrings(String s) {
        int n = s.length();
        int[] sa = doubling(s), rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        long distinct = 0;
        int h = 0;
        for (int i = 0; i < n; i++) {
            if (rank[i] > 0) {
                int j = sa[rank[i] - 1];
                while (i + h < n && j + h < n && s.charAt(i + h) == s.charAt(j + h)) h++;
            } else {
                h = 0;
            }
            distinct += n - i - h;
            if (h > 0) h--;
        }
        return distinct;
    }
}

public class Solution {
    private static long state = 0x2545F4914F6CDD1DL;

    /** splitmix64: deterministic, and only high bits are used so short periods cannot leak in. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static String randomText(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) sb.append(alphabet.charAt((int) ((nextLong() >>> 33) % alphabet.length())));
        return sb.toString();
    }

    static Set<String> allSubstrings(String text) {
        Set<String> found = new HashSet<>();
        for (int i = 0; i < text.length(); i++)
            for (int j = i + 1; j <= text.length(); j++) found.add(text.substring(i, j));
        return found;
    }

    static int bruteOccurrences(String text, String pattern) {
        int count = 0;
        for (int i = 0; i + pattern.length() <= text.length(); i++) if (text.startsWith(pattern, i)) count++;
        return count;
    }

    /** Leaves are counted per suffix, internal nodes must keep at least two of them. */
    static void checkShape(UkkonenTree tree, String label) {
        tree.analyse();
        check(tree.leafCount() == tree.n, "leaves == N for " + label);
        check(tree.nodes <= Math.max(2, 2 * tree.n - 1), "node bound for " + label);
        boolean[] seenDepth = new boolean[tree.n + 1];
        for (int v = 1; v < tree.nodes; v++) {
            if (tree.isLeaf(v)) seenDepth[tree.depth[v]] = true;      // leaf depth == suffix length
            else check(tree.subtreeLeaves[v] >= 2, "internal node has >= 2 leaves in " + label);
        }
        for (int d = 1; d <= tree.n; d++) check(seenDepth[d], "leaf depths are 1..N for " + label);
    }

    /** Structural invariants plus full brute-force cross-checks for an enumerable text. */
    static void verifySmall(String text) {
        UkkonenTree tree = new UkkonenTree(text);
        checkShape(tree, text);
        Set<String> subs = allSubstrings(text);
        check(tree.distinctSubstrings() == subs.size(), "distinct brute " + text);
        check(tree.distinctSubstrings() == SuffixArrayOracle.distinctSubstrings(text), "oracle " + text);
        check(tree.suffixLinksValid(true), "links " + text);
        for (int start = 0; start < text.length(); start++)
            for (int end = start + 1; end <= text.length(); end++) {
                String pattern = text.substring(start, end);
                check(tree.contains(pattern) && tree.occurrences(pattern) == bruteOccurrences(text, pattern),
                        "window " + text + " / " + pattern);
            }
        check(!tree.contains("zzz") && tree.occurrences("zzz") == 0, "absent pattern " + text);
    }

    /** Large-text battery: structure, distinct count against the oracle, sampled window counts and
     *  a near-miss pattern that brute force confirms is really missing. */
    static long checkLarge(String text) {
        UkkonenTree tree = new UkkonenTree(text);
        checkShape(tree, "large");
        long distinct = tree.distinctSubstrings();
        check(distinct == SuffixArrayOracle.distinctSubstrings(text), "large distinct vs SA+LCP");
        check(tree.suffixLinksValid(false), "large suffix links");
        for (int i = 0; i < text.length(); i += 53) {
            String pattern = text.substring(i, Math.min(text.length(), i + 13));
            check(tree.contains(pattern) && tree.occurrences(pattern) == bruteOccurrences(text, pattern),
                    "large window " + i);
        }
        String absent = null;
        for (int k = 0, n = Math.min(40, text.length()); k < n && absent == null; k++) {
            char[] probe = text.substring(0, n).toCharArray();
            probe[k] = probe[k] == 'a' ? 'b' : 'a';
            String candidate = new String(probe);
            if (bruteOccurrences(text, candidate) == 0) absent = candidate;
        }        check(absent != null, "large text admits a near-miss absent pattern");
        check(!tree.contains(absent) && !tree.contains("c"), "large rejects absent patterns");
        System.out.println("distinct substrings : " + distinct + "   nodes: " + tree.nodes);
        return distinct;
    }

    public static void main(String[] args) {
        UkkonenTree banana = new UkkonenTree("banana");
        check(banana.leafCount() == 7 && banana.distinctSubstrings() == 15
                && banana.longestRepeatedLength() == 3, "banana size/repeat");
        check(banana.suffixLinksValid(true) && banana.contains("ban") && banana.contains("nana")
                && !banana.contains("bn") && !banana.contains("bananas"), "banana accept/reject, links");
        check(banana.occurrences("an") == 2 && banana.occurrences("ana") == 2
                && banana.occurrences("banana") == 1 && banana.occurrences("bananas") == 0, "banana counts");
        UkkonenTree empty = new UkkonenTree("");
        check(empty.leafCount() == 1 && empty.distinctSubstrings() == 0 && !empty.contains("a")
                && empty.contains("\u0001"), "empty text rejects text chars");
        check(new UkkonenTree("a").distinctSubstrings() == 1, "single char");
        String[] small = {"a", "b", "aa", "ab", "ba", "aaaa", "abab", "abc", "banana",
                          "mississippi", "abracadabra", "aaabaaaab", "abcabcabc"};
        for (String text : small) verifySmall(text);
        for (String alphabet : new String[]{"ab", "abc"}) {
            for (int trial = 0; trial < 25; trial++) verifySmall(randomText(30, alphabet));
        }
        UkkonenTree mediumTree = new UkkonenTree(randomText(300, "abcdef"));
        String medium = mediumTree.s.substring(0, mediumTree.n - 1);   // s carries the terminator
        check(mediumTree.distinctSubstrings() == SuffixArrayOracle.distinctSubstrings(medium), "medium distinct");
        check(mediumTree.suffixLinksValid(true), "medium links");
        check(checkLarge(randomText(4000, "ab")) > 0, "large distinct count positive");
        UkkonenTree ones = new UkkonenTree("a".repeat(1500));
        checkShape(ones, "aaaa");
        check(ones.distinctSubstrings() == 1500 && ones.longestRepeatedLength() == 1499, "aaaa repeats");
        check(ones.occurrences("aaa") == 1498 && ones.suffixLinksValid(true), "aaaa occurrences/links");
        check(ones.nodes == 2 * 1501 - 1 && ones.nodes - ones.leafCount() == 1500, "aaaa shape");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
