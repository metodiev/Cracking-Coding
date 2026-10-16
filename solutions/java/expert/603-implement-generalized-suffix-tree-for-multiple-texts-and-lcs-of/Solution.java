/** Task 603 - Implement generalized suffix tree for multiple texts and LCS of many strings. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The generalized suffix tree is built with one Ukkonen scan over the concatenation
 * t0 # t1 % t2 ... where every text is closed by its own terminator char (1, 2, 3, ... - characters
 * that cannot occur in the texts). Because each terminator occurs exactly once, no suffix of one
 * text is a prefix of a suffix of another, so every suffix of every text ends at a leaf and the
 * tree really is the generalized tree.
 *
 * Each node stores a bitmask of the texts that have a leaf below it. Then, for any subset S of
 * texts with |S| >= 2: longest common substring of S = max depth over nodes whose mask covers S
 * (a point inside an edge has exactly the leaves of the edge's lower endpoint, so node depths
 * suffice) and the number of distinct substrings common to all texts of S = sum over nodes v whose
 * mask covers S of (depth(v) - depth(parent(v))). Singleton subsets are excluded by definition:
 * with one text the whole text is trivially "common". Both quantities are cross-checked against
 * substring-enumeration brute force on small inputs and on 150+ character random texts.
 */
import java.util.*;

class UkkonenTree {
    static final char FIRST_TERMINATOR = 1;

    final String s;                       // concatenated texts, each closed by its own terminator
    final int n;
    final int[] edgeStart, edgeEnd, suffixLink, parent, depth, subtreeLeaves;
    final Map<Character, Integer>[] kids;
    int nodes, leafEnd;

    @SuppressWarnings("unchecked")
    UkkonenTree(String raw) {
        s = raw;
        n = s.length();
        int capacity = Math.max(2, 2 * n + 2);
        edgeStart = new int[capacity]; edgeEnd = new int[capacity]; suffixLink = new int[capacity];
        parent = new int[capacity]; depth = new int[capacity]; subtreeLeaves = new int[capacity];
        kids = new HashMap[capacity];
        int root = makeNode(-1, -1);
        suffixLink[root] = -1;
        int activeNode = root, activeEdge = -1, activeLength = 0, remaining = 0, lastNew = -1;
        for (int i = 0; i < n; i++) {
            if (i > 0 && isTerminator(s.charAt(i - 1))) {
                // The terminator phase made every suffix explicit, so the next text starts fresh at
                // the root instead of continuing the previous text's active point.
                if (remaining != 0 || activeLength != 0) {
                    throw new IllegalStateException("pending suffixes at a text boundary");
                }
                activeNode = root;
                activeEdge = -1;
            }
            leafEnd = i;
            remaining++;
            lastNew = -1;
            while (remaining > 0) {
                if (activeLength == 0) activeEdge = i;
                char c = s.charAt(activeEdge);
                Integer target = kids[activeNode].get(c);
                if (target == null) {
                    kids[activeNode].put(c, makeNode(i, -1));
                    if (lastNew != -1) { suffixLink[lastNew] = activeNode; lastNew = -1; }
                } else {
                    int w = target, edge = edgeLen(w);
                    if (activeLength >= edge) {
                        activeEdge += edge;
                        activeLength -= edge;
                        activeNode = w;
                        continue;
                    }
                    if (s.charAt(edgeStart[w] + activeLength) == s.charAt(i)) {
                        if (lastNew != -1) { suffixLink[lastNew] = activeNode; lastNew = -1; }
                        activeLength++;
                        break;
                    }
                    int mid = makeNode(edgeStart[w], edgeStart[w] + activeLength - 1);
                    kids[activeNode].put(c, mid);
                    kids[mid].put(s.charAt(i), makeNode(i, -1));
                    edgeStart[w] += activeLength;
                    kids[mid].put(s.charAt(edgeStart[w]), w);
                    if (lastNew != -1) suffixLink[lastNew] = mid;
                    lastNew = mid;
                }
                remaining--;
                if (activeNode == root && activeLength > 0) {
                    activeLength--;
                    activeEdge = i - remaining + 1;
                } else if (activeNode != root) {
                    activeNode = suffixLink[activeNode];
                }
            }
        }
        analyse();
    }

    int makeNode(int start, int end) {
        int id = nodes++;
        edgeStart[id] = start; edgeEnd[id] = end; suffixLink[id] = -1; kids[id] = new HashMap<>();
        return id;
    }

    static boolean isTerminator(char c) { return c == FIRST_TERMINATOR || (c > FIRST_TERMINATOR && c < 0x20); }

    int edgeLen(int v) { return (edgeEnd[v] == -1 ? leafEnd : edgeEnd[v]) - edgeStart[v] + 1; }

    boolean isLeaf(int v) { return v != 0 && edgeEnd[v] == -1; }

    /** Breadth-first order (parents before children) plus accumulated leaf counts. */
    int[] analyse() {
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
            subtreeLeaves[v] += isLeaf(v) ? 1 : 0;
            if (parent[v] != -1) subtreeLeaves[parent[v]] += subtreeLeaves[v];
        }
        return order;
    }

    boolean contains(String p) {
        int v = 0, i = 0;
        while (i < p.length()) {
            Integer t = kids[v].get(p.charAt(i));
            if (t == null) return false;
            int w = t, edge = edgeLen(w);
            for (int k = 0; k < edge && i < p.length(); k++, i++) {
                if (s.charAt(edgeStart[w] + k) != p.charAt(i)) return false;
            }
            v = w;
        }
        return true;
    }
}

class GeneralizedSuffixTree {
    final String[] texts;
    final UkkonenTree tree;
    final int[] textMask;                 // per node: texts having at least one leaf below
    final int[] order;                    // breadth-first node order

    GeneralizedSuffixTree(String[] input) {
        texts = input.clone();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < texts.length; i++) {
            sb.append(texts[i]);
            sb.append((char) (UkkonenTree.FIRST_TERMINATOR + i));
        }
        String joined = sb.toString();
        int[] textOf = new int[joined.length()];
        for (int i = 0, at = 0; i < texts.length; i++) {
            for (int k = 0; k <= texts[i].length(); k++) textOf[at++] = i;   // text plus its terminator
        }
        tree = new UkkonenTree(joined);
        order = tree.analyse();
        textMask = new int[tree.nodes];
        for (int i = tree.nodes - 1; i >= 0; i--) {                          // children before parents
            int v = order[i];
            if (tree.isLeaf(v)) textMask[v] = 1 << textOf[tree.n - tree.depth[v]];
            if (tree.parent[v] != -1) textMask[tree.parent[v]] |= textMask[v];
        }
    }

    int nodeCount() { return tree.nodes; }

    int leafCount() { int c = 0; for (int v = 0; v < tree.nodes; v++) if (tree.isLeaf(v)) c++; return c; }

    /** |S| >= 2: the deepest internal node whose leaves cover every text of the subset. */
    int longestCommonSubstring(int subsetMask) {
        int best = 0;
        for (int v = 1; v < tree.nodes; v++) {
            if (!tree.isLeaf(v) && (textMask[v] & subsetMask) == subsetMask) best = Math.max(best, tree.depth[v]);
        }
        return best;
    }

    /** Distinct substrings occurring in every text of the subset: whole edges ending in a covering
     *  node, since every point inside such an edge has exactly that node's leaves below it. */
    long distinctCommonSubstrings(int subsetMask) {
        long total = 0;
        for (int v = 1; v < tree.nodes; v++) {
            if ((textMask[v] & subsetMask) == subsetMask) total += tree.depth[v] - tree.depth[tree.parent[v]];
        }
        return total;
    }

    boolean contains(String p) { return tree.contains(p); }

    /** Every internal node must keep two or more leaves, and the leaf depths must be the suffix
     *  lengths, i.e. exactly the multiset {1..n} shifted by each suffix start. */
    boolean shapeIsValid() {
        boolean[] depthSeen = new boolean[tree.n + 1];
        for (int v = 1; v < tree.nodes; v++) {
            if (tree.isLeaf(v)) depthSeen[tree.depth[v]] = true;
            else if (tree.subtreeLeaves[v] < 2) return false;
        }
        for (int d = 1; d <= tree.n; d++) if (!depthSeen[d]) return false;
        return true;
    }
}

public class Solution {
    private static long state = 0x5EED5EED5EED5EEDL;

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

    static Set<String> substrings(String text) {
        Set<String> found = new HashSet<>();
        for (int i = 0; i < text.length(); i++)
            for (int j = i + 1; j <= text.length(); j++) found.add(text.substring(i, j));
        return found;
    }

    static int bruteLcs(String[] texts) {
        String shortest = texts[0];
        for (String t : texts) if (t.length() < shortest.length()) shortest = t;
        int best = 0;
        for (String sub : substrings(shortest)) {
            if (sub.length() <= best) continue;
            boolean inAll = true;
            for (String t : texts) if (!t.contains(sub)) { inAll = false; break; }
            if (inAll) best = sub.length();
        }
        return best;
    }

    static long bruteDistinctCommon(String[] texts) {
        Set<String> common = substrings(texts[0]);
        for (int i = 1; i < texts.length; i++) common.retainAll(substrings(texts[i]));
        return common.size();
    }

    static int subsetMask(int... ids) {
        int m = 0;
        for (int id : ids) m |= 1 << id;
        return m;
    }

    /** Structural invariants plus LCS / distinct-common cross-checks for every subset of size 2. */
    static void verify(String[] texts) {
        GeneralizedSuffixTree gst = new GeneralizedSuffixTree(texts);
        check(gst.shapeIsValid(), "shape for " + Arrays.toString(texts));
        check(gst.leafCount() == gst.tree.n, "one leaf per suffix");
        for (int i = 0; i < texts.length; i++) {
            for (int start = 0; start < texts[i].length(); start++) {
                check(gst.contains(texts[i].substring(start)), "suffix of text " + i + " present");
            }
        }
        int full = subsetMask(range(texts.length));
        if (texts.length >= 2) {
            check(gst.longestCommonSubstring(full) == bruteLcs(texts), "LCS all " + Arrays.toString(texts));
            check(gst.distinctCommonSubstrings(full) == bruteDistinctCommon(texts), "common count");
        }
        for (int i = 0; i < texts.length; i++) {
            for (int j = i + 1; j < texts.length; j++) {
                int pair = subsetMask(i, j);
                String[] two = {texts[i], texts[j]};
                check(gst.longestCommonSubstring(pair) == bruteLcs(two), "LCS pair " + i + "," + j);
                check(gst.distinctCommonSubstrings(pair) == bruteDistinctCommon(two), "pair count " + i + "," + j);
            }
        }
    }

    static int[] range(int count) {
        int[] ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = i;
        return ids;
    }

    public static void main(String[] args) {
        GeneralizedSuffixTree classic = new GeneralizedSuffixTree(new String[]{"banana", "ananas"});
        check(classic.longestCommonSubstring(subsetMask(0, 1)) == 5, "banana/ananas lcs");
        check(classic.contains("ananas") && classic.contains("nana") && !classic.contains("bn"), "classic search");
        check(classic.shapeIsValid() && classic.leafCount() == classic.tree.n, "classic shape");

        GeneralizedSuffixTree three = new GeneralizedSuffixTree(new String[]{"abcpq", "pqabc", "zzabczz"});
        check(three.longestCommonSubstring(subsetMask(0, 1, 2)) == 3, "abc in all three");
        check(three.distinctCommonSubstrings(subsetMask(0, 1, 2)) == 6, "abc substrings");
        check(three.longestCommonSubstring(subsetMask(0, 1)) == 3, "abcpq/pqabc");
        check(three.longestCommonSubstring(subsetMask(0, 2)) == 3 && three.longestCommonSubstring(subsetMask(1, 2)) == 3,
                "other pairs");

        GeneralizedSuffixTree emptyMember = new GeneralizedSuffixTree(new String[]{"ab", ""});
        check(emptyMember.longestCommonSubstring(subsetMask(0, 1)) == 0, "empty text kills LCS");
        check(emptyMember.leafCount() == 4, "empty text leaves");

        verify(new String[]{"a", "a"});
        verify(new String[]{"ab", "ba"});
        verify(new String[]{"banana", "ananas"});
        verify(new String[]{"mississippi", "sippim", "issippi"});
        verify(new String[]{"abcdef", "cdefab", "defabc", "efabcd"});
        verify(new String[]{"xyz", "abc", "pqr"});
        verify(new String[]{"aaa", "aa", "a"});

        for (int trial = 0; trial < 12; trial++) {
            verify(new String[]{randomText(26, "abc"), randomText(24, "abc"), randomText(20, "ab")});
        }
        for (int trial = 0; trial < 8; trial++) {
            verify(new String[]{randomText(18, "ab"), randomText(20, "abc"), randomText(22, "abcd"),
                                randomText(16, "abcde")});
        }

        String[] many = {randomText(150, "abc"), randomText(140, "abc"), randomText(130, "ab"), randomText(120, "abc")};
        GeneralizedSuffixTree manyTree = new GeneralizedSuffixTree(many);
        int full = subsetMask(range(many.length));
        int lcs = manyTree.longestCommonSubstring(full);
        check(lcs == bruteLcs(many), "large LCS of four texts");
        check(manyTree.distinctCommonSubstrings(full) == bruteDistinctCommon(many), "large common count");
        check(manyTree.shapeIsValid(), "large shape");
        System.out.println("texts / total length : " + many.length + " / " + manyTree.tree.n);
        System.out.println("tree nodes           : " + manyTree.nodeCount());
        System.out.println("LCS of all texts     : " + lcs);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
