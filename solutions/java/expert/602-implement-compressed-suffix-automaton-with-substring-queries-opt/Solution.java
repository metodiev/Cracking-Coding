/** Task 602 - Implement compressed suffix automaton with substring queries optimized. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * The suffix automaton is the minimal DFA of all suffixes: at most 2n - 1 states and 3n - 4
 * transitions, built incrementally with suffix-link redirection and state cloning. "Compressed"
 * here is the storage form: transitions are not a hash map per state but one packed array of
 * (char, target) pairs in increasing character order plus CSR offsets, so a step costs
 * O(log out-degree) by binary search and the table is a single flat int/char block.
 *
 * Optimized queries: contains / occurrences / firstOccurrence in O(m log sigma); distinct substring
 * count as sum(len[v] - len[link[v]]); longest repeated substring as the deepest state with two or
 * more end positions; longest common substring with another text by a matching-statistics walk that
 * amortises the suffix-link climbs. Every answer is cross-checked against brute force, and the
 * packed layout is verified to be sorted per state.
 */
import java.util.*;

class CompressedSuffixAutomaton {
    final int[] len, link, occ, minEnd;
    final char[] edgeChar;                // packed transitions, grouped by state, chars ascending
    final int[] edgeTo;
    final int[] offset;                   // state s owns edgeChar[offset[s] .. offset[s+1])
    final int states, edges;

    @SuppressWarnings("unchecked")
    CompressedSuffixAutomaton(String text) {
        int n = text.length();

        // Phase 1: build the automaton with one TreeMap per state (sorted transition order).
        int capacity = Math.max(2, 2 * n + 1);
        int[] stateLen = new int[capacity], stateLink = new int[capacity];
        int[] stateOcc = new int[capacity], stateMinEnd = new int[capacity], degree = new int[capacity];
        TreeMap<Character, Integer>[] next = new TreeMap[capacity];
        for (int i = 0; i < capacity; i++) next[i] = new TreeMap<>();
        stateLink[0] = -1;
        int count = 1, last = 0;
        for (int i = 0; i < n; i++) {
            char c = text.charAt(i);
            int cur = count++;
            stateLen[cur] = stateLen[last] + 1;
            stateOcc[cur] = 1;
            stateMinEnd[cur] = i;
            int p = last;
            while (p != -1 && !next[p].containsKey(c)) {
                next[p].put(c, cur);
                p = stateLink[p];
            }
            if (p == -1) {
                stateLink[cur] = 0;
            } else {
                int q = next[p].get(c);
                if (stateLen[p] + 1 == stateLen[q]) {
                    stateLink[cur] = q;
                } else {
                    int clone = count++;
                    stateLen[clone] = stateLen[p] + 1;
                    stateLink[clone] = stateLink[q];
                    stateMinEnd[clone] = stateMinEnd[q];
                    next[clone].putAll(next[q]);
                    while (p != -1 && next[p].get(c) != null && next[p].get(c) == q) {
                        next[p].put(c, clone);
                        p = stateLink[p];
                    }
                    stateLink[q] = clone;
                    stateLink[cur] = clone;
                }
            }
            last = cur;
        }
        // Propagate end-position counts along suffix links, deepest first.
        Integer[] byLen = new Integer[count - 1];
        for (int v = 1; v < count; v++) byLen[v - 1] = v;
        Arrays.sort(byLen, (a, b) -> stateLen[b] - stateLen[a]);
        for (int v : byLen) stateOcc[stateLink[v]] += stateOcc[v];

        // Phase 2: pack the transition table into CSR form.
        int total = 0;
        for (int v = 0; v < count; v++) {
            degree[v] = next[v].size();
            total += degree[v];
        }
        offset = new int[count + 1];
        for (int v = 0; v < count; v++) offset[v + 1] = offset[v] + degree[v];
        edgeChar = new char[total];
        edgeTo = new int[total];
        for (int v = 0; v < count; v++) {
            int at = offset[v];
            for (Map.Entry<Character, Integer> e : next[v].entrySet()) {
                edgeChar[at] = e.getKey();
                edgeTo[at] = e.getValue();
                at++;
            }
        }
        len = Arrays.copyOf(stateLen, count);
        link = Arrays.copyOf(stateLink, count);
        occ = Arrays.copyOf(stateOcc, count);
        minEnd = Arrays.copyOf(stateMinEnd, count);
        states = count;
        edges = total;
    }

    /** Binary search in the state's packed run; returns the target state or -1. */
    private int step(int state, char c) {
        int lo = offset[state], hi = offset[state + 1] - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (edgeChar[mid] < c) lo = mid + 1;
            else if (edgeChar[mid] > c) hi = mid - 1;
            else return edgeTo[mid];
        }
        return -1;
    }

    /** State reached by p, or -1 when p does not occur. */
    int run(String p) {
        int v = 0;
        for (int i = 0; i < p.length(); i++) {
            v = step(v, p.charAt(i));
            if (v == -1) return -1;
        }
        return v;
    }

    boolean contains(String p) { return p.isEmpty() || run(p) != -1; }

    int occurrences(String p) { int v = run(p); return v == -1 ? 0 : occ[v]; }

    int firstOccurrence(String p) { int v = run(p); return v == -1 ? -1 : minEnd[v] - p.length() + 1; }

    long distinctSubstrings() {
        long total = 0;
        for (int v = 1; v < states; v++) total += len[v] - len[link[v]];
        return total;
    }

    int longestRepeatedLength() {
        int best = 0;
        for (int v = 1; v < states; v++) if (occ[v] >= 2) best = Math.max(best, len[v]);
        return best;
    }

    /** Matching statistics: keep the deepest state whose class matches the current suffix of other,
     *  climbing suffix links when the next character is missing (total work stays linear). */
    int longestCommonSubstring(String other) {
        int v = 0, matched = 0, best = 0;
        for (int i = 0; i < other.length(); i++) {
            char c = other.charAt(i);
            while (v != 0 && step(v, c) == -1) {
                v = link[v];
                matched = len[v];
            }
            int u = step(v, c);
            if (u == -1) matched = 0;
            else { v = u; matched++; }
            best = Math.max(best, matched);
        }
        return best;
    }

    /** Invariant of the packed layout: each run holds strictly increasing characters. */
    boolean transitionsSorted() {
        for (int v = 0; v < states; v++) {
            for (int i = offset[v] + 1; i < offset[v + 1]; i++) if (edgeChar[i] <= edgeChar[i - 1]) return false;
        }
        return true;
    }
}

public class Solution {
    private static long state = 0x0BADC0FFEE12345L;

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

    static int bruteOccurrences(String text, String pattern) {
        int count = 0;
        for (int i = 0; i + pattern.length() <= text.length(); i++) if (text.startsWith(pattern, i)) count++;
        return count;
    }

    static long bruteDistinct(String text) {
        Set<String> found = new HashSet<>();
        for (int i = 0; i < text.length(); i++)
            for (int j = i + 1; j <= text.length(); j++) found.add(text.substring(i, j));
        return found.size();
    }

    static int bruteLongestRepeat(String text) {
        int best = 0;
        for (int i = 0; i < text.length(); i++)
            for (int j = i + 1; j <= text.length(); j++) {
                String sub = text.substring(i, j);
                if (sub.length() > best && bruteOccurrences(text, sub) >= 2) best = sub.length();
            }
        return best;
    }

    static int bruteLcs(String a, String b) {
        int best = 0;
        for (int i = 0; i < a.length(); i++)
            for (int j = i + 1; j <= a.length(); j++) {
                String sub = a.substring(i, j);
                if (sub.length() > best && b.contains(sub)) best = sub.length();
            }
        return best;
    }

    /** Every query of the automaton of text must agree with brute force on text. */
    static void verify(String text) {
        CompressedSuffixAutomaton auto = new CompressedSuffixAutomaton(text);
        check(auto.transitionsSorted(), "packed transitions sorted for " + text);
        check(auto.states <= Math.max(2, 2 * text.length() - 1), "state bound for " + text);
        check(auto.edges <= Math.max(3, 3 * text.length() - 4), "edge bound for " + text);
        check(auto.distinctSubstrings() == bruteDistinct(text), "distinct brute " + text);
        check(auto.longestRepeatedLength() == bruteLongestRepeat(text), "longest repeat " + text);
        for (int start = 0; start < text.length(); start++) {
            for (int end = start + 1; end <= text.length(); end++) {
                String pattern = text.substring(start, end);
                check(auto.contains(pattern), "window present " + text);
                check(auto.occurrences(pattern) == bruteOccurrences(text, pattern), "window count " + text);
                check(auto.firstOccurrence(pattern) == text.indexOf(pattern), "first occurrence " + text);
            }
        }
        check(!auto.contains("zzz") && auto.occurrences("zzz") == 0 && auto.firstOccurrence("zzz") == -1,
                "absent pattern " + text);
    }

    public static void main(String[] args) {
        check(new CompressedSuffixAutomaton("").distinctSubstrings() == 0, "empty text");
        CompressedSuffixAutomaton empty = new CompressedSuffixAutomaton("");
        check(empty.contains("") && empty.occurrences("a") == 0 && empty.transitionsSorted(), "empty queries");

        CompressedSuffixAutomaton single = new CompressedSuffixAutomaton("a");
        check(single.distinctSubstrings() == 1 && single.states == 2 && single.edges == 1, "single char");

        CompressedSuffixAutomaton abab = new CompressedSuffixAutomaton("abab");
        check(abab.distinctSubstrings() == 7, "abab distinct");
        check(abab.occurrences("ab") == 2 && abab.occurrences("ba") == 1, "abab counts");
        check(abab.firstOccurrence("ba") == 1 && abab.firstOccurrence("abab") == 0, "abab first");
        check(abab.longestRepeatedLength() == 2 && abab.longestCommonSubstring("baba") == 3, "abab repeats");

        CompressedSuffixAutomaton banana = new CompressedSuffixAutomaton("banana");
        check(banana.distinctSubstrings() == 15 && banana.longestRepeatedLength() == 3, "banana shape");
        check(banana.occurrences("ana") == 2 && banana.occurrences("nan") == 1, "banana counts");
        check(banana.firstOccurrence("na") == 2 && banana.firstOccurrence("banana") == 0, "banana first");

        CompressedSuffixAutomaton aaaa = new CompressedSuffixAutomaton("aaaa");
        check(aaaa.distinctSubstrings() == 4 && aaaa.occurrences("aa") == 3, "aaaa counts");
        check(aaaa.firstOccurrence("aaa") == 0 && aaaa.occurrences("b") == 0, "aaaa first/absent");
        check(aaaa.longestCommonSubstring("aaa") == 3 && aaaa.longestCommonSubstring("b") == 0, "aaaa common");

        for (String text : new String[]{"a", "ab", "ba", "abc", "abab", "banana", "mississippi",
                                        "aaabaaaab", "abcabcabc", "abracadabra"}) {
            verify(text);
        }
        for (String alphabet : new String[]{"ab", "abc"}) {
            for (int trial = 0; trial < 20; trial++) verify(randomText(28, alphabet));
        }
        for (int trial = 0; trial < 20; trial++) {
            String text = randomText(30, "abc");
            String other = randomText(18, "abc");
            CompressedSuffixAutomaton auto = new CompressedSuffixAutomaton(text);
            check(auto.longestCommonSubstring(other) == bruteLcs(text, other), "common substring " + text);
        }

        String large = randomText(400, "ab");
        CompressedSuffixAutomaton largeAuto = new CompressedSuffixAutomaton(large);
        check(largeAuto.distinctSubstrings() == bruteDistinct(large), "large distinct");
        check(largeAuto.transitionsSorted() && largeAuto.states <= 2 * large.length() - 1, "large shape");
        for (int i = 0; i < large.length(); i += 61) {
            String pattern = large.substring(i, Math.min(large.length(), i + 12));
            check(largeAuto.occurrences(pattern) == bruteOccurrences(large, pattern), "large window");
        }
        System.out.println("states / edges      : " + largeAuto.states + " / " + largeAuto.edges);
        System.out.println("distinct substrings : " + largeAuto.distinctSubstrings());
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
