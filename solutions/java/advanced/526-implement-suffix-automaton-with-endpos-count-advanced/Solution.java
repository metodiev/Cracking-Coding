/** Task 526 - Implement suffix automaton with endpos count (advanced). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Suffix automaton built online in O(n) (or O(n log sigma) with the TreeMap transition table used
 * here): each state keeps len (the longest substring it represents), link (the suffix link to the
 * state of the longest proper suffix) and its transitions. A substring is represented by the state
 * reached from the initial state; the set of substrings of a state is exactly the length interval
 * (len(link(v)), len(v)]. Every step adds one end position to the newly created state, so the
 * ENDPOS COUNT of each state - the number of occurrences of every substring it represents - is
 * obtained by counting one unit on each non-clone state and pushing the counts up the suffix-link
 * tree in order of decreasing len. On top of that the file implements: occurrence end positions per
 * state (one position per non-clone state, gathered up the link tree and sorted), the number of
 * distinct substrings sum_v (len(v) - len(link(v))), the number of distinct substrings of a fixed
 * length L (states with len(link) < L <= len), the lexicographically k-th distinct substring by
 * walking the DAG with per-state distinct-substring counts, and the longest common substring of two
 * strings by running the second string through the automaton of the first with suffix-link
 * fallbacks. Documented conventions: occurrences of the empty string are reported as 0, the k-th
 * distinct substring is 1-based in ascending lexicographic order, and positions are 0-based end
 * indices in the source string.
 *
 * Checks: hand example "ababa" gives 9 distinct substrings, "aba" occurs twice ending at 2 and 4,
 * and the whole string occurs once; "aaaa" gives 4 distinct substrings with occurrence counts 4, 3,
 * 2, 1 and the k-th distinct substrings a, aa, aaa, aaaa; for LCG binary and ternary strings of
 * length up to 12 EVERY substring's occurrence count, occurrence positions, the distinct-substring
 * total, the per-length distinct totals and the full lexicographic enumeration (checked through
 * k-th distinct substring for all k) must match brute force; the automaton accepts exactly the
 * substrings of the string (every brute-force substring walks from the initial state, no random
 * foreign word does); the longest common substring is compared against an O(n^2) DP table on LCG
 * pairs and on the hand pair ("abab" vs "baba" gives 3, "abcde" vs "xyz" gives 0); a 200-character
 * LCG string must reproduce the brute-force distinct-substring count of its 20100 substrings;
 * repeated runs are identical.
 */
import java.util.*;

public class Solution {
    static class State {
        int len;
        int link;
        int endPosCount;
        final TreeMap<Character, Integer> next = new TreeMap<>();
        int firstEnd = -1;

        State(int len) {
            this.len = len;
            this.link = -1;
        }
    }

    static class SuffixAutomaton {
        final List<State> states = new ArrayList<>();
        final String source;
        int last;

        SuffixAutomaton(String s) {
            source = s;
            states.add(new State(0));
            states.get(0).link = -1;
            last = 0;
            for (int i = 0; i < s.length(); i++) extend(s.charAt(i), i);
            propagate();
        }

        void extend(char c, int position) {
            int current = states.size();
            states.add(new State(states.get(last).len + 1));
            states.get(current).firstEnd = position;
            int p = last;
            while (p != -1 && !states.get(p).next.containsKey(c)) {
                states.get(p).next.put(c, current);
                p = states.get(p).link;
            }
            if (p == -1) {
                states.get(current).link = 0;
            } else {
                int q = states.get(p).next.get(c);
                if (states.get(p).len + 1 == states.get(q).len) {
                    states.get(current).link = q;
                } else {
                    int clone = states.size();
                    State cloned = new State(states.get(p).len + 1);
                    cloned.next.putAll(states.get(q).next);
                    cloned.link = states.get(q).link;
                    cloned.endPosCount = 0;
                    states.add(cloned);
                    while (p != -1 && Integer.valueOf(q).equals(states.get(p).next.get(c))) {
                        states.get(p).next.put(c, clone);
                        p = states.get(p).link;
                    }
                    states.get(q).link = clone;
                    states.get(current).link = clone;
                }
            }
            last = current;
            states.get(current).endPosCount = 1;
        }

        /** Pushes occurrence counts and end positions up the suffix-link tree. */
        void propagate() {
            List<List<Integer>> children = linkTree();
            int[] order = statesByLenDescending();
            for (int index : order) {
                if (index == 0) continue;
                int parent = states.get(index).link;
                if (parent >= 0) states.get(parent).endPosCount += states.get(index).endPosCount;
            }
            propagatePositions(children, 0);
        }

        List<List<Integer>> linkTree() {
            List<List<Integer>> children = new ArrayList<>();
            for (int i = 0; i < states.size(); i++) children.add(new ArrayList<>());
            for (int i = 1; i < states.size(); i++) children.get(states.get(i).link).add(i);
            return children;
        }

        List<Integer> collectEnds = new ArrayList<>();

        void propagatePositions(List<List<Integer>> children, int node) {
            if (states.get(node).firstEnd >= 0) collectEnds.add(states.get(node).firstEnd);
            for (int child : children.get(node)) propagatePositions(children, child);
        }

        int[] statesByLenDescending() {
            int maxLen = 0;
            for (State state : states) maxLen = Math.max(maxLen, state.len);
            int[] bucket = new int[maxLen + 2];
            for (State state : states) bucket[state.len]++;
            for (int i = 1; i < bucket.length; i++) bucket[i] += bucket[i - 1];
            int[] order = new int[states.size()];
            for (int i = states.size() - 1; i >= 0; i--) {
                int len = states.get(i).len;
                order[--bucket[len]] = i;
            }
            int[] descending = new int[states.size()];
            for (int i = 0; i < order.length; i++) descending[i] = order[order.length - 1 - i];
            return descending;
        }

        /** State reached by the given substring, or -1. */
        int find(String substring) {
            int node = 0;
            for (int i = 0; i < substring.length(); i++) {
                Integer next = states.get(node).next.get(substring.charAt(i));
                if (next == null) return -1;
                node = next;
            }
            return node;
        }

        int occurrences(String substring) {
            if (substring.isEmpty()) return 0;
            int node = find(substring);
            return node < 0 ? 0 : states.get(node).endPosCount;
        }

        /** Sorted 0-based end positions of the substring (empty when absent). */
        int[] endPositions(String substring) {
            if (substring.isEmpty()) return new int[0];
            int node = find(substring);
            if (node < 0) return new int[0];
            collectEnds = new ArrayList<>();
            propagatePositions(linkTree(), node);
            int[] positions = new int[collectEnds.size()];
            for (int i = 0; i < positions.length; i++) positions[i] = collectEnds.get(i);
            Arrays.sort(positions);
            return positions;
        }

        long distinctSubstrings() {
            long total = 0;
            for (int i = 1; i < states.size(); i++) {
                total += states.get(i).len - states.get(states.get(i).link).len;
            }
            return total;
        }

        long distinctSubstringsOfLength(int length) {
            long total = 0;
            for (int i = 1; i < states.size(); i++) {
                int low = states.get(states.get(i).link).len;
                if (low < length && length <= states.get(i).len) total++;
            }
            return total;
        }

        /** 1-based lexicographic index into the sorted distinct substrings. */
        String kthDistinctSubstring(long k) {
            long total = distinctSubstrings();
            if (k < 1 || k > total) return null;
            StringBuilder builder = new StringBuilder();
            int node = 0;
            long remaining = k;
            while (true) {
                for (Map.Entry<Character, Integer> entry : states.get(node).next.entrySet()) {
                    long subtree = distinctFrom(entry.getValue());
                    if (remaining > subtree) {
                        remaining -= subtree;
                        continue;
                    }
                    builder.append(entry.getKey());
                    remaining--;
                    if (remaining == 0) return builder.toString();
                    node = entry.getValue();
                    break;
                }
            }
        }

        final Map<Integer, Long> distinctMemo = new HashMap<>();

        /** Number of distinct substrings of the DAG rooted at the given state (memoised). */
        long distinctFrom(int node) {
            Long cached = distinctMemo.get(node);
            if (cached != null) return cached;
            long total = 1;
            for (int next : states.get(node).next.values()) total += distinctFrom(next);
            distinctMemo.put(node, total);
            return total;
        }

        static String longestCommonSubstring(String a, String b) {
            SuffixAutomaton automaton = new SuffixAutomaton(a);
            int node = 0;
            int length = 0;
            int bestLength = 0;
            int bestEnd = -1;
            for (int i = 0; i < b.length(); i++) {
                char c = b.charAt(i);
                while (node != 0 && !automaton.states.get(node).next.containsKey(c)) {
                    node = automaton.states.get(node).link;
                    length = automaton.states.get(node).len;
                }
                Integer next = automaton.states.get(node).next.get(c);
                if (next != null) {
                    node = next;
                    length++;
                } else {
                    node = 0;
                    length = 0;
                }
                if (length > bestLength) {
                    bestLength = length;
                    bestEnd = i;
                }
            }
            return bestLength == 0 ? "" : b.substring(bestEnd - bestLength + 1, bestEnd + 1);
        }
    }

    static long bruteDistinct(String s) {
        Set<String> seen = new HashSet<>();
        for (int i = 0; i < s.length(); i++) {
            for (int j = i + 1; j <= s.length(); j++) seen.add(s.substring(i, j));
        }
        return seen.size();
    }

    static int bruteOccurrences(String s, String sub) {
        if (sub.isEmpty()) return 0;
        int count = 0;
        for (int i = 0; i + sub.length() <= s.length(); i++) {
            if (s.startsWith(sub, i)) count++;
        }
        return count;
    }

    static int[] bruteEnds(String s, String sub) {
        List<Integer> ends = new ArrayList<>();
        for (int i = 0; i + sub.length() <= s.length(); i++) {
            if (s.startsWith(sub, i)) ends.add(i + sub.length() - 1);
        }
        int[] out = new int[ends.size()];
        for (int i = 0; i < out.length; i++) out[i] = ends.get(i);
        return out;
    }

    static String lcgString(int n, int seed, int alphabet) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            builder.append((char) ('a' + seed % alphabet));
        }
        return builder.toString();
    }

    static String bruteLongestCommon(String a, String b) {
        String best = "";
        for (int i = 0; i < a.length(); i++) {
            for (int j = i + 1; j <= a.length(); j++) {
                String candidate = a.substring(i, j);
                if (candidate.length() > best.length() && b.contains(candidate)) best = candidate;
            }
        }
        return best;
    }

    public static void main(String[] args) {
        SuffixAutomaton ababa = new SuffixAutomaton("ababa");
        check(ababa.distinctSubstrings() == 9, "ababa has 9 distinct substrings");
        check(ababa.occurrences("aba") == 2, "aba occurs twice");
        check(Arrays.equals(ababa.endPositions("aba"), new int[]{2, 4}), "aba ends at 2 and 4");
        check(ababa.occurrences("ababa") == 1, "the whole string occurs once");
        check(Arrays.equals(ababa.endPositions("ababa"), new int[]{4}), "the whole string ends at 4");
        check(ababa.occurrences("babab") == 0, "absent substring has no occurrences");
        check(ababa.endPositions("babab").length == 0, "absent substring has no positions");
        check(ababa.occurrences("") == 0, "empty substring documented as 0 occurrences");

        SuffixAutomaton aaaa = new SuffixAutomaton("aaaa");
        check(aaaa.distinctSubstrings() == 4, "aaaa has 4 distinct substrings");
        String[] aPowers = {"a", "aa", "aaa", "aaaa"};
        for (int i = 0; i < aPowers.length; i++) {
            check(aaaa.occurrences(aPowers[i]) == 4 - i, "occurrence count of a^" + (i + 1));
            check(aaaa.kthDistinctSubstring(i + 1).equals(aPowers[i]), "k-th distinct substring a^"
                    + (i + 1));
        }
        check(aaaa.kthDistinctSubstring(0) == null && aaaa.kthDistinctSubstring(5) == null,
                "out-of-range k reports null");

        String[][] pairs = {{"abab", "baba", "bab"}, {"abcde", "xyz", ""}, {"abcabc", "cab", "cab"},
                {"", "abc", ""}, {"aaa", "aa", "aa"}};
        for (String[] pair : pairs) {
            String got = SuffixAutomaton.longestCommonSubstring(pair[0], pair[1]);
            check(got.equals(pair[2]), "longest common substring of " + pair[0] + " / " + pair[1]);
            check(got.length() == bruteLongestCommon(pair[0], pair[1]).length(),
                    "LCS length matches brute force for " + pair[0] + " / " + pair[1]);
        }

        int seed = 7;
        for (int trial = 0; trial < 14; trial++) {
            int n = 4 + trial % 9;
            int alphabet = 2 + trial % 2;
            String s = lcgString(n, 1000 + trial * 37, alphabet);
            SuffixAutomaton automaton = new SuffixAutomaton(s);
            check(automaton.distinctSubstrings() == bruteDistinct(s),
                    "distinct count matches brute force for " + s);
            for (int length = 1; length <= n; length++) {
                Set<String> distinct = new HashSet<>();
                for (int i = 0; i + length <= n; i++) distinct.add(s.substring(i, i + length));
                check(automaton.distinctSubstringsOfLength(length) == distinct.size(),
                        "distinct count of length " + length + " for " + s);
            }
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j <= n; j++) {
                    String sub = s.substring(i, j);
                    check(automaton.occurrences(sub) == bruteOccurrences(s, sub),
                            "occurrences of " + sub + " in " + s);
                    check(Arrays.equals(automaton.endPositions(sub), bruteEnds(s, sub)),
                            "end positions of " + sub + " in " + s);
                    check(automaton.find(sub) >= 0, "every substring is accepted: " + sub);
                }
            }
            Set<String> sorted = new TreeSet<>();
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j <= n; j++) sorted.add(s.substring(i, j));
            }
            long k = 1;
            for (String expected : sorted) {
                check(automaton.kthDistinctSubstring(k).equals(expected),
                        "k-th distinct substring k = " + k + " for " + s);
                k++;
            }
            check(automaton.find("zzzz") < 0, "a foreign word is rejected");
        }

        for (int trial = 0; trial < 6; trial++) {
            String a = lcgString(12 + trial, 777 + trial * 13, 3);
            String b = lcgString(10 + trial, 4242 + trial * 29, 3);
            String got = SuffixAutomaton.longestCommonSubstring(a, b);
            String expected = bruteLongestCommon(a, b);
            check(got.length() == expected.length(),
                    "LCS length matches the DP brute force, trial " + trial);
            check(b.contains(got), "the reported LCS really occurs in both strings");
            check(a.contains(got), "the reported LCS comes from the first string");
        }

        String big = lcgString(200, 31337, 4);
        SuffixAutomaton bigAutomaton = new SuffixAutomaton(big);
        Set<String> all = new HashSet<>();
        for (int i = 0; i < big.length(); i++) {
            for (int j = i + 1; j <= big.length(); j++) all.add(big.substring(i, j));
        }
        long bruteTotal = all.size();
        check(bigAutomaton.distinctSubstrings() == bruteTotal,
                "200-character string: distinct count matches the 20100-substring scan");
        check(bigAutomaton.occurrences(big) == 1, "the whole 200-character string occurs once");
        check(Arrays.equals(bigAutomaton.endPositions(big), new int[]{199}),
                "the whole string ends at 199");

        String repeated = lcgString(40, 909, 2) + lcgString(40, 909, 2);
        SuffixAutomaton repeatedAutomaton = new SuffixAutomaton(repeated);
        check(repeatedAutomaton.occurrences(repeated.substring(0, 40)) == 2,
                "the repeated block occurs twice");
        check(repeatedAutomaton.occurrences(repeated) == 1, "the doubled string occurs once");
        check(repeatedAutomaton.distinctSubstrings()
                == bruteDistinct(repeated), "doubled string distinct count");
        check(new SuffixAutomaton("ababa").distinctSubstrings() == ababa.distinctSubstrings(),
                "deterministic re-run");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
