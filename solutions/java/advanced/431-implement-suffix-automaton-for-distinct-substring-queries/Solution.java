/** Task 431 - Implement suffix automaton for distinct substring queries. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Query implemented: the distinct-substring count of the automaton's own string, i.e. the sum
 * of len[state] - len[link[state]] over all non-initial states ("abab" -> 7, "aaaa" -> 4).
 * Two further suffix-automaton queries back the accept/reject checks: isSubstring(t) walks the
 * transition function from the initial state, and occurrences(t) reads the end-position count of
 * the reached state (terminal counts are propagated along suffix links in decreasing-length
 * order). Patterns passed to occurrences are non-empty; every answer is cross-checked against a
 * brute-force set of all substrings on LCG-generated strings.
 */
import java.util.*;

class SuffixAutomaton {
    final int[] length;
    final int[] link;
    final int[] ends;
    final List<Map<Character, Integer>> next;
    int states;

    SuffixAutomaton(String text) {
        int capacity = 2 * text.length() + 1;
        length = new int[capacity];
        link = new int[capacity];
        ends = new int[capacity];
        next = new ArrayList<>(capacity);
        states = 1;
        for (int i = 0; i < capacity; i++) next.add(new HashMap<>());
        link[0] = -1;
        int last = 0;
        for (int k = 0; k < text.length(); k++) {
            char ch = text.charAt(k);
            int cur = states++;
            length[cur] = length[last] + 1;
            link[cur] = -1;
            ends[cur] = 1;
            int p = last;
            while (p != -1 && !next.get(p).containsKey(ch)) {
                next.get(p).put(ch, cur);
                p = link[p];
            }
            if (p == -1) {
                link[cur] = 0;
            } else {
                int q = next.get(p).get(ch);
                if (length[p] + 1 == length[q]) {
                    link[cur] = q;
                } else {
                    int clone = states++;
                    length[clone] = length[p] + 1;
                    link[clone] = link[q];
                    next.get(clone).putAll(next.get(q));
                    ends[clone] = 0;
                    while (p != -1 && next.get(p).get(ch) != null && next.get(p).get(ch) == q) {
                        next.get(p).put(ch, clone);
                        p = link[p];
                    }
                    link[q] = clone;
                    link[cur] = clone;
                }
            }
            last = cur;
        }
        Integer[] order = new Integer[states - 1];
        for (int i = 1; i < states; i++) order[i - 1] = i;
        Arrays.sort(order, (a, b) -> length[b] - length[a]);
        for (int state : order) ends[link[state]] += ends[state];
    }

    long distinctSubstrings() {
        long total = 0;
        for (int v = 1; v < states; v++) total += length[v] - length[link[v]];
        return total;
    }

    int walk(String pattern) {
        int state = 0;
        for (int i = 0; i < pattern.length(); i++) {
            Integer nxt = next.get(state).get(pattern.charAt(i));
            if (nxt == null) return -1;
            state = nxt;
        }
        return state;
    }

    boolean isSubstring(String pattern) {
        return walk(pattern) != -1;
    }

    int occurrences(String pattern) {
        int state = walk(pattern);
        return state == -1 ? 0 : ends[state];
    }
}

public class Solution {
    static long bruteDistinctSubstrings(String text) {
        Set<String> found = new HashSet<>();
        for (int i = 0; i < text.length(); i++) {
            for (int j = i + 1; j <= text.length(); j++) found.add(text.substring(i, j));
        }
        return found.size();
    }

    static int bruteOccurrences(String text, String pattern) {
        int count = 0;
        for (int i = 0; i + pattern.length() <= text.length(); i++) {
            if (text.startsWith(pattern, i)) count++;
        }
        return count;
    }

    static List<String> lcgStrings(int count, int seed, int maxLen, String alphabet) {
        List<String> strings = new ArrayList<>();
        for (int s = 0; s < count; s++) {
            int length = 1 + seed % maxLen;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < length; i++) {
                seed = (seed * 75 + 74) % 65537;
                sb.append(alphabet.charAt(seed % alphabet.length()));
            }
            strings.add(sb.toString());
        }
        return strings;
    }

    public static void main(String[] args) {
        check(new SuffixAutomaton("").distinctSubstrings() == 0, "empty");
        check(new SuffixAutomaton("a").distinctSubstrings() == 1, "single char");
        check(new SuffixAutomaton("abab").distinctSubstrings() == 7, "abab");
        check(new SuffixAutomaton("aaaa").distinctSubstrings() == 4, "aaaa");
        check(new SuffixAutomaton("abc").distinctSubstrings() == 6, "abc");
        check(new SuffixAutomaton("ababa").distinctSubstrings() == 9, "ababa");

        SuffixAutomaton banana = new SuffixAutomaton("banana");
        check(banana.distinctSubstrings() == 15, "banana distinct");
        check(banana.isSubstring("ban") && banana.isSubstring("ana") && banana.isSubstring("nan"), "banana accept");
        check(!banana.isSubstring("bn") && !banana.isSubstring("bananas"), "banana reject");
        check(banana.occurrences("an") == 2 && banana.occurrences("ana") == 2, "banana occurrences");
        check(banana.occurrences("x") == 0, "banana absent");

        SuffixAutomaton abab = new SuffixAutomaton("abab");
        check(abab.occurrences("ab") == 2 && abab.occurrences("ba") == 1, "abab occurrences");
        check(abab.occurrences("abab") == 1, "abab whole");

        SuffixAutomaton aaaa = new SuffixAutomaton("aaaa");
        check(aaaa.occurrences("a") == 4 && aaaa.occurrences("aa") == 3, "aaaa short");
        check(aaaa.occurrences("aaa") == 2 && aaaa.occurrences("aaaa") == 1, "aaaa long");

        int rejected = 0;
        int accepted = 0;
        for (String text : lcgStrings(40, 431, 40, "abc")) {
            SuffixAutomaton auto = new SuffixAutomaton(text);
            check(auto.distinctSubstrings() == bruteDistinctSubstrings(text), "random distinct " + text);
            check(auto.isSubstring(text), "whole accepted");
            check(auto.occurrences(text) == 1, "whole once");
            for (int start = 0; start < text.length(); start++) {
                for (int end = start + 1; end <= Math.min(start + 6, text.length()); end++) {
                    String pattern = text.substring(start, end);
                    check(auto.isSubstring(pattern), "window accepted");
                    check(auto.occurrences(pattern) == bruteOccurrences(text, pattern), "window occurrences");
                    accepted++;
                }
            }
            String[] mutations = {"d", "bd", "abz", text + "d"};
            for (String mutation : mutations) {
                if (auto.isSubstring(mutation)) {
                    check(auto.occurrences(mutation) == bruteOccurrences(text, mutation), "mutation occurrences");
                } else {
                    check(bruteOccurrences(text, mutation) == 0, "mutation rejected");
                    rejected++;
                }
            }
        }
        check(rejected > 0 && accepted > 0, "both verdicts exercised");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
