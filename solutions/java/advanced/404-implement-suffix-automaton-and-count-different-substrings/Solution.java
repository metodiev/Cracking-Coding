/** Task 404 - Implement a suffix automaton and count different substrings. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A suffix automaton is the minimal DFA recognizing all substrings of s. Each state stands for
 * an equivalence class of substrings sharing the same end-position set, with `length` the
 * longest member of the class and `link` the state for the longest proper suffix outside it.
 * Every substring in the class is a suffix of the state's longest string with length in
 * (length(link), length], so each class contributes length - length(link) distinct substrings
 * and the total is their sum over all states except the initial one.
 *
 * Anchors: "abab" -> 7, "aaaa" -> 4, "abc" -> 6. Cross-checks: the count matches a brute-force
 * substring set, every substring is accepted by walking transitions from the initial state,
 * generated non-substrings are rejected, and `occurrenceCount` (end-position propagation along
 * suffix links in decreasing-length order) matches a naive scan.
 */
import java.util.*;

class SuffixAutomaton {
    private final List<Map<Character, Integer>> next = new ArrayList<>();
    private final List<Integer> link = new ArrayList<>();
    private final List<Integer> length = new ArrayList<>();
    private final List<Boolean> isClone = new ArrayList<>();
    private int last = 0;

    SuffixAutomaton() {
        next.add(new HashMap<>());
        link.add(-1);
        length.add(0);
        isClone.add(false);
    }

    void extend(char ch) {
        int cur = length.size();
        next.add(new HashMap<>());
        link.add(0);
        length.add(length.get(last) + 1);
        isClone.add(false);
        int p = last;
        while (p != -1 && !next.get(p).containsKey(ch)) {
            next.get(p).put(ch, cur);
            p = link.get(p);
        }
        if (p == -1) {
            link.set(cur, 0);
        } else {
            int q = next.get(p).get(ch);
            if (length.get(p) + 1 == length.get(q)) {
                link.set(cur, q);
            } else {
                int clone = length.size();
                next.add(new HashMap<>(next.get(q)));
                link.add(link.get(q));
                length.add(length.get(p) + 1);
                isClone.add(true);
                while (p != -1) {
                    Integer target = next.get(p).get(ch);
                    if (target == null || target.intValue() != q) break;
                    next.get(p).put(ch, clone);
                    p = link.get(p);
                }
                link.set(q, clone);
                link.set(cur, clone);
            }
        }
        last = cur;
    }

    int stateFor(String t) {
        int state = 0;
        for (int i = 0; i < t.length(); i++) {
            Integer nxt = next.get(state).get(t.charAt(i));
            if (nxt == null) return -1;
            state = nxt;
        }
        return state;
    }

    boolean contains(String t) {
        return stateFor(t) != -1;
    }

    long distinctSubstrings() {
        long total = 0;
        for (int v = 1; v < length.size(); v++) total += length.get(v) - length.get(link.get(v));
        return total;
    }

    int[] endposSizes() {
        int states = length.size();
        Integer[] order = new Integer[states - 1];
        for (int i = 1; i < states; i++) order[i - 1] = i;
        Arrays.sort(order, (a, b) -> Integer.compare(length.get(b), length.get(a)));
        int[] count = new int[states];
        for (int v = 1; v < states; v++) count[v] = isClone.get(v) ? 0 : 1;
        for (int v : order) count[link.get(v)] += count[v];
        return count;
    }

    int occurrenceCount(String t) {
        int state = stateFor(t);
        return state == -1 ? 0 : endposSizes()[state];
    }

    static SuffixAutomaton build(String s) {
        SuffixAutomaton automaton = new SuffixAutomaton();
        for (int i = 0; i < s.length(); i++) automaton.extend(s.charAt(i));
        return automaton;
    }
}

class SubstringTools {
    static Set<String> allSubstrings(String s) {
        Set<String> seen = new LinkedHashSet<>();
        for (int i = 0; i < s.length(); i++)
            for (int j = i + 1; j <= s.length(); j++)
                seen.add(s.substring(i, j));
        return seen;
    }

    static int countOccurrences(String s, String t) {
        if (t.isEmpty()) return 0;
        int count = 0;
        for (int i = 0; i + t.length() <= s.length(); i++)
            if (s.startsWith(t, i)) count++;
        return count;
    }

    static int[] suffixArrayNaive(String s) {
        int n = s.length();
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (a, b) -> compare(s, a, b));
        int[] out = new int[n];
        for (int i = 0; i < n; i++) out[i] = idx[i];
        return out;
    }

    static int compare(String s, int i, int j) {
        int n = s.length();
        while (i < n && j < n) {
            if (s.charAt(i) != s.charAt(j)) return s.charAt(i) < s.charAt(j) ? -1 : 1;
            i++;
            j++;
        }
        if (i == j) return 0;
        return i == n ? -1 : 1;
    }

    static int[] kasai(String s, int[] sa) {
        int n = s.length();
        int[] lcp = new int[n];
        int[] rank = new int[n];
        for (int i = 0; i < n; i++) rank[sa[i]] = i;
        int h = 0;
        for (int i = 0; i < n; i++) {
            int r = rank[i];
            if (r == 0) {
                h = 0;
                continue;
            }
            int j = sa[r - 1];
            while (i + h < n && j + h < n && s.charAt(i + h) == s.charAt(j + h)) h++;
            lcp[r] = h;
            if (h > 0) h--;
        }
        return lcp;
    }

    static int sum(int[] values) {
        int total = 0;
        for (int value : values) total += value;
        return total;
    }
}

public class Solution {
    private static int seed = 2718;

    static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        SuffixAutomaton abab = SuffixAutomaton.build("abab");
        check(abab.distinctSubstrings() == 7, "abab count");
        check(SubstringTools.allSubstrings("abab").size() == 7, "abab brute");

        SuffixAutomaton aaaa = SuffixAutomaton.build("aaaa");
        check(aaaa.distinctSubstrings() == 4, "aaaa count");

        check(SuffixAutomaton.build("").distinctSubstrings() == 0, "empty");
        check(SuffixAutomaton.build("a").distinctSubstrings() == 1, "single");
        check(SuffixAutomaton.build("abc").distinctSubstrings() == 6, "abc");
        check(SuffixAutomaton.build("banana").distinctSubstrings() == 15, "banana");
        check(abab.contains(""), "empty accepted");
        check(!abab.contains("baba"), "baba rejected");
        check(abab.stateFor("abab") != -1 && abab.stateFor("ababa") == -1, "abab state");

        String[] small = {"", "a", "aa", "ab", "banana", "mississippi", "abababab", "aabbaabb"};
        for (String s : small) {
            SuffixAutomaton automaton = SuffixAutomaton.build(s);
            Set<String> substrings = SubstringTools.allSubstrings(s);
            check(automaton.distinctSubstrings() == substrings.size(), "distinct " + s);
            int[] lcp = SubstringTools.kasai(s, SubstringTools.suffixArrayNaive(s));
            check(automaton.distinctSubstrings() == s.length() * (s.length() + 1) / 2 - SubstringTools.sum(lcp),
                    "identity " + s);
        }

        Set<String> ababSubstrings = SubstringTools.allSubstrings("abab");
        for (String t : ababSubstrings) check(abab.contains(t), "accept " + t);
        for (String t : new String[]{"c", "bb", "abb", "baa", "ababa", "babab", "xyz"}) {
            check(!abab.contains(t), "reject " + t);
        }
        for (String a : ababSubstrings) {
            for (String b : ababSubstrings) {
                String joined = a + b;
                check(abab.contains(joined) == ababSubstrings.contains(joined), "join " + joined);
            }
        }

        for (String s : new String[]{"banana", "abab", "aaaa", "mississippi", "abababab"}) {
            SuffixAutomaton automaton = SuffixAutomaton.build(s);
            for (String t : SubstringTools.allSubstrings(s)) {
                check(automaton.occurrenceCount(t) == SubstringTools.countOccurrences(s, t), "occ " + s + "/" + t);
            }
            check(automaton.occurrenceCount("xx") == 0, "absent " + s);
        }
        check(SuffixAutomaton.build("banana").occurrenceCount("ana") == 2, "ana");
        check(SuffixAutomaton.build("banana").occurrenceCount("a") == 3, "a");
        check(SuffixAutomaton.build("mississippi").occurrenceCount("issi") == 2, "issi");

        for (String alphabet : new String[]{"ab", "abc"}) {
            for (int trial = 0; trial < 30; trial++) {
                String s = lcgString(20, alphabet);
                SuffixAutomaton automaton = SuffixAutomaton.build(s);
                Set<String> substrings = SubstringTools.allSubstrings(s);
                check(automaton.distinctSubstrings() == substrings.size(), "lcg distinct " + s);
                for (String t : substrings) {
                    check(automaton.contains(t), "lcg accept " + t);
                    check(automaton.occurrenceCount(t) == SubstringTools.countOccurrences(s, t), "lcg occ " + t);
                }
                for (String t : new String[]{alphabet.charAt(0) + "" + alphabet.charAt(alphabet.length() - 1)
                        + alphabet.charAt(0), "zzz", "q", s + alphabet.charAt(0)}) {
                    if (!substrings.contains(t)) check(!automaton.contains(t), "lcg reject " + t);
                }
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
