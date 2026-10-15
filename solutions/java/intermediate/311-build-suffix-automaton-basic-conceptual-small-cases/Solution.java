/** Task 311 - Suffix automaton (basic conceptual, small cases). (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

class SuffixAutomaton {
    final List<Integer> length = new ArrayList<>();              // len[v]
    final List<Integer> link = new ArrayList<>();                 // suffix link; -1 marks the initial state
    final List<Map<Character, Integer>> transitions = new ArrayList<>();
    int last;

    SuffixAutomaton() {
        length.add(0);
        link.add(-1);
        transitions.add(new HashMap<>());
        last = 0;
    }

    void extend(char ch) {
        int cur = length.size();
        length.add(length.get(last) + 1);
        link.add(0);
        transitions.add(new HashMap<>());
        int p = last;
        while (p != -1 && !transitions.get(p).containsKey(ch)) {
            transitions.get(p).put(ch, cur);
            p = link.get(p);
        }
        if (p == -1) {
            link.set(cur, 0);
        } else {
            int q = transitions.get(p).get(ch);
            if (length.get(p) + 1 == length.get(q)) {
                link.set(cur, q);
            } else {
                int clone = length.size();
                length.add(length.get(p) + 1);
                link.add(link.get(q));
                transitions.add(new HashMap<>(transitions.get(q)));
                while (p != -1 && transitions.get(p).getOrDefault(ch, -1) == q) {
                    transitions.get(p).put(ch, clone);
                    p = link.get(p);
                }
                link.set(q, clone);
                link.set(cur, clone);
            }
        }
        last = cur;
    }

    /** DFA walk from the initial state; accepts "" vacuously (the empty substring). */
    boolean accepts(String word) {
        int state = 0;
        for (int i = 0; i < word.length(); i++) {
            Integer next = transitions.get(state).get(word.charAt(i));
            if (next == null) return false;
            state = next;
        }
        return true;
    }

    int distinctSubstrings() {
        int total = 0;
        for (int v = 1; v < length.size(); v++) total += length.get(v) - length.get(link.get(v));
        return total;
    }
}

public class Solution {
    static SuffixAutomaton buildSuffixAutomaton(String s) {
        SuffixAutomaton automaton = new SuffixAutomaton();
        for (int i = 0; i < s.length(); i++) automaton.extend(s.charAt(i));
        return automaton;
    }

    /** The independent oracle: every substring of s, collected in a set. */
    static Set<String> bruteSubstrings(String s) {
        Set<String> out = new HashSet<>();
        for (int i = 0; i < s.length(); i++)
            for (int j = i + 1; j <= s.length(); j++) out.add(s.substring(i, j));
        return out;
    }

    static List<String> allWords(String alphabet, int maxLength) {
        List<String> words = new ArrayList<>();
        words.add("");
        List<String> frontier = new ArrayList<>();
        frontier.add("");
        for (int depth = 0; depth < maxLength; depth++) {
            List<String> next = new ArrayList<>();
            for (String word : frontier)
                for (int i = 0; i < alphabet.length(); i++) next.add(word + alphabet.charAt(i));
            words.addAll(next);
            frontier = next;
        }
        return words;
    }

    static void verify(String s, String alphabet) {
        SuffixAutomaton automaton = buildSuffixAutomaton(s);
        Set<String> substrings = bruteSubstrings(s);
        // 1) the len/link formula agrees with the brute-force substring set
        check(automaton.distinctSubstrings() == substrings.size(), "count " + s);
        // 2) every substring of s is accepted by walking the automaton
        for (String word : substrings) check(automaton.accepts(word), "accept " + s + " " + word);
        // 3) and every short word over the alphabet is accepted iff it really is a substring
        //    (the empty word is accepted vacuously; the brute set only holds non-empty ones)
        for (String word : allWords(alphabet, 4)) {
            if (word.isEmpty()) check(automaton.accepts(word), "empty " + s);
            else check(automaton.accepts(word) == substrings.contains(word), "member " + s + " " + word);
        }
        // 4) the state count stays within the textbook 2n bound
        check(automaton.length.size() <= 2 * s.length() + 1, "state bound " + s);
    }

    private static int seed = 3110;

    private static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // "abab" has exactly the 7 distinct substrings a, b, ab, ba, aba, bab, abab
        SuffixAutomaton abab = buildSuffixAutomaton("abab");
        check(bruteSubstrings("abab").equals(new HashSet<>(Arrays.asList("a", "b", "ab", "ba", "aba", "bab", "abab"))), "abab set");
        check(abab.distinctSubstrings() == 7, "abab formula");

        // "aaaa" has 4: a, aa, aaa, aaaa
        SuffixAutomaton aaaa = buildSuffixAutomaton("aaaa");
        check(bruteSubstrings("aaaa").size() == 4, "aaaa brute");
        check(aaaa.distinctSubstrings() == 4, "aaaa formula");

        SuffixAutomaton banana = buildSuffixAutomaton("banana");
        check(banana.distinctSubstrings() == 15 && bruteSubstrings("banana").size() == 15, "banana");

        SuffixAutomaton empty = buildSuffixAutomaton("");
        check(empty.distinctSubstrings() == 0, "empty count");
        check(empty.accepts(""), "empty word accepted");

        check(!abab.accepts("bb") && !abab.accepts("baa") && !abab.accepts("baab"), "rejections");
        check(abab.accepts("bab"), "bab accepted");

        String[] fixed = {"a", "ab", "abab", "banana", "mississippi"};
        String[] fixedAlphabet = {"ab", "ab", "ab", "ban", "mips"};
        for (int i = 0; i < fixed.length; i++) verify(fixed[i], fixedAlphabet[i]);
        for (int i = 0; i < 12; i++) verify(lcgString(10, "ab"), "ab");
        for (int i = 0; i < 8; i++) verify(lcgString(9, "abc"), "abc");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
