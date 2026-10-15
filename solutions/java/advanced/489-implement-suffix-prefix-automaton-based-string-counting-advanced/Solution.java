/** Task 489 - Implement suffix-prefix automaton based string counting (advanced). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Documented clean task: given a finite set of strings S, count the distinct strings x that are both
 * a non-empty prefix of some s in S and a non-empty suffix of some s in S, i.e. |P ∩ Q| where
 * P = {s[:i] : s in S, 1 <= i <= |s|} and Q = {s[j:] : s in S, 0 <= j < |s|}.
 *
 * Automaton approach: build a trie over all prefixes of S (every trie node corresponds to a member
 * of P, and the trie accepts exactly P - it is the "prefix automaton" of S). For each suffix
 * x = s[j:] of every s in S, feed x to the automaton from the root in one walk: x is accepted iff
 * x in P, so the walk decides membership x in P ∩ Q. A set of the accepted x counts each string
 * once. The walk costs O(|x|) per suffix, O(sum |s|^2) overall - documented as fine for the small
 * sets used here.
 *
 * The result is asserted against a brute-force set intersection on small LCG-generated sets
 * (30 sets of 5 strings over {a, b}, lengths 1..12) and on hand cases; the random-set count total is
 * pinned at 330 and the larger {a,b,c} case at 11. Hand case: {ababa, baba} -> 5
 * (a, ba, aba, baba, ababa).
 */
import java.util.*;

class TrieNode {
    final Map<Character, TrieNode> children = new HashMap<>();
}

class Trie {
    final TrieNode root = new TrieNode();
    int size;

    void insert(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            TrieNode next = node.children.get(ch);
            if (next == null) {
                next = new TrieNode();
                node.children.put(ch, next);
                size++;
            }
            node = next;
        }
    }

    boolean accepts(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            node = node.children.get(word.charAt(i));
            if (node == null) return false;
        }
        return true;
    }
}

public class Solution {
    static Trie buildPrefixAutomaton(String[] strings) {
        Trie trie = new Trie();
        for (String s : strings) {
            for (int i = 1; i <= s.length(); i++) trie.insert(s.substring(0, i));
        }
        return trie;
    }

    static int[] countPrefixSuffixIntersection(String[] strings) {
        Trie trie = buildPrefixAutomaton(strings);
        Set<String> matched = new HashSet<>();
        int walked = 0;
        for (String s : strings) {
            for (int j = 0; j < s.length(); j++) {
                String x = s.substring(j);
                walked++;
                if (trie.accepts(x)) matched.add(x);
            }
        }
        return new int[]{matched.size(), walked, trie.size};
    }

    static Set<String> brutePrefixSuffixIntersection(String[] strings) {
        Set<String> prefixes = new HashSet<>();
        Set<String> suffixes = new HashSet<>();
        for (String s : strings) {
            for (int i = 1; i <= s.length(); i++) prefixes.add(s.substring(0, i));
            for (int j = 0; j < s.length(); j++) suffixes.add(s.substring(j));
        }
        prefixes.retainAll(suffixes);
        return prefixes;
    }

    static String[] lcgStrings(int count, int seed, int maxLength, String alphabet) {
        String[] strings = new String[count];
        for (int k = 0; k < count; k++) {
            seed = (seed * 75 + 74) % 65537;
            int length = 1 + seed % maxLength;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < length; i++) {
                seed = (seed * 75 + 74) % 65537;
                sb.append(alphabet.charAt(seed % alphabet.length()));
            }
            strings[k] = sb.toString();
        }
        return strings;
    }

    static int totalSuffixes(String[] strings) {
        int total = 0;
        for (String s : strings) total += s.length();
        return total;
    }

    static int prefixCount(String[] strings) {
        Set<String> prefixes = new HashSet<>();
        for (String s : strings) {
            for (int i = 1; i <= s.length(); i++) prefixes.add(s.substring(0, i));
        }
        return prefixes.size();
    }

    public static void main(String[] args) {
        check(countPrefixSuffixIntersection(new String[]{"ababa", "baba"})[0] == 5, "ababa hand");
        check(brutePrefixSuffixIntersection(new String[]{"ababa", "baba"})
                .equals(new TreeSet<>(Arrays.asList("a", "ba", "aba", "baba", "ababa"))), "ababa brute");
        check(countPrefixSuffixIntersection(new String[]{"abc"})[0] == 1, "abc hand");
        check(brutePrefixSuffixIntersection(new String[]{"abc"}).equals(new TreeSet<>(Arrays.asList("abc"))), "abc brute");
        check(countPrefixSuffixIntersection(new String[]{"a"})[0] == 1, "single char");
        check(countPrefixSuffixIntersection(new String[]{"ab", "ba"})[0] == 4, "ab ba hand");
        check(brutePrefixSuffixIntersection(new String[]{"ab", "ba"})
                .equals(new TreeSet<>(Arrays.asList("a", "b", "ab", "ba"))), "ab ba brute");
        check(countPrefixSuffixIntersection(new String[]{"aaaa"})[0] == 4, "aaaa hand");

        String[][] automatonCases = {
                {"ababa", "baba"},
                {"abc", "cba", "bac"}
        };
        for (String[] strings : automatonCases) {
            Trie trie = buildPrefixAutomaton(strings);
            check(trie.size == prefixCount(strings), "automaton size");
            Set<String> prefixes = new HashSet<>();
            for (String s : strings) {
                for (int i = 1; i <= s.length(); i++) prefixes.add(s.substring(0, i));
            }
            for (String x : prefixes) check(trie.accepts(x), "accepts prefix " + x);
            check(!trie.accepts("zzz"), "rejects zzz");
        }

        int total = 0;
        for (int k = 0; k < 30; k++) {
            String[] strings = lcgStrings(5, 489 + k, 12, "ab");
            int[] result = countPrefixSuffixIntersection(strings);
            Set<String> expected = brutePrefixSuffixIntersection(strings);
            check(result[0] == expected.size(), "random set " + k + " " + result[0] + " " + expected.size());
            check(result[1] == totalSuffixes(strings), "walked " + k);
            check(result[2] == prefixCount(strings), "nodes " + k);
            total += result[0];
        }
        check(total == 330, "random total " + total);

        String[] strings = lcgStrings(6, 777, 14, "abc");
        int[] result = countPrefixSuffixIntersection(strings);
        Set<String> expected = brutePrefixSuffixIntersection(strings);
        check(result[0] == expected.size() && result[0] == 11, "large case " + result[0]);
        check(result[1] == totalSuffixes(strings), "large walked");
        check(result[2] == prefixCount(strings), "large nodes");

        check(countPrefixSuffixIntersection(new String[]{"abc", "abc", "abc"})[0]
                == countPrefixSuffixIntersection(new String[]{"abc"})[0], "duplicates collapse");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
