/** Task 340 - Implement palindrome tree (Eertree) overview & small test. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class Eertree {
    // Two roots: node 0 has length -1 (imaginary), node 1 has length 0 (empty). Every other
    // node is a distinct palindromic substring with a suffix link to its longest proper
    // palindromic suffix. Counts start as "how often this node was the longest palindromic
    // suffix" and are then propagated along suffix links, longest node first.
    private final String text;
    private final List<Integer> length = new ArrayList<>(Arrays.asList(-1, 0));
    private final List<Integer> link = new ArrayList<>(Arrays.asList(0, 0));
    private final List<Map<Character, Integer>> next = new ArrayList<>();
    private final List<Integer> occurrences = new ArrayList<>(Arrays.asList(0, 0));
    private final List<String> palindrome = new ArrayList<>(Arrays.asList(null, ""));
    private int last = 1;

    Eertree(String text) {
        this.text = text;
        next.add(new HashMap<>());
        next.add(new HashMap<>());
        for (int position = 0; position < text.length(); position++) {
            extend(position, text.charAt(position));
        }
        Integer[] nodes = new Integer[length.size() - 2];
        for (int i = 0; i < nodes.length; i++) nodes[i] = i + 2;
        Arrays.sort(nodes, (a, b) -> length.get(b) - length.get(a));
        for (int node : nodes) {
            int parent = link.get(node);
            occurrences.set(parent, occurrences.get(parent) + occurrences.get(node));
        }
    }

    private void extend(int position, char ch) {
        int current = last;
        while (true) {
            int index = position - 1 - length.get(current);
            if (index >= 0 && text.charAt(index) == ch) break;
            current = link.get(current);
        }

        if (next.get(current).containsKey(ch)) {
            last = next.get(current).get(ch);
            occurrences.set(last, occurrences.get(last) + 1);
            return;
        }

        int node = length.size();
        length.add(length.get(current) + 2);
        next.add(new HashMap<>());
        palindrome.add(length.get(node) == 1
                ? String.valueOf(ch) : ch + palindrome.get(current) + ch);
        if (length.get(node) == 1) {
            link.add(1);
        } else {
            int candidate = link.get(current);
            while (true) {
                int index = position - 1 - length.get(candidate);
                if (index >= 0 && text.charAt(index) == ch) break;
                candidate = link.get(candidate);
            }
            link.add(next.get(candidate).get(ch));
        }
        next.get(current).put(ch, node);
        occurrences.add(1);
        last = node;
    }

    int distinctCount() {
        return length.size() - 2;
    }

    /** Map palindrome -> full occurrence count, for every distinct palindromic substring. */
    Map<String, Integer> counts() {
        Map<String, Integer> result = new TreeMap<>();
        for (int node = 2; node < length.size(); node++) {
            result.put(palindrome.get(node), occurrences.get(node));
        }
        return result;
    }
}

public class Solution {
    private static int seed = 1717;

    static Set<String> bruteDistinct(String text) {
        Set<String> found = new HashSet<>();
        for (int start = 0; start < text.length(); start++) {
            for (int end = start + 1; end <= text.length(); end++) {
                String piece = text.substring(start, end);
                if (piece.contentEquals(new StringBuilder(piece).reverse())) found.add(piece);
            }
        }
        return found;
    }

    static Map<String, Integer> bruteCounts(String text) {
        Map<String, Integer> counts = new TreeMap<>();
        for (String piece : bruteDistinct(text)) {
            int size = piece.length();
            int total = 0;
            for (int start = 0; start + size <= text.length(); start++) {
                if (text.substring(start, start + size).equals(piece)) total++;
            }
            counts.put(piece, total);
        }
        return counts;
    }

    private static String join(Map<String, Integer> counts) {
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<String, Integer> entry : new TreeMap<>(counts).entrySet()) {
            builder.append(entry.getKey()).append('=').append(entry.getValue()).append(';');
        }
        return builder.toString();
    }

    public static void main(String[] args) {
        // Hand-checked: "abba" has distinct palindromes {a, b, bb, abba}.
        Eertree tree = new Eertree("abba");
        check(tree.distinctCount() == 4, "abba distinct count");
        check(join(tree.counts()).equals("a=2;abba=1;b=2;bb=1;"), "abba counts");
        check(join(tree.counts()).equals(join(bruteCounts("abba"))), "abba matches brute force");
        check(tree.distinctCount() == bruteDistinct("abba").size(), "abba distinct vs brute");

        // "aaaa": four distinct palindromes with counts 4, 3, 2, 1.
        Eertree four = new Eertree("aaaa");
        check(four.distinctCount() == 4, "aaaa distinct count");
        check(join(four.counts()).equals("a=4;aa=3;aaa=2;aaaa=1;"), "aaaa counts");
        check(join(four.counts()).equals(join(bruteCounts("aaaa"))), "aaaa matches brute force");

        // Edge cases.
        check(new Eertree("").distinctCount() == 0, "empty string");
        check(new Eertree("").counts().isEmpty(), "empty counts");
        Eertree single = new Eertree("x");
        check(single.distinctCount() == 1 && join(single.counts()).equals("x=1;"), "single char");
        check(new Eertree("abcd").distinctCount() == 4, "no long palindromes");
        check(join(new Eertree("abcd").counts()).equals("a=1;b=1;c=1;d=1;"), "abcd counts");
        check(join(new Eertree("aba").counts()).equals("a=2;aba=1;b=1;"), "aba counts");
        check(join(new Eertree("abccba").counts()).equals("a=2;abccba=1;b=2;bccb=1;c=2;cc=1;"),
                "abccba counts");

        // LCG strings: distinct palindromes and every occurrence count must match brute force.
        for (int trial = 0; trial < 60; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int size = seed % 40;
            seed = (seed * 75 + 74) % 65537;
            String alphabet = seed % 2 == 0 ? "ab" : "abc";
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < size; i++) {
                seed = (seed * 75 + 74) % 65537;
                builder.append(alphabet.charAt(seed % alphabet.length()));
            }
            String text = builder.toString();
            Eertree random = new Eertree(text);
            Map<String, Integer> expected = bruteCounts(text);
            check(join(random.counts()).equals(join(expected)), "random counts " + trial + " " + text);
            check(random.distinctCount() == expected.size(), "random distinct " + trial + " " + text);
            for (Map.Entry<String, Integer> entry : random.counts().entrySet()) {
                String piece = entry.getKey();
                check(piece.equals(new StringBuilder(piece).reverse().toString()), "palindrome");
                check(entry.getValue() >= 1 && piece.length() <= text.length(), "count bounds");
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
