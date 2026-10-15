/** Task 253 - Implement compressed trie (radix tree) basics. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class RadixNode {
    String label;
    final Map<Character, RadixNode> children = new TreeMap<>();
    boolean isWord;

    RadixNode(String label) {
        this.label = label;
    }
}

public class Solution {
    static boolean insert(RadixNode root, String word) {
        RadixNode node = root;
        String rest = word;
        while (true) {
            if (rest.isEmpty()) {
                boolean wasNew = !node.isWord;
                node.isWord = true;
                return wasNew;
            }
            char first = rest.charAt(0);
            RadixNode child = node.children.get(first);
            if (child == null) {
                RadixNode fresh = new RadixNode(rest);
                fresh.isWord = true;
                node.children.put(first, fresh);
                return true;
            }
            int common = 0;
            while (common < rest.length() && common < child.label.length()
                    && rest.charAt(common) == child.label.charAt(common)) common++;
            if (common == child.label.length()) {              // consumed the whole edge
                node = child;
                rest = rest.substring(common);
                continue;
            }
            RadixNode split = new RadixNode(child.label.substring(0, common));   // split edge
            child.label = child.label.substring(common);
            split.children.put(child.label.charAt(0), child);
            node.children.put(first, split);
            rest = rest.substring(common);
            if (rest.isEmpty()) {
                split.isWord = true;
            } else {
                RadixNode fresh = new RadixNode(rest);
                fresh.isWord = true;
                split.children.put(rest.charAt(0), fresh);
            }
            return true;
        }
    }

    static boolean contains(RadixNode root, String word) {
        RadixNode node = root;
        int i = 0;
        while (i < word.length()) {
            RadixNode child = node.children.get(word.charAt(i));
            if (child == null || !word.substring(i).startsWith(child.label)) return false;
            i += child.label.length();
            node = child;
        }
        return node.isWord;
    }

    static List<String> wordsWithPrefix(RadixNode root, String prefix) {
        RadixNode node = root;
        String soFar = "";
        int i = 0;
        while (i < prefix.length()) {
            RadixNode child = node.children.get(prefix.charAt(i));
            if (child == null) return new ArrayList<>();
            int want = prefix.length() - i;
            int take = Math.min(want, child.label.length());
            if (!child.label.substring(0, take).equals(prefix.substring(i, i + take))) {
                return new ArrayList<>();
            }
            node = child;
            soFar += child.label;
            if (want <= child.label.length()) break;           // prefix ends inside this edge
            i += child.label.length();
        }
        List<String> out = new ArrayList<>();
        collect(node, soFar, out);
        return out;
    }

    static void collect(RadixNode node, String soFar, List<String> out) {
        if (node.isWord) out.add(soFar);
        for (RadixNode child : node.children.values()) collect(child, soFar + child.label, out);
    }

    static boolean startsWith(RadixNode root, String prefix) {
        return !wordsWithPrefix(root, prefix).isEmpty();
    }

    /** Deterministic compressed rendering: labels glued by commas, '*' = stored word. */
    static String shape(RadixNode node) {
        String text = node.label + (node.isWord ? "*" : "");
        if (node.children.isEmpty()) return text;
        List<String> parts = new ArrayList<>();
        for (RadixNode child : node.children.values()) parts.add(shape(child));
        return text + "[" + String.join(",", parts) + "]";
    }

    /** Number of nodes excluding the root, i.e. the number of edges. */
    static int nodeCount(RadixNode root) {
        int total = root.children.size();
        for (RadixNode child : root.children.values()) total += nodeCount(child);
        return total;
    }

    static int plainTrieEdges(String[] words) {
        Set<String> prefixes = new TreeSet<>();
        for (String word : words) {
            for (int i = 1; i <= word.length(); i++) prefixes.add(word.substring(0, i));
        }
        return prefixes.size();
    }

    public static void main(String[] args) {
        RadixNode root = new RadixNode("");
        check(insert(root, "cart"), "cart inserted");
        check(shape(root).equals("[cart*]"), "cart alone");
        check(insert(root, "car"), "car splits the edge");
        check(shape(root).equals("[car*[t*]]"), "split: car* + t*");
        check(insert(root, "care"), "care joins under car");
        check(shape(root).equals("[car*[e*,t*]]"), "compressed shape car*[e*,t*]");
        check(!insert(root, "car"), "duplicate word");

        check(contains(root, "car") && contains(root, "care") && contains(root, "cart"), "lookups");
        for (String miss : new String[]{"ca", "c", "carts", "caret", "card", ""}) {
            check(!contains(root, miss), "not a word: " + miss);
        }

        check(nodeCount(root) == 3, "three edges");
        check(plainTrieEdges(new String[]{"cart", "car", "care"}) == 5, "plain trie edges");
        check(nodeCount(root) < plainTrieEdges(new String[]{"cart", "car", "care"}),
                "compressed smaller than plain");

        check(insert(root, "do") && insert(root, "dot") && insert(root, "dog"), "do family");
        check(shape(root).equals("[car*[e*,t*],do*[g*,t*]]"), "two families");

        check(wordsWithPrefix(root, "car").toString().equals("[car, care, cart]"), "car prefix");
        check(wordsWithPrefix(root, "ca").toString().equals("[car, care, cart]"), "prefix inside edge");
        check(wordsWithPrefix(root, "care").toString().equals("[care]"), "care prefix");
        check(wordsWithPrefix(root, "do").toString().equals("[do, dog, dot]"), "do prefix");
        check(wordsWithPrefix(root, "d").toString().equals("[do, dog, dot]"), "d prefix");
        check(wordsWithPrefix(root, "").toString().equals("[car, care, cart, do, dog, dot]"), "all words");
        check(wordsWithPrefix(root, "z").isEmpty(), "no z words");
        check(wordsWithPrefix(root, "carts").isEmpty(), "carts missing");
        check(startsWith(root, "car") && !startsWith(root, "cat"), "startsWith");

        check(nodeCount(root) == 6, "six edges total");
        check(plainTrieEdges(new String[]{"car", "care", "cart", "do", "dog", "dot"}) == 9,
                "plain trie has 9");
        for (String word : new String[]{"car", "care", "cart", "do", "dog", "dot"}) {
            check(contains(root, word), "still stored: " + word);
        }
        for (String miss : new String[]{"co", "dat", "dodge", "done"}) {
            check(!contains(root, miss), "still missing: " + miss);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
