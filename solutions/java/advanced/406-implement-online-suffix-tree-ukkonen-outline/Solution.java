/** Task 406 - Implement an online suffix tree (Ukkonen outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Outline: a real Ukkonen tree builds the *compressed* suffix tree online in O(n) by extending
 * the last suffix, following a suffix link when a split happens, and keeping the suffix link
 * of every new node correct. This outline keeps the same skeleton for tiny strings: an
 * explicit suffix trie of the text (every node is a distinct substring, edges are single
 * characters) plus the suffix link of each node, which points at the node of its path label
 * minus the first character. That link is exactly the invariant Ukkonen restores when it
 * splits an edge, and it is what makes "ends with suffix link" navigation possible here:
 * following links from a node walks through all suffixes of its path in O(depth) steps.
 *
 * Links are wired breadth-first: a depth-1 node links to the root, and a node whose parent p
 * is reached over character c links to children[link[p]][c], which always exists in a suffix
 * trie.
 *
 * Checks on tiny texts only (banana, mississippi, abab, aaaa): every substring is findable
 * and every generated non-substring fails, node count matches the brute-force distinct
 * substring set, each link target really is the node of path[1:], link chains enumerate all
 * suffixes, and subtree terminal counts reproduce naive occurrence counts.
 */
import java.util.*;

class SuffixTreeOutline {
    final String text;
    final List<Map<Character, Integer>> children = new ArrayList<>();
    final List<Integer> parent = new ArrayList<>();
    final List<Character> parentChar = new ArrayList<>();
    final List<Integer> depth = new ArrayList<>();
    final List<Boolean> terminal = new ArrayList<>();
    final List<Integer> link = new ArrayList<>();

    SuffixTreeOutline(String text) {
        this.text = text;
        children.add(new HashMap<>());
        parent.add(-1);
        parentChar.add('\0');
        depth.add(0);
        terminal.add(false);
        link.add(-1);
        for (int start = 0; start < text.length(); start++) {
            int node = 0;
            for (int pos = start; pos < text.length(); pos++) {
                char ch = text.charAt(pos);
                Integer child = children.get(node).get(ch);
                if (child == null) {
                    child = depth.size();
                    children.get(node).put(ch, child);
                    children.add(new HashMap<>());
                    parent.add(node);
                    parentChar.add(ch);
                    depth.add(depth.get(node) + 1);
                    terminal.add(false);
                    link.add(-1);
                }
                node = child;
            }
            terminal.set(node, true);
        }
        wireSuffixLinks();
    }

    private void wireSuffixLinks() {
        List<Integer> order = new ArrayList<>();
        order.add(0);
        for (int i = 0; i < order.size(); i++) order.addAll(children.get(order.get(i)).values());
        for (int node : order) {
            if (node == 0) continue;
            if (depth.get(node) == 1) {
                link.set(node, 0);
            } else {
                int p = parent.get(node);
                link.set(node, children.get(link.get(p)).get(parentChar.get(node)));
            }
        }
    }

    String pathLabel(int node) {
        StringBuilder sb = new StringBuilder();
        while (node != 0) {
            sb.append(parentChar.get(node));
            node = parent.get(node);
        }
        return sb.reverse().toString();
    }

    int find(String pattern) {
        int node = 0;
        for (int i = 0; i < pattern.length(); i++) {
            Integer child = children.get(node).get(pattern.charAt(i));
            if (child == null) return -1;
            node = child;
        }
        return node;
    }

    boolean contains(String pattern) {
        return find(pattern) != -1;
    }

    List<Integer> suffixChain(int node) {
        List<Integer> chain = new ArrayList<>();
        while (node != -1) {
            chain.add(node);
            node = link.get(node);
        }
        return chain;
    }

    int occurrences(String pattern) {
        int node = find(pattern);
        if (node == -1) return 0;
        int count = 0;
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(node);
        while (!stack.isEmpty()) {
            int current = stack.pop();
            if (terminal.get(current)) count++;
            stack.addAll(children.get(current).values());
        }
        return count;
    }

    int distinctSubstrings() {
        return depth.size() - 1;
    }
}

class OutlineTools {
    static Set<String> allSubstrings(String s) {
        Set<String> seen = new LinkedHashSet<>();
        for (int i = 0; i < s.length(); i++)
            for (int j = i + 1; j <= s.length(); j++)
                seen.add(s.substring(i, j));
        return seen;
    }

    static int countOccurrences(String s, String pattern) {
        if (pattern.isEmpty()) return 0;
        int count = 0;
        for (int i = 0; i + pattern.length() <= s.length(); i++)
            if (s.startsWith(pattern, i)) count++;
        return count;
    }

    static List<String> enumerateStrings(String alphabet, int maxLen) {
        List<String> out = new ArrayList<>();
        List<String> frontier = new ArrayList<>();
        frontier.add("");
        for (int depth = 0; depth <= maxLen; depth++) {
            out.addAll(frontier);
            List<String> nextFrontier = new ArrayList<>();
            for (String word : frontier)
                for (int c = 0; c < alphabet.length(); c++) nextFrontier.add(word + alphabet.charAt(c));
            frontier = nextFrontier;
        }
        return out;
    }

    static void checkOutline(String text, String alphabet, int probeLen) {
        SuffixTreeOutline tree = new SuffixTreeOutline(text);
        Set<String> substrings = allSubstrings(text);
        Solution.check(tree.distinctSubstrings() == substrings.size(), "distinct " + text);
        Solution.check(tree.contains("") && tree.find("") == 0, "root " + text);
        for (String t : substrings) {
            int node = tree.find(t);
            Solution.check(node != -1, "find " + t);
            Solution.check(tree.pathLabel(node).equals(t), "label " + t);
            Solution.check(tree.occurrences(t) == countOccurrences(text, t), "occ " + t);
        }
        for (String t : enumerateStrings(alphabet, probeLen)) {
            if (!t.isEmpty() && !substrings.contains(t)) Solution.check(!tree.contains(t), "reject " + t);
        }
        for (int node = 1; node < tree.depth.size(); node++) {
            String label = tree.pathLabel(node);
            int link = tree.link.get(node);
            Solution.check(link != -1, "link exists " + label);
            Solution.check(tree.pathLabel(link).equals(label.substring(1)), "link label " + label);
            List<Integer> chain = tree.suffixChain(node);
            Solution.check(chain.size() == label.length() + 1, "chain size " + label);
            List<String> labels = new ArrayList<>();
            for (int n : chain) labels.add(tree.pathLabel(n));
            List<String> expected = new ArrayList<>();
            for (int k = 0; k < label.length(); k++) expected.add(label.substring(k));
            expected.add("");
            Solution.check(labels.equals(expected), "chain labels " + label);
            Solution.check(tree.depth.get(link) == tree.depth.get(node) - 1, "link depth " + label);
        }
    }
}

public class Solution {
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

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
        OutlineTools.checkOutline("banana", "abn", 4);
        SuffixTreeOutline banana = new SuffixTreeOutline("banana");
        check(banana.occurrences("ana") == 2, "ana");
        check(banana.occurrences("a") == 3, "a");
        check(banana.occurrences("na") == 2, "na");
        check(banana.occurrences("banana") == 1, "banana");
        check(!banana.contains("bananas") && !banana.contains("nana-"), "rejects");

        OutlineTools.checkOutline("mississippi", "imps", 3);
        SuffixTreeOutline mississippi = new SuffixTreeOutline("mississippi");
        check(mississippi.occurrences("issi") == 2, "issi");
        check(mississippi.occurrences("ssi") == 2, "ssi");
        check(mississippi.occurrences("i") == 4, "i");

        OutlineTools.checkOutline("abab", "ab", 6);
        OutlineTools.checkOutline("aaaa", "a", 6);
        OutlineTools.checkOutline("abcab", "abc", 5);
        OutlineTools.checkOutline("", "ab", 2);
        OutlineTools.checkOutline("z", "ab", 2);

        for (int trial = 0; trial < 20; trial++) OutlineTools.checkOutline(lcgString(8, "ab"), "ab", 3);
        for (int trial = 0; trial < 10; trial++) OutlineTools.checkOutline(lcgString(7, "abc"), "abc", 3);

        // online flavour: one more character extends the outline and keeps every invariant
        SuffixTreeOutline base = new SuffixTreeOutline("aban");
        check(base.occurrences("a") == 2, "base a");
        SuffixTreeOutline extended = new SuffixTreeOutline("abana");
        check(extended.distinctSubstrings() == OutlineTools.allSubstrings("abana").size(), "extended distinct");
        check(extended.occurrences("a") == 3, "extended a");
        check(extended.pathLabel(extended.link.get(extended.find("bana"))).equals("ana"), "extended link");
        System.out.println("PASS");
    }
}
