/** Task 116 - Insert and search in trie. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

class ArrayNode {
    final ArrayNode[] kids = new ArrayNode[26];
    boolean end;
}

class WordTrie {
    final ArrayNode root = new ArrayNode();

    void insert(String word) {
        ArrayNode node = root;
        for (char c : word.toCharArray()) {
            int i = c - 'a';
            if (node.kids[i] == null) node.kids[i] = new ArrayNode();
            node = node.kids[i];
        }
        node.end = true;
    }

    boolean search(String word) {
        ArrayNode node = nodeFor(word);
        return node != null && node.end;
    }

    int countWithPrefix(String prefix) {
        ArrayNode node = nodeFor(prefix);
        return node == null ? 0 : count(node);
    }

    List<String> wordsWithPrefix(String prefix) {
        List<String> out = new ArrayList<>();
        ArrayNode node = nodeFor(prefix);
        if (node != null) collect(node, new StringBuilder(prefix), out);
        return out;
    }

    private ArrayNode nodeFor(String s) {
        ArrayNode node = root;
        for (char c : s.toCharArray()) {
            node = node.kids[c - 'a'];
            if (node == null) return null;
        }
        return node;
    }

    private int count(ArrayNode node) {
        int total = node.end ? 1 : 0;
        for (ArrayNode kid : node.kids) if (kid != null) total += count(kid);
        return total;
    }

    private void collect(ArrayNode node, StringBuilder path, List<String> out) {
        if (node.end) out.add(path.toString());
        for (int i = 0; i < 26; i++)
            if (node.kids[i] != null) {
                path.append((char) ('a' + i));
                collect(node.kids[i], path, out);
                path.deleteCharAt(path.length() - 1);
            }
    }
}

public class Solution {
    public static void main(String[] args) {
        WordTrie t = new WordTrie();
        for (String w : new String[]{"app", "apple", "apply", "apt", "banana"}) t.insert(w);
        check(t.search("app") && t.search("apply"), "search hits");
        check(!t.search("ap") && !t.search("apples"), "search misses");
        check(t.countWithPrefix("app") == 3, "count app");
        check(t.countWithPrefix("b") == 1 && t.countWithPrefix("z") == 0, "count b/z");
        check(t.wordsWithPrefix("app").toString().equals("[app, apple, apply]"), "collect");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
