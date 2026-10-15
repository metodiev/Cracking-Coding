/** Task 115 - Implement a basic trie for lowercase words. (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

class TrieNode {
    final Map<Character, TrieNode> children = new HashMap<>();
    boolean isWord;
}

class Trie {
    final TrieNode root = new TrieNode();
    int wordCount = 0;

    void insert(String word) {
        TrieNode node = root;
        for (char c : word.toCharArray())
            node = node.children.computeIfAbsent(c, k -> new TrieNode());
        if (!node.isWord) {
            node.isWord = true;
            wordCount++;
        }
    }

    boolean contains(String word) {
        TrieNode node = walk(word);
        return node != null && node.isWord;
    }

    boolean startsWith(String prefix) {
        return walk(prefix) != null;
    }

    private TrieNode walk(String s) {
        TrieNode node = root;
        for (char c : s.toCharArray()) {
            node = node.children.get(c);
            if (node == null) return null;
        }
        return node;
    }
}

public class Solution {
    public static void main(String[] args) {
        Trie t = new Trie();
        for (String w : new String[]{"cat", "car", "card", "dog"}) t.insert(w);
        check(t.wordCount == 4, "four words");
        t.insert("cat");
        check(t.wordCount == 4, "duplicate not double counted");
        check(t.contains("cat") && t.contains("card"), "contains");
        check(!t.contains("ca") && !t.contains("care"), "prefixes are not words");
        check(t.startsWith("ca") && t.startsWith("") && !t.startsWith("z"), "startsWith");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
