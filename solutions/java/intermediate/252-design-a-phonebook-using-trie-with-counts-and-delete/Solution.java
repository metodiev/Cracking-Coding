/** Task 252 - Design a phonebook using trie with counts and delete. (Tier: Intermediate, CTCI Ch 4) */
import java.util.*;

class TrieNode {
    final Map<Character, TrieNode> children = new TreeMap<>();
    int count;
}

class Phonebook {
    final TrieNode root = new TrieNode();

    private TrieNode walk(String name) {
        TrieNode node = root;
        for (int i = 0; i < name.length(); i++) {
            node = node.children.get(name.charAt(i));
            if (node == null) return null;
        }
        return node;
    }

    int insert(String name) {
        TrieNode node = root;
        for (int i = 0; i < name.length(); i++) {
            char ch = name.charAt(i);
            TrieNode child = node.children.get(ch);
            if (child == null) {
                child = new TrieNode();
                node.children.put(ch, child);
            }
            node = child;
        }
        return ++node.count;
    }

    int count(String name) {
        TrieNode node = walk(name);
        return node == null ? 0 : node.count;
    }

    boolean delete(String name) {
        if (name.isEmpty()) return false;
        return deleteRec(root, name, 0);
    }

    private boolean deleteRec(TrieNode node, String name, int i) {
        if (i == name.length()) {
            if (node.count == 0) return false;
            node.count--;
            return true;
        }
        char ch = name.charAt(i);
        TrieNode child = node.children.get(ch);
        if (child == null) return false;
        if (!deleteRec(child, name, i + 1)) return false;
        if (child.count == 0 && child.children.isEmpty()) node.children.remove(ch);
        return true;
    }

    /** All stored (name, count) pairs starting with prefix, lexicographically sorted. */
    List<String> listPrefix(String prefix) {
        List<String> out = new ArrayList<>();
        TrieNode node = walk(prefix);
        if (node == null) return out;
        collect(node, prefix, out);
        return out;
    }

    private void collect(TrieNode node, String path, List<String> out) {
        if (node.count > 0) out.add(path + "=" + node.count);
        for (Map.Entry<Character, TrieNode> e : node.children.entrySet()) {
            collect(e.getValue(), path + e.getKey(), out);
        }
    }

    List<String> words() {
        return listPrefix("");
    }

    int nodeCount() {
        int total = 0;
        Deque<TrieNode> stack = new ArrayDeque<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TrieNode node = stack.pop();
            total++;
            for (TrieNode child : node.children.values()) stack.push(child);
        }
        return total;
    }
}

public class Solution {
    static final String ALL = "[alan=1, albert=1, alice=2, b=1, barb=1, bob=1]";

    static Phonebook load() {
        Phonebook book = new Phonebook();
        check(book.insert("alice") == 1, "first alice");
        check(book.insert("alice") == 2, "duplicate increments");
        check(book.insert("alan") == 1, "alan");
        check(book.insert("albert") == 1, "albert");
        check(book.insert("bob") == 1, "bob");
        check(book.insert("barb") == 1, "barb");
        check(book.insert("b") == 1, "b");
        return book;
    }

    public static void main(String[] args) {
        Phonebook empty = new Phonebook();
        check(empty.nodeCount() == 1, "empty phonebook is a bare root");

        Phonebook book = load();
        // 1 root + 17 shared-prefix nodes: a,al,ala,alan,alb,albe,alber,albert,ali,alic,
        // alice,b,ba,bar,barb,bo,bob
        check(book.nodeCount() == 18, "shared prefixes share nodes");

        check(book.count("alice") == 2, "alice exact count");
        check(book.count("alan") == 1, "alan exact count");
        check(book.count("alicia") == 0, "unknown name");
        check(book.count("al") == 0, "prefix is not a name");
        check(book.count("alic") == 0, "deep prefix is not a name");
        check(book.count("") == 0, "empty name");

        check(book.listPrefix("al").toString().equals("[alan=1, albert=1, alice=2]"), "al prefix");
        check(book.listPrefix("ali").toString().equals("[alice=2]"), "ali prefix");
        check(book.listPrefix("b").toString().equals("[b=1, barb=1, bob=1]"), "b prefix");
        check(book.listPrefix("z").isEmpty(), "no z names");
        check(book.listPrefix("").toString().equals(ALL), "all names sorted");
        check(book.words().toString().equals(ALL), "words()");

        check(book.delete("alice"), "first alice delete");
        check(book.count("alice") == 1 && book.nodeCount() == 18, "count decremented");
        check(book.listPrefix("al").toString().equals("[alan=1, albert=1, alice=1]"), "alice still listed");
        check(book.delete("alice"), "second alice delete");
        check(book.count("alice") == 0 && book.nodeCount() == 15, "alice/alic/ali pruned");
        check(book.listPrefix("al").toString().equals("[alan=1, albert=1]"), "alice removed");
        check(book.listPrefix("ali").isEmpty(), "ali branch pruned");
        check(!book.delete("alice"), "third alice delete fails");
        check(!book.delete("zzz"), "unknown delete fails");
        check(!book.delete(""), "empty delete fails");

        check(book.delete("b"), "b count hits zero");
        check(book.count("b") == 0 && book.nodeCount() == 15, "b node kept for bob/barb");
        check(book.listPrefix("b").toString().equals("[barb=1, bob=1]"), "b no longer a name");

        check(book.delete("barb"), "barb delete");
        check(book.nodeCount() == 12, "barb/bar/ba pruned");
        check(book.words().toString().equals("[alan=1, albert=1, bob=1]"), "remaining words");

        check(book.delete("alan"), "alan delete");
        check(book.nodeCount() == 10, "alan/ala pruned");
        check(book.delete("albert"), "albert delete");
        check(book.nodeCount() == 4, "root + b + bo + bob remain");
        check(book.words().toString().equals("[bob=1]"), "only bob left");

        check(book.delete("bob"), "bob delete");
        check(book.nodeCount() == 1, "back to the empty state");
        check(book.words().isEmpty() && book.listPrefix("").isEmpty(), "nothing left");
        check(!book.delete("bob"), "double delete fails");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
