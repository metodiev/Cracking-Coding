/** Task 117 - Autocomplete top-k prefix suggestions (trie). (Tier: Beginner, CTCI Ch 4) */
import java.util.*;

class AcNode {
    final Map<Character, AcNode> kids = new HashMap<>();
    int weight;
    int best;
}

class Autocomplete {
    final AcNode root = new AcNode();

    void add(String word, int weight) {
        AcNode node = root;
        for (char c : word.toCharArray())
            node = node.kids.computeIfAbsent(c, k -> new AcNode());
        node.weight += weight;
        recompute(root);
    }

    private int recompute(AcNode node) {
        int best = node.weight;
        for (AcNode kid : node.kids.values()) best = Math.max(best, recompute(kid));
        node.best = best;
        return best;
    }

    List<String> suggest(String prefix, int k) {
        AcNode node = root;
        for (char c : prefix.toCharArray()) {
            node = node.kids.get(c);
            if (node == null) return List.of();
        }
        List<String[]> found = new ArrayList<>();
        collect(node, new StringBuilder(prefix), found);
        found.sort((a, b) -> {
            int w = Integer.compare(Integer.parseInt(b[1]), Integer.parseInt(a[1]));
            return w != 0 ? w : a[0].compareTo(b[0]);
        });
        List<String> out = new ArrayList<>();
        for (int i = 0; i < Math.min(k, found.size()); i++) out.add(found.get(i)[0]);
        return out;
    }

    private void collect(AcNode node, StringBuilder path, List<String[]> out) {
        if (node.weight > 0) out.add(new String[]{path.toString(), String.valueOf(node.weight)});
        for (Map.Entry<Character, AcNode> e : node.kids.entrySet()) {
            path.append(e.getKey());
            collect(e.getValue(), path, out);
            path.deleteCharAt(path.length() - 1);
        }
    }
}

public class Solution {
    public static void main(String[] args) {
        Autocomplete ac = new Autocomplete();
        ac.add("car", 5);
        ac.add("cart", 3);
        ac.add("carbon", 2);
        ac.add("care", 1);
        ac.add("dog", 9);

        check(ac.suggest("car", 2).toString().equals("[car, cart]"), "top 2 by popularity");
        check(ac.suggest("car", 10).toString().equals("[car, cart, carbon, care]"), "all four");
        check(ac.suggest("do", 5).toString().equals("[dog]"), "do -> dog");
        check(ac.suggest("z", 3).isEmpty(), "unknown prefix");
        check(ac.suggest("", 1).toString().equals("[dog]"), "top overall");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
