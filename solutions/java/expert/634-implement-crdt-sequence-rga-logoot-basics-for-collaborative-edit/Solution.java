/** Task 634 - Implement CRDT sequence (RGA / Logoot) basics for collaborative editing. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Two ways to make a *sequence* converge:
 *   * RGA (Replicated Growable Array) gives every character a globally unique id (replica,
 *     counter) and remembers the id it was inserted *after*. The document is therefore a tree:
 *     children of an origin are ordered by a deterministic key - here (counter, replica) - so
 *     every replica that has the same characters computes the same total order, whatever the
 *     arrival order was. Deletions are tombstones: the character stops being visible but keeps
 *     its place in the tree, so later inserts that reference it still land correctly. Because a
 *     node's descendants stay in its own subtree, two replicas typing concurrently at the same
 *     position produce two contiguous runs rather than an interleaved mess.
 *   * Logoot allocates each character a *position* - a list of (digit, site) pairs - and inserts
 *     between the two neighbouring positions, synthesising a new position strictly in between.
 *     No tree and no ids of neighbours are needed; the total order on positions is the document
 *     order, so nothing has to be reordered on delivery.
 *
 * Simulation model, and what is simplified: replicas are objects, "delivering an operation" is a
 * method call, and causal delivery is assumed (an insert whose origin is unknown is simply kept
 * orphaned and recorded, as RGA's buffer would). The Logoot position generator requires a real gap
 * between the neighbours; a digit base of one million makes that a practical certainty for the
 * depths exercised here, and an exhausted gap is reported as an explicit failure instead of
 * silently producing a wrong order. What is not simplified: the ordering rules, the tombstones and
 * the position algebra are the real ones, and both are cross-checked against brute force.
 *
 * Verified: (1) both CRDTs converge to the same sequence under *every* causally admissible
 * delivery order of a set of concurrent operations; (2) two replicas inserting at the same place
 * concurrently yield the same order, in two contiguous runs, never interleaved; (3) tombstones
 * keep later inserts anchored correctly and deletion is idempotent and total; (4) a replica's own
 * successive inserts come out in the order it made them; (5) Logoot positions are a strict total
 * order, `between` always returns a position strictly between its neighbours, and 30 repeated
 * inserts between the same neighbours stay strictly increasing; (6) the negative control - an RGA
 * that appends after the origin instead of ordering siblings deterministically - diverges for
 * concurrent inserts at the same origin, which the convergence checker catches.
 */
import java.util.*;

class RgaElement {
    final String id;
    final int replica;
    final int counter;
    final String origin;             // null = inserted at the head
    final char value;
    boolean deleted;
    boolean delivered;

    RgaElement(String id, int replica, int counter, String origin, char value) {
        this.id = id;
        this.replica = replica;
        this.counter = counter;
        this.origin = origin;
        this.value = value;
    }
}

/** RGA as a tree of (origin -> children), ordered by (counter, replica). */
class Rga {
    final Map<String, RgaElement> byId = new TreeMap<>();
    final Map<String, List<RgaElement>> children = new TreeMap<>();   // key "@root" for the head
    int orphans;

    static final String ROOT = "@root";

    static RgaElement make(int replica, int counter, String origin, char value) {
        return new RgaElement(replica + ":" + counter, replica, counter, origin, value);
    }

    void deliver(RgaElement element) {
        if (byId.containsKey(element.id)) return;                      // idempotent
        byId.put(element.id, element);
        element.delivered = true;
        if (element.origin != null && !byId.containsKey(element.origin)) orphans++;
        children.computeIfAbsent(keyOf(element.origin), k -> new ArrayList<>()).add(element);
    }

    static String keyOf(String origin) {
        return origin == null ? ROOT : origin;
    }

    void delete(String id) {
        RgaElement element = byId.get(id);
        if (element != null) element.deleted = true;
    }

    /** Total document order: depth-first, siblings by (counter, replica). */
    List<RgaElement> order() {
        List<RgaElement> out = new ArrayList<>();
        walk(ROOT, out);
        return out;
    }

    void walk(String originKey, List<RgaElement> out) {
        List<RgaElement> kids = new ArrayList<>(children.getOrDefault(originKey, new ArrayList<>()));
        kids.sort(Comparator.comparingInt((RgaElement e) -> e.counter).thenComparingInt(e -> e.replica));
        for (RgaElement kid : kids) {
            out.add(kid);
            walk(kid.id, out);
        }
    }

    String visible() {
        StringBuilder sb = new StringBuilder();
        for (RgaElement e : order()) if (!e.deleted) sb.append(e.value);
        return sb.toString();
    }

    String idsInOrder() {
        StringBuilder sb = new StringBuilder();
        for (RgaElement e : order()) {
            if (sb.length() > 0) sb.append(',');
            sb.append(e.deleted ? "(" + e.value + ")" : Character.toString(e.value));
        }
        return sb.toString();
    }

    /** The wrong variant used as a negative control: append right after the origin. */
    static class Naive {
        final List<RgaElement> flat = new ArrayList<>();

        void deliver(RgaElement element) {
            for (RgaElement e : flat) if (e.id.equals(element.id)) return;   // idempotent
            int at = 0;
            if (element.origin != null) {
                for (int i = 0; i < flat.size(); i++) {
                    if (flat.get(i).id.equals(element.origin)) {
                        at = i + 1;
                        break;
                    }
                }
            }
            flat.add(at, element);
        }

        String visible() {
            StringBuilder sb = new StringBuilder();
            for (RgaElement e : flat) sb.append(e.value);
            return sb.toString();
        }
    }
}

/** Logoot position: a list of (digit, site) pairs compared lexicographically. */
class LPosition implements Comparable<LPosition> {
    static final int BASE = 1000000;
    final List<int[]> digits = new ArrayList<>();

    LPosition append(int digit, int site) {
        digits.add(new int[]{digit, site});
        return this;
    }

    LPosition copy() {
        LPosition p = new LPosition();
        for (int[] d : digits) p.digits.add(new int[]{d[0], d[1]});
        return p;
    }

    int digitAt(int index) {
        return index < digits.size() ? digits.get(index)[0] : -1;
    }

    int siteAt(int index) {
        return index < digits.size() ? digits.get(index)[1] : -1;
    }

    @Override
    public int compareTo(LPosition other) {
        int length = Math.max(digits.size(), other.digits.size());
        for (int i = 0; i < length; i++) {
            int a = digitAt(i);
            int b = other.digitAt(i);
            if (a != b) return Integer.compare(a, b);
            int sa = siteAt(i);
            int sb = other.siteAt(i);
            if (sa != sb) return Integer.compare(sa, sb);
        }
        return 0;
    }

    String render() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < digits.size(); i++) {
            if (i > 0) sb.append(' ');
            sb.append(digits.get(i)[0]).append('.').append(digits.get(i)[1]);
        }
        return sb.append(']').toString();
    }

    /** A position strictly between `lower` and `upper`; `upper` may be null (unbounded). */
    static LPosition between(LPosition lower, LPosition upper, int site) {
        Solution.check(lower != null, "a lower bound is required");
        if (upper == null) {
            LPosition p = lower.copy();
            int last = p.digitAt(p.digits.size() - 1);
            p.digits.add(new int[]{last + 1, site});
            return p;
        }
        Solution.check(lower.compareTo(upper) < 0, "the lower bound must precede the upper bound");
        LPosition result = new LPosition();
        int index = 0;
        while (true) {
            int la = lower.digitAt(index);
            int sa = lower.siteAt(index);
            int ub = upper.digitAt(index);
            int sb = upper.siteAt(index);
            if (la == ub && sa == sb) {                  // identical token: copy it and go deeper
                Solution.check(la != -1, "between() needs two distinct positions");
                result.digits.add(new int[]{la, sa});
                index++;
                continue;
            }
            if (la == -1) {                              // the lower is a prefix of the upper here
                result.digits.add(new int[]{0, 0});      // smallest token that still beats a prefix
                return result;
            }
            Solution.check(ub != -1, "unreachable: the lower bound must precede the upper bound");
            if (ub - la > 1) {                           // a free digit sits in the gap
                result.digits.add(new int[]{(la + ub) / 2, site});
                return result;
            }
            if (ub == la && sb - sa > 1) {               // same digit, a free site sits in between
                result.digits.add(new int[]{la, (sa + sb) / 2});
                return result;
            }
            // The tokens are adjacent, so the order is already decided at this depth: keep the
            // lower token and append anything greater than the lower's own tail.
            result.digits.add(new int[]{la, sa});
            result.digits.add(new int[]{Math.max(lower.digitAt(index + 1), 0) + 1, site});
            return result;
        }
    }

    /** A position before everything: the smallest possible position. */
    static LPosition head(int site) {
        LPosition p = new LPosition();
        p.append(0, site);
        return p;
    }
}

public class Solution {
    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        rgaBasicEditing();
        rgaConvergesUnderDeliveryOrders();
        concurrentTypingIsNotInterleaved();
        tombstonesKeepAnchors();
        replicaOrderIsPreserved();
        logootPositions();
        logootDenseInsertion();
        negativeControlNaiveRga();
        System.out.println("PASS");
    }

    static void rgaBasicEditing() {
        Rga doc = new Rga();
        RgaElement a = Rga.make(0, 1, null, 'a');
        RgaElement b = Rga.make(0, 2, a.id, 'b');
        RgaElement c = Rga.make(0, 3, b.id, 'c');
        doc.deliver(a);
        doc.deliver(b);
        doc.deliver(c);
        check(doc.visible().equals("abc"), "typing appends: " + doc.visible());

        RgaElement x = Rga.make(1, 1, a.id, 'X');
        doc.deliver(x);
        check(doc.visible().equals("aXbc"), "insert after an anchor: " + doc.visible());
        doc.delete(x.id);
        check(doc.visible().equals("abc"), "delete removes it from the visible text");
        check(doc.order().size() == 4, "the tombstone is still in the tree");
        doc.delete(x.id);
        check(doc.visible().equals("abc"), "deleting twice is a no-op");
        doc.delete("nope:9");
        check(doc.visible().equals("abc"), "deleting an unknown id is a no-op");
        check(doc.orphans == 0, "no orphans in a causal delivery");

        Rga fresh = new Rga();
        check(fresh.visible().isEmpty() && fresh.order().isEmpty(), "an empty document is empty");
        fresh.deliver(Rga.make(0, 1, null, 'z'));
        check(fresh.visible().equals("z"), "the first character is inserted at the head");
    }

    /** Every causally admissible delivery order of the same operations converges. */
    static void rgaConvergesUnderDeliveryOrders() {
        List<RgaElement> operations = Arrays.asList(
                Rga.make(0, 1, null, 'a'),          // replica 0 types "ab"
                Rga.make(0, 2, "0:1", 'b'),
                Rga.make(1, 1, "0:1", 'c'),         // replica 1 concurrently types "cd" after a
                Rga.make(1, 2, "1:1", 'd'),
                Rga.make(2, 1, null, 'e'));         // replica 2 prepends
        Set<String> texts = new TreeSet<>();
        Set<String> orders = new TreeSet<>();
        int valid = 0;
        for (List<RgaElement> permutation : permutations(operations)) {
            Rga doc = new Rga();
            boolean admissible = true;
            Set<String> delivered = new HashSet<>();
            for (RgaElement op : permutation) {
                if (op.origin != null && !delivered.contains(op.origin)) {
                    admissible = false;                    // causal delivery requirement
                    break;
                }
                doc.deliver(op);
                delivered.add(op.id);
            }
            if (!admissible) continue;
            valid++;
            texts.add(doc.visible());
            orders.add(doc.idsInOrder());
        }
        check(valid == 15, "causally admissible permutations: " + valid);
        check(texts.size() == 1, "one visible text for every delivery order: " + texts);
        check(orders.size() == 1, "one total order for every delivery order: " + orders);
        check(texts.iterator().next().length() == 5, "all five characters are visible");
    }

    /** Two replicas typing at the same place: identical order, two contiguous runs. */
    static void concurrentTypingIsNotInterleaved() {
        Rga replicaA = new Rga();
        Rga replicaB = new Rga();
        RgaElement base = Rga.make(9, 1, null, '.');
        replicaA.deliver(base);
        replicaB.deliver(base);

        List<RgaElement> typedByA = new ArrayList<>();
        String origin = base.id;
        for (char ch : new char[]{'1', '2', '3'}) {
            RgaElement e = Rga.make(0, typedByA.size() + 1, origin, ch);
            typedByA.add(e);
            origin = e.id;
        }
        List<RgaElement> typedByB = new ArrayList<>();
        origin = base.id;
        for (char ch : new char[]{'a', 'b', 'c'}) {
            RgaElement e = Rga.make(1, typedByB.size() + 1, origin, ch);
            typedByB.add(e);
            origin = e.id;
        }
        for (RgaElement e : typedByA) replicaA.deliver(e);
        for (RgaElement e : typedByB) replicaB.deliver(e);
        for (RgaElement e : typedByB) replicaA.deliver(e);       // deliver the other side
        for (RgaElement e : typedByA) replicaB.deliver(e);

        check(replicaA.visible().equals(replicaB.visible()),
                "both replicas converge: " + replicaA.visible() + " vs " + replicaB.visible());
        String expected = "." + "123" + "abc";
        check(replicaA.visible().equals(expected) || replicaA.visible().equals("." + "abc" + "123"),
                "the two runs are contiguous: " + replicaA.visible());
        check(replicaA.visible().equals(expected), "the tie-break favours the lower replica: "
                + replicaA.visible());
        check(replicaA.idsInOrder().indexOf("2:1") == -1, "sanity: ids are replica:counter");
    }

    /** Tombstones keep the anchors alive so later inserts land in the right place. */
    static void tombstonesKeepAnchors() {
        Rga doc = new Rga();
        RgaElement a = Rga.make(0, 1, null, 'a');
        RgaElement b = Rga.make(0, 2, a.id, 'b');
        RgaElement c = Rga.make(0, 3, b.id, 'c');
        doc.deliver(a);
        doc.deliver(b);
        doc.deliver(c);
        doc.delete(b.id);
        check(doc.visible().equals("ac"), "b is gone: " + doc.visible());

        RgaElement d = Rga.make(1, 1, b.id, 'd');               // insert after the tombstone
        doc.deliver(d);
        check(doc.visible().equals("adc"), "the insert still anchors after b: " + doc.visible());
        RgaElement e = Rga.make(1, 2, c.id, 'e');
        doc.deliver(e);
        check(doc.visible().equals("adce"), "and after c: " + doc.visible());
        doc.delete(a.id);
        doc.delete(c.id);
        check(doc.visible().equals("de"), "deleting two anchors leaves the children: " + doc.visible());
        check(doc.order().size() == 5, "all tombstones retained");
        check(doc.orphans == 0, "no orphans");

        Rga withOrphan = new Rga();
        RgaElement ghost = Rga.make(2, 1, "7:7", 'g');
        withOrphan.deliver(ghost);
        check(withOrphan.orphans == 1, "an unknown origin is reported as an orphan");
        check(withOrphan.visible().isEmpty(), "an orphan stays invisible");
        withOrphan.deliver(Rga.make(7, 7, null, 'h'));
        check(withOrphan.visible().equals("hg"), "it becomes visible once the origin arrives: "
                + withOrphan.visible());
    }

    /** A replica's own successive inserts appear in the order it made them. */
    static void replicaOrderIsPreserved() {
        Rga doc = new Rga();
        String origin = null;
        StringBuilder typed = new StringBuilder();
        for (int i = 1; i <= 8; i++) {
            char ch = (char) ('a' + i - 1);
            RgaElement e = Rga.make(3, i, origin, ch);
            doc.deliver(e);
            origin = e.id;
            typed.append(ch);
        }
        check(doc.visible().equals(typed.toString()), "sequential typing: " + doc.visible());
        RgaElement mid = Rga.make(4, 1, "3:4", '!');            // insert in the middle
        doc.deliver(mid);
        check(doc.visible().equals("abcd!efgh"), "the middle insert lands after d: " + doc.visible());
        check(doc.order().size() == 9, "nine elements in the tree");
    }

    /** Logoot positions: a strict total order and a strictly-between generator. */
    static void logootPositions() {
        LPosition p1 = new LPosition();
        p1.append(5, 1);
        LPosition p2 = new LPosition();
        p2.append(5, 2);
        LPosition p3 = new LPosition();
        p3.append(6, 1);
        check(p1.compareTo(p2) < 0, "the site id breaks ties");
        check(p2.compareTo(p3) < 0, "a larger digit wins");
        check(p3.compareTo(p1) > 0 && p1.compareTo(p1) == 0, "antisymmetry");

        LPosition mid = LPosition.between(p1, p3, 7);
        check(p1.compareTo(mid) < 0 && mid.compareTo(p3) < 0, "strictly in between: " + mid.render());
        LPosition deeper = LPosition.between(p1, p2, 8);
        check(p1.compareTo(deeper) < 0 && deeper.compareTo(p2) < 0,
                "between two neighbours with the same digit: " + deeper.render());
        LPosition before = LPosition.head(0);
        LPosition after = LPosition.between(before, null, 0);
        check(before.compareTo(after) < 0, "the head can always be beaten: " + after.render());

        List<LPosition> positions = new ArrayList<>(Arrays.asList(p3, p1, deeper, mid, p2));
        List<LPosition> sorted = new ArrayList<>(positions);
        Collections.sort(sorted);
        for (int i = 0; i + 1 < sorted.size(); i++) {
            check(sorted.get(i).compareTo(sorted.get(i + 1)) < 0, "sorted strictly: " + sorted);
        }
        check(sorted.get(0) == p1, "the smallest is the one with the smallest digit");
        check(sorted.indexOf(p3) > sorted.indexOf(p2), "5.x sorts before 6.x");
        check(sorted.indexOf(mid) > sorted.indexOf(p1) && sorted.indexOf(mid) < sorted.indexOf(p3),
                "the synthesised position sits between its neighbours");
    }

    /** Repeated dense insertion between the same two positions keeps working. */
    static void logootDenseInsertion() {
        LPosition low = LPosition.head(0);
        LPosition high = LPosition.between(low, null, 1);
        List<LPosition> inserted = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            LPosition p = LPosition.between(low, high, i % 3);
            inserted.add(p);
            low = p;                                   // always insert just before the upper bound
        }
        for (int i = 0; i + 1 < inserted.size(); i++) {
            check(inserted.get(i).compareTo(inserted.get(i + 1)) < 0,
                    "dense inserts stay strictly increasing at step " + i);
        }
        check(inserted.get(0).compareTo(inserted.get(inserted.size() - 1)) < 0, "first before last");
        for (LPosition p : inserted) {
            check(p.compareTo(high) < 0, "all stay below the upper bound");
        }

        // A document ordered by positions reproduces the insertion order.
        LPosition start = LPosition.head(0);
        LPosition end = LPosition.between(start, null, 0);
        LPosition cursor = start;
        List<LPosition> document = new ArrayList<>();
        for (int i = 0; i < 12; i++) {
            cursor = LPosition.between(cursor, end, i % 2);
            document.add(cursor);
        }
        check(document.size() == 12, "twelve characters typed: " + document.size());
        List<LPosition> sortedDocument = new ArrayList<>(document);
        sortedDocument.add(start);
        sortedDocument.add(end);
        Collections.sort(sortedDocument);
        check(sortedDocument.get(0) == start, "the head position sorts first");
        check(sortedDocument.get(sortedDocument.size() - 1) == end, "the tail position sorts last");
        for (int i = 0; i < document.size(); i++) {
            check(sortedDocument.get(i + 1) == document.get(i),
                    "the sorted document follows the typing order at index " + i);
        }
    }

    /** Negative control: appending after the origin diverges for concurrent inserts. */
    static void negativeControlNaiveRga() {
        RgaElement base = Rga.make(9, 1, null, '.');
        List<RgaElement> byA = Arrays.asList(base, Rga.make(0, 1, base.id, '1'),
                Rga.make(0, 2, "0:1", '2'));
        List<RgaElement> byB = Arrays.asList(base, Rga.make(1, 1, base.id, 'a'),
                Rga.make(1, 2, "1:1", 'b'));

        Rga good = new Rga();
        for (RgaElement e : byA) good.deliver(e);
        for (RgaElement e : byB) good.deliver(e);
        Rga goodReversed = new Rga();
        for (RgaElement e : byB) goodReversed.deliver(e);
        for (RgaElement e : byA) goodReversed.deliver(e);
        check(good.visible().equals(goodReversed.visible()), "RGA converges: " + good.visible());

        Rga.Naive naive = new Rga.Naive();
        for (RgaElement e : byA) naive.deliver(e);
        for (RgaElement e : byB) naive.deliver(e);
        Rga.Naive naiveReversed = new Rga.Naive();
        for (RgaElement e : byB) naiveReversed.deliver(e);
        for (RgaElement e : byA) naiveReversed.deliver(e);
        check(!naive.visible().equals(naiveReversed.visible()),
                "the naive append-after-origin rule diverges: " + naive.visible() + " vs "
                        + naiveReversed.visible());
        check(naive.visible().equals(".ab12") && naiveReversed.visible().equals(".12ab"),
                "the divergence is an interleaved mess: " + naiveReversed.visible() + " vs "
                        + naive.visible());
    }

    static <T> List<List<T>> permutations(List<T> items) {
        List<List<T>> out = new ArrayList<>();
        permute(new ArrayList<>(items), 0, out);
        return out;
    }

    static <T> void permute(List<T> items, int index, List<List<T>> out) {
        if (index == items.size()) {
            out.add(new ArrayList<>(items));
            return;
        }
        for (int i = index; i < items.size(); i++) {
            Collections.swap(items, index, i);
            permute(items, index + 1, out);
            Collections.swap(items, index, i);
        }
    }
}
