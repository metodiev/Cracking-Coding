/** Task 665 - Implement operational transform with intention preservation tests. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Operational transformation over linear text. Characters are atoms with globally unique ids; an
 * operation is INSERT(visible position, payload) or DELETE(the atom ids it removes). Transformations
 * move an operation past a concurrent one:
 *
 *   insert vs insert - the loser of the tie (position, then replica priority) is shifted right by the
 *                      winner's length, so both sites agree on the interleaving;
 *   insert vs delete - the insert shifts left by the number of atoms the delete removes before it;
 *   delete vs delete - the later delete drops the atoms the earlier one already removed (set union);
 *   delete vs insert - unaffected, because deletions name atoms rather than positions.
 *
 * Correctness is pinned by the convergence property TP1: for every pair of operations generated from a
 * common base document, applying a then transform(b,a) must equal applying b then transform(a,b). The
 * check runs TP1 exhaustively over every insert/delete pair producible from "abc" and "abcd" (hundreds
 * of pairs), and then over three-site scenarios where every site applies its own operation first and
 * transforms each incoming one against everything it has already applied - all sites must converge to
 * the same text. Intention preservation is checked separately: an insert's payload must survive as a
 * contiguous substring and every atom a delete named must stay invisible in the converged document.
 *
 * Simplification (documented): deletions name character ids, so overlapping concurrent deletes need no
 * splitting and no delete/delete priority resolution; positions are only used for inserts.
 */
import java.util.*;

final class OtAtom {
    final int id;
    final int replica;
    final int sequence;
    final char ch;
    boolean visible = true;

    OtAtom(int id, int replica, int sequence, char ch) {
        this.id = id;
        this.replica = replica;
        this.sequence = sequence;
        this.ch = ch;
    }
}

final class TextOp {
    static final int INSERT = 0, DELETE = 1;
    final int type;
    final int replica;
    final int sequence;
    final int position;
    final String payload;
    final int[] targets;

    private TextOp(int type, int replica, int sequence, int position, String payload, int[] targets) {
        this.type = type;
        this.replica = replica;
        this.sequence = sequence;
        this.position = position;
        this.payload = payload;
        this.targets = targets;
    }

    static TextOp insert(int replica, int sequence, int position, String payload) {
        if (payload.isEmpty()) throw new IllegalArgumentException("empty insert payload");
        return new TextOp(INSERT, replica, sequence, position, payload, new int[0]);
    }

    static TextOp delete(int replica, int sequence, int[] targets) {
        if (targets.length == 0) throw new IllegalArgumentException("empty delete");
        return new TextOp(DELETE, replica, sequence, -1, "", targets);
    }

    TextOp withPosition(int newPosition) {
        return new TextOp(type, replica, sequence, newPosition, payload, targets);
    }

    TextOp withTargets(int[] newTargets) {
        return new TextOp(type, replica, sequence, position, payload, newTargets);
    }

    long priority() { return replica * 1_000_000L + sequence; }

    int[] createdIds() {
        int[] ids = new int[payload.length()];
        for (int i = 0; i < ids.length; i++) ids[i] = TextOp.atomId(replica, sequence + i);
        return ids;
    }

    static int atomId(int replica, int sequence) { return replica * 1_000_000 + sequence; }

    String describe() {
        return (type == INSERT ? "ins(" + payload + ")" : "del(" + targets.length + ")") + "@" + replica + "/"
                + sequence + " pos=" + position;
    }
}

final class OtDocument {
    final List<OtAtom> atoms = new ArrayList<>();

    static OtDocument of(String text) {
        OtDocument document = new OtDocument();
        for (int i = 0; i < text.length(); i++) {   // base characters live on a reserved replica id
            document.atoms.add(new OtAtom(TextOp.atomId(9, i), 9, i, text.charAt(i)));
        }
        return document;
    }

    OtDocument copy() {
        OtDocument clone = new OtDocument();
        for (OtAtom atom : atoms) {
            OtAtom copy = new OtAtom(atom.id, atom.replica, atom.sequence, atom.ch);
            copy.visible = atom.visible;
            clone.atoms.add(copy);
        }
        return clone;
    }

    String text() {
        StringBuilder sb = new StringBuilder();
        for (OtAtom atom : atoms) if (atom.visible) sb.append(atom.ch);
        return sb.toString();
    }

    List<Integer> visibleIds() {
        List<Integer> ids = new ArrayList<>();
        for (OtAtom atom : atoms) if (atom.visible) ids.add(atom.id);
        return ids;
    }

    Set<Integer> visibleIdSet() {
        return new HashSet<>(visibleIds());
    }

    int indexOf(int atomId) {
        for (int i = 0; i < atoms.size(); i++) if (atoms.get(i).id == atomId) return i;
        return -1;
    }

    private int listIndexOfVisible(int visibleIndex) {
        if (visibleIndex < 0) throw new IllegalArgumentException("negative position");
        int seen = 0;
        for (int i = 0; i < atoms.size(); i++) {
            if (!atoms.get(i).visible) continue;
            if (seen == visibleIndex) return i;
            seen++;
        }
        if (seen == visibleIndex) return atoms.size();
        throw new IllegalArgumentException("position " + visibleIndex + " past the end");
    }

    void apply(TextOp op) {
        if (op.type == TextOp.INSERT) {
            int at = listIndexOfVisible(op.position);
            for (int i = 0; i < op.payload.length(); i++) {
                int id = TextOp.atomId(op.replica, op.sequence + i);
                if (indexOf(id) >= 0) continue;
                atoms.add(at++, new OtAtom(id, op.replica, op.sequence + i, op.payload.charAt(i)));
            }
        } else {
            for (int id : op.targets) {
                int at = indexOf(id);
                if (at >= 0) atoms.get(at).visible = false;
            }
        }
    }

    boolean isVisible(int atomId) {
        int at = indexOf(atomId);
        return at >= 0 && atoms.get(at).visible;
    }

    /** Visible index of an atom, or -1 when it is invisible or absent. */
    int visibleIndexOf(int atomId) {
        int seen = 0;
        for (OtAtom atom : atoms) {
            if (!atom.visible) continue;
            if (atom.id == atomId) return seen;
            seen++;
        }
        return -1;
    }
}

final class Transform {
    /** Number of atoms `other` deletes that sit before `position` in the base document. */
    private static int deletedBefore(TextOp other, int position, OtDocument base) {
        int count = 0;
        for (int id : other.targets) {
            int index = base.visibleIndexOf(id);
            if (index >= 0 && index < position) count++;
        }
        return count;
    }

    /** `op` transformed to apply after `other`; both were generated against `base`. */
    static TextOp transform(TextOp op, TextOp other, OtDocument base) {
        if (op.type == TextOp.INSERT && other.type == TextOp.INSERT) {
            boolean otherPrecedes = other.position < op.position
                    || (other.position == op.position && other.priority() < op.priority());
            int shift = otherPrecedes ? other.payload.length() : 0;
            return op.withPosition(op.position + shift);
        }
        if (op.type == TextOp.INSERT) {
            return op.withPosition(op.position - deletedBefore(other, op.position, base));
        }
        if (other.type == TextOp.INSERT) {
            return op;                                        // deletes name atoms, inserts create new ones
        }
        Set<Integer> removed = new HashSet<>();
        for (int id : other.targets) removed.add(id);
        List<Integer> keep = new ArrayList<>();
        for (int id : op.targets) if (!removed.contains(id)) keep.add(id);
        if (keep.size() == op.targets.length) return op;
        if (keep.isEmpty()) return op.withTargets(new int[0]);
        int[] targets = new int[keep.size()];
        for (int i = 0; i < targets.length; i++) targets[i] = keep.get(i);
        return op.withTargets(targets);
    }

    static List<Integer> canonicalOrder(List<TextOp> ops) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < ops.size(); i++) order.add(i);
        order.sort((a, b) -> Long.compare(ops.get(a).priority(), ops.get(b).priority()));
        return order;
    }
}

public class Solution {
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** Every insert and delete operation producible from the given base by both replicas. */
    static List<TextOp> allOps(OtDocument base, int length) {
        List<TextOp> ops = new ArrayList<>();
        for (int replica = 0; replica < 2; replica++) {
            for (int position = 0; position <= length; position++) {
                ops.add(TextOp.insert(replica, 100 + position * 10, position, "x"));
                ops.add(TextOp.insert(replica, 200 + position * 10, position, "yz"));
            }
            for (int position = 0; position < length; position++) {
                int id = base.visibleIds().get(position);
                ops.add(TextOp.delete(replica, 30 + position, new int[]{id}));
            }
            if (length >= 2) {
                List<Integer> ids = base.visibleIds();
                ops.add(TextOp.delete(replica, 40, new int[]{ids.get(0), ids.get(1)}));
                ops.add(TextOp.delete(replica, 41, new int[]{ids.get(length - 1)}));
            }
        }
        return ops;
    }


    static void checkTp1(OtDocument base, List<TextOp> ops, String label) {
        int pairs = 0;
        for (TextOp a : ops) {
            for (TextOp b : ops) {
                OtDocument left = base.copy();
                left.apply(a);
                TextOp bAfterA = Transform.transform(b, a, base);
                left.apply(bAfterA);

                OtDocument right = base.copy();
                right.apply(b);
                TextOp aAfterB = Transform.transform(a, b, base);
                right.apply(aAfterB);

                check(left.text().equals(right.text()), label + ": TP1 text mismatch for " + a.describe()
                        + " vs " + b.describe() + " -> " + left.text() + " / " + right.text());
                check(left.visibleIdSet().equals(right.visibleIdSet()), label + ": TP1 atom set mismatch for "
                        + a.describe() + " vs " + b.describe());
                pairs++;
            }
        }
        check(pairs == ops.size() * ops.size(), label + ": every pair was exercised, " + pairs + " pairs");
    }

    /** Site `site` applies its own operation first, then the others transformed against everything applied. */
    static OtDocument convergeAt(OtDocument base, List<TextOp> ops, int site) {
        OtDocument document = base.copy();
        List<TextOp> applied = new ArrayList<>();
        TextOp mine = ops.get(site);
        document.apply(mine);
        applied.add(mine);
        for (int index : Transform.canonicalOrder(ops)) {
            if (index == site) continue;
            TextOp remote = ops.get(index);
            for (TextOp earlier : applied) remote = Transform.transform(remote, earlier, base);
            document.apply(remote);
            applied.add(remote);
        }
        return document;
    }

    public static void main(String[] args) {
        // --- trivial transforms ------------------------------------------------
        OtDocument abc = OtDocument.of("abc");
        TextOp insertHead = TextOp.insert(0, 0, 0, "X");
        TextOp insertTail = TextOp.insert(1, 0, 3, "Y");
        check(Transform.transform(insertHead, insertTail, abc).position == 0,
                "an insert before a later insert is not shifted");
        check(Transform.transform(insertTail, insertHead, abc).position == 4,
                "an insert after a preceding insert is shifted by its length");
        TextOp deleteB = TextOp.delete(0, 0, new int[]{abc.visibleIds().get(1)});
        check(Transform.transform(insertTail, deleteB, abc).position == 2,
                "an insert after a deleted character moves left, got "
                        + Transform.transform(insertTail, deleteB, abc).position);
        check(Transform.transform(insertHead, deleteB, abc).position == 0,
                "an insert before a deleted character stays put");
        check(Transform.transform(deleteB, insertHead, abc) == deleteB
                        || Transform.transform(deleteB, insertHead, abc).targets.length == 1,
                "a delete is unaffected by an insert");

        // --- delete vs delete --------------------------------------------------
        TextOp deleteAll = TextOp.delete(1, 0, new int[]{abc.visibleIds().get(0), abc.visibleIds().get(1)});
        TextOp deleteFirst = TextOp.delete(0, 0, new int[]{abc.visibleIds().get(0)});
        check(Transform.transform(deleteAll, deleteFirst, abc).targets.length == 1,
                "the overlapping part of a concurrent delete is dropped");
        check(Transform.transform(deleteFirst, deleteAll, abc).targets.length == 0,
                "a delete fully covered by another becomes a no-op");

        // --- exhaustive TP1 ----------------------------------------------------
        checkTp1(OtDocument.of("abc"), allOps(OtDocument.of("abc"), 3), "TP1/abc");
        checkTp1(OtDocument.of("abcd"), allOps(OtDocument.of("abcd"), 4), "TP1/abcd");
        checkTp1(OtDocument.of("a"), allOps(OtDocument.of("a"), 1), "TP1/single");

        // --- three site convergence --------------------------------------------
        List<List<TextOp>> scenarios = new ArrayList<>();
        OtDocument base = OtDocument.of("cracking");
        scenarios.add(List.of(
                TextOp.insert(0, 0, 0, "X"),
                TextOp.insert(1, 0, 0, "Y"),
                TextOp.insert(2, 0, 0, "Z")));
        scenarios.add(List.of(
                TextOp.insert(0, 0, 3, "abc"),
                TextOp.delete(1, 0, new int[]{base.visibleIds().get(3)}),
                TextOp.insert(2, 0, 8, "!")));
        scenarios.add(List.of(
                TextOp.delete(0, 0, new int[]{base.visibleIds().get(0)}),
                TextOp.delete(1, 0, new int[]{base.visibleIds().get(0), base.visibleIds().get(1)}),
                TextOp.delete(2, 0, new int[]{base.visibleIds().get(1)})));
        scenarios.add(List.of(
                TextOp.insert(0, 0, 4, "mi"),
                TextOp.insert(1, 5, 4, "MO"),
                TextOp.delete(2, 0, new int[]{base.visibleIds().get(4)})));
        scenarios.add(List.of(
                TextOp.insert(0, 0, 8, "END"),
                TextOp.insert(1, 0, 8, "end"),
                TextOp.insert(2, 0, 8, "|")));
        int scenarioIndex = 0;
        for (List<TextOp> ops : scenarios) {
            List<String> results = new ArrayList<>();
            for (int site = 0; site < 3; site++) results.add(convergeAt(base, ops, site).text());
            check(results.get(0).equals(results.get(1)) && results.get(1).equals(results.get(2)),
                    "scenario " + scenarioIndex + " converges: " + results);
            OtDocument converged = convergeAt(base, ops, 0);
            for (TextOp op : ops) {
                if (op.type == TextOp.INSERT) {
                    check(converged.text().contains(op.payload),
                            "scenario " + scenarioIndex + " keeps the payload " + op.payload
                                    + " contiguous in " + converged.text());
                } else {
                    for (int id : op.targets) {
                        check(!converged.isVisible(id), "scenario " + scenarioIndex
                                + " keeps the atom deleted by " + op.describe() + " invisible");
                    }
                }
            }
            scenarioIndex++;
        }

        // --- deterministic tie break for equal positions -----------------------
        OtDocument tie = OtDocument.of("ab");
        TextOp fromZero = TextOp.insert(0, 0, 1, "0");
        TextOp fromOne = TextOp.insert(1, 0, 1, "1");
        OtDocument tieSiteZero = tie.copy();
        tieSiteZero.apply(fromZero);
        tieSiteZero.apply(Transform.transform(fromOne, fromZero, tie));
        OtDocument tieSiteOne = tie.copy();
        tieSiteOne.apply(fromOne);
        tieSiteOne.apply(Transform.transform(fromZero, fromOne, tie));
        check(tieSiteZero.text().equals(tieSiteOne.text()), "equal positions converge");
        check(tieSiteZero.text().equals("a01b"), "the lower replica id goes first, got " + tieSiteZero.text());

        // --- edge cases --------------------------------------------------------
        OtDocument empty = OtDocument.of("");
        TextOp onlyInsert = TextOp.insert(0, 0, 0, "hi");
        check(Transform.transform(TextOp.insert(1, 0, 0, "yo"), onlyInsert, empty).position == 2,
                "in an empty document the later insert is shifted past the winner");
        check(Transform.transform(onlyInsert, TextOp.insert(1, 0, 0, "yo"), empty).position == 0,
                "the priority winner keeps its position");
        OtDocument whole = OtDocument.of("xy");
        TextOp deleteEverything = TextOp.delete(0, 0, new int[]{whole.visibleIds().get(0), whole.visibleIds().get(1)});
        OtDocument cleared = whole.copy();
        cleared.apply(deleteEverything);
        check(cleared.text().isEmpty(), "deleting every atom empties the text");
        check(Transform.transform(TextOp.insert(1, 0, 0, "z"), deleteEverything, whole).position == 0,
                "an insert in front of a full delete keeps position 0");
        OtDocument appended = OtDocument.of("ab");
        TextOp atEnd = TextOp.insert(0, 0, 2, "Z");
        appended.apply(atEnd);
        check(appended.text().equals("abZ"), "an insert at the end of the document appends");

        // --- rejected operations ------------------------------------------------
        boolean rejected = false;
        try {
            TextOp.insert(0, 0, 0, "");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty insert must be rejected");
        rejected = false;
        try {
            TextOp.delete(0, 0, new int[0]);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty delete must be rejected");
        rejected = false;
        try {
            OtDocument.of("ab").apply(TextOp.insert(0, 0, 5, "x"));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an insert past the end must be rejected");
        rejected = false;
        try {
            OtDocument.of("ab").apply(TextOp.insert(0, 0, -1, "x"));
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a negative position must be rejected");

        System.out.println("PASS");
    }
}
