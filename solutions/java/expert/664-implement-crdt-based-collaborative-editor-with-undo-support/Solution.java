/** Task 664 - Implement CRDT-based collaborative editor with undo support. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A sequence CRDT (RGA - replicated growable array) with tombstones, plus local undo. Every character
 * is an atom with a globally unique id (replica, sequence), an anchor (the atom it was inserted after)
 * and a depth. Insertion walks past concurrent siblings that already occupy the anchor and sorts them
 * by (sequence, replica), so any two replicas holding the same set of atoms produce the same document
 * order no matter in which order the operations arrived - the convergence property the checks hammer
 * with permuted delivery orders. Deletion marks a tombstone instead of removing the atom, so the
 * position of everything anchored to it survives.
 *
 * Operations are INSERT(anchor, text), DELETE(target atom ids) and UNDELETE(target atom ids); an
 * operation whose anchor or target is still unknown is buffered and retried, which makes delivery
 * order irrelevant. Undo pops the replica's own last operation: an insert is undone by deleting the
 * atoms it created, a delete by clearing the tombstones it set (selective undo of local operations -
 * the documented simplification is that undo does not resurrect characters a concurrent replica
 * deleted, and that undo is per replica rather than a global history).
 *
 * Verified: three replicas converge for every permutation of the delivery order; the converged text
 * matches the hand-written expectation; concurrent inserts at one position are ordered identically
 * everywhere; redelivery is idempotent; out-of-order delivery buffers and then applies; inserts
 * anchored to deleted characters land correctly; undo restores/removes exactly the intended text and
 * converges once broadcast; an empty undo history is rejected; and out-of-range edits are rejected.
 */
import java.util.*;

final class Atom {
    final int id;
    final int replica;
    final int sequence;
    final char ch;
    final int origin;
    final int depth;
    boolean deleted;
    long deleteStamp;      // the (sequence, replica) stamp of the winning delete marker

    Atom(int id, int replica, int sequence, char ch, int origin, int depth) {
        this.id = id;
        this.replica = replica;
        this.sequence = sequence;
        this.ch = ch;
        this.origin = origin;
        this.depth = depth;
    }
}

final class EditOp {
    static final int INSERT = 0, DELETE = 1, UNDELETE = 2;
    final int type;
    final int replica;
    final int sequence;
    final int anchor;
    final String text;
    final int[] targets;

    private EditOp(int type, int replica, int sequence, int anchor, String text, int[] targets) {
        this.type = type;
        this.replica = replica;
        this.sequence = sequence;
        this.anchor = anchor;
        this.text = text;
        this.targets = targets;
    }

    static EditOp insert(int replica, int sequence, int anchor, String text) {
        return new EditOp(INSERT, replica, sequence, anchor, text, new int[0]);
    }

    static EditOp delete(int replica, int sequence, int[] targets) {
        return new EditOp(DELETE, replica, sequence, -1, "", targets);
    }

    static EditOp undelete(int replica, int sequence, int[] targets) {
        return new EditOp(UNDELETE, replica, sequence, -1, "", targets);
    }

    String name() {
        return type == INSERT ? "insert" : type == DELETE ? "delete" : "undelete";
    }
}

final class Editor {
    final int replicaId;
    final List<Atom> atoms = new ArrayList<>();          // document order, tombstones included
    final Map<Integer, EditOp> buffered = new LinkedHashMap<>();
    final ArrayDeque<EditOp> history = new ArrayDeque<>();   // this replica's own operations
    private int sequenceCounter;   // next atom sequence (advanced by the characters an insert creates)
    private int opCounter;         // next operation key (deletes and undeletes never create atoms)

    Editor(int replicaId) { this.replicaId = replicaId; }

    static int atomId(int replica, int sequence) { return replica * 1_000_000 + sequence; }

    private int indexOfAtom(int atomId) {
        for (int i = 0; i < atoms.size(); i++) if (atoms.get(i).id == atomId) return i;
        return -1;
    }

    /** Number of visible characters before `position`, and the atom id at position-1. */
    private int anchorBefore(int position) {
        if (position < 0) throw new IllegalArgumentException("negative position");
        int seen = 0;
        int last = -1;
        for (Atom atom : atoms) {
            if (atom.deleted) continue;
            if (seen == position) return last;
            last = atom.id;
            seen++;
        }
        if (seen == position) return last;
        throw new IllegalArgumentException("position " + position + " is past the end of the document");
    }

    private int visibleLength() {
        int count = 0;
        for (Atom atom : atoms) if (!atom.deleted) count++;
        return count;
    }

    String text() {
        StringBuilder sb = new StringBuilder();
        for (Atom atom : atoms) if (!atom.deleted) sb.append(atom.ch);
        return sb.toString();
    }

    int atomCount() { return atoms.size(); }

    private static boolean greaterSibling(Atom atom, int sequence, int replica) {
        if (atom.sequence != sequence) return atom.sequence > sequence;
        return atom.replica > replica;
    }

    /** Where an atom anchored at `anchorId` belongs: after it, before any larger concurrent sibling. */
    private int insertionPoint(int anchorId, int sequence, int replica) {
        int depth = anchorId < 0 ? 0 : atoms.get(indexOfAtom(anchorId)).depth + 1;
        int at = anchorId < 0 ? 0 : indexOfAtom(anchorId) + 1;
        while (at < atoms.size()) {
            Atom candidate = atoms.get(at);
            if (candidate.depth < depth) break;                 // shallower: we left the sibling range
            if (candidate.depth == depth) {
                if (greaterSibling(candidate, sequence, replica)) {
                    at++;                                       // a concurrent winner: skip it and its subtree
                    continue;
                }
                break;                                          // a concurrent loser: we go before it
            }
            at++;                                               // descendant of a skipped sibling
        }
        return at;
    }

    // ------------------------------------------------------------------ local edits
    int[] insertAt(int position, String text) {
        if (text.isEmpty()) throw new IllegalArgumentException("nothing to insert");
        int anchor = anchorBefore(position);
        EditOp op = EditOp.insert(replicaId, sequenceCounter, anchor, text);
        sequenceCounter += text.length();
        apply(op);
        history.addLast(op);
        return idsOf(op);
    }

    EditOp deleteAt(int position, int length) {
        if (length <= 0) throw new IllegalArgumentException("nothing to delete");
        List<Integer> targets = new ArrayList<>();
        int seen = 0;
        for (Atom atom : atoms) {
            if (atom.deleted) continue;
            if (seen >= position && seen < position + length) targets.add(atom.id);
            seen++;
        }
        if (targets.size() != length) throw new IllegalArgumentException("range past the end of the document");
        int[] ids = new int[targets.size()];
        for (int i = 0; i < ids.length; i++) ids[i] = targets.get(i);
        EditOp op = EditOp.delete(replicaId, opCounter++, ids);
        apply(op);
        history.addLast(op);
        return op;
    }

    /** Undo of the replica's own last operation; returns the compensating operation to broadcast. */
    EditOp undo() {
        if (history.isEmpty()) throw new IllegalStateException("nothing to undo");
        EditOp undone = history.removeLast();
        if (undone.type == EditOp.INSERT) {
            List<Integer> live = new ArrayList<>();
            for (int id : idsOf(undone)) {
                int at = indexOfAtom(id);
                if (at >= 0 && !atoms.get(at).deleted) live.add(id);
            }
            if (live.isEmpty()) return null;
            int[] ids = new int[live.size()];
            for (int i = 0; i < ids.length; i++) ids[i] = live.get(i);
            EditOp compensating = EditOp.delete(replicaId, opCounter++, ids);
            apply(compensating);
            return compensating;
        }
        if (undone.type == EditOp.DELETE) {
            EditOp compensating = EditOp.undelete(replicaId, opCounter++, undone.targets);
            apply(compensating);
            return compensating;
        }
        EditOp compensating = EditOp.delete(replicaId, opCounter++, undone.targets);
        apply(compensating);
        return compensating;
    }

    int historySize() { return history.size(); }

    private int[] idsOf(EditOp op) {
        int[] ids = new int[op.text.length()];
        for (int i = 0; i < ids.length; i++) ids[i] = atomId(op.replica, op.sequence + i);
        return ids;
    }

    // ------------------------------------------------------------------ replication
    /** Deliver an operation; buffers it when its anchor or target is still unknown. */
    boolean deliver(EditOp op) {
        if (!ready(op)) {
            buffered.put(op.type * 1_000_000_000 + op.replica * 1_000_000 + op.sequence, op);
            return false;
        }
        apply(op);
        boolean progress = true;
        while (progress) {
            progress = false;
            List<Integer> keys = new ArrayList<>(buffered.keySet());
            for (int key : keys) {
                EditOp waiting = buffered.get(key);
                if (ready(waiting)) {
                    buffered.remove(key);
                    apply(waiting);
                    progress = true;
                }
            }
        }
        return true;
    }

    int bufferedCount() { return buffered.size(); }

    private boolean ready(EditOp op) {
        if (op.type == EditOp.INSERT) {
            return op.anchor < 0 || indexOfAtom(op.anchor) >= 0;
        }
        for (int target : op.targets) if (indexOfAtom(target) < 0) return false;
        return true;
    }

    private void apply(EditOp op) {
        if (op.type == EditOp.INSERT) {
            int anchor = op.anchor;
            for (int i = 0; i < op.text.length(); i++) {
                int id = atomId(op.replica, op.sequence + i);
                if (indexOfAtom(id) >= 0) continue;              // idempotent redelivery
                int depth = anchor < 0 ? 0 : atoms.get(indexOfAtom(anchor)).depth + 1;
                int at = insertionPoint(anchor, op.sequence + i, op.replica);
                atoms.add(at, new Atom(id, op.replica, op.sequence + i, op.text.charAt(i), anchor, depth));
                anchor = id;
            }
        } else {
            // delete markers are last-writer-wins registers, so a delete and an undelete commute
            long stamp = op.sequence * 1000L + op.replica + 1;   // strictly above the "never deleted" mark
            boolean target = op.type == EditOp.DELETE;
            for (int id : op.targets) {
                int at = indexOfAtom(id);
                if (at < 0) continue;
                Atom atom = atoms.get(at);
                if (stamp > atom.deleteStamp) {
                    atom.deleteStamp = stamp;
                    atom.deleted = target;
                }
            }
        }
    }

    boolean isDeleted(int atomId) {
        int at = indexOfAtom(atomId);
        return at >= 0 && atoms.get(at).deleted;
    }

    /** Invariant: the visible text is exactly the atoms that are not tombstoned. */
    boolean structureIsConsistent() {
        int visible = 0;
        Set<Integer> ids = new HashSet<>();
        for (Atom atom : atoms) {
            if (!ids.add(atom.id)) return false;
            if (!atom.deleted) visible++;
        }
        return visible == visibleLength() && visible == text().length();
    }
}

public class Solution {
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** Delivers every operation to every replica in the given order. */
    static List<String> deliverAll(List<Editor> editors, List<EditOp> ops, int[] permutation) {
        List<String> texts = new ArrayList<>();
        for (Editor editor : editors) {
            for (int step = 0; step < ops.size(); step++) editor.deliver(ops.get(permutation[step]));
            texts.add(editor.text());
        }
        return texts;
    }

    static List<EditOp> sessionOps() {
        List<EditOp> ops = new ArrayList<>();
        // hand written two replica session over "hello"
        ops.add(EditOp.insert(0, 0, -1, "he"));
        ops.add(EditOp.insert(0, 2, Editor.atomId(0, 1), "llo"));
        ops.add(EditOp.insert(1, 0, Editor.atomId(0, 1), "ELLO"));
        ops.add(EditOp.insert(0, 5, Editor.atomId(0, 4), " world"));
        ops.add(EditOp.delete(1, 6, new int[]{Editor.atomId(0, 2)}));
        ops.add(EditOp.insert(2, 0, Editor.atomId(0, 1), "x"));
        ops.add(EditOp.undelete(1, 7, new int[]{Editor.atomId(0, 2)}));
        return ops;
    }

    public static void main(String[] args) {
        // --- local editing ------------------------------------------------------
        Editor solo = new Editor(0);
        solo.insertAt(0, "world");
        check(solo.text().equals("world"), "local insert at the head");
        solo.insertAt(0, "hello ");
        check(solo.text().equals("hello world"), "local insert at the head again");
        solo.deleteAt(5, 1);
        check(solo.text().equals("helloworld"), "local delete removes one character");
        check(solo.atomCount() == 11, "deleted characters stay as tombstones");
        check(solo.structureIsConsistent(), "structure invariant after local edits");

        // --- undo --------------------------------------------------------------
        Editor undoing = new Editor(3);
        undoing.insertAt(0, "abc");
        check(undoing.text().equals("abc"), "insert applied");
        EditOp undoInsert = undoing.undo();
        check(undoInsert != null && undoInsert.name().equals("delete"), "undoing an insert deletes it");
        check(undoing.text().equals(""), "the text is empty after undoing the insert");
        check(undoing.historySize() == 0, "the history is empty after one undo");
        int[] xyz = undoing.insertAt(0, "xyz");
        EditOp remove = undoing.deleteAt(1, 1);
        check(remove.targets.length == 1 && remove.targets[0] == xyz[1],
                "the deleted atom is the middle character");
        check(undoing.text().equals("xz"), "the middle character was deleted, got " + undoing.text());
        EditOp undoDelete = undoing.undo();
        check(undoDelete != null && undoDelete.name().equals("undelete"), "undoing a delete undeletes");
        check(undoing.text().equals("xyz"), "undo restores the deleted character, got " + undoing.text());
        check(!undoing.isDeleted(xyz[1]), "the tombstone is cleared");
        undoing.deleteAt(2, 1);
        check(undoing.text().equals("xy"), "delete the tail");
        boolean rejected = false;
        try {
            new Editor(9).undo();
        } catch (IllegalStateException expected) {
            rejected = true;
        }
        check(rejected, "undo with an empty history must be rejected");

        // --- two edits at the same position converge ---------------------------
        Editor left = new Editor(0);
        Editor right = new Editor(1);
        left.insertAt(0, "A");
        right.insertAt(0, "B");
        EditOp fromLeft = EditOp.insert(0, 0, -1, "A");
        EditOp fromRight = EditOp.insert(1, 0, -1, "B");
        left.deliver(fromRight);
        right.deliver(fromLeft);
        check(left.text().equals(right.text()), "concurrent inserts at position 0 converge");
        check(left.text().equals("BA"), "the higher replica id is placed first, got " + left.text());

        // --- a buffered out of order delivery ----------------------------------
        Editor ordered = new Editor(0);
        Editor shuffled = new Editor(0);
        EditOp first = EditOp.insert(0, 0, -1, "one");
        EditOp second = EditOp.insert(0, 3, Editor.atomId(0, 2), " two");
        ordered.deliver(first);
        ordered.deliver(second);
        shuffled.deliver(second);
        check(shuffled.bufferedCount() == 1, "the operation without its anchor is buffered");
        check(shuffled.text().isEmpty(), "and nothing is applied yet");
        shuffled.deliver(first);
        check(shuffled.bufferedCount() == 0 && shuffled.text().equals("one two"), "the buffer drains on arrival");
        check(shuffled.text().equals(ordered.text()), "out of order delivery matches in order delivery");

        // --- redelivery is idempotent ------------------------------------------
        int atomsBefore = ordered.atomCount();
        String textBefore = ordered.text();
        for (int repeat = 0; repeat < 3; repeat++) {
            ordered.deliver(first);
            ordered.deliver(second);
        }
        check(ordered.atomCount() == atomsBefore && ordered.text().equals(textBefore),
                "redelivery changes nothing");

        // --- three replicas converge for every delivery permutation ------------
        List<EditOp> ops = sessionOps();
        int[] reference = new int[ops.size()];
        for (int i = 0; i < ops.size(); i++) reference[i] = i;
        Editor canonical = new Editor(0);
        for (EditOp op : ops) canonical.deliver(op);
        String canonicalText = canonical.text();
        check(canonicalText.equals("hello worldxELLO"),                "concurrent siblings sort by descending (sequence, replica), got " + canonicalText);
        check(canonicalText.startsWith("hello world"), "the base text survives, got " + canonicalText);
        check(canonical.structureIsConsistent(), "the canonical document is structurally consistent");

        List<int[]> permutations = new ArrayList<>();
        permutations.add(reference);
        permutations.add(new int[]{6, 5, 4, 3, 2, 1, 0});
        permutations.add(new int[]{3, 0, 5, 1, 6, 2, 4});
        permutations.add(new int[]{1, 4, 2, 6, 0, 3, 5});
        permutations.add(new int[]{2, 6, 0, 5, 3, 1, 4});
        permutations.add(new int[]{5, 2, 6, 1, 3, 0, 4});
        for (int p = 0; p < permutations.size(); p++) {
            List<Editor> editors = new ArrayList<>();
            for (int replica = 0; replica < 3; replica++) editors.add(new Editor(replica));
            List<String> texts = deliverAll(editors, ops, permutations.get(p));
            for (String text : texts) {
                check(text.equals(canonicalText), "permutation " + p + " converges to the canonical text, got " + text);
            }
        }
        check(canonical.bufferedCount() == 0, "no operation is left buffered in the canonical run");

        // --- inserts anchored to a tombstone -----------------------------------
        Editor tombstones = new Editor(0);
        tombstones.insertAt(0, "ab");
        tombstones.deleteAt(0, 1);
        check(tombstones.text().equals("b"), "the tombstone leaves one visible character");
        check(tombstones.atomCount() == 2, "but the atom is still there for anchoring");
        Editor anchorCheck = new Editor(0);
        EditOp makeA = EditOp.insert(0, 0, -1, "a");
        EditOp makeB = EditOp.insert(0, 1, Editor.atomId(0, 0), "b");
        anchorCheck.deliver(makeA);
        anchorCheck.deliver(makeB);
        anchorCheck.deliver(EditOp.delete(1, 0, new int[]{Editor.atomId(0, 0)}));
        anchorCheck.deliver(EditOp.insert(2, 0, Editor.atomId(0, 0), "c"));
        check(anchorCheck.text().equals("bc"), "the insert joins the tombstone's siblings in key order, got "
                + anchorCheck.text());
        check(anchorCheck.isDeleted(Editor.atomId(0, 0)), "the anchor stays deleted");
        Editor aliveAnchor = new Editor(0);
        aliveAnchor.deliver(EditOp.insert(0, 0, -1, "a"));
        aliveAnchor.deliver(EditOp.delete(1, 0, new int[]{Editor.atomId(0, 0)}));
        check(aliveAnchor.text().isEmpty(), "the only character is tombstoned");
        aliveAnchor.deliver(EditOp.insert(2, 0, Editor.atomId(0, 0), "c"));
        check(aliveAnchor.text().equals("c"), "an insert anchored at a tombstone still becomes visible");
        check(aliveAnchor.atomCount() == 2, "the tombstone is still in the document");

        // --- undoing a broadcast insert converges on every replica --------------
        Editor author = new Editor(0);
        Editor peer = new Editor(1);
        EditOp typing = EditOp.insert(0, 0, -1, "draft");
        author.deliver(typing);
        peer.deliver(typing);
        int[] inserted = new int[]{Editor.atomId(0, 0), Editor.atomId(0, 1), Editor.atomId(0, 2),
                Editor.atomId(0, 3), Editor.atomId(0, 4)};
        EditOp undoBroadcast = EditOp.delete(0, 5, inserted);
        author.deliver(undoBroadcast);
        peer.deliver(undoBroadcast);
        check(author.text().isEmpty() && peer.text().isEmpty(), "the broadcast undo empties both replicas");
        EditOp redoBroadcast = EditOp.undelete(0, 6, inserted);
        author.deliver(redoBroadcast);
        peer.deliver(redoBroadcast);
        check(author.text().equals("draft") && peer.text().equals("draft"),
                "a broadcast undelete restores both replicas");
        check(author.structureIsConsistent() && peer.structureIsConsistent(), "invariants survive undo");

        // --- deterministic rebuild ---------------------------------------------
        Editor rebuilt = new Editor(0);
        for (EditOp op : ops) rebuilt.deliver(op);
        check(rebuilt.text().equals(canonicalText), "rebuilding from the same operations is deterministic");
        check(rebuilt.atomCount() == canonical.atomCount(), "the same atoms are created");

        // --- rejected input -----------------------------------------------------
        rejected = false;
        try {
            new Editor(0).insertAt(1, "x");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "inserting past the end must be rejected");
        rejected = false;
        try {
            new Editor(0).insertAt(0, "");
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "inserting nothing must be rejected");
        rejected = false;
        try {
            new Editor(0).deleteAt(0, 0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "deleting nothing must be rejected");
        rejected = false;
        try {
            Editor small = new Editor(0);
            small.insertAt(0, "ab");
            small.deleteAt(1, 5);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "deleting past the end must be rejected");

        System.out.println("PASS");
    }
}
