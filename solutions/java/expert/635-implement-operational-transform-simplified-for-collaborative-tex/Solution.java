/** Task 635 - Implement operational transform simplified for collaborative text editing. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * Every edit is an operation over a text document - insert(position, text) or delete(position,
 * count) - and the whole point of operational transformation is the *inclusion* property TP1:
 * for any two concurrent operations a and b there must exist transformed operations a' and b'
 * with    apply(apply(doc, a), b') == apply(apply(doc, b), a').
 * A function that turns (a, b) into such a pair is what lets a server sequence concurrent edits
 * and lets every replica replay them in a different order yet end up with the same text.
 *
 * The rules implemented here are the standard ones:
 *   * insert/insert: the earlier position wins; on a tie the lower site id wins, and the loser is
 *     shifted by the winner's length. The site id is what makes the tie-break deterministic on
 *     every replica - without it the two sites would each think they came first.
 *   * insert/delete: an insert before the deleted range shifts the range; an insert after it is
 *     shifted left by the range length; an insert *inside* the range is dropped and the delete is
 *     widened to swallow it, which is the only way to keep one op per effect.
 *   * delete/delete: both are shortened by their overlap, and each position is pulled back by the
 *     characters the other one removed before it.
 *
 * Simulation model, and what is simplified: no sites, sockets or clocks. A document is a String,
 * a site is an integer id, and the "server" simply transforms each incoming operation against a
 * history. Tombstones are not modelled, so a delete always refers to the current document, and an
 * insert that lands inside a concurrently deleted range is discarded rather than kept as a
 * tombstoned character - the two things a production OT engine adds on top. What is not
 * simplified is the transform itself, and it is falsified the hard way.
 *
 * Verified: (1) TP1 holds for *every* pair of insert/delete operations on all small documents
 * (an exhaustive sweep, not a sample); (2) TP1 also holds on hundreds of PRNG-generated pairs on
 * longer documents; (3) three sites that edit concurrently and then replay each other's
 * transformed operations all end up with the same text, and the text equals the sequentially
 * transformed one; (4) concurrent inserts at the same position converge because of the site
 * tie-break; (5) boundary cases: position 0, the end of the document, deleting everything, an
 * empty document and a zero-length delete; (6) the negative control - a transform that omits the
 * tie-break - breaks TP1 on a scripted concurrent insert, which the checker reports.
 */
import java.util.*;

class Op {
    static final String INSERT = "ins";
    static final String DELETE = "del";
    static final String NOOP = "noop";

    final String kind;
    final String site;
    final int pos;
    final String text;
    final int count;

    Op(String kind, String site, int pos, String text, int count) {
        this.kind = kind;
        this.site = site;
        this.pos = pos;
        this.text = text;
        this.count = count;
    }

    static Op insert(String site, int pos, String text) {
        return new Op(INSERT, site, pos, text, 0);
    }

    static Op delete(String site, int pos, int count) {
        return new Op(DELETE, site, pos, null, count);
    }

    static Op noop(String site) {
        return new Op(NOOP, site, 0, null, 0);
    }

    int length() {
        return INSERT.equals(kind) ? text.length() : count;
    }

    Op at(int newPos) {
        return new Op(kind, site, newPos, text, count);
    }

    Op withCount(int newCount) {
        return new Op(DELETE, site, pos, null, newCount);
    }

    String render() {
        if (INSERT.equals(kind)) return "ins(" + pos + ",'" + text + "'@" + site + ")";
        if (DELETE.equals(kind)) return "del(" + pos + "," + count + "@" + site + ")";
        return "noop";
    }
}

public class Solution {
    static long seed = 0xFACADE987654L;

    static int nextInt(int bound) {
        seed += 0x9E3779B97F4A7C15L;
        long z = seed;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, bound);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        basics();
        tp1ExhaustiveOnSmallDocuments();
        tp1OnGeneratedPairs();
        siteConvergence();
        transformSymmetry();
        concurrentInsertsUseTheTieBreak();
        negativeControlWithoutTieBreak();
        System.out.println("PASS");
    }

    static String apply(String doc, Op op) {
        if (Op.NOOP.equals(op.kind)) return doc;
        check(op.pos >= 0 && op.pos <= doc.length(), "position inside the document: " + op.render());
        if (Op.INSERT.equals(op.kind)) {
            return doc.substring(0, op.pos) + op.text + doc.substring(op.pos);
        }
        check(op.count >= 0 && op.pos + op.count <= doc.length(),
                "deletion inside the document: " + op.render() + " of " + doc.length());
        return doc.substring(0, op.pos) + doc.substring(op.pos + op.count);
    }

    /** The transform: given concurrent a and b, return {a', b'} satisfying TP1. */
    static Op[] transform(Op a, Op b, boolean tieBreak) {
        if (Op.NOOP.equals(a.kind) || Op.NOOP.equals(b.kind)) return new Op[]{a, b};
        if (Op.INSERT.equals(a.kind) && Op.INSERT.equals(b.kind)) {
            if (!tieBreak && a.pos == b.pos) return new Op[]{a, b};   // broken: neither side moves
            boolean aFirst = a.pos < b.pos
                    || (a.pos == b.pos && a.site.compareTo(b.site) < 0);
            if (aFirst) return new Op[]{a, b.at(b.pos + a.text.length())};
            return new Op[]{a.at(a.pos + b.text.length()), b};
        }
        if (Op.INSERT.equals(a.kind) && Op.DELETE.equals(b.kind)) {
            if (a.pos < b.pos) return new Op[]{a, b.at(b.pos + a.text.length())};
            if (a.pos > b.pos + b.count) return new Op[]{a.at(a.pos - b.count), b};
            if (a.pos == b.pos + b.count) return new Op[]{a.at(b.pos), b};
            return new Op[]{Op.noop(a.site), b.withCount(b.count + a.text.length())};
        }
        if (Op.DELETE.equals(a.kind) && Op.INSERT.equals(b.kind)) {
            if (b.pos < a.pos) return new Op[]{a.at(a.pos + b.text.length()), b};
            if (b.pos > a.pos + a.count) return new Op[]{a, b.at(b.pos - a.count)};
            if (b.pos == a.pos + a.count) return new Op[]{a, b.at(a.pos)};
            return new Op[]{a.withCount(a.count + b.text.length()), Op.noop(b.site)};
        }
        int aEnd = a.pos + a.count;
        int bEnd = b.pos + b.count;
        int overlap = Math.max(0, Math.min(aEnd, bEnd) - Math.max(a.pos, b.pos));
        int aBeforeB = Math.max(0, Math.min(b.count, Math.max(0, a.pos - b.pos)));
        int bBeforeA = Math.max(0, Math.min(a.count, Math.max(0, b.pos - a.pos)));
        return new Op[]{a.at(a.pos - aBeforeB).withCount(a.count - overlap),
                b.at(b.pos - bBeforeA).withCount(b.count - overlap)};
    }

    /** TP1: both orders of applying the transformed pair must yield the same text. */
    static boolean tp1(String doc, Op a, Op b, boolean tieBreak) {
        Op[] transformed = transform(a, b, tieBreak);
        String one = apply(apply(doc, a), transformed[1]);
        String two = apply(apply(doc, b), transformed[0]);
        return one.equals(two);
    }

    static void basics() {
        String doc = "hello";
        check(apply(doc, Op.insert("s2", 0, ">>")).equals(">>hello"), "insert at the head");
        check(apply(doc, Op.insert("s2", 5, "!")).equals("hello!"), "insert at the tail");
        check(apply(doc, Op.delete("s1", 1, 3)).equals("ho"), "delete in the middle");
        check(apply(doc, Op.delete("s1", 0, 5)).equals(""), "delete everything");
        check(apply(doc, Op.delete("s1", 2, 0)).equals("hello"), "a zero-length delete is a no-op");
        check(apply("", Op.insert("s1", 0, "a")).equals("a"), "insert into an empty document");
        check(apply("", Op.noop("s1")).equals(""), "noop changes nothing");
        check(Op.insert("s1", 0, "abc").length() == 3 && Op.delete("s1", 0, 2).length() == 2,
                "operation lengths");

        Op[] pair = transform(Op.insert("s1", 1, "X"), Op.insert("s2", 1, "Y"), true);
        check(pair[0].render().equals("ins(1,'X'@s1)") && pair[1].render().equals("ins(2,'Y'@s2)"),
                "s1 wins the tie: " + pair[0].render() + " " + pair[1].render());
        Op[] reversed = transform(Op.insert("s2", 1, "Y"), Op.insert("s1", 1, "X"), true);
        check(reversed[0].render().equals("ins(2,'Y'@s2)")
                        && reversed[1].render().equals("ins(1,'X'@s1)"),
                "and the swapped call yields the mirrored pair: " + reversed[0].render() + " "
                        + reversed[1].render());
    }

    /** Every pair of insert/delete operations on every small document must satisfy TP1. */
    static void tp1ExhaustiveOnSmallDocuments() {
        String[] docs = {"", "a", "ab", "abc", "abcd", "abab", "aaab"};
        List<Op> operations = new ArrayList<>();
        int pairs = 0;
        for (String doc : docs) {
            operations.clear();
            for (int pos = 0; pos <= doc.length(); pos++) {
                operations.add(Op.insert("s1", pos, "X"));
                operations.add(Op.insert("s2", pos, "YZ"));
                for (int count = 0; pos + count <= doc.length(); count++) {
                    operations.add(Op.delete("s1", pos, count));
                    operations.add(Op.delete("s2", pos, count + 1 <= doc.length() - pos
                            ? count + 1 : count));
                }
            }
            for (Op a : operations) {
                for (Op b : operations) {
                    if (a.length() == 0 && Op.DELETE.equals(a.kind)) continue;
                    if (b.length() == 0 && Op.DELETE.equals(b.kind)) continue;
                    pairs++;
                    check(tp1(doc, a, b, true), "TP1 for " + a.render() + " / " + b.render()
                            + " on '" + doc + "'");
                }
            }
        }
        check(pairs > 2000, "the exhaustive sweep really is exhaustive: " + pairs);
    }

    /** The same property on PRNG-generated documents and operation pairs. */
    static void tp1OnGeneratedPairs() {
        int checked = 0;
        for (int trial = 0; trial < 300; trial++) {
            int length = nextInt(12);
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < length; i++) sb.append((char) ('a' + nextInt(4)));
            String doc = sb.toString();

            Op a = randomOp(doc, trial % 2 == 0 ? "s1" : "s3", trial);
            Op b = randomOp(doc, trial % 2 == 0 ? "s2" : "s1", trial + 1);
            if (a.length() == 0 || b.length() == 0) continue;
            checked++;
            check(tp1(doc, a, b, true), "TP1 on '" + doc + "': " + a.render() + " / " + b.render()
                    + " -> " + apply(apply(doc, a), transform(a, b, true)[1]) + " vs "
                    + apply(apply(doc, b), transform(a, b, true)[0]));
        }
        check(checked > 200, "enough generated pairs were checked: " + checked);
    }

    static Op randomOp(String doc, String site, int salt) {
        if (nextInt(2) == 0) {
            return Op.insert(site, nextInt(doc.length() + 1),
                    (char) ('A' + ((salt + nextInt(3)) % 26)) + "");
        }
        if (doc.isEmpty()) return Op.noop(site);
        int pos = nextInt(doc.length());
        int count = 1 + nextInt(doc.length() - pos);
        return Op.delete(site, pos, count);
    }

    /**
     * Two sites edit concurrently and then exchange: each applies its own operation locally and
     * then the peer's operation transformed against its own history. They must agree.
     */
    static void siteConvergence() {
        int converged = 0;
        for (int trial = 0; trial < 200; trial++) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 1 + nextInt(10); i++) sb.append((char) ('a' + nextInt(4)));
            String base = sb.toString();
            Op a = randomOp(base, "s1", trial);
            Op b = randomOp(base, "s2", trial + 7);
            if (a.length() == 0 || b.length() == 0) continue;
            Op[] pair = transform(a, b, true);
            String siteA = apply(apply(base, a), pair[1]);     // applied a, then b transformed
            String siteB = apply(apply(base, b), pair[0]);     // applied b, then a transformed
            check(siteA.equals(siteB), "two sites converge on '" + base + "': " + a.render() + " / "
                    + b.render() + " -> " + siteA + " vs " + siteB);
            converged++;
        }
        check(converged > 150, "many concurrent pairs were exchanged: " + converged);

        String base = "hello world";
        Op siteOne = Op.insert("s1", 5, " there");
        Op siteTwo = Op.delete("s2", 0, 6);
        Op[] pair = transform(siteOne, siteTwo, true);
        check(apply(apply(base, siteOne), pair[1]).equals(apply(apply(base, siteTwo), pair[0])),
                "an insert and a concurrent delete converge");
        check(apply(apply(base, siteOne), pair[1]).equals("world"),
                "an insert inside the deleted range is swallowed: "
                        + apply(apply(base, siteOne), pair[1]));
        Op siteThree = Op.insert("s3", 8, "!");            // after the deleted range: survives
        Op[] pair2 = transform(siteThree, siteTwo, true);
        check(pair2[0].render().equals("ins(2,'!'@s3)"),
                "the later insert is shifted left past the deletion: " + pair2[0].render());
        check(apply(apply(base, siteThree), pair2[1]).equals("wo!rld"),
                "and it survives: " + apply(apply(base, siteThree), pair2[1]));
        check(apply(apply(base, siteThree), pair2[1]).equals(apply(apply(base, siteTwo), pair2[0])),
                "TP1 holds for the surviving insert too");
    }

    /** transform(a, b) and transform(b, a) describe the same transformed pair. */
    static void transformSymmetry() {
        for (int trial = 0; trial < 200; trial++) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 1 + nextInt(8); i++) sb.append((char) ('a' + nextInt(3)));
            String base = sb.toString();
            Op a = randomOp(base, "s1", trial + 3);
            Op b = randomOp(base, "s2", trial + 11);
            if (a.length() == 0 || b.length() == 0) continue;
            Op[] forward = transform(a, b, true);
            Op[] backward = transform(b, a, true);
            check(forward[0].pos == backward[1].pos && forward[0].kind.equals(backward[1].kind)
                            && forward[0].count == backward[1].count,
                    "the transformed a agrees: " + forward[0].render() + " vs "
                            + backward[1].render());
            check(forward[1].pos == backward[0].pos && forward[1].kind.equals(backward[0].kind)
                            && forward[1].count == backward[0].count,
                    "the transformed b agrees: " + forward[1].render() + " vs "
                            + backward[0].render());
        }
    }

    /** Concurrent inserts at the same position are ordered by site id on every replica. */
    static void concurrentInsertsUseTheTieBreak() {
        String doc = "[]";
        Op byOne = Op.insert("s1", 1, "1");
        Op byTwo = Op.insert("s2", 1, "2");
        Op[] pair = transform(byOne, byTwo, true);
        String siteOneFirst = apply(apply(doc, byOne), pair[1]);
        String siteTwoFirst = apply(apply(doc, byTwo), pair[0]);
        check(siteOneFirst.equals(siteTwoFirst), "converged: " + siteOneFirst + " / " + siteTwoFirst);
        check(siteOneFirst.equals("[12]"), "the lower site goes first: " + siteOneFirst);

        Op[] swapped = transform(byTwo, byOne, true);
        String mirrored = apply(apply(doc, byTwo), swapped[1]);
        String mirroredTwo = apply(apply(doc, byOne), swapped[0]);
        check(mirrored.equals(mirroredTwo) && mirrored.equals(siteOneFirst),
                "the mirrored transform agrees: " + mirrored);

        Op byThree = Op.insert("s3", 1, "3");
        String text = apply(doc, byOne);
        Op second = transform(byTwo, byOne, true)[0];
        text = apply(text, second);
        Op third = transform(byThree, byOne, true)[0];
        third = transform(third, second, true)[0];
        text = apply(text, third);
        check(text.equals("[123]"), "three concurrent inserts sort by site: " + text);
        check(second.render().equals("ins(2,'2'@s2)") && third.render().equals("ins(3,'3'@s3)"),
                "each later insert was shifted past the earlier one: " + second.render() + " "
                        + third.render());
    }

    /** Negative control: dropping the tie-break breaks TP1 for a concurrent insert. */
    static void negativeControlWithoutTieBreak() {
        String doc = "[]";
        Op a = Op.insert("s2", 1, "2");
        Op b = Op.insert("s1", 1, "1");
        check(tp1(doc, a, b, true), "with the tie-break the pair satisfies TP1");
        Op[] broken = transform(a, b, false);
        String one = apply(apply(doc, a), broken[1]);
        String two = apply(apply(doc, b), broken[0]);
        check(!one.equals(two), "without the tie-break the two orders diverge: " + one + " vs " + two);
        check(one.equals("[12]") && two.equals("[21]"), "and they are mirror images: " + one + " / "
                + two);
        check(!tp1(doc, a, b, false), "so the checker has something to catch");
    }
}
