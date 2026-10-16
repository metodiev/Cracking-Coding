/** Task 985 - Implement contribution recognition system with verified citations for authors.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Every contribution records the artifact body it touched plus a weight by kind (commit 3, review 2,
 *  edit 1). A citation claim names a contribution, the exact excerpt the author wants credited and the
 *  digest of that excerpt. A claim is accepted only when (a) the contribution exists, (b) the digest
 *  equals the FNV-1a 64 digest this system recomputes over the cited excerpt and (c) the same
 *  contribution has not already been credited; duplicates and bad proofs are recorded but never paid.
 *  Accepted credits are summed per author, share = credits / total credits, and the recognition tier is
 *  Platinum >= 0.30, Gold >= 0.15, Silver >= 0.05, else Contributor (all bounds inclusive).
 *  Simplified: in-memory strings, a fixed four-kind weight table and no persistence layer; the proof
 *  verification, the idempotent crediting and the share arithmetic are the real ones.
 *  Verified: tampered digests and off-body citations are rejected, duplicates never double count, the
 *  shares always partition the credit (sum == 1), threshold edges fall on the inclusive side, and an
 *  independent brute-force tally reproduces every share.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double EPS = 1e-12;
    static final Map<String, Integer> WEIGHT = Map.of("commit", 3, "review", 2, "edit", 1);

    /** FNV-1a 64 over the UTF-16 units of `s`, rendered as 16 lower-case hex digits. */
    static String digest(String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= 0x100000001b3L; }
        return String.format("%016x", h);
    }

    static final class Contribution {
        final String id, author, kind, body;
        Contribution(String id, String author, String kind, String body) {
            this.id = id; this.author = author; this.kind = kind; this.body = body;
        }
        int weight() { return WEIGHT.getOrDefault(kind, 1); }
    }

    /** A claim: "excerpt X of contribution C is mine, and here is its digest". */
    static final class Claim {
        final String contributionId, excerpt, digest;
        Claim(String contributionId, String excerpt, String digest) {
            this.contributionId = contributionId; this.excerpt = excerpt; this.digest = digest;
        }
        Claim(String contributionId, String excerpt) { this(contributionId, excerpt, digest(excerpt)); }
    }

    static final class Ledger {
        final Map<String, Contribution> contributions = new LinkedHashMap<>();
        final Map<String, Integer> credits = new TreeMap<>();
        final Set<String> creditedContributions = new TreeSet<>();
        final List<String> audit = new ArrayList<>();
        int accepted, rejected, duplicates;

        void add(Contribution c) { contributions.put(c.id, c); }

        /** Returns true when the claim was credited; every outcome is written to the audit log. */
        boolean verify(Claim claim) {
            Contribution c = contributions.get(claim.contributionId);
            if (c == null) {
                rejected++; audit.add("reject " + claim.contributionId + ": unknown contribution");
                return false;
            }
            if (!digest(claim.excerpt).equals(claim.digest)) {
                rejected++; audit.add("reject " + c.id + ": digest does not match excerpt");
                return false;
            }
            if (!c.body.contains(claim.excerpt)) {
                rejected++; audit.add("reject " + c.id + ": excerpt not present in artifact body");
                return false;
            }
            if (!creditedContributions.add(c.id)) {
                duplicates++; audit.add("ignore " + c.id + ": already credited, duplicate claim");
                return false;
            }
            accepted++;
            credits.merge(c.author, c.weight(), Integer::sum);
            audit.add("accept " + c.id + " -> " + c.author + " (+" + c.weight() + ")");
            return true;
        }

        int totalCredits() {
            int t = 0;
            for (int v : credits.values()) t += v;
            return t;
        }

        Map<String, Double> shares() {
            Map<String, Double> out = new TreeMap<>();
            int total = totalCredits();
            if (total == 0) return out;
            for (Map.Entry<String, Integer> e : credits.entrySet()) out.put(e.getKey(), e.getValue() / (double) total);
            return out;
        }

        Map<String, String> tiers() {
            Map<String, String> out = new TreeMap<>();
            for (Map.Entry<String, Double> e : shares().entrySet()) out.put(e.getKey(), tierOf(e.getValue()));
            return out;
        }
    }

    static String tierOf(double share) {
        if (share >= 0.30) return "Platinum";
        if (share >= 0.15) return "Gold";
        if (share >= 0.05) return "Silver";
        return "Contributor";
    }

    /** Independent oracle: replays the same rule set with its own bookkeeping (no Ledger state). */
    static Map<String, Double> bruteForceShares(List<Contribution> all, List<Claim> claims) {
        Map<String, Integer> credit = new TreeMap<>();
        Map<String, Contribution> byId = new HashMap<>();
        for (Contribution c : all) byId.put(c.id, c);
        Set<String> used = new HashSet<>();
        for (Claim claim : claims) {
            Contribution c = byId.get(claim.contributionId);
            if (c == null || !digest(claim.excerpt).equals(claim.digest) || !c.body.contains(claim.excerpt)) continue;
            if (!used.add(c.id)) continue;
            credit.merge(c.author, c.weight(), Integer::sum);
        }
        int total = 0;
        for (int v : credit.values()) total += v;
        Map<String, Double> shares = new TreeMap<>();
        if (total == 0) return shares;
        for (Map.Entry<String, Integer> e : credit.entrySet()) shares.put(e.getKey(), e.getValue() / (double) total);
        return shares;
    }

    static List<Contribution> corpus() {
        return List.of(
            new Contribution("c1", "ada", "commit", "List<String> bfsOrder(Graph g, String start) { ... }"),
            new Contribution("c2", "ada", "edit", "boolean hasCycle(Graph g) { ... }"),
            new Contribution("c3", "grace", "commit", "int[] mergeSort(int[] a) { ... }"),
            new Contribution("c4", "grace", "review", "nit: rename bfsOrder to bfsFrontier in tests"),
            new Contribution("c5", "linus", "edit", "String renderReport(List<Row> rows) { ... }"));
    }

    static List<Claim> claims() {
        return List.of(
            new Claim("c1", "bfsOrder"),
            new Claim("c2", "hasCycle"),
            new Claim("c3", "mergeSort"),
            new Claim("c4", "bfsFrontier"),
            new Claim("c5", "renderReport"));
    }

    public static void main(String[] args) {
        List<Contribution> corpus = corpus();
        List<Claim> claims = claims();

        Ledger ledger = new Ledger();
        for (Contribution c : corpus) ledger.add(c);
        for (Claim claim : claims) check(ledger.verify(claim), "every well-formed claim must be credited: " + claim.contributionId);

        check(ledger.accepted == 5 && ledger.rejected == 0 && ledger.duplicates == 0
                        && ledger.totalCredits() == 3 + 1 + 3 + 2 + 1,
                "five valid claims must yield 10 credits and no rejections: " + ledger.audit);

        Map<String, Double> shares = ledger.shares();
        double sum = 0;
        for (double v : shares.values()) sum += v;
        check(Math.abs(sum - 1.0) < EPS && shares.size() == 3,
                "shares must partition the credit pool: " + shares);
        check(Math.abs(shares.get("ada") - 0.4) < EPS && Math.abs(shares.get("grace") - 0.5) < EPS
                        && Math.abs(shares.get("linus") - 0.1) < EPS,
                "credit weights must be summed per author, not per claim: " + shares);
        check(ledger.tiers().equals(Map.of("ada", "Platinum", "grace", "Platinum", "linus", "Silver")),
                "tiers must follow the inclusive thresholds: " + ledger.tiers());

        // --- verification is a real gate: tampered proofs, off-body excerpts and replays are refused.
        Ledger hostile = new Ledger();
        for (Contribution c : corpus) hostile.add(c);
        check(hostile.verify(new Claim("c1", "bfsOrder")), "control claim before tampering");
        check(!hostile.verify(new Claim("c2", "hasCycle", digest("hasCycle ").substring(0, 16))),
                "a digest over the wrong excerpt must be rejected");
        check(!hostile.verify(new Claim("c3", "mergeSort(fake)")),
                "an excerpt that does not occur in the body must be rejected");
        check(!hostile.verify(new Claim("c1", "bfsOrder")) && hostile.duplicates == 1,
                "a replayed claim for an already credited contribution must not pay twice");
        check(!hostile.verify(new Claim("c9", "ghost")) && hostile.rejected == 3,
                "claims against unknown contributions must be rejected");
        check(hostile.totalCredits() == 3 && hostile.credits.get("ada") == 3,
                "rejected and duplicate claims must leave the credit pool untouched: " + hostile.credits);
        check(hostile.tiers().equals(Map.of("ada", "Platinum")),
                "a single credited author owns the whole pool and ranks Platinum: " + hostile.tiers());

        // --- inclusive boundary: a 100-credit pool lets every threshold be hit exactly (0.30/0.15/0.05/0.04).
        Ledger boundary = new Ledger();
        for (int i = 0; i < 100; i++) {
            String author = i < 30 ? "plat" : i < 45 ? "gold" : i < 50 ? "silver" : i < 54 ? "small" : "bulk";
            boundary.add(new Contribution("b" + i, author, "edit", "snippet" + i));
        }
        for (int i = 0; i < 100; i++) boundary.verify(new Claim("b" + i, "snippet" + i));
        Map<String, Double> boundaryShares = boundary.shares();
        check(Math.abs(boundaryShares.get("plat") - 0.30) < EPS && Math.abs(boundaryShares.get("gold") - 0.15) < EPS
                        && Math.abs(boundaryShares.get("silver") - 0.05) < EPS && Math.abs(boundaryShares.get("small") - 0.04) < EPS,
                "boundary shares must land exactly on the thresholds: " + boundaryShares);
        check(boundary.tiers().equals(Map.of("plat", "Platinum", "gold", "Gold", "silver", "Silver",
                        "small", "Contributor", "bulk", "Platinum")),
                "threshold comparisons must be inclusive: " + boundary.tiers());

        // --- degenerate input: nothing verified means no shares and no division by zero.
        Ledger empty = new Ledger();
        empty.verify(new Claim("c1", "bfsOrder"));
        check(empty.shares().isEmpty() && empty.tiers().isEmpty() && empty.totalCredits() == 0,
                "an empty credit pool must produce no shares at all");

        // --- independent oracle: brute-force replay must agree on the whole main ledger.
        Map<String, Double> oracle = bruteForceShares(corpus, claims);
        check(oracle.keySet().equals(shares.keySet()), "oracle must name the same authors: " + oracle);
        for (String author : oracle.keySet())
            check(Math.abs(oracle.get(author) - shares.get(author)) < EPS,
                    "oracle share mismatch for " + author + ": " + oracle.get(author) + " vs " + shares.get(author));

        System.out.println("authors recognised: " + shares.size() + ", credits: " + ledger.totalCredits());
        System.out.println("shares: " + shares);
        System.out.println("tiers: " + ledger.tiers());
        System.out.println("audit: accepted=" + ledger.accepted + " rejected=" + ledger.rejected
                + " duplicates=" + ledger.duplicates);
        System.out.println("PASS");
    }
}
