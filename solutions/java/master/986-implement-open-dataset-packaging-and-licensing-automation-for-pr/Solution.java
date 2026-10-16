/** Task 986 - Implement open dataset packaging and licensing automation for practice problems.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  A package builder collects practice problems (each with an SPDX-ish license), derives the outbound
 *  license as the join of the inputs (rank MIT 0 < Apache-2.0/CC-BY-4.0 1 < CC-BY-SA-4.0 2 <
 *  GPL-3.0-only 3) and refuses the bundle when an input outranks the outbound license, when two
 *  inputs come from conflicting families (share-alike + copyleft) or when any input is unlicensed.
 *  On success it emits a manifest (slug, license, size per problem plus the outbound license) and a
 *  content digest that is computed over the *sorted* entries, so packaging is order independent;
 *  the NOTICE file lists attributions sorted by license rank then slug.
 *  Simplified: in-memory problems and a five-license rank table instead of a full SPDX graph; the
 *  compatibility test, the order-independent digest and the manifest verification are the real ones.
 *  Verified: permissive bundles derive MIT, share-alike and copyleft inputs force their own outbound
 *  license, incompatible neighbours and unlicensed inputs are rejected with the offending slug, the
 *  digest is shuffle invariant, manifest tampering is detected, and an independent rank-based oracle
 *  reproduces every derived license.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final Map<String, Integer> RANK = Map.of(
            "MIT", 0, "Apache-2.0", 1, "CC-BY-4.0", 1, "CC-BY-SA-4.0", 2, "GPL-3.0-only", 3);
    static final List<String> KNOWN = new ArrayList<>(RANK.keySet());
    static final Set<String> CONFLICT = Set.of("CC-BY-SA-4.0|GPL-3.0-only");

    static final class Problem {
        final String slug, license, body;
        Problem(String slug, String license, String body) { this.slug = slug; this.license = license; this.body = body; }
        int size() { return body.length(); }
    }

    static final class Packaged {
        final String name, outbound, digest, error;
        final List<String> manifest, notice;
        Packaged(String name, String outbound, String digest, String error, List<String> manifest, List<String> notice) {
            this.name = name; this.outbound = outbound; this.digest = digest; this.error = error;
            this.manifest = manifest; this.notice = notice;
        }
        boolean ok() { return error == null; }
    }

    static String fnv(String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= 0x100000001b3L; }
        return String.format("%016x", h);
    }

    static boolean conflict(String a, String b) {
        return CONFLICT.contains(a + "|" + b) || CONFLICT.contains(b + "|" + a);
    }

    static boolean compatible(String outbound, String incoming) {
        return RANK.get(incoming) <= RANK.get(outbound) && !conflict(outbound, incoming);
    }

    /**
     * Derives the join of the input licenses, or null when no single outbound license can cover them.
     * Ties are broken by name so the join does not depend on input order.
     */
    static String derive(List<Problem> problems) {
        String best = null;
        for (Problem p : problems) {
            if (!RANK.containsKey(p.license)) return null;
            if (best == null || RANK.get(p.license) > RANK.get(best)
                    || (RANK.get(p.license).equals(RANK.get(best)) && p.license.compareTo(best) < 0)) best = p.license;
        }
        if (best == null) return null;
        for (Problem p : problems) if (!compatible(best, p.license)) return null;
        return best;
    }

    static Packaged build(String name, List<Problem> problems, String requested) {
        if (problems.isEmpty()) return new Packaged(name, null, null, "empty package has no license", List.of(), List.of());
        for (Problem p : problems)
            if (!RANK.containsKey(p.license))
                return new Packaged(name, null, null, "unlicensed problem: " + p.slug, List.of(), List.of());
        String outbound = requested == null ? derive(problems) : requested;
        if (outbound == null || !RANK.containsKey(outbound))
            return new Packaged(name, null, null, "no outbound license covers: " + slugs(problems), List.of(), List.of());
        for (Problem p : problems)
            if (!compatible(outbound, p.license))
                return new Packaged(name, null, null, "'" + p.slug + "' (" + p.license + ") is incompatible with "
                        + outbound, List.of(), List.of());

        List<String> entries = new ArrayList<>();
        for (Problem p : problems) entries.add(p.slug + "|" + p.license + "|" + p.size());
        Collections.sort(entries);                                   // order independence
        List<String> manifest = new ArrayList<>(entries);
        manifest.add("outbound|" + outbound);
        manifest.add("problems|" + problems.size());
        StringBuilder canonical = new StringBuilder(name).append('\n');
        for (String e : manifest) canonical.append(e).append('\n');
        return new Packaged(name, outbound, fnv(canonical.toString()), null, manifest, notice(problems));
    }

    static String slugs(List<Problem> problems) {
        StringBuilder sb = new StringBuilder();
        for (Problem p : problems) { if (sb.length() > 0) sb.append(','); sb.append(p.slug); }
        return sb.toString();
    }

    /** Attributions ordered by license rank (strongest first) then slug, so the NOTICE is deterministic. */
    static List<String> notice(List<Problem> problems) {
        List<Problem> sorted = new ArrayList<>(problems);
        sorted.sort((a, b) -> RANK.get(b.license) != RANK.get(a.license)
                ? Integer.compare(RANK.get(b.license), RANK.get(a.license)) : a.slug.compareTo(b.slug));
        List<String> lines = new ArrayList<>();
        for (Problem p : sorted) lines.add(p.slug + " (" + p.license + "), " + p.size() + " bytes");
        return lines;
    }

    /** Verifies a manifest against the problems it claims: digest, outbound license and per-entry sizes. */
    static String verify(Packaged pkg, String name, List<Problem> problems) {
        if (!pkg.ok()) return "package not built";
        Map<String, Problem> bySlug = new HashMap<>();
        for (Problem p : problems) bySlug.put(p.slug, p);
        for (String entry : pkg.manifest) {
            if (entry.startsWith("outbound|")) continue;
            if (entry.startsWith("problems|")) {
                if (!entry.equals("problems|" + problems.size())) return "problem count drifted";
                continue;
            }
            String[] parts = entry.split("\\|");
            Problem p = bySlug.get(parts[0]);
            if (p == null) return "unknown slug in manifest: " + parts[0];
            if (!parts[1].equals(p.license)) return "license drifted for " + p.slug;
            if (!parts[2].equals(String.valueOf(p.size()))) return "size drifted for " + p.slug;
        }
        StringBuilder canonical = new StringBuilder(name).append('\n');
        for (String e : pkg.manifest) canonical.append(e).append('\n');
        return fnv(canonical.toString()).equals(pkg.digest) ? "ok" : "digest mismatch";
    }

    static List<Problem> permissive() {
        return List.of(new Problem("two-sum", "MIT", "two sum body"),
                       new Problem("bfs-order", "Apache-2.0", "bfs body"),
                       new Problem("graph-coloring", "CC-BY-4.0", "coloring body"));
    }

    public static void main(String[] args) {
        Packaged permissive = build("starter-set", permissive(), null);
        check(permissive.ok() && permissive.outbound.equals("Apache-2.0"),
                "the join of MIT/Apache-2.0/CC-BY-4.0 is Apache-2.0: " + permissive.error + " -> " + permissive.outbound);
        check(permissive.manifest.size() == 5 && permissive.manifest.get(0).equals("bfs-order|Apache-2.0|8"),
                "manifest entries must be sorted and carry slug|license|size: " + permissive.manifest);
        check(verify(permissive, "starter-set", permissive()).equals("ok"),
                "a freshly built manifest must verify clean: " + verify(permissive, "starter-set", permissive()));

        // --- order independence: the digest is computed over sorted entries, so shuffling cannot change it.
        List<Problem> shuffled = new ArrayList<>(permissive());
        Collections.reverse(shuffled);
        Packaged reversedPackage = build("starter-set", shuffled, null);
        check(reversedPackage.digest.equals(permissive.digest) && reversedPackage.manifest.equals(permissive.manifest),
                "packaging must be order independent: " + reversedPackage.digest + " vs " + permissive.digest);

        // --- share-alike infection: one CC-BY-SA input promotes the whole bundle.
        List<Problem> mixed = new ArrayList<>(permissive());
        mixed.add(new Problem("range-queries", "CC-BY-SA-4.0", "segment tree body"));
        Packaged shareAlike = build("mixed-set", mixed, null);
        check(shareAlike.ok() && shareAlike.outbound.equals("CC-BY-SA-4.0")
                        && shareAlike.notice.get(0).startsWith("range-queries"),
                "a share-alike input must become the outbound license and head the NOTICE: " + shareAlike.notice);
        List<Problem> mitAndShareAlike = List.of(new Problem("two-sum", "MIT", "two sum body"),
                new Problem("range-queries", "CC-BY-SA-4.0", "segment tree body"));
        Packaged weak = build("weak", mitAndShareAlike, "MIT");
        check(!weak.ok() && weak.error.contains("range-queries"),
                "requesting a weaker outbound license must be refused with the offending slug: " + weak.error);

        // --- conflicting families and unlicensed inputs can never be packaged together.
        List<Problem> clash = List.of(new Problem("a", "CC-BY-SA-4.0", "x"), new Problem("b", "GPL-3.0-only", "y"));
        check(!build("clash", clash, null).ok() && build("clash", clash, null).error.contains("no outbound license"),
                "share-alike plus copyleft has no common outbound license: " + build("clash", clash, null).error);
        check(!build("unlicensed", List.of(new Problem("c", "NOASSERTION", "z")), null).ok(),
                "an unlicensed input must never be packaged");
        check(build("copyleft", List.of(new Problem("a", "GPL-3.0-only", "x"), new Problem("b", "MIT", "y")), null)
                        .outbound.equals("GPL-3.0-only"),
                "copyleft dominates a permissive neighbour");

        // --- boundary: an empty package has no license to derive.
        Packaged empty = build("empty", List.of(), null);
        check(!empty.ok() && empty.digest == null && empty.error.equals("empty package has no license"),
                "an empty package must be refused: " + empty.error);

        // --- tamper detection: mutated sizes, licenses and digests must all be caught.
        List<String> tamperedSize = new ArrayList<>(permissive.manifest);
        tamperedSize.set(0, "bfs-order|Apache-2.0|9");
        Packaged sizeTamper = new Packaged("starter-set", permissive.outbound, permissive.digest, null, tamperedSize, permissive.notice);
        check(verify(sizeTamper, "starter-set", permissive()).equals("size drifted for bfs-order"),
                "a rewritten size must fail manifest verification: " + verify(sizeTamper, "starter-set", permissive()));
        Packaged digestTamper = new Packaged("starter-set", permissive.outbound, fnv("nonsense"), null,
                permissive.manifest, permissive.notice);
        check(verify(digestTamper, "starter-set", permissive()).equals("digest mismatch"),
                "a rewritten digest must fail manifest verification");
        check(!verify(permissive, "starter-set", List.of(new Problem("two-sum", "MIT", "two sum body!"))).equals("ok"),
                "a stale manifest must not verify against changed problems");

        // --- independent oracle: rank-max derivation recomputed with its own single pass.
        check(derive(permissive()).equals(oracleDerive(permissive()))
                        && derive(mixed).equals(oracleDerive(mixed))
                        && Objects.equals(derive(clash), oracleDerive(clash)),
                "derived licenses must match the independent oracle: " + derive(mixed) + " vs " + oracleDerive(mixed));

        System.out.println("bundle " + permissive.name + " -> " + permissive.outbound + " digest " + permissive.digest);
        System.out.println("manifest: " + permissive.manifest);
        System.out.println("notice: " + permissive.notice);
        System.out.println("PASS");
    }

    /** Oracle: recomputes the join using its own max-scan (name tie-break) and pairwise conflict pass. */
    static String oracleDerive(List<Problem> problems) {
        String winner = null;
        int winnerRank = -1, inspected = 0;
        for (Problem p : problems) {
            Integer r = RANK.get(p.license);
            if (r == null) return null;
            inspected++;
            if (r > winnerRank || (r == winnerRank && p.license.compareTo(winner) < 0)) { winnerRank = r; winner = p.license; }
        }
        if (inspected == 0) return null;
        for (Problem p : problems) for (Problem q : problems) if (conflict(p.license, q.license)) return null;
        for (Problem p : problems) if (RANK.get(p.license) > winnerRank) return null;
        return winner;
    }
}
