/** Task 990 - Implement reproducible Docker images for canonical environment with JDK/test harness.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  A build is a validated instruction list: FROM must reference a base by digest (never a floating
 *  tag), every apt package inside RUN must be pinned as "name=version", COPY must carry the content
 *  digest of the file, ENV entries are canonicalised by sorting, and any other instruction (a
 *  timestamp hook, a health probe) is refused. The whole list is validated *before* any digest is
 *  computed, so a broken build produces no layers at all. A valid build then folds into a Merkle
 *  chain: digest(i) = FNV-1a(parent | canonical step). That gives byte-identical inputs an identical
 *  image digest, lets a content-addressed cache reuse the unchanged prefix when only the trailing
 *  harness layer changes, and lets a verifier recompute the chain to catch reordered or forged layers.
 *  Simplified: instruction strings instead of a Docker daemon and a synthetic base digest; the
 *  pinning rules, the Merkle chain, the prefix cache and the verification are the real ones.
 *  Verified: identical builds are bit-identical, tag-only bases, unpinned packages, timestamped and
 *  unknown instructions are rejected before any layer exists, appending reuses exactly the prefix,
 *  an early change invalidates the suffix and caps the cache hits, reordered/forged chains fail
 *  verification, the minimal and empty boundaries behave, and a from-scratch reverse oracle agrees.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final String BASE = "eclipse-temurin@sha256:" + "ab".repeat(32);

    static String fnv(String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= 0x100000001b3L; }
        return String.format("%016x", h);
    }

    /** Normalises one instruction, or returns null when the instruction cannot be reproduced. */
    static String canonicalise(String instruction) {
        String line = instruction.trim();
        if (line.isEmpty()) return null;
        String kind = line.split("\\s+", 2)[0];
        String rest = line.length() > kind.length() ? line.substring(kind.length()).trim() : "";
        switch (kind) {
            case "FROM":
                return rest.matches("[a-z0-9._/-]+@sha256:[0-9a-f]{64}") ? "FROM " + rest : null;
            case "RUN": {
                String prefix = "apt-get install -y ";
                if (!rest.startsWith(prefix)) return null;
                List<String> packages = new ArrayList<>(Arrays.asList(rest.substring(prefix.length()).trim().split("\\s+")));
                for (String pkg : packages) if (!pkg.contains("=") || pkg.startsWith("=")) return null;
                Collections.sort(packages);                     // mirror ordering must not change the layer
                return "RUN " + prefix + String.join(" ", packages);
            }
            case "COPY": {
                String[] parts = rest.split("\\s+");
                return parts.length == 2 && parts[1].matches("sha256:[0-9a-f]{64}")
                        ? "COPY " + parts[0] + " " + parts[1] : null;
            }
            case "ENV": {
                List<String> vars = new ArrayList<>(Arrays.asList(rest.split("\\s+")));
                for (String v : vars) if (!v.contains("=")) return null;
                Collections.sort(vars);
                return "ENV " + String.join(" ", vars);
            }
            default:
                return null;                                    // TIMESTAMP, HEALTHCHECK, ... are not reproducible
        }
    }

    static final class Build {
        final List<String> canonical;
        final List<String> digests;
        final Map<String, Integer> cache;
        final String error;
        final int cacheHits;

        Build(List<String> instructions, Map<String, Integer> warmCache) {
            List<String> validated = new ArrayList<>();
            String failure = null;
            for (String instruction : instructions) {
                String step = canonicalise(instruction);
                if (step == null) { failure = "irreproducible instruction: " + instruction; break; }
                validated.add(step);
            }
            this.error = failure;
            this.canonical = failure == null ? validated : List.of();
            this.digests = new ArrayList<>();
            this.cache = new LinkedHashMap<>();
            int hits = 0;
            if (failure == null) {
                String parent = "";
                for (String step : validated) {
                    parent = fnv(parent + "|" + step);
                    digests.add(parent);
                    if (warmCache != null && warmCache.containsKey(parent)) hits++;
                    cache.put(parent, cache.getOrDefault(parent, 0) + 1);
                }
            }
            this.cacheHits = hits;
        }

        boolean ok() { return error == null; }
        String imageDigest() { return digests.isEmpty() ? null : digests.get(digests.size() - 1); }
    }

    static List<String> jdkBuild() {
        return List.of(
            "FROM " + BASE,
            "RUN apt-get install -y openjdk-17-jdk=17.0.9+9 junit5=5.10.0 git=1:2.39.2-1",
            "COPY grader sha256:" + "cd".repeat(32),
            "ENV JAVA_HOME=/opt/java/openjdk PATH=/opt/java/openjdk/bin:/usr/bin",
            "COPY harness sha256:" + "ef".repeat(32));
    }

    static String verify(List<String> instructions, String expectedDigest) {
        Build rebuild = new Build(instructions, null);
        if (!rebuild.ok()) return rebuild.error;
        return rebuild.imageDigest().equals(expectedDigest) ? "ok" : "digest mismatch";
    }

    static Map<String, Integer> warmFrom(Build previous) {
        Map<String, Integer> warm = new LinkedHashMap<>();
        for (String digest : previous.digests) warm.put(digest, 1);
        return warm;
    }

    public static void main(String[] args) {
        List<String> build = jdkBuild();
        Build first = new Build(build, null);
        check(first.ok() && first.canonical.size() == 5,
                "a fully pinned build must validate every instruction: " + first.error);
        check(first.canonical.get(1).equals("RUN apt-get install -y git=1:2.39.2-1 junit5=5.10.0 openjdk-17-jdk=17.0.9+9"),
                "package lists must be sorted so mirror order cannot change the layer: " + first.canonical.get(1));
        Build second = new Build(build, null);
        check(second.imageDigest().equals(first.imageDigest()),
                "byte-identical builds must be bit-identical: " + second.imageDigest() + " vs " + first.imageDigest());
        check(verify(build, first.imageDigest()).equals("ok"),
                "a verifier recomputing the chain must accept a fresh image: " + verify(build, first.imageDigest()));
        check(first.digests.size() == 5 && new HashSet<>(first.digests).size() == 5,
                "each layer must have its own digest: " + first.digests);

        // --- reproducibility gates: floating tags, unpinned packages, timestamped or unknown steps.
        Build tagOnly = new Build(List.of("FROM eclipse-temurin:17-jdk", "RUN apt-get install -y junit5=5.10.0"), null);
        check(!tagOnly.ok() && tagOnly.canonical.isEmpty(),
                "a base referenced by tag must be refused before any layer: " + tagOnly.error);
        Build unpinned = new Build(List.of("FROM " + BASE, "RUN apt-get install -y openjdk-17-jdk"), null);
        check(!unpinned.ok() && unpinned.canonical.isEmpty() && unpinned.error.contains("openjdk-17-jdk"),
                "an unpinned package must abort the whole build with no layers: " + unpinned.error);
        Build timestamped = new Build(List.of("FROM " + BASE, "RUN TIMESTAMP(now) touch /build-stamp"), null);
        check(!timestamped.ok() && timestamped.error.contains("irreproducible instruction"),
                "embedded wall-clock state must be refused: " + timestamped.error);
        Build unknown = new Build(List.of("FROM " + BASE, "HEALTHCHECK curl localhost"), null);
        check(!unknown.ok() && unknown.digests.isEmpty(),
                "unknown instructions must be refused rather than silently digested");
        check(!new Build(List.of("   "), null).ok(), "a blank instruction must be refused");

        // --- prefix cache: appending the perf-report layer reuses the first five digests unchanged.
        List<String> extended = new ArrayList<>(build);
        extended.add("COPY perf-report sha256:" + "11".repeat(32));
        Build appended = new Build(extended, warmFrom(first));
        check(appended.ok() && appended.cacheHits == 5 && appended.digests.subList(0, 5).equals(first.digests),
                "appending a layer must reuse the whole prefix: " + appended.cacheHits + " hits");
        check(!appended.imageDigest().equals(first.imageDigest()) && appended.digests.size() == 6,
                "adding a layer must change the image digest: " + appended.imageDigest());
        check(appended.cache.get(appended.digests.get(5)) == 1,
                "the new layer must enter the content-addressed cache exactly once");

        // --- Merkle behaviour: an early change invalidates the suffix and caps the reusable hits at 2.
        List<String> changed = new ArrayList<>(build);
        changed.set(2, "COPY grader sha256:" + "99".repeat(32));
        Build mutated = new Build(changed, warmFrom(first));
        check(mutated.ok() && mutated.digests.get(0).equals(first.digests.get(0))
                        && mutated.digests.get(1).equals(first.digests.get(1)) && mutated.cacheHits == 2,
                "a change in layer 3 must leave layers 1-2 reusable: " + mutated.cacheHits + " hits");
        for (int i = 2; i < mutated.digests.size(); i++)
            check(!mutated.digests.get(i).equals(first.digests.get(i)), "layer " + i + " must shift after an early change");

        // --- tamper detection: reordered layers and forged digests must fail verification.
        List<String> swapped = new ArrayList<>(build);
        Collections.swap(swapped, 2, 4);
        check(verify(swapped, first.imageDigest()).equals("digest mismatch"),
                "reordered layers must fail verification: " + verify(swapped, first.imageDigest()));
        check(verify(build, fnv("forged")).equals("digest mismatch"), "a forged image digest must fail verification");
        check(verify(List.of("HOOK now"), first.imageDigest()).contains("irreproducible instruction"),
                "a malformed build must be reported before any digest comparison");

        // --- boundaries: a single pinned FROM is an image; an empty build is not.
        Build minimal = new Build(List.of("FROM " + BASE), null);
        Build empty = new Build(List.of(), null);
        check(minimal.ok() && minimal.digests.size() == 1 && minimal.imageDigest().equals(fnv("|FROM " + BASE)),
                "a FROM-only image must digest its single layer: " + minimal.imageDigest());
        check(empty.imageDigest() == null && empty.digests.isEmpty(),
                "an empty build must not produce an image digest");

        // --- independent oracle: recompute every layer from scratch (no rolling state reused).
        check(oracleDigest(build).equals(first.imageDigest()),
                "from-scratch oracle mismatch: " + oracleDigest(build) + " vs " + first.imageDigest());
        check(oracleDigest(extended).equals(appended.imageDigest()),
                "from-scratch oracle must reproduce the extended image too");

        System.out.println("image digest: " + first.imageDigest() + " over " + first.digests.size() + " layers");
        System.out.println("cache hits: appended=" + appended.cacheHits + " early-change=" + mutated.cacheHits);
        System.out.println("PASS");
    }

    /** Oracle: rebuilds the chain without any shared state - each layer's prefix is folded from scratch. */
    static String oracleDigest(List<String> instructions) {
        List<String> steps = new ArrayList<>();
        for (String instruction : instructions) {
            String step = canonicalise(instruction);
            if (step == null) return "invalid";
            steps.add(step);
        }
        String last = "";
        for (int i = 0; i < steps.size(); i++) {
            String prefix = "";
            for (int j = 0; j < i; j++) prefix = fnv(prefix + "|" + steps.get(j));
            last = fnv(prefix + "|" + steps.get(i));
        }
        return last;
    }
}
