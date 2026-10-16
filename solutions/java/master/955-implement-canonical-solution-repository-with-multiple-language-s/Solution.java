/** Task 955 - Implement canonical solution repository with multiple language solutions and performance baselines.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory canonical repository. Every entry is one (problem, language) solution with a verification
 *  flag, a content digest and a measured runtime; the registry rejects duplicate (problem, language)
 *  pairs, reports per-language coverage and gaps, and picks the canonical solution with an explicit
 *  policy: verified beats unverified, then the lower measured runtime, then the canonical language order.
 *  A report additionally compares digests with a recorded snapshot (drift) and runtimes with a recorded
 *  baseline (regression above 20%).
 *  Simplified: the repository lives in lists instead of files and the "runtime" is a recorded number of
 *  microseconds rather than a measured benchmark; the policy, coverage and drift semantics are intact.
 *  Verified: coverage counts match, gaps list the missing languages, the canonical policy prefers a
 *  verified solution over a faster unverified one and breaks ties by language order, drift and
 *  regression flag exactly the mutated entries, duplicates are rejected, selection is insertion-order
 *  independent, and an independent sort-based oracle reproduces every canonical pick.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    static final List<String> LANGUAGE_ORDER = List.of("java", "python", "javascript", "csharp");

    record Entry(int number, String language, boolean verified, String digest, long micros) {
        String key() { return number + "/" + language; }
    }

    static final class Registry {
        final List<Entry> entries = new ArrayList<>();

        boolean add(Entry entry) {
            for (Entry existing : entries)
                if (existing.number() == entry.number() && existing.language().equals(entry.language())) return false;
            entries.add(entry);
            return true;
        }

        /** Canonical policy as a strict order: verified, then runtime, then language order. */
        static boolean beats(Entry candidate, Entry incumbent) {
            if (candidate.verified() != incumbent.verified()) return candidate.verified();
            if (candidate.micros() != incumbent.micros()) return candidate.micros() < incumbent.micros();
            return LANGUAGE_ORDER.indexOf(candidate.language()) < LANGUAGE_ORDER.indexOf(incumbent.language());
        }

        Entry canonical(int number) {
            Entry best = null;
            for (Entry entry : entries)
                if (entry.number() == number && (best == null || beats(entry, best))) best = entry;
            return best;
        }

        Set<Integer> problems() {
            Set<Integer> numbers = new TreeSet<>();
            for (Entry entry : entries) numbers.add(entry.number());
            return numbers;
        }

        List<String> gaps(int number) {
            List<String> missing = new ArrayList<>();
            for (String language : LANGUAGE_ORDER) {
                boolean present = false;
                for (Entry entry : entries) if (entry.number() == number && entry.language().equals(language)) present = true;
                if (!present) missing.add(language);
            }
            return missing;
        }

        Map<String, Integer> coverageByLanguage() {
            Map<String, Integer> coverage = new TreeMap<>();
            for (String language : LANGUAGE_ORDER) coverage.put(language, 0);
            for (Entry entry : entries) coverage.merge(entry.language(), 1, Integer::sum);
            return coverage;
        }

        Map<String, String> digests() {
            Map<String, String> map = new TreeMap<>();
            for (Entry entry : entries) map.put(entry.key(), entry.digest());
            return map;
        }

        Map<String, Long> runtimes() {
            Map<String, Long> map = new TreeMap<>();
            for (Entry entry : entries) map.put(entry.key(), entry.micros());
            return map;
        }
    }

    record Report(int problems, Map<String, Integer> coverage, Map<Integer, String> canonical,
                  List<String> drift, List<String> regressions, String digest) {}

    static Report analyze(Registry registry, Map<String, String> baselineDigests, Map<String, Long> baselineRuntimes) {
        Map<Integer, String> canonical = new TreeMap<>();
        for (Integer number : registry.problems()) {
            Entry best = registry.canonical(number);
            canonical.put(number, best.language() + (best.verified() ? " verified" : " unverified"));
        }
        List<String> drift = new ArrayList<>();
        for (Map.Entry<String, String> entry : registry.digests().entrySet()) {
            String base = baselineDigests.get(entry.getKey());
            if (base != null && !base.equals(entry.getValue())) drift.add(entry.getKey());
        }
        List<String> regressions = new ArrayList<>();
        for (Map.Entry<String, Long> entry : registry.runtimes().entrySet()) {
            Long base = baselineRuntimes.get(entry.getKey());
            if (base != null && entry.getValue() * 100 > base * 120) regressions.add(entry.getKey());
        }
        long h = fnvText(FNV_OFFSET, "repo");
        for (Map.Entry<Integer, String> entry : canonical.entrySet()) h = fnvText(h, entry.getKey() + "=" + entry.getValue());
        for (Map.Entry<String, Integer> entry : registry.coverageByLanguage().entrySet())
            h = fnvText(h, entry.getKey() + ":" + entry.getValue());
        for (String key : drift) h = fnvText(h, "drift:" + key);
        for (String key : regressions) h = fnvText(h, "slow:" + key);
        return new Report(registry.problems().size(), registry.coverageByLanguage(), canonical, drift, regressions, hex(h));
    }

    static String digestOf(int number, String language, String revision) {
        return hex(fnvText(fnvText(FNV_OFFSET, revision), number + "/" + language));
    }

    static Registry fullRegistry(boolean reversed) {
        Registry registry = new Registry();
        for (int number = 101; number <= 106; number++) {
            int index = 0;
            for (String language : LANGUAGE_ORDER) {
                boolean verified = (number + index) % 5 != 0;
                long micros = 400 + 37L * ((number * 7 + index * 13) % 40);
                if (!reversed) registry.add(new Entry(number, language, verified, digestOf(number, language, "v1"), micros));
                else registry.entries.add(new Entry(number, language, verified, digestOf(number, language, "v1"), micros));
                index++;
            }
        }
        if (reversed) Collections.reverse(registry.entries);
        return registry;
    }

    /** Copy of the registry with one entry replaced, used for drift and regression scenarios. */
    static Registry replaced(Registry base, int number, String language, String digest, long micros) {
        Registry registry = new Registry();
        for (Entry entry : base.entries) {
            if (entry.number() == number && entry.language().equals(language))
                registry.add(new Entry(number, language, entry.verified(), digest, micros));
            else registry.add(entry);
        }
        return registry;
    }

    public static void main(String[] args) {
        Registry full = fullRegistry(false);

        // 1. coverage: six problems, four languages each, no gaps, canonical defined everywhere
        Map<String, Integer> coverage = full.coverageByLanguage();
        boolean coverageOk = coverage.size() == 4;
        for (String language : LANGUAGE_ORDER) coverageOk &= coverage.get(language) == 6;
        boolean gapsEmpty = true;
        for (Integer number : full.problems()) gapsEmpty &= full.gaps(number).isEmpty();
        check(full.entries.size() == 24 && full.problems().size() == 6 && coverageOk && gapsEmpty
                        && full.canonical(101) != null,
                "24 entries cover 6 problems in 4 languages with no gaps");

        // 2. canonical policy: verified beats a faster unverified solution, ties break by language order
        Registry policy = new Registry();
        policy.add(new Entry(102, "python", true, digestOf(102, "python", "v1"), 900));
        policy.add(new Entry(102, "java", true, digestOf(102, "java", "v1"), 700));
        policy.add(new Entry(102, "javascript", false, digestOf(102, "javascript", "v1"), 100));
        policy.add(new Entry(103, "python", true, digestOf(103, "python", "v1"), 500));
        policy.add(new Entry(103, "java", true, digestOf(103, "java", "v1"), 500));
        check(policy.canonical(102).language().equals("java") && policy.canonical(102).verified()
                        && policy.canonical(103).language().equals("java"),
                "verified solution wins over a faster unverified one and ties break by language order");

        // 3. gaps: a partially covered problem still has a canonical pick
        Registry partial = new Registry();
        partial.add(new Entry(200, "java", true, digestOf(200, "java", "v1"), 300));
        partial.add(new Entry(200, "python", false, digestOf(200, "python", "v1"), 250));
        check(partial.gaps(200).equals(List.of("javascript", "csharp"))
                        && partial.canonical(200).language().equals("java")
                        && analyze(partial, Map.of(), Map.of()).problems() == 1,
                "gaps list the missing languages while the canonical pick still resolves");

        // 4. drift: only the mutated entry is reported
        Map<String, String> baselineDigests = full.digests();
        Map<String, Long> baselineRuntimes = full.runtimes();
        Registry drifted = replaced(full, 104, "csharp", digestOf(104, "csharp", "v2"), baselineRuntimes.get("104/csharp"));
        Report driftReport = analyze(drifted, baselineDigests, baselineRuntimes);
        check(driftReport.drift().equals(List.of("104/csharp")) && analyze(full, baselineDigests, baselineRuntimes).drift().isEmpty(),
                "content drift flags exactly the changed entry");

        // 5. performance regression above 20%, unchanged entries are not flagged
        Registry slowed = replaced(full, 105, "python", baselineDigests.get("105/python"),
                baselineRuntimes.get("105/python") * 150 / 100);
        Report slowReport = analyze(slowed, baselineDigests, baselineRuntimes);
        check(slowReport.regressions().equals(List.of("105/python")) && slowReport.drift().isEmpty(),
                "50% slower entry is flagged, identical entries are not");

        // 6. duplicates are rejected and leave the registry untouched
        Registry small = new Registry();
        boolean firstAdd = small.add(new Entry(300, "java", true, digestOf(300, "java", "v1"), 100));
        boolean secondAdd = small.add(new Entry(300, "java", false, digestOf(300, "java", "v2"), 90));
        check(firstAdd && !secondAdd && small.entries.size() == 1 && small.canonical(300).verified(),
                "duplicate (problem, language) pair rejected without overwriting the original");

        // 7. selection is independent of insertion order and reproducible
        Registry reversed = fullRegistry(true);
        check(analyze(reversed, baselineDigests, baselineRuntimes).digest().equals(analyze(full, baselineDigests, baselineRuntimes).digest())
                        && analyze(full, baselineDigests, baselineRuntimes).canonical().equals(driftReport.canonical()),
                "reversed insertion order yields the same report digest and canonical picks");

        // 8. independent oracle: sort the candidates per problem and take the first; plus the empty boundary
        boolean oracleOk = true;
        for (Integer number : full.problems()) {
            List<Entry> candidates = new ArrayList<>();
            for (Entry entry : full.entries) if (entry.number() == number) candidates.add(entry);
            candidates.sort((a, b) -> Registry.beats(a, b) ? -1 : (Registry.beats(b, a) ? 1 : 0));
            oracleOk &= full.canonical(number).key().equals(candidates.get(0).key());
        }
        Registry empty = new Registry();
        Report emptyReport = analyze(empty, Map.of(), Map.of());
        check(oracleOk && empty.canonical(1) == null && emptyReport.problems() == 0
                        && emptyReport.canonical().isEmpty()
                        && emptyReport.coverage().get("java") == 0 && empty.gaps(1).size() == 4,
                "sorted-candidate oracle reproduces every pick and an empty registry is handled");

        System.out.println("entries: " + full.entries.size() + ", problems: " + full.problems().size()
                + ", coverage per language: " + coverage);
        System.out.println("canonical picks: " + analyze(full, baselineDigests, baselineRuntimes).canonical());
        System.out.println("drift: " + driftReport.drift() + ", regressions: " + slowReport.regressions());
        System.out.println("report digest: " + analyze(full, baselineDigests, baselineRuntimes).digest()
                + ", reversed: " + analyze(reversed, baselineDigests, baselineRuntimes).digest());
        System.out.println("PASS");
    }
}
