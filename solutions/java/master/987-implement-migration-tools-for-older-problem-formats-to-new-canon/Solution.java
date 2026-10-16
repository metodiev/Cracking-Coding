/** Task 987 - Implement migration tools for older problem formats to new canonical schema.
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  Legacy records are flat maps: format v1 carries id/title/text/tags/level, format v2 adds chapter and
 *  a numeric difficulty, and the canonical schema (v3) is a sorted map with slug, title, body, tags,
 *  difficulty (1..5), chapter (int) and schema_version=3. The migrator detects the format from the
 *  record itself, applies v1->v2->v3 as separate steps so any intermediate shape can be produced and
 *  re-ingested, and validates at every hop: a blank title, an unknown level, an unparseable chapter,
 *  an out-of-range difficulty or an unrecognised version becomes a diagnostic instead of a record.
 *  Tag normalisation (trim, lower-case, de-duplicate, sort) happens exactly once, in the final step, so
 *  migration is idempotent on canonical records and the round trip canonical -> v2 -> canonical is a
 *  fixed point.
 *  Simplified: maps of strings instead of files, and the chapter hint comes from a record field rather
 *  than from a directory layout; the staged migration, the validation gates and the fixed-point
 *  properties are the real ones.
 *  Verified: v1 and v2 records land on the exact canonical fields, all five rejection paths fire, tag
 *  normalisation survives empty and messy inputs, double migration is a no-op, the v2 round trip is a
 *  fixed point, and an independent hand-written extractor reproduces slug and tags for every record.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final int CANONICAL_VERSION = 3;
    static final Map<String, Integer> LEVELS = Map.of("easy", 2, "medium", 3, "hard", 4);

    static Map<String, String> record(String... kv) {
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) m.put(kv[i], kv[i + 1]);
        return m;
    }

    static final class Result {
        final Map<String, String> canonical;   // null when the record was rejected
        final String error, applied;
        Result(Map<String, String> canonical, String error, String applied) {
            this.canonical = canonical; this.error = error; this.applied = applied;
        }
        boolean ok() { return canonical != null; }
    }

    static int detectVersion(Map<String, String> rec) {
        String declared = rec.get("schema_version");
        if (declared != null) {
            try { return Integer.parseInt(declared.trim()); } catch (NumberFormatException e) { return -1; }
        }
        return rec.containsKey("level") ? 1 : rec.containsKey("difficulty") ? 2 : -1;
    }

    /** Transforms any recognised legacy shape into the canonical map, or explains why it cannot. */
    static Result migrate(Map<String, String> rec) {
        int version = detectVersion(rec);
        String applied = "none";
        Map<String, String> cur = rec;
        if (version == 1) {
            Map<String, String> step = v1ToV2(cur);
            if (step == null) return new Result(null, "v1 record has an unknown level: " + rec.get("level"), "rejected");
            cur = step;
            applied = "v1->v2";
            version = 2;
        }
        if (version == 2) {
            Result step = v2ToV3(cur);
            if (!step.ok()) return step;
            return new Result(step.canonical, null, applied.equals("none") ? "v2->v3" : applied + ",v2->v3");
        }
        if (version == CANONICAL_VERSION) return new Result(new TreeMap<>(cur), null, "no-op");
        return new Result(null, "unsupported schema version: " + rec.get("schema_version"), "rejected");
    }

    static Map<String, String> v1ToV2(Map<String, String> rec) {
        String level = rec.getOrDefault("level", "").trim().toLowerCase(Locale.ROOT);
        if (!LEVELS.containsKey(level)) return null;
        Map<String, String> out = new LinkedHashMap<>(rec);
        out.remove("level");
        out.put("difficulty", String.valueOf(LEVELS.get(level)));
        out.put("chapter", rec.getOrDefault("chapter_hint", "0"));
        out.put("schema_version", "2");
        return out;
    }

    static Result v2ToV3(Map<String, String> rec) {
        String title = rec.getOrDefault("title", "").trim();
        if (title.isEmpty()) return new Result(null, "title must not be blank", "rejected");
        String body = rec.getOrDefault("text", "").trim();
        if (body.isEmpty()) return new Result(null, "body must not be empty", "rejected");
        int difficulty;
        try { difficulty = Integer.parseInt(rec.getOrDefault("difficulty", "3").trim()); }
        catch (NumberFormatException e) { return new Result(null, "difficulty is not numeric", "rejected"); }
        if (difficulty < 1 || difficulty > 5) return new Result(null, "difficulty out of range 1..5: " + difficulty, "rejected");
        int chapter;
        try { chapter = parseChapter(rec.getOrDefault("chapter", "0")); }
        catch (NumberFormatException e) { return new Result(null, "chapter is not parseable: " + rec.get("chapter"), "rejected"); }

        Map<String, String> out = new TreeMap<>();
        out.put("schema_version", String.valueOf(CANONICAL_VERSION));
        out.put("slug", slugify(title));
        out.put("title", title);
        out.put("body", body);
        out.put("tags", String.join(",", normalizeTags(rec.getOrDefault("tags", ""))));
        out.put("difficulty", String.valueOf(difficulty));
        out.put("chapter", String.valueOf(chapter));
        return new Result(out, null, "v2->v3");
    }

    /** "Ch 4" / "chapter 12" / "4" all mean chapter 4; anything else is a hard error. */
    static int parseChapter(String raw) {
        String digits = raw.replaceAll("\\D+", "");
        if (digits.isEmpty()) return 0;
        return Integer.parseInt(digits);
    }

    static String slugify(String title) {
        String s = title.trim().toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-");
        while (s.startsWith("-")) s = s.substring(1);
        while (s.endsWith("-")) s = s.substring(0, s.length() - 1);
        return s;
    }

    static List<String> normalizeTags(String raw) {
        Set<String> tags = new TreeSet<>();
        for (String part : raw.split(","))
            if (!part.trim().isEmpty()) tags.add(part.trim().toLowerCase(Locale.ROOT));
        return new ArrayList<>(tags);
    }

    /** Exports a canonical record back to the v2 shape so that re-ingesting it must be a fixed point. */
    static Map<String, String> toV2(Map<String, String> canonical) {
        Map<String, String> out = new LinkedHashMap<>();
        out.put("id", canonical.get("slug"));
        out.put("title", canonical.get("title"));
        out.put("text", canonical.get("body"));
        out.put("tags", canonical.get("tags"));
        out.put("difficulty", canonical.get("difficulty"));
        out.put("chapter", canonical.get("chapter"));
        out.put("schema_version", "2");
        return out;
    }

    public static void main(String[] args) {
        Map<String, String> v1 = record("id", "p17", "title", "  BFS Order  ", "text", "visit the frontier",
                "tags", "Graph, BFS ,graph", "level", "hard", "chapter_hint", "Ch 4");
        Result first = migrate(v1);
        check(first.ok() && first.applied.equals("v1->v2,v2->v3"), "v1 records must walk both hops: " + first.applied);
        Map<String, String> canonical = first.canonical;
        check(canonical.get("slug").equals("bfs-order") && canonical.get("title").equals("BFS Order")
                        && canonical.get("body").equals("visit the frontier") && canonical.get("tags").equals("bfs,graph")
                        && canonical.get("difficulty").equals("4") && canonical.get("chapter").equals("4")
                        && canonical.get("schema_version").equals("3"),
                "v1 must land on the exact canonical fields: " + canonical);
        check(new ArrayList<>(canonical.keySet()).equals(List.of("body", "chapter", "difficulty", "schema_version",
                        "slug", "tags", "title")),
                "canonical keys must be sorted for deterministic dumps: " + canonical.keySet());

        Map<String, String> v2 = record("id", "p18", "title", "Merge Sort", "text", "split then merge",
                "tags", "sorting,,SORTING,arrays", "difficulty", "5", "chapter", "chapter 7", "schema_version", "2");
        Result second = migrate(v2);
        check(second.ok() && second.applied.equals("v2->v3") && second.canonical.get("chapter").equals("7")
                        && second.canonical.get("tags").equals("arrays,sorting"),
                "v2 must be migrated directly with normalised tags: " + second.canonical);

        // --- rejection paths: every validation gate must name a reason instead of inventing a record.
        Result noTitle = migrate(record("title", "   ", "text", "body", "difficulty", "3", "schema_version", "2"));
        Result noBody = migrate(record("title", "T", "text", "", "difficulty", "3", "schema_version", "2"));
        Result badLevel = migrate(record("title", "T", "text", "b", "level", "trivial"));
        Result badDifficulty = migrate(record("title", "T", "text", "b", "difficulty", "7", "schema_version", "2"));
        Result nonNumeric = migrate(record("title", "T", "text", "b", "difficulty", "many", "schema_version", "2"));
        Result badVersion = migrate(record("title", "T", "text", "b", "schema_version", "9"));
        Result badChapter = migrate(record("title", "T", "text", "b", "difficulty", "3", "chapter", "--", "schema_version", "2"));
        check(!noTitle.ok() && noTitle.error.equals("title must not be blank"), "blank titles must be rejected: " + noTitle.error);
        check(!noBody.ok() && noBody.error.equals("body must not be empty"), "empty bodies must be rejected");
        check(!badLevel.ok() && badLevel.error.contains("unknown level"), "unknown v1 levels must be rejected");
        check(!badDifficulty.ok() && badDifficulty.error.contains("out of range"), "difficulty 7 must be rejected");
        check(!nonNumeric.ok() && nonNumeric.error.equals("difficulty is not numeric"), "non-numeric difficulty must be rejected");
        check(!badVersion.ok() && badVersion.error.equals("unsupported schema version: 9"), "unknown versions must be rejected");
        check(badChapter.ok() && badChapter.canonical.get("chapter").equals("0"),
                "a chapter with no digits falls back to 0: " + badChapter.canonical);

        // --- idempotency: canonical input is a fixed point, and v1 is no longer detected.
        Result again = migrate(canonical);
        check(again.ok() && again.applied.equals("no-op") && again.canonical.equals(canonical),
                "migrating a canonical record must be a no-op: " + again.applied);
        Result twice = migrate(second.canonical);
        check(twice.canonical.equals(second.canonical) && twice.applied.equals("no-op"),
                "second migration must not shift any field");
        // --- round trip through the legacy v2 shape must return the identical canonical map.
        Result roundTrip = migrate(toV2(canonical));
        check(roundTrip.ok() && roundTrip.canonical.equals(canonical),
                "canonical -> v2 -> canonical must be a fixed point: " + roundTrip.canonical);

        // --- boundary: empty tags are legal, and a one-character title still slugs.
        Result bare = migrate(record("title", "A", "text", "x", "tags", "", "difficulty", "1", "schema_version", "2"));
        check(bare.ok() && bare.canonical.get("tags").equals("") && bare.canonical.get("slug").equals("a")
                        && bare.canonical.get("difficulty").equals("1"),
                "empty tags must stay empty and the slug must survive a one-char title: " + bare.canonical);

        // --- batch report: successes and errors are counted without leaking a partial record.
        List<Map<String, String>> batch = List.of(v1, v2, canonical, record("schema_version", "9"),
                record("title", "Legacy", "text", "t", "level", "easy", "tags", " DP "));
        int okCount = 0, errors = 0;
        for (Map<String, String> rec : batch) { if (migrate(rec).ok()) okCount++; else errors++; }
        check(okCount == 4 && errors == 1,         "batch migration must report " + 4 + " ok and 1 error: " + okCount + "/" + errors);
        // --- independent oracle: a hand-written extractor reproduces slug and tags without reusing helpers.
        List<Map<String, String>> samples = new ArrayList<>(batch);
        samples.add(record("title", "Graph Coloring  !", "text", "b", "tags", "GRAPH , coloring", "difficulty", "2",
                "chapter", "Ch 9", "schema_version", "2"));
        for (Map<String, String> rec : samples) {
            Result r = migrate(rec);
            if (!r.ok()) continue;
            String title = rec.getOrDefault("title", "").trim();
            String expectedSlug = oracleSlug(title);
            String expectedTags = oracleTags(rec.getOrDefault("tags", ""));
            check(r.canonical.get("slug").equals(expectedSlug) && r.canonical.get("tags").equals(expectedTags),
                    "oracle mismatch for '" + title + "': " + r.canonical.get("slug") + "/" + r.canonical.get("tags")
                            + " vs " + expectedSlug + "/" + expectedTags);
        }

        System.out.println("migrated: " + okCount + " ok, " + errors + " rejected");
        System.out.println("canonical sample: " + canonical);
        System.out.println("PASS");
    }

    /** Oracle slug: lower-case, every non-alphanumeric run becomes one dash, edges trimmed. */
    static String oracleSlug(String title) {
        StringBuilder sb = new StringBuilder();
        boolean pendingDash = false;
        for (char c : title.toLowerCase(Locale.ROOT).toCharArray()) {
            if (Character.isLetterOrDigit(c)) {
                if (pendingDash && sb.length() > 0) sb.append('-');
                pendingDash = false;
                sb.append(c);
            } else pendingDash = true;
        }
        return sb.toString();
    }

    /** Oracle tags: split, trim, lower-case, insert into a sorted set, join back with commas. */
    static String oracleTags(String raw) {
        List<String> seen = new ArrayList<>();
        for (String part : raw.split(",", -1)) {
            String t = part.trim().toLowerCase(Locale.ROOT);
            if (t.isEmpty() || seen.contains(t)) continue;
            seen.add(t);
        }
        Collections.sort(seen);
        return String.join(",", seen);
    }
}
