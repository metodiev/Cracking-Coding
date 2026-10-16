/** Task 979 - Implement certification exam generator and proctoring integration.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  The exam side assembles blueprinted forms: a quota per topic, candidate items ordered by (times used,
 *  seed-fixed tie-break key), an item never exceeding its exposure limit and every form sorted by
 *  difficulty with no repeats; an impossible quota is reported as a shortfall instead of silently shrinking
 *  the blueprint. The proctoring side scores a session's anomaly events (tab switch 2, paste 3, multiple
 *  faces 5, face absent 4, network 0) with a per-kind contribution cap, maps the risk to ok / review /
 *  invalidate and records every action in a hash chain so a tampered log no longer verifies.
 *  Simplified: an in-memory exposure counter over a few forms, a static severity table and a polynomial
 *  hash chain instead of a real item bank, ML proctoring and a keyed ledger; the quota, exposure, ordering,
 *  risk-threshold and chain-verification properties are the real ones.
 *  Verified: quotas, a brute-force best-key oracle for the first form, exposure limits, ordering and
 *  uniqueness, form rotation with seed-stable checksums, deduped and capped risk scoring with threshold
 *  boundaries, tamper detection, the shortfall and empty-input boundaries and unknown-topic rejection.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final Map<String, Double> SEVERITY = Map.of(
            "tab-switch", 2.0, "copy-paste", 3.0, "multiple-faces", 5.0, "face-absent", 4.0, "network-drop", 0.0);
    static final double KIND_CAP = 10.0, REVIEW_AT = 8.0, INVALIDATE_AT = 12.0;

    static final class Item {
        final String id, topic; final double difficulty; final int exposureLimit;
        Item(String id, String topic, double difficulty, int exposureLimit) {
            this.id = id; this.topic = topic; this.difficulty = difficulty; this.exposureLimit = exposureLimit;
        }
    }

    static final class Form {
        final List<Item> items = new ArrayList<>(); final Map<String, Integer> shortfall = new TreeMap<>();
        String checksum() {
            long hash = 1125899906842597L;
            for (Item item : items) for (int i = 0; i < item.id.length(); i++) hash = hash * 31 + item.id.charAt(i);
            return Long.toHexString(hash);
        }
    }

    /** Seed-fixed tie-break key per item: the only place the seed enters the assembly. */
    static Map<String, Long> tieBreakKeys(List<Item> pool, long seed) {
        Map<String, Long> keys = new TreeMap<>();
        long state = seed;
        for (Item item : pool) {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            keys.put(item.id, Math.floorMod(z ^ (z >>> 31), 1_000_003L));
        }
        return keys;
    }
    /** Assembles `forms` blueprinted forms respecting quotas, exposure limits and difficulty order. */
    static List<Form> generateForms(List<Item> pool, Map<String, Integer> blueprint, int forms, long seed) {
        for (String topic : blueprint.keySet()) {
            boolean known = false;
            for (Item item : pool) known |= item.topic.equals(topic);
            if (!known) throw new IllegalArgumentException("unknown topic in blueprint: " + topic);
        }
        Map<String, Long> keys = tieBreakKeys(pool, seed);
        Map<String, Integer> used = new TreeMap<>();
        for (Item item : pool) used.put(item.id, 0);
        List<Form> out = new ArrayList<>();
        for (int f = 0; f < forms; f++) {
            Form form = new Form();
            for (Map.Entry<String, Integer> quota : new TreeMap<>(blueprint).entrySet()) {
                int remaining = quota.getValue();
                while (remaining > 0) {
                    Item best = null;
                    for (Item item : pool) {
                        if (!item.topic.equals(quota.getKey()) || used.get(item.id) >= item.exposureLimit) continue;
                        if (form.items.contains(item)) continue;
                        if (best == null || used.get(item.id) < used.get(best.id)
                                || (used.get(item.id).equals(used.get(best.id)) && keys.get(item.id) < keys.get(best.id))) {
                            best = item;
                        }
                    }
                    if (best == null) break;
                    form.items.add(best);
                    used.merge(best.id, 1, Integer::sum);
                    remaining--;
                }
                if (remaining > 0) form.shortfall.put(quota.getKey(), remaining);
            }
            form.items.sort(Comparator.comparingDouble((Item i) -> i.difficulty).thenComparing(i -> i.id));
            out.add(form);
        }
        return out;
    }
    /** Independent oracle for the first form: per topic the quota items with the smallest tie-break keys. */
    static Set<String> firstFormOracle(List<Item> pool, Map<String, Integer> blueprint, long seed) {
        Map<String, Long> keys = tieBreakKeys(pool, seed);
        Set<String> chosen = new TreeSet<>();
        for (Map.Entry<String, Integer> quota : blueprint.entrySet()) {
            List<Item> candidates = new ArrayList<>();
            for (Item item : pool) if (item.topic.equals(quota.getKey())) candidates.add(item);
            candidates.sort(Comparator.comparingLong((Item i) -> keys.get(i.id)).thenComparing(i -> i.id));
            for (int i = 0; i < Math.min(quota.getValue(), candidates.size()); i++) chosen.add(candidates.get(i).id);
        }
        return chosen;
    }
    static final class Event {
        final String session, kind; final int minute;
        Event(String session, String kind, int minute) {
            if (!SEVERITY.containsKey(kind)) throw new IllegalArgumentException("unknown anomaly " + kind);
            this.session = session; this.kind = kind; this.minute = minute;
        }
        String key() { return session + "/" + kind + "/" + minute; }
    }

    /** Risk score: identical (session, kind, minute) events collapse, each kind is capped. */
    static double risk(String session, List<Event> events) {
        Map<String, Integer> seen = new TreeMap<>();
        Map<String, Double> totals = new TreeMap<>();
        for (Event event : events) {
            if (!event.session.equals(session)) continue;
            if (seen.merge(event.key(), 1, Integer::sum) > 1) continue;
            totals.merge(event.kind, SEVERITY.get(event.kind), Double::sum);
        }
        double total = 0;
        for (double contribution : totals.values()) total += Math.min(KIND_CAP, contribution);
        return total;
    }
    static String verdict(double risk) {
        if (risk >= INVALIDATE_AT) return "invalidate";
        if (risk >= REVIEW_AT) return "review";
        return "ok";
    }

    /** Hash chain over the proctoring log: entry i is bound to every previous entry. */
    static final class Ledger {
        final List<String> entries = new ArrayList<>();
        final List<Long> hashes = new ArrayList<>();
        void append(String entry) {
            long previous = hashes.isEmpty() ? 0L : hashes.get(hashes.size() - 1);
            long hash = previous * 1000003L + 7;
            for (int i = 0; i < entry.length(); i++) hash = hash * 31 + entry.charAt(i);
            entries.add(entry);
            hashes.add(hash);
        }
        boolean verify() {
            long previous = 0L;
            for (int i = 0; i < entries.size(); i++) {
                long hash = previous * 1000003L + 7;
                for (int j = 0; j < entries.get(i).length(); j++) hash = hash * 31 + entries.get(i).charAt(j);
                if (hash != hashes.get(i)) return false;
                previous = hash;
            }
            return true;
        }
    }

    static List<Item> pool() {
        return List.of(new Item("a1", "arrays", 1.0, 2), new Item("a2", "arrays", 1.5, 3),
                new Item("a3", "arrays", 2.0, 3), new Item("a4", "arrays", 2.5, 2),
                new Item("g1", "graphs", 3.0, 3), new Item("g2", "graphs", 3.5, 3), new Item("g3", "graphs", 4.0, 3),
                new Item("d1", "dp", 4.5, 3), new Item("d2", "dp", 5.0, 3));
    }

    static List<Event> session() {
        return List.of(new Event("s1", "tab-switch", 4), new Event("s1", "copy-paste", 5),
                new Event("s1", "tab-switch", 4), new Event("s1", "network-drop", 6),
                new Event("s2", "multiple-faces", 3), new Event("s2", "tab-switch", 9));
    }

    public static void main(String[] args) {
        List<Item> pool = pool();
        Map<String, Integer> blueprint = new LinkedHashMap<>(Map.of("arrays", 2, "graphs", 2, "dp", 1));
        List<Form> forms = generateForms(pool, blueprint, 3, 2024L);

        Map<String, Integer> counts = new TreeMap<>();
        for (Item item : forms.get(0).items) counts.merge(item.topic, 1, Integer::sum);
        Set<String> actual = new TreeSet<>();
        for (Item item : forms.get(0).items) actual.add(item.id);
        check(counts.equals(new TreeMap<>(blueprint)) && forms.get(0).items.size() == 5
                        && forms.get(0).shortfall.isEmpty(),
                "every form must satisfy the blueprint quotas exactly: " + counts);
        check(actual.equals(firstFormOracle(pool, blueprint, 2024L)),
                "the first form must match the independent best-key oracle: " + actual);
        Map<String, Integer> usage = new TreeMap<>();
        for (Form form : forms) for (Item item : form.items) usage.merge(item.id, 1, Integer::sum);
        boolean limits = true;
        for (Item item : pool) limits &= usage.getOrDefault(item.id, 0) <= item.exposureLimit;
        check(limits && usage.values().stream().mapToInt(Integer::intValue).sum() == 15,
                "no item may exceed its exposure limit across forms: " + usage);
        boolean ordered = true;
        for (Form form : forms) {
            for (int i = 1; i < form.items.size(); i++) {
                ordered &= form.items.get(i).difficulty >= form.items.get(i - 1).difficulty;
            }
            ordered &= new HashSet<>(form.items).size() == form.items.size();
        }
        check(ordered && forms.get(0).items.get(0).difficulty == forms.get(0).items.stream()
                        .mapToDouble(i -> i.difficulty).min().orElse(-99)
                        && forms.get(1).items.get(0).difficulty == forms.get(1).items.stream()
                        .mapToDouble(i -> i.difficulty).min().orElse(-99),
                "forms must be difficulty-ordered and duplicate-free");

        Set<String> first = new TreeSet<>(forms.get(0).items.stream().map(i -> i.id).toList());
        Set<String> second = new TreeSet<>(forms.get(1).items.stream().map(i -> i.id).toList());
        check(!first.equals(second) && generateForms(pool, blueprint, 3, 2024L).get(2).checksum()
                        .equals(forms.get(2).checksum()),
                "consecutive forms must rotate items while a re-run with the same seed reproduces them");
        check(risk("s1", session()) == 5.0 && risk("s2", session()) == 7.0
                        && verdict(risk("s2", session())).equals("ok") && verdict(7.999).equals("ok")
                        && verdict(8.0).equals("review") && verdict(12.0).equals("invalidate")
                        && verdict(30.0).equals("invalidate"),
                "risk scoring must dedup, cap per kind and map to the right verdict: " + risk("s2", session()));

        List<Event> flooded = new ArrayList<>(session());
        for (int minute = 0; minute < 20; minute++) flooded.add(new Event("s3", "tab-switch", minute));
        double capped = risk("s3", flooded);
        check(capped == KIND_CAP && verdict(capped).equals("review") && risk("s3", flooded) == KIND_CAP,
                "a flood of one anomaly kind must be capped instead of unbounded: " + capped);

        Ledger ledger = new Ledger();
        for (Event event : session()) ledger.append(event.key());
        ledger.append("verdict s1 " + verdict(risk("s1", session())));
        Ledger tampered = new Ledger();
        for (Event event : session()) tampered.append(event.key());
        tampered.append("verdict s1 " + verdict(risk("s1", session())));
        tampered.entries.set(2, "s1/network-drop/4");
        check(ledger.verify() && ledger.entries.size() == 7 && !tampered.verify(),
                "an untouched proctoring ledger must verify and an edited entry must break the chain");

        Map<String, Integer> huge = Map.of("arrays", 99, "graphs", 2, "dp", 1);
        List<Form> shortForms = generateForms(pool, huge, 1, 7L);
        check(shortForms.get(0).shortfall.get("arrays") == 95 && shortForms.get(0).items.size() == 7
                        && !shortForms.get(0).shortfall.containsKey("graphs"),
                "an impossible quota must be reported as a shortfall: " + shortForms.get(0).shortfall);

        boolean rejected = false;
        try {
            generateForms(pool, Map.of("geometry", 1), 1, 1L);
        } catch (IllegalArgumentException e) {
            rejected = e.getMessage().contains("unknown topic");
        }
        check(rejected && generateForms(pool, Map.of(), 2, 1L).get(0).items.isEmpty()
                        && generateForms(List.of(), Map.of(), 1, 1L).get(0).items.isEmpty(),
                "an unknown topic must be rejected and empty inputs must yield an empty form");

        System.out.println("forms=" + forms.size() + " perForm=" + forms.get(0).items.size()
                + " usage=" + usage.values().stream().mapToInt(Integer::intValue).sum());
        System.out.println("risk s1=" + risk("s1", session()) + " s2=" + risk("s2", session())
                + " verdict=" + verdict(risk("s2", session())) + " capped=" + capped);
        System.out.println("PASS");
    }
}
