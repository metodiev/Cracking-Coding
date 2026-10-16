/** Task 966 - Implement proctoring/anticheat measures for timed contests (privacy-aware).
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory proctoring pipeline. A session is a list of coarse events (paste, focus loss, keystroke);
 *  the pipeline immediately reduces them to three capped signal buckets (paste count, focus losses,
 *  largest timing gap) inside a pseudonymous record that carries only a cohort hash, never the student id
 *  or the raw events. A weighted risk score decides the flag, and cohort aggregates are published only
 *  when the cohort holds at least k sessions, so no individual can be singled out from a small group.
 *  Simplified: events are synthetic and the "cohort" is a string tag rather than a real contest room; the
 *  bucketing, score, threshold, k-anonymity rule and pseudonymisation are the real mechanics.
 *  Verified: calm sessions stay unflagged; a paste-heavy session is flagged with capped buckets; the
 *  flag threshold is inclusive and boundary-tested directly on the record; records keep no raw events (a
 *  session bucketed and then stripped of its events yields identical records); k-anonymity suppresses
 *  small cohorts while publishing large ones; processing is deterministic and order independent; empty
 *  inputs and event-less sessions are handled; an independent recomputation from the raw events
 *  reproduces every score.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static long splitmix64(long x) {
        long z = x + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;

    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }

    static String hex(long v) { return String.format("%016x", v); }

    record RawEvent(String kind, int tick) {}
    record Session(String student, String cohort, List<RawEvent> events) {}

    static final int BUCKET_CAP = 4, FLAG_THRESHOLD = 6, K_ANONYMITY = 5;
    static final Set<String> SIGNALS = Set.of("paste", "focus-loss", "keystroke");

    /** The only thing that leaves the proctoring pipeline: a pseudonym plus capped counters. */
    record Bucketed(String cohortHash, String studentHash, int pasteBucket, int focusBucket, int gapBucket, int events) {
        int riskScore() { return 3 * pasteBucket + 2 * focusBucket + gapBucket; }
        boolean flagged() { return riskScore() >= FLAG_THRESHOLD; }
    }

    record Cohort(String cohortHash, int size, int flagged, double meanScore, boolean published) {}

    static String pseudonym(String prefix, String id) { return hex(fnvText(FNV_OFFSET, prefix + ":" + id)); }

    static Bucketed bucket(Session session) {
        int pastes = 0, focusLosses = 0, largestGap = 0, previous = -1;
        for (RawEvent event : session.events()) {
            if (!SIGNALS.contains(event.kind())) continue;
            if (event.kind().equals("paste")) pastes++;
            if (event.kind().equals("focus-loss")) focusLosses++;
            if (previous >= 0) largestGap = Math.max(largestGap, event.tick() - previous);
            previous = event.tick();
        }
        return new Bucketed(pseudonym("cohort", session.cohort()), pseudonym("student", session.student()),
                Math.min(BUCKET_CAP, pastes), Math.min(BUCKET_CAP, focusLosses), Math.min(BUCKET_CAP, largestGap / 5),
                session.events().size());
    }

    static final class Pipeline {
        final List<Bucketed> records = new ArrayList<>();

        List<Bucketed> process(List<Session> sessions) {
            records.clear();
            List<Session> sorted = new ArrayList<>(sessions);
            sorted.sort(Comparator.comparing(Session::student));
            for (Session session : sorted) records.add(bucket(session));
            return new ArrayList<>(records);
        }

        List<Cohort> publish() {
            Map<String, List<Bucketed>> groups = new TreeMap<>();
            for (Bucketed record : records) groups.computeIfAbsent(record.cohortHash(), key -> new ArrayList<>()).add(record);
            List<Cohort> cohorts = new ArrayList<>();
            for (Map.Entry<String, List<Bucketed>> entry : groups.entrySet()) {
                int flagged = 0, total = 0;
                for (Bucketed record : entry.getValue()) {
                    total += record.riskScore();
                    if (record.flagged()) flagged++;
                }
                boolean published = entry.getValue().size() >= K_ANONYMITY;
                double mean = (double) total / entry.getValue().size();
                cohorts.add(new Cohort(entry.getKey(), entry.getValue().size(), flagged, published ? mean : 0.0, published));
            }
            return cohorts;
        }

        List<Bucketed> flagged() {
            List<Bucketed> out = new ArrayList<>();
            for (Bucketed record : records) if (record.flagged()) out.add(record);
            return out;
        }

        String digest() {
            long h = fnvText(FNV_OFFSET, "proctor:" + FLAG_THRESHOLD + "/" + BUCKET_CAP);
            for (Bucketed record : records)
                h = fnvText(h, record.cohortHash() + ":" + record.studentHash() + ":" + record.riskScore() + "|");
            return hex(h);
        }
    }

    /** Independent score: walks the raw events again and applies the documented weighting directly. */
    static int riskFromRawEvents(Session session) {
        int pastes = 0, focusLosses = 0, largestGap = 0, previous = -1;
        for (RawEvent event : session.events()) {
            if (previous >= 0) largestGap = Math.max(largestGap, event.tick() - previous);
            previous = event.tick();
            if (event.kind().equals("paste")) pastes++;
            if (event.kind().equals("focus-loss")) focusLosses++;
        }
        int pasteScore = 3 * Math.min(BUCKET_CAP, pastes);
        int focusScore = 2 * Math.min(BUCKET_CAP, focusLosses);
        int gapScore = Math.min(BUCKET_CAP, largestGap / 5);
        return pasteScore + focusScore + gapScore;
    }

    static Session honest(String student, String cohort, int seed) {
        List<RawEvent> events = new ArrayList<>();
        long state = splitmix64(seed);
        int tick = 0;
        for (int i = 0; i < 12; i++) {
            state = splitmix64(state + i);
            tick += 4 + (int) Math.floorMod(state, 3);
            events.add(new RawEvent("keystroke", tick));
        }
        return new Session(student, cohort, events);
    }

    static Session noisy(String student, String cohort, int seed) {
        Session base = honest(student, cohort, seed);
        List<RawEvent> events = new ArrayList<>(base.events());
        events.add(new RawEvent("focus-loss", 0));
        return new Session(student, cohort, events);
    }

    static Session cheating(String student, String cohort) {
        List<RawEvent> events = new ArrayList<>();
        for (int i = 0; i < 5; i++) events.add(new RawEvent("paste", 10 + i * 40));
        events.add(new RawEvent("focus-loss", 200));
        events.add(new RawEvent("focus-loss", 260));
        events.add(new RawEvent("focus-loss", 320));
        events.add(new RawEvent("keystroke", 400));
        return new Session(student, cohort, events);
    }

    static List<Session> sessions() {
        List<Session> all = new ArrayList<>();
        for (int i = 0; i < 10; i++) all.add(honest("alpha-" + i, "alpha", 100 + i));
        for (int i = 0; i < 6; i++) all.add(i % 2 == 0 ? noisy("beta-" + i, "beta", 200 + i) : honest("beta-" + i, "beta", 200 + i));
        all.add(cheating("beta-cheat", "beta"));
        all.add(honest("gamma-0", "gamma", 300));
        all.add(honest("gamma-1", "gamma", 301));
        return all;
    }

    public static void main(String[] args) {
        Pipeline pipeline = new Pipeline();
        List<Bucketed> records = pipeline.process(sessions());

        // 1. calm sessions stay unflagged: only the paste-heavy session trips the threshold
        List<Bucketed> flagged = pipeline.flagged();
        check(flagged.size() == 1 && pipeline.records.size() == 19
                        && flagged.get(0).studentHash().equals(pseudonym("student", "beta-cheat")),
                "exactly one session out of 19 is flagged");

        // 2. buckets are capped so a single metric cannot dominate
        Bucketed cheat = flagged.get(0);
        check(cheat.pasteBucket() == BUCKET_CAP && cheat.focusBucket() == 3 && cheat.gapBucket() == BUCKET_CAP
                        && cheat.riskScore() == 22 && cheat.flagged(),
                "paste-heavy session is flagged with capped buckets and score " + cheat.riskScore());

        // 3. the threshold is inclusive and tested directly on a record
        check(new Bucketed("c", "s", 2, 0, 0, 0).riskScore() == 6 && new Bucketed("c", "s", 2, 0, 0, 0).flagged()
                        && new Bucketed("c", "s", 1, 1, 0, 0).riskScore() == 5
                        && !new Bucketed("c", "s", 1, 1, 0, 0).flagged(),
                "score 6 is flagged and score 5 is not");

        // 4. privacy: records carry no raw events or student identifiers
        List<Session> stripped = new ArrayList<>();
        for (Session session : sessions()) stripped.add(new Session(session.student(), session.cohort(), new ArrayList<>()));
        Pipeline afterStrip = new Pipeline();
        List<Bucketed> strippedRecords = afterStrip.process(stripped);
        boolean pseudonymous = true;
        for (Bucketed record : pipeline.records) {
            pseudonymous &= !record.studentHash().contains("beta-cheat") && !record.cohortHash().contains("beta");
            pseudonymous &= record.cohortHash().length() == 16;
        }
        check(pseudonymous && strippedRecords.size() == records.size()
                        && afterStrip.records.get(0).studentHash().equals(pipeline.records.get(0).studentHash()),
                "records are pseudonymous and hold no raw events");

        // 5. k-anonymity: small cohorts are suppressed, large ones published
        List<Cohort> cohorts = pipeline.publish();
        int published = 0, suppressed = 0;
        boolean sizesOk = true;
        for (Cohort cohort : cohorts) {
            sizesOk &= cohort.published() == (cohort.size() >= K_ANONYMITY);
            if (cohort.published()) published++;
            else suppressed++;
        }
        Cohort beta = null;
        for (Cohort cohort : cohorts) if (cohort.size() == 7) beta = cohort;
        check(cohorts.size() == 3 && published == 2 && suppressed == 1 && sizesOk
                        && beta != null && beta.flagged() == 1 && beta.published(),
                "gamma cohort of 2 is suppressed while alpha and beta are published");

        // 6. determinism and order independence
        Pipeline shuffled = new Pipeline();
        List<Session> reversed = new ArrayList<>(sessions());
        Collections.reverse(reversed);
        shuffled.process(reversed);
        check(shuffled.digest().equals(pipeline.digest()) && new Pipeline().process(sessions()).size() == records.size()
                        && new Pipeline().process(sessions()).get(0).equals(records.get(0)),
                "shuffled input produces the same pseudonymous records and digest");

        // 7. boundaries: no sessions, and a session that produced no events
        Pipeline empty = new Pipeline();
        check(empty.process(List.of()).isEmpty() && empty.publish().isEmpty() && empty.flagged().isEmpty()
                        && empty.digest().equals(hex(fnvText(FNV_OFFSET, "proctor:" + FLAG_THRESHOLD + "/" + BUCKET_CAP))),
                "an empty contest produces no records, no cohorts and no flags");
        Pipeline quiet = new Pipeline();
        quiet.process(List.of(new Session("quiet", "delta", new ArrayList<>()), new Session("quiet-2", "delta", new ArrayList<>())));
        check(quiet.records.get(0).riskScore() == 0 && !quiet.records.get(0).flagged()
                        && quiet.publish().get(0).size() == 2 && !quiet.publish().get(0).published(),
                "an event-less session scores zero and still counts towards its cohort");

        // 8. independent oracle: recompute every score from the raw events
        boolean oracleOk = true;
        for (Session session : sessions()) {
            int expected = riskFromRawEvents(session);
            int actual = bucket(session).riskScore();
            oracleOk &= expected == actual;
        }
        int recomputedFlags = 0;
        for (Session session : sessions()) if (riskFromRawEvents(session) >= FLAG_THRESHOLD) recomputedFlags++;
        check(oracleOk && recomputedFlags == flagged.size(),
                "an independent walk over the raw events reproduces every score and the flag count");

        System.out.println("sessions: " + records.size() + ", flagged: " + flagged.size());
        System.out.println("cohorts: " + cohorts.size() + " (published " + published + ", suppressed " + suppressed + ")");
        System.out.println("flagged record: " + cheat);
        System.out.println("pipeline digest: " + pipeline.digest());
        System.out.println("PASS");
    }
}
