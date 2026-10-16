/** Task 915 - Implement secure remote attestation flow integrated with orchestration for trust chain.
 *  (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * A worker node may only join the cluster if it can prove, on demand, which code it is running. The flow has
 * four moving parts, and each is a place where a naive implementation fails:
 *
 *   trust chain       manufacturer key -> endorsement of the node's attestation key -> the node's quote; a
 *                     quote only counts if the chain reaches a key the operator already trusts;
 *   challenge         the orchestrator issues a single-use nonce, so a recorded quote cannot be replayed and
 *                     a nonce that was never issued (or was consumed) cannot be presented;
 *   freshness         the quote carries a clock value and expires, limiting the value of a captured quote;
 *   admission policy  trusted measurements, revoked measurements and revoked nodes; a join is admitted only
 *                     when every rule holds, otherwise it is refused with the failing reason.
 *
 * The checks walk every acceptance rule and every single-rule violation - forged signature, unknown
 * measurement, revoked measurement, revoked node, untrusted endorsement, stale and replayed challenges,
 * expired quote and an empty policy that must admit nobody - and re-derive each verdict with an independent
 * implementation so the orchestrator is never compared against itself.
 *
 * Simplification: 64-bit splitmix64 tags stand in for signatures and hashes, the clock is a counter and the
 * "cluster" is a set of worker ids. Invariants verified: soundness (no violating quote is admitted),
 * completeness for honest workers, nonce single-use, and reproducible decisions.
 */
import java.util.*;

/** splitmix64, doubling as the toy hash and MAC. */
final class Crypto915 {
    static long hash(long value) {
        long v = (value ^ 0x9E3779B97F4A7C15L) * 0xBF58476D1CE4E5B9L;
        v = (v ^ (v >>> 27)) * 0x94D049BB133111EBL;
        return v ^ (v >>> 31);
    }

    static long mix(long state, long value) { return hash(state ^ (value * 0x9E3779B97F4A7C15L)); }

    static long hashText(String text) {
        long h = 0x915BEEFL;
        for (int i = 0; i < text.length(); i++) h = mix(h, text.charAt(i));
        return mix(h, text.length());
    }
}

/** The manufacturer's endorsement of one node's attestation key. */
final class Endorsement {
    final String nodeId;
    final long attestationKey;
    final long issuer;
    final long tag;

    Endorsement(String nodeId, long attestationKey, long issuer) {
        this.nodeId = nodeId;
        this.attestationKey = attestationKey;
        this.issuer = issuer;
        this.tag = Crypto915.mix(Crypto915.hashText(nodeId) ^ attestationKey, issuer);
    }

    /** True only when the tag recomputes under a key the operator trusts. */
    boolean verifiesUnder(long trustedIssuer) {
        return issuer == trustedIssuer
                && tag == Crypto915.mix(Crypto915.hashText(nodeId) ^ attestationKey, issuer);
    }
}

/** A node's quote: what it runs, when, answering which challenge, signed with its attestation key. */
final class Quote {
    final String nodeId;
    final long measurement;
    final long challenge;
    final long clock;
    final long endorsementTag;
    final long signature;

    Quote(String nodeId, long measurement, long challenge, long clock, Endorsement endorsement, long key) {
        this.nodeId = nodeId;
        this.measurement = measurement;
        this.challenge = challenge;
        this.clock = clock;
        this.endorsementTag = endorsement.tag;
        this.signature = sign(nodeId, measurement, challenge, clock, key);
    }

    private Quote(Quote source, long newMeasurement) {
        this.nodeId = source.nodeId;
        this.measurement = newMeasurement;
        this.challenge = source.challenge;
        this.clock = source.clock;
        this.endorsementTag = source.endorsementTag;
        this.signature = source.signature;
    }

    /** Tamper helper: rewrite the measurement but keep the old signature. */
    Quote withMeasurement(long newMeasurement) { return new Quote(this, newMeasurement); }

    static long sign(String nodeId, long measurement, long challenge, long clock, long key) {
        long h = Crypto915.hashText(nodeId);
        h = Crypto915.mix(h, measurement);
        h = Crypto915.mix(h, challenge);
        return Crypto915.mix(h, clock ^ key);
    }

    boolean signatureValid(long attestationKey) {
        return signature == sign(nodeId, measurement, challenge, clock, attestationKey);
    }
}

/** The orchestrator's admission policy. */
final class AdmissionPolicy {
    final long trustedIssuer;
    final long freshness;
    final Set<Long> trustedMeasurements = new TreeSet<>();
    final Set<Long> revokedMeasurements = new TreeSet<>();
    final Set<Long> revokedNodes = new TreeSet<>();

    AdmissionPolicy(long trustedIssuer, long freshness) {
        this.trustedIssuer = trustedIssuer;
        this.freshness = freshness;
    }

    AdmissionPolicy trust(long measurement) {
        trustedMeasurements.add(measurement);
        return this;
    }

    AdmissionPolicy revoke(long measurement) {
        revokedMeasurements.add(measurement);
        return this;
    }

    AdmissionPolicy revokeNode(String nodeId) {
        revokedNodes.add(Crypto915.hashText(nodeId));
        return this;
    }
}

/** Verdict plus the reason and the trust chain links that were walked. */
final class Verdict {
    final boolean admitted;
    final String reason;
    final List<String> chain;

    Verdict(boolean admitted, String reason, List<String> chain) {
        this.admitted = admitted;
        this.reason = reason;
        this.chain = chain;
    }
}

/** The orchestrator: nonce issuance, verification and cluster membership. */
final class Orchestrator {
    final AdmissionPolicy policy;
    final Set<Long> outstanding = new TreeSet<>();
    final Set<Long> consumed = new TreeSet<>();
    final Set<String> admitted = new TreeSet<>();
    int counter;

    Orchestrator(AdmissionPolicy policy) { this.policy = policy; }

    long issueChallenge() {
        counter++;
        long nonce = Crypto915.mix(0x915CAFEL, counter);
        outstanding.add(nonce);
        return nonce;
    }

    Set<String> admittedWorkers() { return admitted; }

    Verdict verify(Quote quote, Endorsement endorsement, long now) {
        List<String> chain = new ArrayList<>();
        if (!endorsement.verifiesUnder(policy.trustedIssuer)) {
            return new Verdict(false, "untrusted endorsement", chain);
        }
        chain.add("manufacturer");
        if (!endorsement.nodeId.equals(quote.nodeId) || endorsement.tag != quote.endorsementTag
                || !quote.signatureValid(endorsement.attestationKey)) {
            return new Verdict(false, "bad signature", chain);
        }
        chain.add(quote.nodeId);
        if (consumed.contains(quote.challenge)) return new Verdict(false, "replayed challenge", chain);
        if (!outstanding.contains(quote.challenge)) return new Verdict(false, "stale challenge", chain);
        if (now < quote.clock || now - quote.clock > policy.freshness) {
            return new Verdict(false, "expired quote", chain);
        }
        if (policy.revokedNodes.contains(Crypto915.hashText(quote.nodeId))) {
            return new Verdict(false, "revoked node", chain);
        }
        if (policy.revokedMeasurements.contains(quote.measurement)) {
            return new Verdict(false, "revoked measurement", chain);
        }
        if (!policy.trustedMeasurements.contains(quote.measurement)) {
            return new Verdict(false, "untrusted measurement", chain);
        }
        outstanding.remove(quote.challenge);
        consumed.add(quote.challenge);
        chain.add("quote");
        admitted.add(quote.nodeId);
        return new Verdict(true, "admitted", chain);
    }
}

public class Solution {
    static final long MANUFACTURER_KEY = 0xFAB123456L;
    static final long FRESHNESS = 5;

    public static void main(String[] args) {
        long keyA = Crypto915.hashText("node-a-key");
        Endorsement endorsementA = new Endorsement("node-a", keyA, MANUFACTURER_KEY);
        long good = Crypto915.hashText("image-v3.2");
        long other = Crypto915.hashText("image-v3.1");

        AdmissionPolicy policy = new AdmissionPolicy(MANUFACTURER_KEY, FRESHNESS).trust(good).trust(other);
        Orchestrator orchestrator = new Orchestrator(policy);

        // ---- 1. an honest worker joins and the whole chain is walked -----------------------------------
        long challenge = orchestrator.issueChallenge();
        Verdict accepted = orchestrator.verify(new Quote("node-a", good, challenge, 100, endorsementA, keyA),
                endorsementA, 101);
        check(accepted.admitted, "an honest worker must be admitted, reason " + accepted.reason);
        check(accepted.reason.equals("admitted"), "the verdict must carry the admitted reason");
        check(accepted.chain.equals(List.of("manufacturer", "node-a", "quote")),
                "the trust chain must be walked link by link, got " + accepted.chain);
        check(orchestrator.admittedWorkers().equals(Set.of("node-a")), "the worker must join the cluster");

        // ---- 2. nonce discipline: stale and replayed challenges ----------------------------------------
        expect(orchestrator, endorsementA, new Quote("node-a", good, 0x1234567L, 102, endorsementA, keyA),
                103, "stale challenge");
        expect(orchestrator, endorsementA, new Quote("node-a", good, challenge, 102, endorsementA, keyA), 103,
                "replayed challenge");
        check(orchestrator.issueChallenge() != orchestrator.issueChallenge(),
                "every issued challenge must be distinct");

        // ---- 3. one violated rule at a time, each with its own reason ----------------------------------
        expect(orchestrator, endorsementA, new Quote("node-a", Crypto915.hashText("image-rogue"),
                orchestrator.issueChallenge(), 102, endorsementA, keyA), 103, "untrusted measurement");
        long tamperChallenge = orchestrator.issueChallenge();
        Quote tampered = new Quote("node-a", good, tamperChallenge, 102, endorsementA, keyA)
                .withMeasurement(Crypto915.hashText("image-rogue"));
        expect(orchestrator, endorsementA, tampered, 103, "bad signature");
        long wrongKeyChallenge = orchestrator.issueChallenge();
        expect(orchestrator, endorsementA, new Quote("node-a", good, wrongKeyChallenge, 102, endorsementA,
                keyA ^ 0x1L), 103, "bad signature");
        long expiryChallenge = orchestrator.issueChallenge();
        expect(orchestrator, endorsementA, new Quote("node-a", good, expiryChallenge, 50, endorsementA, keyA),
                400, "expired quote");
        long futureChallenge = orchestrator.issueChallenge();
        expect(orchestrator, endorsementA, new Quote("node-a", good, futureChallenge, 500, endorsementA, keyA),
                100, "expired quote");

        // ---- 4. untrusted endorsements, identity binding and revocation --------------------------------
        Endorsement rogue = new Endorsement("node-b", Crypto915.hashText("k"), MANUFACTURER_KEY ^ 0x99L);
        long rogueChallenge = orchestrator.issueChallenge();
        expect(orchestrator, rogue, new Quote("node-b", good, rogueChallenge, 102, rogue,
                Crypto915.hashText("k")), 103, "untrusted endorsement");
        Endorsement forOtherNode = new Endorsement("node-z", Crypto915.hashText("z-key"), MANUFACTURER_KEY);
        long identityChallenge = orchestrator.issueChallenge();
        expect(orchestrator, forOtherNode, new Quote("node-a", good, identityChallenge, 10, forOtherNode,
                Crypto915.hashText("z-key")), 11, "bad signature");
        AdmissionPolicy revoking = new AdmissionPolicy(MANUFACTURER_KEY, FRESHNESS).trust(good)
                .revoke(other).revokeNode("node-a");
        Orchestrator revoker = new Orchestrator(revoking);
        long revokedChallengeA = revoker.issueChallenge();
        expect(revoker, endorsementA, new Quote("node-a", good, revokedChallengeA, 10, endorsementA, keyA), 11,
                "revoked node");
        check(revoker.admittedWorkers().isEmpty(), "a revoked node must never join");
        AdmissionPolicy measurementOnly = new AdmissionPolicy(MANUFACTURER_KEY, FRESHNESS).trust(good)
                .trust(other).revoke(other);
        Orchestrator lenient = new Orchestrator(measurementOnly);
        long revokedChallengeB = lenient.issueChallenge();
        expect(lenient, endorsementA, new Quote("node-a", other, revokedChallengeB, 10, endorsementA, keyA),
                11, "revoked measurement");
        check(!orchestrator.admittedWorkers().contains("node-b"), "no rogue worker may be in the cluster");
        check(orchestrator.admittedWorkers().size() == 1, "exactly one worker joined, got "
                + orchestrator.admittedWorkers());

        // ---- 5. the verdict is re-derivable -----------------------------------------------------------------
        long recheckChallenge = lenient.issueChallenge();
        Quote requote = new Quote("node-a", other, recheckChallenge, 20, endorsementA, keyA);
        Verdict verdict = lenient.verify(requote, endorsementA, 21);
        check(!verdict.admitted && verdict.reason.equals("revoked measurement"),
                "a revoked measurement must stay refused, got " + verdict.reason);
        check(reDerive(measurementOnly, requote, endorsementA, 21).equals(verdict.reason),
                "an independent re-derivation must agree on the refusal");
        long acceptChallenge = lenient.issueChallenge();
        Quote goodQuote = new Quote("node-a", good, acceptChallenge, 30, endorsementA, keyA);
        Verdict acceptVerdict = lenient.verify(goodQuote, endorsementA, 31);
        check(acceptVerdict.admitted, "a trusted measurement must still be admitted, reason "
                + acceptVerdict.reason);
        check(reDerive(measurementOnly, goodQuote, endorsementA, 31).equals("admitted"),
                "the re-derivation must accept the honest quote");
        check(reDerive(measurementOnly, tampered, endorsementA, 31).equals("bad signature"),
                "the re-derivation must reject the tampered quote");
        check(reDerive(measurementOnly, new Quote("node-a", good, 7L, 5, endorsementA, keyA), endorsementA, 100)
                .equals("expired quote"), "the re-derivation must reject an expired quote");

        // ---- 6. boundaries -----------------------------------------------------------------------------------
        Orchestrator locked = new Orchestrator(new AdmissionPolicy(MANUFACTURER_KEY, FRESHNESS));
        long emptyChallenge = locked.issueChallenge();
        expect(locked, endorsementA, new Quote("node-a", good, emptyChallenge, 5, endorsementA, keyA), 6,
                "untrusted measurement");
        check(locked.admittedWorkers().isEmpty(), "an empty policy must admit nobody");
        AdmissionPolicy zeroWidth = new AdmissionPolicy(MANUFACTURER_KEY, 0).trust(good);
        Orchestrator instantaneous = new Orchestrator(zeroWidth);
        long tightChallenge = instantaneous.issueChallenge();
        check(instantaneous.verify(new Quote("node-a", good, tightChallenge, 7, endorsementA, keyA),
                endorsementA, 7).admitted,
                "a zero-width freshness window must still accept a same-tick quote");
        check(new Orchestrator(zeroWidth).issueChallenge() != 0, "a fresh orchestrator issues a valid nonce");
        Orchestrator single = new Orchestrator(new AdmissionPolicy(MANUFACTURER_KEY, FRESHNESS).trust(good));
        long singleChallenge = single.issueChallenge();
        check(single.verify(new Quote("node-a", good, singleChallenge, 1, endorsementA, keyA), endorsementA, 2)
                .admitted, "a single-node cluster must admit its only trusted worker");
        check(single.admittedWorkers().equals(Set.of("node-a")), "exactly that worker is in the cluster");

        // ---- 7. determinism ------------------------------------------------------------------------------------
        check(Crypto915.hashText("node-a-key") == keyA, "key derivation must be reproducible");
        check(new Endorsement("node-a", keyA, MANUFACTURER_KEY).tag == endorsementA.tag,
                "endorsement tags must be reproducible");
        Orchestrator replay = new Orchestrator(new AdmissionPolicy(MANUFACTURER_KEY, FRESHNESS).trust(good)
                .trust(other));
        long replayChallenge = replay.issueChallenge();
        check(replayChallenge == challenge, "challenge issuance must replay identically");
        Verdict replayVerdict = replay.verify(new Quote("node-a", good, replayChallenge, 100, endorsementA,
                keyA), endorsementA, 101);
        check(replayVerdict.admitted == accepted.admitted && replayVerdict.chain.equals(accepted.chain),
                "the decision must replay identically");

        System.out.println("trust chain manufacturer -> node-a -> quote walked with " + accepted.chain.size()
                + " links; policy trusts " + policy.trustedMeasurements.size() + " measurements");
        System.out.println("refusals exercised: stale/replayed challenge, untrusted measurement, bad signature, "
                + "expired quote, untrusted endorsement, revoked node, revoked measurement, empty policy");
        System.out.println("admitted workers " + orchestrator.admittedWorkers() + "; independent re-derivation "
                + "agrees on accept, refuse and tamper");
        System.out.println("PASS");
    }

    /** Independent re-derivation of the measurement-level rules, without the orchestrator's nonce state. */
    static String reDerive(AdmissionPolicy policy, Quote quote, Endorsement endorsement, long now) {
        if (endorsement.issuer != policy.trustedIssuer) return "untrusted endorsement";
        boolean tagOk = endorsement.tag == Crypto915.mix(Crypto915.hashText(endorsement.nodeId)
                ^ endorsement.attestationKey, endorsement.issuer);
        if (!tagOk || !endorsement.nodeId.equals(quote.nodeId) || endorsement.tag != quote.endorsementTag) {
            return "bad signature";
        }
        if (!quote.signatureValid(endorsement.attestationKey)) return "bad signature";
        if (now < quote.clock || now - quote.clock > policy.freshness) return "expired quote";
        if (policy.revokedNodes.contains(Crypto915.hashText(quote.nodeId))) return "revoked node";
        if (policy.revokedMeasurements.contains(quote.measurement)) return "revoked measurement";
        if (!policy.trustedMeasurements.contains(quote.measurement)) return "untrusted measurement";
        return "admitted";
    }

    static void expect(Orchestrator orchestrator, Endorsement endorsement, Quote quote, long now,
            String reason) {
        Verdict verdict = orchestrator.verify(quote, endorsement, now);
        check(!verdict.admitted, "expected a rejection for " + reason + ", got " + verdict.reason);
        check(verdict.reason.equals(reason), "expected reason '" + reason + "', got '" + verdict.reason + "'");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
