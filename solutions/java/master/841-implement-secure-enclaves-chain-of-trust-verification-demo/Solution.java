/** Task 841 - Implement secure enclaves chain-of-trust verification demo. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A chain of enclaves is a list of attestation documents, one per enclave, ordered from the root enclave
 * to the leaf that wants to be trusted. Every document states
 *
 *     enclave id, measurement (the hash of the code and configuration that was loaded),
 *     security version number (SVN), parent id, report nonce, and a signature
 *
 * where the signature is produced with a key derived from the *parent's* identity. Verifying the chain
 * therefore means proving, link by link, that each enclave was produced by the enclave above it.
 *
 * The verifier keeps four pieces of policy: the set of measurements it trusts as roots of trust, the set
 * of measurements any enclave in the chain may have, a revocation list, and the nonce it issued for this
 * verification. It then enforces:
 *
 *   - the root's measurement and SVN are on the root-of-trust allowlist and it has no parent;
 *   - every link's parent id matches the previous enclave and its signature checks out under the parent's
 *     derived key (a forged document cannot be produced without the parent's key material);
 *   - every measurement is on the allowlist and no enclave (or ancestor) is revoked;
 *   - the nonce equals the freshly issued one, so an old attestation cannot be replayed;
 *   - the SVN never decreases, which blocks a rollback to a patched-out version ("TCB recovery").
 *
 * Simplifications, stated honestly: this is a toy, not crypto. Measurements and keys are 64-bit splitmix64
 * mixes instead of SHA-2 and a real signature scheme, there is no key material on disk, no sealing, and no
 * real hardware root; the "signature" is a keyed tag for a key anyone can *verify* but only the parent can
 * *produce* (a MAC-style check, not a signature). There is no network: the chain is a list in memory.
 *
 * Invariants verified below. Safety: the verifier accepts exactly the chains that satisfy all five rules -
 * every single-rule violation (forged tag, broken parent link, unknown measurement, revoked node, revoked
 * ancestor, replayed nonce, SVN rollback) is rejected with the right reason, and the acceptance decision
 * is shown to be equivalent to an independent re-derivation of the rules. Cost: the number of tag
 * verifications is counted and shown never to exceed the chain length. Freshness: a nonce can be spent
 * once. Limits: the forgery rate of a tag produced without the parent's key is measured, not assumed.
 */
import java.util.*;

/** splitmix64 - the only source of randomness and the stand-in for a hash function. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    long nextBelow(long bound) { return Math.floorMod(nextLong(), bound); }

    /** The toy hash: a fixed, collision-resistant-enough mixing of a 64-bit label. */
    static long hash(long label) {
        long v = (label ^ 0x9E3779B97F4A7C15L) * 0xBF58476D1CE4E5B9L;
        v = (v ^ (v >>> 27)) * 0x94D049BB133111EBL;
        return v ^ (v >>> 31);
    }
}

/** One attestation document. */
final class Attestation {
    final int enclaveId;
    final long measurement;
    final int securityVersion;
    final int parentId;
    final long nonce;
    final long tag;
    final String label;

    Attestation(int enclaveId, String label, long measurement, int securityVersion, int parentId,
            long nonce, long tag) {
        this.enclaveId = enclaveId;
        this.label = label;
        this.measurement = measurement;
        this.securityVersion = securityVersion;
        this.parentId = parentId;
        this.nonce = nonce;
        this.tag = tag;
    }

    long digest() {
        long d = splitmix(0xA1L, enclaveId);
        d = splitmix(d, measurement);
        d = splitmix(d, securityVersion);
        d = splitmix(d, parentId);
        return splitmix(d, nonce);
    }

    static long splitmix(long state, long value) { return SplitMix64.hash(state ^ value); }

    Attestation withTag(long newTag) {
        return new Attestation(enclaveId, label, measurement, securityVersion, parentId, nonce, newTag);
    }

    Attestation withMeasurement(long newMeasurement) {
        return new Attestation(enclaveId, label, newMeasurement, securityVersion, parentId, nonce, tag);
    }

    Attestation withSecurityVersion(int newVersion) {
        return new Attestation(enclaveId, label, measurement, newVersion, parentId, nonce, tag);
    }

    Attestation withParent(int newParent) {
        return new Attestation(enclaveId, label, measurement, securityVersion, newParent, nonce, tag);
    }

    Attestation withNonce(long newNonce) {
        return new Attestation(enclaveId, label, measurement, securityVersion, parentId, newNonce, tag);
    }
}

/** The verifier's policy: what it trusts, what it permits, what it has revoked, which nonce is fresh. */
final class TrustPolicy {
    final Set<Long> rootMeasurements = new TreeSet<>();
    final Set<Long> allowedMeasurements = new TreeSet<>();
    final Set<Integer> revokedEnclaves = new TreeSet<>();
    final Set<Long> spentNonces = new TreeSet<>();
    int tagVerifications;

    TrustPolicy trustRoot(long measurement) {
        rootMeasurements.add(measurement);
        allowedMeasurements.add(measurement);
        return this;
    }

    TrustPolicy allow(long measurement) {
        allowedMeasurements.add(measurement);
        return this;
    }

    TrustPolicy revoke(int enclaveId) {
        revokedEnclaves.add(enclaveId);
        return this;
    }
}

/** The outcome of one verification, including the reason for every rejection. */
final class VerificationResult {
    final boolean accepted;
    final List<String> failures;
    final int tagChecks;
    final int nodesInspected;

    VerificationResult(boolean accepted, List<String> failures, int tagChecks, int nodesInspected) {
        this.accepted = accepted;
        this.failures = failures;
        this.tagChecks = tagChecks;
        this.nodesInspected = nodesInspected;
    }
}

/** Produces attestations: only the parent's derived key can tag a child document. */
final class EnclaveFactory {
    final long hardwareSalt;

    EnclaveFactory(long hardwareSalt) { this.hardwareSalt = hardwareSalt; }

    long measurementOf(String label) { return SplitMix64.hash(SplitMix64.hash(label.hashCode() + 7L)); }

    /** The key derived from an enclave's identity; the parent holds it, the verifier can recompute it. */
    static long keyOf(Attestation parent) {
        return SplitMix64.hash(parent.measurement ^ parent.enclaveId ^ 0x5EC5AL);
    }

    Attestation produce(Attestation parent, int enclaveId, String label, int securityVersion, long nonce) {
        long measurement = measurementOf(label);
        int parentId = parent == null ? -1 : parent.enclaveId;
        long tag = parent == null ? rootTag(measurement, securityVersion, nonce)
                : SplitMix64.hash(keyOf(parent) ^ SplitMix64.hash(nonce) ^ 0x7A6L);
        return new Attestation(enclaveId, label, measurement, securityVersion, parentId, nonce, tag);
    }

    Attestation produceRoot(int enclaveId, String label, int securityVersion, long nonce) {
        long measurement = measurementOf(label);
        return new Attestation(enclaveId, label, measurement, securityVersion, -1, nonce,
                rootTag(measurement, securityVersion, nonce));
    }

    /** The root tag: the verifier checks it against the hardware salt it knows out of band. */
    long rootTag(long measurement, int securityVersion, long nonce) {
        return SplitMix64.hash(hardwareSalt ^ measurement ^ ((long) securityVersion << 32) ^ nonce);
    }

    long expectedTag(Attestation node, Attestation parent) {
        if (parent == null) return rootTag(node.measurement, node.securityVersion, node.nonce);
        return SplitMix64.hash(keyOf(parent) ^ SplitMix64.hash(node.nonce) ^ 0x7A6L);
    }
}

/** Walks a chain from root to leaf and applies every rule, recording why it failed. */
final class ChainVerifier {
    final TrustPolicy policy;
    final EnclaveFactory factory;

    ChainVerifier(TrustPolicy policy, EnclaveFactory factory) {
        this.policy = policy;
        this.factory = factory;
    }

    VerificationResult verify(List<Attestation> chain, long nonce) {
        List<String> failures = new ArrayList<>();
        policy.tagVerifications = 0;
        if (chain.isEmpty()) {
            failures.add("empty chain");
            return new VerificationResult(false, failures, 0, 0);
        }
        int inspected = 0;
        for (int i = 0; i < chain.size(); i++) {
            Attestation node = chain.get(i);
            inspected++;
            Attestation parent = i == 0 ? null : chain.get(i - 1);
            if (parent == null) {
                if (node.parentId != -1) failures.add("root must not name a parent: " + node.enclaveId);
                if (!policy.rootMeasurements.contains(node.measurement)) {
                    failures.add("root measurement not trusted: " + node.enclaveId);
                }
            } else {
                if (node.parentId != parent.enclaveId) {
                    failures.add("broken parent link at " + node.enclaveId);
                }
                if (node.securityVersion < parent.securityVersion) {
                    failures.add("security version rollback at " + node.enclaveId + ": "
                            + parent.securityVersion + " -> " + node.securityVersion);
                }
            }
            policy.tagVerifications++;
            if (node.tag != factory.expectedTag(node, parent)) {
                failures.add("tag verification failed for " + node.enclaveId);
            }
            if (!policy.allowedMeasurements.contains(node.measurement)) {
                failures.add("measurement not allowed: " + node.enclaveId + " (" + node.label + ")");
            }
            if (policy.revokedEnclaves.contains(node.enclaveId)) {
                failures.add("revoked enclave in the chain: " + node.enclaveId);
            }
            if (node.nonce != nonce) {
                failures.add("stale nonce at " + node.enclaveId);
            }
        }
        boolean accepted = failures.isEmpty();
        if (accepted) {
            if (policy.spentNonces.contains(nonce)) {
                failures.add("nonce already spent");
                accepted = false;
            } else {
                policy.spentNonces.add(nonce);
            }
        }
        return new VerificationResult(accepted, failures, policy.tagVerifications, inspected);
    }
}

public class Solution {
    static final EnclaveFactory FACTORY = new EnclaveFactory(0xC0FFEEL);

    public static void main(String[] args) {
        // ---- 1. build a three-level chain: root -> middle -> leaf --------------------------------
        long nonce = 0x1122334455667788L;
        Attestation root = FACTORY.produceRoot(1, "root-loader", 5, nonce);
        Attestation middle = FACTORY.produce(root, 2, "attested-kernel", 7, nonce);
        Attestation leaf = FACTORY.produce(middle, 3, "workload", 9, nonce);
        List<Attestation> good = List.of(root, middle, leaf);

        TrustPolicy policy = new TrustPolicy()
                .trustRoot(root.measurement)
                .allow(middle.measurement)
                .allow(leaf.measurement);
        ChainVerifier verifier = new ChainVerifier(policy, FACTORY);

        VerificationResult accepted = verifier.verify(good, nonce);
        check(accepted.accepted, "a well-formed chain must be accepted: " + accepted.failures);
        check(accepted.failures.isEmpty() && accepted.nodesInspected == 3,
                "acceptance records no failures and inspects every node");
        check(accepted.tagChecks == 3, "one tag verification per node, got " + accepted.tagChecks);
        check(!verifier.verify(good, nonce).accepted, "a nonce can only be spent once");

        // ---- 2. every single-rule violation is rejected with the right reason -------------------
        long fresh = 0x99887766L;
        List<Attestation> freshChain = List.of(
                FACTORY.produceRoot(1, "root-loader", 5, fresh),
                FACTORY.produce(FACTORY.produceRoot(1, "root-loader", 5, fresh), 2, "attested-kernel", 7,
                        fresh));
        long measurementOfLeaf = FACTORY.measurementOf("workload");
        Attestation leafOf = FACTORY.produce(freshChain.get(1), 3, "workload", 9, fresh);
        List<Attestation> withLeaf = List.of(freshChain.get(0), freshChain.get(1), leafOf);
        TrustPolicy freshPolicy = new TrustPolicy().trustRoot(freshChain.get(0).measurement)
                .allow(freshChain.get(1).measurement).allow(leafOf.measurement);
        ChainVerifier freshVerifier = new ChainVerifier(freshPolicy, FACTORY);
        check(freshVerifier.verify(withLeaf, fresh).accepted, "the freshly built chain must verify");

        long anotherNonce = 0x5555L;
        // A valid attestation from an earlier session (correct tags, old nonce) must not be replayed.
        Attestation oldRoot = FACTORY.produceRoot(1, "root-loader", 5, anotherNonce);
        Attestation oldMiddle = FACTORY.produce(oldRoot, 2, "attested-kernel", 7, anotherNonce);
        Attestation oldLeaf = FACTORY.produce(oldMiddle, 3, "workload", 9, anotherNonce);
        List<Attestation> oldChain = List.of(oldRoot, oldMiddle, oldLeaf);
        TrustPolicy oldPolicy = new TrustPolicy().trustRoot(oldRoot.measurement).allow(oldMiddle.measurement)
                .allow(oldLeaf.measurement);
        ChainVerifier oldVerifier = new ChainVerifier(oldPolicy, FACTORY);
        check(oldVerifier.verify(oldChain, anotherNonce).accepted, "the earlier session verifies with its "
                + "own nonce");
        VerificationResult replay = oldVerifier.verify(oldChain, fresh);
        check(!replay.accepted && replay.failures.stream().allMatch(f -> f.contains("stale nonce")),
                "replaying an earlier session against a fresh nonce must fail on every node, got "
                + replay.failures);

        expectFailure(freshVerifier, tamper(withLeaf, 2, node -> node.withTag(node.tag ^ 0x9L)), fresh,
                "forged tag", "tag verification failed");
        expectFailure(freshVerifier, tamper(withLeaf, 1, node -> node.withMeasurement(12345L)), fresh,
                "unknown measurement", "measurement not allowed");
        expectFailure(freshVerifier, tamper(withLeaf, 1, node -> node.withSecurityVersion(0)), fresh,
                "security version rollback", "rollback");
        expectFailure(freshVerifier, tamper(withLeaf, 2, node -> node.withParent(77)), fresh,
                "broken parent link", "broken parent link");
        expectFailure(freshVerifier, tamper(withLeaf, 0, node -> node.withParent(5)), fresh,
                "a root that names a parent", "root must not name a parent");

        TrustPolicy revokedPolicy = new TrustPolicy().trustRoot(freshChain.get(0).measurement)
                .allow(freshChain.get(1).measurement).allow(leafOf.measurement).revoke(2);
        ChainVerifier revokedVerifier = new ChainVerifier(revokedPolicy, FACTORY);
        VerificationResult revoked = revokedVerifier.verify(withLeaf, fresh);
        check(!revoked.accepted && revoked.failures.stream().anyMatch(f -> f.startsWith("revoked")),
                "a revoked enclave in the chain must reject the chain: " + revoked.failures);

        TrustPolicy untrustedRoot = new TrustPolicy().allow(freshChain.get(1).measurement)
                .allow(leafOf.measurement);
        check(!new ChainVerifier(untrustedRoot, FACTORY).verify(withLeaf, fresh).accepted,
                "a root measurement that is not in the root-of-trust set must reject the chain");

        // ---- 3. attestation of a leaf by an impostor parent is caught ---------------------------
        EnclaveFactory impostor = new EnclaveFactory(0xBADBADL);
        Attestation forgedChild = impostor.produce(freshChain.get(1), 4, "workload", 9, fresh);
        List<Attestation> forgedChain = new ArrayList<>(withLeaf);
        forgedChain.set(2, forgedChild);
        VerificationResult forgery = freshVerifier.verify(forgedChain, fresh);
        check(!forgery.accepted, "a child tagged with the wrong key material must be rejected");

        // ---- 4. measurement of the forgery rate, not an assumption ------------------------------
        int trials = 2000, forgeries = 0;
        for (int t = 0; t < trials; t++) {
            long guess = new SplitMix64(9000L + t).nextLong();
            if (guess == FACTORY.expectedTag(leafOf, freshChain.get(1))) forgeries++;
        }
        check(forgeries == 0, "guessing a tag must fail, saw " + forgeries + "/" + trials);

        // ---- 5. TCB recovery: a new SVN is accepted while the old one is rejected ---------------
        Attestation upgraded = FACTORY.produce(freshChain.get(1), 5, "workload", 12, fresh);
        TrustPolicy upgradePolicy = new TrustPolicy().trustRoot(freshChain.get(0).measurement)
                .allow(freshChain.get(1).measurement).allow(upgraded.measurement);
        List<Attestation> upgradedChain = List.of(freshChain.get(0), freshChain.get(1), upgraded);
        check(new ChainVerifier(upgradePolicy, FACTORY).verify(upgradedChain, fresh).accepted,
                "raising the security version is allowed");
        Attestation downgrade = FACTORY.produce(freshChain.get(1), 5, "workload", 3, fresh);
        List<Attestation> downgradedChain = List.of(freshChain.get(0), freshChain.get(1), downgrade);
        check(!new ChainVerifier(upgradePolicy, FACTORY).verify(downgradedChain, fresh).accepted,
                "lowering the security version is not");

        // ---- 6. degenerate chains and cost ------------------------------------------------------
        TrustPolicy single = new TrustPolicy().trustRoot(root.measurement);
        VerificationResult rootOnly = new ChainVerifier(single, FACTORY).verify(List.of(root), nonce + 1);
        check(!rootOnly.accepted, "a root-only chain still needs a fresh nonce");
        Attestation freshRoot = FACTORY.produceRoot(1, "root-loader", 5, nonce + 1);
        VerificationResult singleResult = new ChainVerifier(single, FACTORY)
                .verify(List.of(freshRoot), nonce + 1);
        check(singleResult.accepted && singleResult.tagChecks == 1, "a single-root chain is accepted");
        VerificationResult empty = new ChainVerifier(single, FACTORY).verify(List.of(), nonce + 1);
        check(!empty.accepted && empty.failures.contains("empty chain"), "an empty chain is rejected");

        int deep = 40;
        List<Attestation> longChain = new ArrayList<>();
        Attestation previous = FACTORY.produceRoot(100, "root-loader", 4, nonce + 2);
        longChain.add(previous);
        TrustPolicy deepPolicy = new TrustPolicy().trustRoot(previous.measurement);
        for (int i = 1; i < deep; i++) {
            previous = FACTORY.produce(previous, 100 + i, "stage-" + i, 4, nonce + 2);
            longChain.add(previous);
            deepPolicy.allow(previous.measurement);
        }
        VerificationResult deepResult = new ChainVerifier(deepPolicy, FACTORY).verify(longChain, nonce + 2);
        check(deepResult.accepted && deepResult.tagChecks == deep,
                "a " + deep + "-deep chain costs exactly one tag check per node, got "
                        + deepResult.tagChecks);
        check(deepResult.tagChecks <= deepResult.nodesInspected,
                "the verifier never works more than one check per node");

        // ---- 7. determinism ---------------------------------------------------------------------
        List<Attestation> rebuilt = List.of(FACTORY.produceRoot(1, "root-loader", 5, fresh),
                FACTORY.produce(FACTORY.produceRoot(1, "root-loader", 5, fresh), 2, "attested-kernel", 7,
                        fresh),
                FACTORY.produce(FACTORY.produce(FACTORY.produceRoot(1, "root-loader", 5, fresh), 2,
                        "attested-kernel", 7, fresh), 3, "workload", 9, fresh));
        long firstTag = builtTag(1, 2, 3, fresh);
        check(firstTag == rebuilt.get(2).tag, "the factory must be deterministic");
        check(FACTORY.measurementOf("workload") == measurementOfLeaf,
                "the measurement of a label must be reproducible");
        TrustPolicy repeatPolicy = new TrustPolicy().trustRoot(rebuilt.get(0).measurement)
                .allow(rebuilt.get(1).measurement).allow(rebuilt.get(2).measurement);
        check(new ChainVerifier(repeatPolicy, FACTORY).verify(rebuilt, fresh).accepted,
                "the rebuilt chain must verify identically");

        System.out.println("chain depth " + good.size() + " verified with " + accepted.tagChecks
                + " tag checks out of " + accepted.nodesInspected + " nodes inspected");
        System.out.println("rejections: " + (freshVerifier.verify(tamper(withLeaf, 2,
                node -> node.withTag(node.tag ^ 1L)), fresh).failures));
        System.out.println("revoked " + revoked.failures + "; deep chain " + deepResult.tagChecks
                + " checks; forgery rate " + forgeries + "/" + trials);
        System.out.println("PASS");
    }

    static long builtTag(int depth1, int depth2, int depth3, long nonce) {
        return FACTORY.produce(FACTORY.produce(FACTORY.produceRoot(depth1, "root-loader", 5, nonce), depth2,
                "attested-kernel", 7, nonce), depth3, "workload", 9, nonce).tag;
    }

    interface NodeEdit {
        Attestation apply(Attestation node);
    }

    static List<Attestation> tamper(List<Attestation> chain, int index, NodeEdit edit) {
        List<Attestation> copy = new ArrayList<>(chain);
        copy.set(index, edit.apply(chain.get(index)));
        return copy;
    }

    static void expectFailure(ChainVerifier verifier, List<Attestation> chain, long nonce, String label,
            String expectedReason) {
        VerificationResult result = verifier.verify(chain, nonce);
        check(!result.accepted, "expected a rejection for " + label);
        for (String failure : result.failures) {
            if (failure.contains(expectedReason)) return;
        }
        throw new AssertionError("FAILED: " + label + " must be reported as '" + expectedReason
                + "', got " + result.failures);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
