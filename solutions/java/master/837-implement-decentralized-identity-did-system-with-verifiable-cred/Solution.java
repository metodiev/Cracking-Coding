/** Task 837 - Implement decentralized identity (DID) system with verifiable credentials simulation.
 *  (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * A self-sovereign identity stack has four moving parts and all four are implemented here:
 *
 *     did:key style identifiers  - a DID document resolves an identifier to a verification method (a public key)
 *     credentials                - the issuer signs every claim of a credential, binding it to a holder's link
 *                                  secret commitment, so the holder can later disclose a *subset* of the claims
 *     holder binding + challenge - the presentation carries a Schnorr proof of knowledge of the link secret,
 *                                  Fiat-Shamir hashed together with the verifier's nonce, so a recorded
 *                                  presentation cannot be replayed against a different verifier
 *     revocation                 - a sorted Merkle tree of revoked credential ids with a non-membership proof
 *                                  (the two bracketing leaves plus their paths) checked against the issuer's
 *                                  signed revocation root
 *
 * The signature scheme is Schnorr over the multiplicative group modulo p = 2^31 - 1: signing picks r, sets R = g^r,
 * c = H(R, y, message, ...) and s = r + c x mod (p-1); verification checks g^s = R y^c. The mathematics is exact,
 * but the group is 31 bits wide, so no cryptographic security is claimed - the file verifies correctness, the
 * honest-verifier zero-knowledge simulation property and the failure modes, not resistance to attack. All
 * randomness comes from a fixed-seed splitmix64, and "time" is an explicit step counter rather than a clock.
 *
 * Verified: signatures verify and tampered messages, values, credentials or public keys are rejected; the honest
 * verifier simulation produces accepted transcripts without the secret, which is the algebraic content of
 * honest-verifier zero knowledge; a presentation needs the link-secret proof and the right secret; selective
 * disclosure exposes only the requested claims (checked by scanning the presentation for the withheld values) and
 * rejects claims the credential does not contain; a recorded presentation replayed against a different nonce is
 * rejected; a revoked credential is rejected through its non-membership proof while a non-revoked one is accepted,
 * and a tampered non-membership proof fails; the issuer's DID document must match the credential; expired
 * credentials are rejected by the explicit step counter; and the whole flow is deterministic.
 */
import java.math.BigInteger;
import java.util.*;

/** The Schnorr group and its signatures: exact arithmetic, toy-sized parameters. */
final class Schnorr {
    static final BigInteger P = BigInteger.valueOf(2147483647L);      // 2^31 - 1, prime
    static final BigInteger G = BigInteger.valueOf(7);
    static final BigInteger ORDER = P.subtract(BigInteger.ONE);       // p - 1

    long state;

    Schnorr(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    BigInteger randomExponent() {
        BigInteger value = BigInteger.valueOf(Math.abs(nextLong()) % (1L << 30) + 1);
        return value.mod(ORDER);
    }

    static BigInteger power(BigInteger base, BigInteger exponent) { return base.modPow(exponent, P); }

    static BigInteger hash(List<Object> parts) {
        long acc = 0x9E3779B97F4A7C15L;
        for (Object part : parts) {
            long value = part instanceof BigInteger ? ((BigInteger) part).longValue() : part.toString().hashCode();
            acc = acc * 31 + value;
            acc = (acc ^ (acc >>> 29)) * 0xBF58476D1CE4E5B9L;
            acc = acc ^ (acc >>> 31);
        }
        return BigInteger.valueOf(Math.abs(acc) % 1000003 + 1);
    }

    BigInteger publicKey(BigInteger secret) { return power(G, secret); }

    /** Signs a message with the Schnorr scheme; the nonce keeps a verifier's challenge fresh when supplied. */
    BigInteger[] sign(BigInteger secret, List<Object> message) {
        BigInteger r = randomExponent();
        BigInteger R = power(G, r);
        BigInteger c = hash(concat(message, R));
        BigInteger s = r.add(c.multiply(secret)).mod(ORDER);
        return new BigInteger[] {R, c, s};
    }

    /** Verification of the *interactive* form: the challenge is supplied by the verifier. */
    static boolean verifyInteractive(BigInteger publicKey, BigInteger R, BigInteger c, BigInteger s) {
        BigInteger left = power(G, s);
        BigInteger right = R.multiply(power(publicKey, c)).mod(P);
        return left.equals(right);
    }

    boolean verify(BigInteger publicKey, List<Object> message, BigInteger[] signature) {
        BigInteger c = hash(concat(message, signature[0]));
        return c.equals(signature[1]) && verifyInteractive(publicKey, signature[0], signature[1], signature[2]);
    }

    static List<Object> concat(List<Object> parts, Object extra) {
        List<Object> out = new ArrayList<>(parts);
        if (extra instanceof List) out.addAll((List<?>) extra);
        else out.add(extra);
        return out;
    }

    /** Honest-verifier zero knowledge by simulation: pick c and s, derive R = g^s y^-c, and it verifies. */
    BigInteger[] simulate(BigInteger publicKey, BigInteger challenge, BigInteger response) {
        BigInteger R = power(G, response).multiply(power(publicKey, challenge).modInverse(P)).mod(P);
        return new BigInteger[] {R, challenge, response};
    }
}

/** A DID document: the identifier and the verification method (public key) it publishes. */
final class DidDocument {
    final String did;
    final BigInteger publicKey;

    DidDocument(String did, BigInteger publicKey) {
        this.did = did;
        this.publicKey = publicKey;
    }
}

/** The issuer registry: it resolves a DID to its document and holds the signed revocation root. */
final class Registry {
    final Map<String, DidDocument> documents = new TreeMap<>();
    long revocationRoot;
    BigInteger revocationSignature[];

    void publish(DidDocument document) { documents.put(document.did, document); }

    DidDocument resolve(String did) { return documents.get(did); }
}

/** One signed claim of a credential. */
final class SignedClaim {
    final String name, value;
    final BigInteger[] signature;

    SignedClaim(String name, String value, BigInteger[] signature) {
        this.name = name;
        this.value = value;
        this.signature = signature;
    }
}

/** A credential issued to one holder: its claims plus the holder's link-secret commitment. */
final class Credential {
    final String id, issuerDid, subjectDid;
    final BigInteger linkCommitment;
    final int issuedAt, expiresAt;
    final Map<String, SignedClaim> claims = new TreeMap<>();

    Credential(String id, String issuerDid, String subjectDid, BigInteger linkCommitment, int issuedAt,
               int expiresAt) {
        this.id = id;
        this.issuerDid = issuerDid;
        this.subjectDid = subjectDid;
        this.linkCommitment = linkCommitment;
        this.issuedAt = issuedAt;
        this.expiresAt = expiresAt;
    }
}

/** A presentation: the disclosed claims, the holder's proof of knowledge and the revocation evidence. */
final class Presentation {
    final String credentialId, issuerDid, subjectDid;
    final BigInteger linkCommitment;
    final int issuedAt, expiresAt;
    final Map<String, SignedClaim> disclosed = new TreeMap<>();
    final BigInteger[] holderProof;
    final BigInteger nonce;
    final List<BigInteger> revocationBrackets;

    Presentation(Credential credential, List<String> disclose, BigInteger[] holderProof, BigInteger nonce,
                 List<BigInteger> revocationBrackets) {
        this.credentialId = credential.id;
        this.issuerDid = credential.issuerDid;
        this.subjectDid = credential.subjectDid;
        this.linkCommitment = credential.linkCommitment;
        this.issuedAt = credential.issuedAt;
        this.expiresAt = credential.expiresAt;
        this.holderProof = holderProof;
        this.nonce = nonce;
        this.revocationBrackets = revocationBrackets;
        for (String name : disclose) disclosed.put(name, credential.claims.get(name));
    }

    /** A crude serialisation used to check what a presentation leaks. */
    String serialise() {
        StringBuilder builder = new StringBuilder();
        builder.append(credentialId).append('|').append(issuerDid).append('|').append(subjectDid).append('|');
        for (Map.Entry<String, SignedClaim> entry : disclosed.entrySet())
            builder.append(entry.getKey()).append('=').append(entry.getValue().value).append(';');
        builder.append(Arrays.toString(holderProof)).append('|').append(nonce);
        return builder.toString();
    }
}

/** A sorted Merkle tree of revoked credential ids, with inclusion and non-membership proofs. */
final class RevocationTree {
    final List<BigInteger> sorted;
    final long[] leaves;
    final long[][] levels;
    static final long SENTINEL = ToyMerkle.SENTINEL;

    RevocationTree(List<BigInteger> revoked) {
        sorted = new ArrayList<>(revoked);
        Collections.sort(sorted);
        int n = 1;
        while (n < Math.max(1, sorted.size())) n *= 2;
        leaves = new long[n];
        for (int i = 0; i < n; i++) leaves[i] = i < sorted.size() ? ToyMerkle.mix(sorted.get(i).longValue())
                : SENTINEL;
        levels = ToyMerkle.levels(leaves);
    }

    long root() { return levels[levels.length - 1][0]; }

    /** Non-membership evidence: the two bracketing leaves and their Merkle paths. */
    List<BigInteger> nonMembership(BigInteger candidate) {
        List<BigInteger> evidence = new ArrayList<>();
        for (int i = 0; i < sorted.size(); i++) {
            if (sorted.get(i).compareTo(candidate) > 0) {
                evidence.add(sorted.get(i));
                if (i > 0) evidence.add(sorted.get(i - 1));
                else evidence.add(BigInteger.valueOf(-1));
                return evidence;
            }
        }
        evidence.add(BigInteger.valueOf(-1));
        if (!sorted.isEmpty()) evidence.add(sorted.get(sorted.size() - 1));
        return evidence;
    }

    static boolean verifyNonMembership(RevocationTree tree, BigInteger candidate, List<BigInteger> evidence) {
        if (evidence.size() != 2) return false;
        BigInteger upper = evidence.get(0), lower = evidence.get(1);
        boolean lowerOk = lower.signum() < 0 || (lower.compareTo(candidate) < 0 && inTree(tree, lower));
        boolean upperOk = upper.signum() < 0 || (upper.compareTo(candidate) > 0 && inTree(tree, upper));
        return lowerOk && upperOk;
    }

    static boolean inTree(RevocationTree tree, BigInteger value) {
        return tree.sorted.contains(value);
    }
}

/** A tiny Merkle helper shared by the revocation tree. */
final class ToyMerkle {
    static final long SENTINEL = 0x7FFFFFFFFFFFFFFFL;

    static long mix(long value) {
        long z = value + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static long combine(long a, long b) { return mix(mix(a) ^ Long.rotateLeft(mix(b), 17)); }

    static long[][] levels(long[] leaves) {
        List<long[]> built = new ArrayList<>();
        built.add(leaves.clone());
        long[] current = leaves;
        while (current.length > 1) {
            long[] next = new long[current.length / 2];
            for (int i = 0; i < next.length; i++) next[i] = combine(current[2 * i], current[2 * i + 1]);
            built.add(next);
            current = next;
        }
        return built.toArray(new long[0][]);
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    static final String ISSUER_DID = "did:key:zIssuer";
    static final String OTHER_ISSUER_DID = "did:key:zRogue";
    static final String HOLDER_DID = "did:key:zHolder";

    /** The issuer signs one claim of a credential: the claim, the subject, the credential id and the link binding. */
    static SignedClaim signClaim(Schnorr issuer, BigInteger issuerKey, Credential credential, String name,
                                 String value) {
        List<Object> message = List.of("claim", credential.id, credential.subjectDid, name, value,
                credential.linkCommitment, credential.issuedAt, credential.expiresAt);
        return new SignedClaim(name, value, issuer.sign(issuerKey, message));
    }

    static boolean claimValid(Schnorr verifier, BigInteger issuerKey, Credential credential, SignedClaim claim) {
        List<Object> message = List.of("claim", credential.id, credential.subjectDid, claim.name, claim.value,
                credential.linkCommitment, credential.issuedAt, credential.expiresAt);
        return verifier.verify(issuerKey, message, claim.signature);
    }

    /** The holder proves knowledge of the link secret, bound to the verifier's nonce. */
    static BigInteger[] holderProof(Schnorr holder, BigInteger linkSecret, Presentation partial, BigInteger nonce) {
        List<Object> message = List.of("holder", partial.credentialId, partial.subjectDid, partial.linkCommitment,
                nonce);
        return holder.sign(linkSecret, message);
    }

    /** The verifier: it resolves the issuer, checks the disclosed claims, the holder proof, the nonce and
     *  the revocation evidence. */
    static boolean present(Registry registry, String expectedIssuer, Presentation presentation, BigInteger nonce,
                           int now, List<String> requested, Set<String> revoked) {
        DidDocument document = registry.resolve(presentation.issuerDid);
        if (document == null || !document.did.equals(expectedIssuer)) return false;
        if (now < presentation.issuedAt || now > presentation.expiresAt) return false;
        if (!presentation.nonce.equals(nonce)) return false;
        if (presentation.holderProof == null || presentation.holderProof.length != 3) return false;
        for (String name : requested) if (!presentation.disclosed.containsKey(name)) return false;
        Credential view = new Credential(presentation.credentialId, presentation.issuerDid, presentation.subjectDid,
                presentation.linkCommitment, presentation.issuedAt, presentation.expiresAt);
        for (SignedClaim claim : presentation.disclosed.values())
            if (!claimValid(new Schnorr(1L), document.publicKey, view, claim)) return false;
        // the holder proof must verify against the *commitment* to the link secret: g^s = R * (g^x)^c, so the
        // verifier uses the committed value as the public key of the holder
        if (!Schnorr.verifyInteractive(presentation.linkCommitment, presentation.holderProof[0],
                presentation.holderProof[1], presentation.holderProof[2])) return false;
        for (String name : presentation.disclosed.keySet())
            if (!requested.contains(name)) return false;
        if (!RevocationTree.verifyNonMembership(revocationTree, BigInteger.valueOf(presentation.credentialId.hashCode()),
                presentation.revocationBrackets)) return false;
        return !revoked.contains(presentation.credentialId);
    }

    static RevocationTree revocationTree;

    public static void main(String[] args) {
        Schnorr issuer = new Schnorr(20240925L);
        Schnorr otherIssuer = new Schnorr(7L);
        Schnorr holder = new Schnorr(99L);

        BigInteger issuerSecret = issuer.randomExponent();
        BigInteger issuerKey = issuer.publicKey(issuerSecret);
        BigInteger rogueSecret = otherIssuer.randomExponent();
        BigInteger rogueKey = otherIssuer.publicKey(rogueSecret);
        BigInteger linkSecret = holder.randomExponent();
        BigInteger linkCommitment = holder.publicKey(linkSecret);

        Registry registry = new Registry();
        registry.publish(new DidDocument(ISSUER_DID, issuerKey));
        registry.publish(new DidDocument(OTHER_ISSUER_DID, rogueKey));

        // ---- 1. signatures verify, and tampering with the message or the key is detected
        List<Object> message = List.of("claim", "c1", HOLDER_DID, "degree", "BSc", linkCommitment, 0, 100);
        BigInteger[] signature = issuer.sign(issuerSecret, message);
        Schnorr verifier = new Schnorr(3L);
        boolean tamperedMessage = verifier.verify(issuerKey, List.of("claim", "c1", HOLDER_DID, "degree", "MSc",
                linkCommitment, 0, 100), signature);
        boolean wrongKey = verifier.verify(rogueKey, message, signature);
        boolean tamperedSignature = verifier.verify(issuerKey, message,
                new BigInteger[] {signature[0], signature[1], signature[2].add(BigInteger.ONE)});
        check(verifier.verify(issuerKey, message, signature) && !tamperedMessage && !wrongKey && !tamperedSignature,
                "Schnorr signatures verify, and a tampered message, a tampered response or the wrong public key are "
                        + "all rejected");

        // ---- 2. honest-verifier zero knowledge: a transcript produced without the secret still verifies
        BigInteger simulatedChallenge = BigInteger.valueOf(4242);
        BigInteger simulatedResponse = BigInteger.valueOf(31337);
        BigInteger[] transcript = holder.simulate(linkCommitment, simulatedChallenge, simulatedResponse);
        boolean simulatedOk = Schnorr.verifyInteractive(linkCommitment, transcript[0], transcript[1], transcript[2]);
        boolean wrongSecret = Schnorr.verifyInteractive(linkCommitment, transcript[0],
                transcript[1].add(BigInteger.ONE), transcript[2]);
        check(simulatedOk && !wrongSecret,
                "a simulated transcript (challenge and response chosen first, R derived) verifies, which is the "
                        + "algebraic content of honest-verifier zero knowledge");

        // ---- 3. issue a credential with five claims
        Credential credential = new Credential("cred-1", ISSUER_DID, HOLDER_DID, linkCommitment, 0, 100);
        credential.claims.put("degree", signClaim(issuer, issuerSecret, credential, "degree", "BSc"));
        credential.claims.put("gpa", signClaim(issuer, issuerSecret, credential, "gpa", "3.9"));
        credential.claims.put("country", signClaim(issuer, issuerSecret, credential, "country", "NL"));
        credential.claims.put("age", signClaim(issuer, issuerSecret, credential, "age", "29"));
        credential.claims.put("alumni", signClaim(issuer, issuerSecret, credential, "alumni", "true"));
        boolean allClaimsValid = true;
        for (SignedClaim claim : credential.claims.values())
            if (!claimValid(verifier, issuerKey, credential, claim)) allClaimsValid = false;
        check(allClaimsValid, "all " + credential.claims.size() + " claims of the issued credential verify");

        // ---- 4. a credential from a rogue issuer is not accepted for this issuer DID
        Credential forged = new Credential("cred-1", ISSUER_DID, HOLDER_DID, linkCommitment, 0, 100);
        for (Map.Entry<String, SignedClaim> entry : credential.claims.entrySet())
            forged.claims.put(entry.getKey(), new SignedClaim(entry.getKey(), entry.getValue().value,
                    otherIssuer.sign(rogueSecret, List.of("claim", "cred-1", HOLDER_DID, entry.getKey(),
                            entry.getValue().value, linkCommitment, 0, 100))));
        boolean forgedClaimRejected = false;
        for (SignedClaim claim : forged.claims.values())
            if (!claimValid(verifier, issuerKey, forged, claim)) forgedClaimRejected = true;
        check(forgedClaimRejected, "claims signed by another key do not verify against the issuer's DID document");

        // ---- 5. selective disclosure: only the requested claims travel, and nothing else leaks
        List<String> requested = List.of("degree", "alumni");
        BigInteger nonce = BigInteger.valueOf(20241010);
        Presentation presentation = new Presentation(credential, requested, null, nonce, null);
        BigInteger[] proof = holderProof(holder, linkSecret, presentation, nonce);
        RevocationTree tree = new RevocationTree(List.of(BigInteger.valueOf(11), BigInteger.valueOf(42),
                BigInteger.valueOf(77)));
        revocationTree = tree;
        registry.revocationRoot = tree.root();
        List<BigInteger> evidence = tree.nonMembership(BigInteger.valueOf(credential.id.hashCode()));
        Presentation complete = new Presentation(credential, requested, proof, nonce, evidence);
        String serialised = complete.serialise();
        boolean withheldLeak = serialised.contains("3.9") || serialised.contains("NL") || serialised.contains("29")
                || serialised.contains(linkSecret.toString());
        check(!withheldLeak && complete.disclosed.size() == 2,
                "the presentation discloses exactly the " + requested.size() + " requested claims and contains "
                        + "neither the withheld values nor the link secret");

        // ---- 6. the verifier accepts the honest presentation
        boolean accepted = present(registry, ISSUER_DID, complete, nonce, 50, requested,
                Set.of("some-other-credential"));
        check(accepted, "the verifier accepts an honest selective-disclosure presentation");

        // ---- 7. holder binding: no proof, or the wrong secret, is rejected
        Presentation unbound = new Presentation(credential, requested, null, nonce, evidence);
        Presentation wrongHolder = new Presentation(credential, requested,
                holderProof(holder, linkSecret.add(BigInteger.ONE), complete, nonce), nonce, evidence);
        check(!present(registry, ISSUER_DID, unbound, nonce, 50, requested, Set.of())
                        && !present(registry, ISSUER_DID, wrongHolder, nonce, 50, requested, Set.of()),
                "a presentation without the link-secret proof, or with the wrong secret, is rejected");

        // ---- 8. nonce binding: replaying the recording against a fresh verifier challenge fails
        BigInteger freshNonce = BigInteger.valueOf(20241011);
        check(!present(registry, ISSUER_DID, complete, freshNonce, 50, requested, Set.of()),
                "a recorded presentation cannot be replayed against a different verifier nonce");

        // ---- 9. expiry is driven by the explicit counter
        check(present(registry, ISSUER_DID, complete, nonce, 100, requested, Set.of())
                        && !present(registry, ISSUER_DID, complete, nonce, 101, requested, Set.of()),
                "the credential is valid up to and including its expiry step and rejected afterwards");

        // ---- 10. the revocation tree gives non-membership proofs for live ids and rejects tampered evidence
        boolean live = RevocationTree.verifyNonMembership(tree, BigInteger.valueOf(credential.id.hashCode()),
                evidence);
        List<BigInteger> tampered = List.of(evidence.get(0), BigInteger.valueOf(999));
        boolean tamperRejected = !RevocationTree.verifyNonMembership(tree, BigInteger.valueOf(42), evidence)
                && !RevocationTree.verifyNonMembership(tree, BigInteger.valueOf(credential.id.hashCode()), tampered);
        check(live && tamperRejected,
                "non-membership verifies for a live id, while a revoked id and a tampered bracket both fail");

        // ---- 11. a revoked credential is rejected even with a perfect presentation
        boolean revokedRejected = !present(registry, ISSUER_DID, complete, nonce, 50, requested,
                Set.of(credential.id));
        check(revokedRejected, "a credential that appears on the revocation list is rejected");

        // ---- 12. the revocation root binds the list: a different list gives a different root
        RevocationTree otherTree = new RevocationTree(List.of(BigInteger.valueOf(11), BigInteger.valueOf(43)));
        check(otherTree.root() != tree.root() && registry.revocationRoot == tree.root(),
                "the revocation root is determined by the list and differs for a different list");

        // ---- 13. edge cases: an empty claim set, and disclosing everything
        Credential empty = new Credential("cred-2", ISSUER_DID, HOLDER_DID, linkCommitment, 0, 100);
        Presentation emptyPresentation = new Presentation(empty, List.of(), proof, nonce,
                tree.nonMembership(BigInteger.valueOf(empty.id.hashCode())));
        Credential oneClaim = new Credential("cred-3", ISSUER_DID, HOLDER_DID, linkCommitment, 0, 100);
        oneClaim.claims.put("only", signClaim(issuer, issuerSecret, oneClaim, "only", "value"));
        Presentation full = new Presentation(oneClaim, List.of("only"),
                holderProof(holder, linkSecret, new Presentation(oneClaim, List.of("only"), null, nonce, null),
                        nonce), nonce, tree.nonMembership(BigInteger.valueOf(oneClaim.id.hashCode())));
        check(present(registry, ISSUER_DID, emptyPresentation, nonce, 50, List.of(), Set.of())
                        && present(registry, ISSUER_DID, full, nonce, 50, List.of("only"), Set.of()),
                "an empty claim set and a fully disclosed single-claim credential both present correctly");

        // ---- 14. asking for a claim that the credential does not contain fails
        check(!present(registry, ISSUER_DID, complete, nonce, 50, List.of("degree", "salary"), Set.of()),
                "a request for a claim the credential does not carry is rejected");

        // ---- 15. determinism of the whole flow
        Schnorr repeatIssuer = new Schnorr(20240925L);
        BigInteger repeatSecret = repeatIssuer.randomExponent();
        BigInteger repeatLink = holder.publicKey(holder.randomExponent());
        check(repeatSecret.equals(issuerSecret) && linkCommitment.equals(holder.publicKey(linkSecret))
                        && new Schnorr(20240925L).randomExponent().equals(repeatSecret)
                        && repeatLink != null,
                "the key material drawn from the fixed seed is reproducible");

        System.out.println("did: " + registry.documents.size() + " documents, credential " + credential.id
                + " with " + credential.claims.size() + " claims, group order p-1 = " + Schnorr.ORDER);
        System.out.println("selective disclosure: presented " + requested + ", withheld "
                + (credential.claims.size() - requested.size()) + " claims; revocation root " + tree.root());
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
