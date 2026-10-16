/** Task 712 - Implement privacy-preserving set intersection via hashing sketch.
 * (Tier: Expert, CTCI Beyond -- Security & Privacy)
 *
 * The protocol is the classical Diffie-Hellman style PSI: both sides map their elements into the
 * multiplicative group modulo a prime p (here the Mersenne prime 2^31 - 1, so every product fits in a
 * long and no BigInteger is needed), then blind them with their own secret exponent,
 *
 *     client -> server :  x^a  for every client element x        server -> client :  y^b  for every server element y
 *
 * after which each side raises what it received to its own secret. Both sides now hold the *same*
 * doubly blinded values z = hash(element)^(a b) for exactly the elements of the intersection, because
 * exponentiation commutes; matching the two sets reveals the intersection without either side learning
 * the other's non-intersecting elements - in the honest-but-curious model this is (up to the size of
 * the domain, which is why the demo modulus is small and no security is claimed) exactly PSI.
 *
 * The "hashing sketch" half of the task is the communication saving: instead of sending all of its
 * doubly blinded values the client sends only the k numerically smallest ones. Any value that matches
 * is still a genuine intersection element, so the sketch can never produce a false positive; it can
 * only miss elements, and its recall grows monotonically with k, which is what the checks measure.
 *
 * Verified: the exact protocol reproduces the brute-force intersection on random, disjoint, nested,
 * identical, empty and singleton inputs; every matched value maps back to a real element; blinded
 * values are never the raw hashes and change with the secret; the sketch is contained in the true
 * intersection, its size is monotone in k, and it equals the exact answer when k covers the set;
 * duplicates, empty elements and out-of-range secrets are rejected.
 */
import java.util.*;

class PsiProtocol {
    static final long MODULUS = 2147483647L;          // 2^31 - 1, prime

    static final class Transcript {
        final List<String> intersection;
        final int clientToServerValues;
        final int serverToClientValues;

        Transcript(List<String> intersection, int clientToServerValues, int serverToClientValues) {
            this.intersection = Collections.unmodifiableList(intersection);
            this.clientToServerValues = clientToServerValues;
            this.serverToClientValues = serverToClientValues;
        }

        int totalValues() {
            return clientToServerValues + serverToClientValues;
        }
    }

    private static void validateSecret(long secret) {
        if (secret < 2 || secret > MODULUS - 2) {
            throw new IllegalArgumentException("secret exponent out of range: " + secret);
        }
    }

    private static void validateSet(List<String> elements, String label) {
        Set<String> seen = new HashSet<>();
        for (String element : elements) {
            if (element == null || element.isEmpty()) {
                throw new IllegalArgumentException(label + " contains an empty element");
            }
            if (!seen.add(element)) {
                throw new IllegalArgumentException(label + " contains the duplicate " + element);
            }
        }
    }

    /** FNV-1a plus a splitmix finaliser, mapped into [1, p - 1]. */
    static long hashToField(String element) {
        long hash = 0xCBF29CE484222325L;
        for (int i = 0; i < element.length(); i++) {
            hash ^= element.charAt(i);
            hash *= 0x100000001B3L;
        }
        hash += 0x9E3779B97F4A7C15L;
        hash = (hash ^ (hash >>> 30)) * 0xBF58476D1CE4E5B9L;
        hash = (hash ^ (hash >>> 27)) * 0x94D049BB133111EBL;
        hash = hash ^ (hash >>> 31);
        return 1 + Math.floorMod(hash, MODULUS - 1);
    }

    /** Both factors stay below 2^31, so the product stays below 2^62 and the reduction is exact. */
    static long mulmod(long a, long b) {
        return (a * b) % MODULUS;
    }

    static long modPow(long base, long exponent) {
        long result = 1;
        long current = base % MODULUS;
        long power = exponent;
        while (power > 0) {
            if ((power & 1) != 0) result = mulmod(result, current);
            current = mulmod(current, current);
            power >>= 1;
        }
        return result;
    }

    /**
     * Honest-but-curious two-party PSI. `sketchKeep` caps how many doubly blinded values the client
     * transmits; pass the client set size for the exact protocol.
     */
    static Transcript run(List<String> clientSet, List<String> serverSet,
                          long clientSecret, long serverSecret, int sketchKeep) {
        validateSecret(clientSecret);
        validateSecret(serverSecret);
        validateSet(clientSet, "client set");
        validateSet(serverSet, "server set");
        if (sketchKeep < 1 || sketchKeep > Math.max(1, clientSet.size())) {
            throw new IllegalArgumentException("sketchKeep must be in [1, client set size]");
        }

        // round one: the client blinds its own elements
        long[] clientBlinded = new long[clientSet.size()];
        for (int i = 0; i < clientBlinded.length; i++) {
            clientBlinded[i] = modPow(hashToField(clientSet.get(i)), clientSecret);
        }
        // round two: the server blinds its own elements and exponentiates the client's values
        Set<Long> serverDoubly = new HashSet<>();
        for (String element : serverSet) {
            long blinded = modPow(hashToField(element), serverSecret);
            serverDoubly.add(modPow(blinded, clientSecret));
        }
        Map<Long, String> clientDoubly = new HashMap<>();
        for (int i = 0; i < clientBlinded.length; i++) {
            clientDoubly.put(modPow(clientBlinded[i], serverSecret), clientSet.get(i));
        }

        // round three: the client transmits (at most) the sketchKeep smallest doubly blinded values,
        // the server replies with those that also occur in its own doubly blinded set
        List<Long> transmitted = new ArrayList<>(clientDoubly.keySet());
        transmitted.sort(Comparator.naturalOrder());
        int send = Math.min(sketchKeep, transmitted.size());
        List<Long> sketch = transmitted.subList(0, send);
        List<String> intersection = new ArrayList<>();
        int replies = 0;
        for (long value : sketch) {
            if (serverDoubly.contains(value)) {
                intersection.add(clientDoubly.get(value));
                replies++;
            }
        }
        Collections.sort(intersection);
        // message accounting: round one sends every blinded client element, round two every blinded
        // server element, round three only the sketch and the matching replies
        return new Transcript(intersection, clientSet.size() + send, serverSet.size() + replies);
    }

    static List<String> bruteForceIntersection(List<String> clientSet, List<String> serverSet) {
        Set<String> server = new HashSet<>(serverSet);
        List<String> intersection = new ArrayList<>();
        for (String element : clientSet) {
            if (server.contains(element)) intersection.add(element);
        }
        Collections.sort(intersection);
        return intersection;
    }
}

public class Solution {
    private static long state = 0x7127AB1EL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), bound);
    }

    static List<String> randomSet(int size, int range, int offset) {
        Set<String> elements = new TreeSet<>();
        for (int i = 0; i < size; i++) {
            elements.add("item-" + (offset + nextInt(range)));
        }
        return new ArrayList<>(elements);
    }

    public static void main(String[] args) {
        // --- the field hashing --------------------------------------------------
        long first = PsiProtocol.hashToField("alpha");
        check(first >= 1 && first < PsiProtocol.MODULUS, "hash must land in the multiplicative group");
        check(first == PsiProtocol.hashToField("alpha"), "hashing must be deterministic");
        check(PsiProtocol.hashToField("beta") != first, "different elements must hash apart");
        Set<Long> hashes = new HashSet<>();
        for (int i = 0; i < 500; i++) hashes.add(PsiProtocol.hashToField("key-" + i));
        check(hashes.size() == 500, "no collisions among five hundred keys");
        check(PsiProtocol.modPow(2, 0) == 1, "x^0 is one");
        check(PsiProtocol.mulmod(PsiProtocol.MODULUS - 1, PsiProtocol.MODULUS - 1) == 1,
                "the largest residue is its own inverse");
        check(PsiProtocol.modPow(2, PsiProtocol.MODULUS - 1) == 1,
                "Fermat: 2^(p-1) is one for the Mersenne prime");

        // --- the exact protocol against brute force -----------------------------
        int matched = 0;
        for (int trial = 0; trial < 12; trial++) {
            List<String> clientSet = randomSet(4 + nextInt(9), 24, 0);
            List<String> serverSet = randomSet(4 + nextInt(9), 24, nextInt(3) * 6);
            long clientSecret = 3 + nextInt(100000);
            long serverSecret = 3 + nextInt(100000);
            PsiProtocol.Transcript transcript = PsiProtocol.run(clientSet, serverSet, clientSecret,
                    serverSecret, clientSet.size());
            List<String> expected = PsiProtocol.bruteForceIntersection(clientSet, serverSet);
            check(transcript.intersection.equals(expected),
                    "trial " + trial + ": " + transcript.intersection + " vs " + expected);
            check(transcript.clientToServerValues == 2 * clientSet.size(),
                    "the exact protocol transmits the whole client side twice");
            check(transcript.serverToClientValues == serverSet.size() + expected.size(),
                    "the server sends its blinded set plus one reply per match");
            matched += expected.size();
        }
        check(matched > 5, "the random instances really did intersect: " + matched);

        // --- structured instances -----------------------------------------------
        List<String> base = List.of("a", "b", "c", "d", "e");
        check(PsiProtocol.run(base, base, 7, 11, base.size()).intersection.equals(base),
                "identical sets intersect completely");
        check(PsiProtocol.run(base, List.of("c", "d"), 7, 11, base.size()).intersection
                        .equals(List.of("c", "d")),
                "a nested set intersects to the smaller one");
        check(PsiProtocol.run(base, List.of("x", "y"), 7, 11, base.size()).intersection.isEmpty(),
                "disjoint sets intersect to nothing");
        check(PsiProtocol.run(List.of("solo"), List.of("solo"), 13, 17, 1).intersection
                        .equals(List.of("solo")),
                "singletons intersect");
        PsiProtocol.Transcript emptyRun = PsiProtocol.run(new ArrayList<>(), base, 7, 11, 1);
        check(emptyRun.intersection.isEmpty() && emptyRun.totalValues() == base.size(),
                "an empty client set sends nothing");
        check(PsiProtocol.bruteForceIntersection(new ArrayList<>(), base).isEmpty(),
                "brute force agrees about the empty set");

        // --- blinding hides the raw hashes --------------------------------------
        long rawHash = PsiProtocol.hashToField("secret-element");
        long blinded = PsiProtocol.modPow(rawHash, 999983);
        check(blinded != rawHash, "the transmitted value must not be the raw hash");
        long otherSecret = PsiProtocol.modPow(rawHash, 999979);
        check(blinded != otherSecret, "different secrets must give different blindings");
        Set<Long> blindedSet = new HashSet<>();
        for (String value : base) blindedSet.add(PsiProtocol.modPow(PsiProtocol.hashToField(value), 4242));
        check(blindedSet.size() == base.size(), "blinding must stay injective on the set");
        check(PsiProtocol.modPow(PsiProtocol.modPow(rawHash, 999983), 999979)
                        == PsiProtocol.modPow(PsiProtocol.modPow(rawHash, 999979), 999983),
                "exponentiation commutes, which is what makes the match work");

        // --- the hashing sketch: contained, monotone, and exact when complete ---
        List<String> clientSet = randomSet(20, 30, 0);
        List<String> serverSet = randomSet(20, 30, 0);
        int clientSize = clientSet.size();
        check(clientSize >= 12, "the sketch experiment needs a large client set: " + clientSize);
        List<String> truth = PsiProtocol.bruteForceIntersection(clientSet, serverSet);
        check(truth.size() >= 6, "the sketch experiment needs a non-trivial intersection: " + truth);
        int previousSize = -1;
        for (int keep : new int[]{1, 4, 10, clientSize}) {
            PsiProtocol.Transcript sketch = PsiProtocol.run(clientSet, serverSet, 12345, 54321, keep);
            check(truth.containsAll(sketch.intersection),
                    "a sketch can never invent an element: " + sketch.intersection);
            check(sketch.intersection.size() >= previousSize,
                    "recall must be monotone in the sketch size at k = " + keep);
            previousSize = sketch.intersection.size();
            check(sketch.clientToServerValues == clientSize + keep,
                    "the sketch transmits the client set plus exactly k values");
        }
        PsiProtocol.Transcript full = PsiProtocol.run(clientSet, serverSet, 12345, 54321, clientSize);
        check(full.intersection.equals(truth), "a complete sketch equals the exact intersection");
        PsiProtocol.Transcript tiny = PsiProtocol.run(clientSet, serverSet, 12345, 54321, 1);
        check(tiny.intersection.size() <= 1, "a one-value sketch cannot report more than one match");
        check(tiny.totalValues() < full.totalValues(),
                "the sketch must save communication: " + tiny.totalValues() + " vs " + full.totalValues());

        // --- rejected input ------------------------------------------------------
        check(rejects(() -> PsiProtocol.run(List.of("a", "a"), base, 7, 11, 1)),
                "duplicate client elements rejected");
        check(rejects(() -> PsiProtocol.run(base, List.of("a", ""), 7, 11, 1)),
                "empty elements rejected");
        check(rejects(() -> PsiProtocol.run(base, base, 1, 11, 1)), "secret one rejected");
        check(rejects(() -> PsiProtocol.run(base, base, 7, PsiProtocol.MODULUS - 1, 1)),
                "trivial secret rejected");
        check(rejects(() -> PsiProtocol.run(base, base, 7, 11, 0)), "zero sketch size rejected");
        check(rejects(() -> PsiProtocol.run(base, base, 7, 11, base.size() + 1)),
                "sketch larger than the set rejected");

        // --- determinism ---------------------------------------------------------
        PsiProtocol.Transcript again = PsiProtocol.run(clientSet, serverSet, 12345, 54321, clientSet.size());
        check(again.intersection.equals(full.intersection), "the protocol must be deterministic");

        System.out.println("exact transcript values = " + full.totalValues() + ", one-value sketch = "
                + tiny.totalValues() + ", intersection = " + truth.size() + " of " + clientSet.size());
        System.out.println("PASS");
    }

    static boolean rejects(Runnable action) {
        try {
            action.run();
        } catch (IllegalArgumentException | IllegalStateException expected) {
            return true;
        }
        return false;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
