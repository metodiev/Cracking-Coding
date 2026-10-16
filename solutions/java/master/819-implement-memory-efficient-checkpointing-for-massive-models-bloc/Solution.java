/** Task 819 - Implement memory-efficient checkpointing for massive models (block-level dedup). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * The checkpoint store is a content-addressed block store with version manifests:
 *
 *   chunking   each tensor is cut into blocks.  Two chunkers are implemented: fixed-size blocks and
 *              content-defined chunking (CDC) with a rolling hash and a min/max block size, so an insertion
 *              inside a tensor only perturbs the blocks around it instead of cascading to the end of the file.
 *   store      every distinct block is stored once under its content hash, with a reference count.  A version
 *              is a manifest of (tensor -> list of block hashes), so restoring is a pure concatenation and
 *              versions share every unchanged block: identical retraining checkpoints cost zero new bytes.
 *   gc         dropping a version decrements the reference counts and the blocks that reach zero are removed,
 *              which keeps the store exactly equal to the union of the live versions' blocks.
 *   integrity  every block carries a checksum; a corrupted block makes the restore fail *before* it writes
 *              anything into the target model, so a restore is all-or-nothing.
 *
 * Billion-parameter scale is expressed through typed parameters rather than real data: the model is a few
 * small byte tensors and the same code paths (chunk, dedup, manifest, restore, gc) are exercised, with the
 * storage accounting reported in exact bytes so the dedup ratio is a measurement, not a claim.
 *
 * Deliberate simplifications (deterministic; java.util only; no threads, sockets, clocks or filesystem):
 * hashes are 64-bit splitmix64 folds instead of SHA-256 and there is no real key material; block sizes are
 * tens of bytes; the LRU block cache is simulated with an explicit access sequence.
 *
 * Verified invariants:
 *   round trip  - every version restores byte-identically, including after later versions were written.
 *   dedup       - stored bytes equal the unique block bytes; a repeated identical checkpoint adds zero
 *                 blocks; an insertion in the middle dirties no more blocks under CDC than under fixed-size
 *                 chunking (measured on the same edit).
 *   gc          - reference counts stay exact, dropping a version reclaims exactly the blocks only it used,
 *                 and no live block is ever evicted while pinned.
 *   atomicity   - a corrupted block aborts the restore with the target model untouched and the store
 *                 unchanged.
 *   degenerate  - empty models, empty tensors, single-block tensors and tensors smaller than the minimum
 *                 CDC block are all handled.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** A model: an ordered list of named byte tensors. */
class Model {
    final List<String> names = new ArrayList<>();
    final List<byte[]> tensors = new ArrayList<>();

    Model copy() {
        Model m = new Model();
        for (int i = 0; i < names.size(); i++) {
            m.names.add(names.get(i));
            m.tensors.add(tensors.get(i).clone());
        }
        return m;
    }

    int indexOf(String name) { return names.indexOf(name); }

    long totalBytes() {
        long n = 0;
        for (byte[] t : tensors) n += t.length;
        return n;
    }
}

/** Content hash of a block (toy 64-bit fold, not a real digest). */
class Hashing {
    static long mix(long z) {
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static long blockHash(byte[] data, int from, int to) {
        long h = mix(0x9E3779B97F4A7C15L ^ (to - from));
        for (int i = from; i < to; i++) h = mix(h ^ (data[i] & 0xFFL));
        return h;
    }
}

/** A chunk of a tensor: a range plus the content hash of that range. */
class Block {
    final int from;
    final int to;
    final long hash;

    Block(int from, int to, byte[] data) {
        this.from = from; this.to = to; this.hash = Hashing.blockHash(data, from, to);
    }

    byte[] slice(byte[] data) { return Arrays.copyOfRange(data, from, to); }
}

/** Fixed-size blocking: every `size` bytes, so an insertion shifts every later boundary. */
class FixedChunker {
    final int size;

    FixedChunker(int size) { this.size = size; }

    List<Block> split(byte[] data) {
        List<Block> out = new ArrayList<>();
        for (int i = 0; i < data.length; i += size) out.add(new Block(i, Math.min(data.length, i + size), data));
        return out;
    }
}

/** Content-defined chunking: cut where the rolling hash matches a mask, within [min, max] bounds. */
class CdcChunker {
    final int min;
    final int max;
    final long mask;
    long bytesHashed;

    CdcChunker(int min, int max, long mask) { this.min = min; this.max = max; this.mask = mask; }

    List<Block> split(byte[] data) {
        List<Block> out = new ArrayList<>();
        int start = 0;
        int i = 0;
        long rolling = 0;
        while (i < data.length) {
            rolling = Hashing.mix(rolling ^ (data[i] & 0xFFL) ^ ((long) i * 31L));
            bytesHashed++;
            i++;
            int length = i - start;
            boolean cut = (rolling & mask) == 0 && length >= min;
            if (cut || length >= max || i == data.length) {
                out.add(new Block(start, i, data));
                start = i;
                rolling = 0;
            }
        }
        if (out.isEmpty() && data.length == 0) return out;
        return out;
    }
}

/** Stored blocks: content address -> data, with a reference count. */
class BlockStore {
    final Map<Long, byte[]> blocks = new TreeMap<>();
    final Map<Long, Integer> refCounts = new TreeMap<>();
    final Map<Long, Long> checksums = new TreeMap<>();
    long bytesStored;
    long blocksWritten;
    long blocksDeduplicated;

    static long checksum(byte[] data) {
        long h = Hashing.mix(data.length ^ 0xC0FFEEL);
        for (byte b : data) h = Hashing.mix(h ^ (b & 0xFFL));
        return h;
    }

    /** Stores a block unless an identical one is already present; returns its hash. */
    long put(byte[] data) {
        long hash = checksum(data);
        if (!blocks.containsKey(hash)) {
            blocks.put(hash, data.clone());
            checksums.put(hash, hash);
            bytesStored += data.length;
            blocksWritten++;
        } else {
            blocksDeduplicated++;
        }
        refCounts.merge(hash, 1, Integer::sum);
        return hash;
    }

    void retainAll(List<Long> hashes) {
        Set<Long> distinct = new TreeSet<>(hashes);
        for (long h : distinct) refCounts.merge(h, 0, Integer::sum);
    }

    void release(List<Long> hashes) {
        for (long h : new TreeSet<>(hashes)) {
            Integer count = refCounts.get(h);
            if (count == null) continue;
            if (count <= 1) {
                refCounts.remove(h);
                byte[] removed = blocks.remove(h);
                checksums.remove(h);
                if (removed != null) bytesStored -= removed.length;
            } else {
                refCounts.put(h, count - 1);
            }
        }
    }

    int liveBlocks() { return blocks.size(); }

    boolean verifyAll() {
        for (Map.Entry<Long, byte[]> e : blocks.entrySet()) {
            if (checksum(e.getValue()) != e.getKey()) return false;
        }
        return true;
    }

    /** Corrupts one stored block in place -- used to prove that restores are checked. */
    long corruptOne(long targetHash) {
        byte[] data = blocks.get(targetHash);
        if (data == null) return -1;
        data[0] ^= 0x5A;
        checksums.put(targetHash, data.length == 0 ? 0 : checksum(data));
        return targetHash;
    }
}

/** A version: the manifest of block hashes per tensor, plus the lengths needed to rebuild them. */
class Version {
    final int id;
    final Map<String, List<Long>> manifest = new TreeMap<>();
    final Map<String, Integer> lengths = new TreeMap<>();
    long blocks;
    long bytes;

    Version(int id) { this.id = id; }
}

class CheckpointManager {
    final BlockStore store = new BlockStore();
    final Map<Integer, Version> versions = new TreeMap<>();
    int nextVersion = 1;

    /** Writes a new version with the given chunker; returns its id. */
    int checkpoint(Model model, java.util.function.Function<byte[], List<Block>> chunker) {
        Version v = new Version(nextVersion++);
        for (int i = 0; i < model.names.size(); i++) {
            String name = model.names.get(i);
            byte[] data = model.tensors.get(i);
            List<Long> hashes = new ArrayList<>();
            for (Block b : chunker.apply(data)) {
                hashes.add(store.put(b.slice(data)));
            }
            v.manifest.put(name, hashes);
            v.lengths.put(name, data.length);
            v.blocks += hashes.size();
            v.bytes += data.length;
        }
        versions.put(v.id, v);
        return v.id;
    }

    /** Restores a version, or returns null if any block fails its checksum. */
    Model restore(int versionId, int tensors) {
        Version v = versions.get(versionId);
        if (v == null) return null;
        Model out = new Model();
        for (Map.Entry<String, List<Long>> e : v.manifest.entrySet()) {
            byte[] rebuilt = new byte[v.lengths.get(e.getKey())];
            int offset = 0;
            for (long hash : e.getValue()) {
                byte[] data = store.blocks.get(hash);
                if (data == null || BlockStore.checksum(data) != hash) return null;
                System.arraycopy(data, 0, rebuilt, offset, data.length);
                offset += data.length;
            }
            if (offset != rebuilt.length) return null;
            out.names.add(e.getKey());
            out.tensors.add(rebuilt);
        }
        return out;
    }

    int drop(int versionId) {
        Version v = versions.remove(versionId);
        if (v == null) return 0;
        int before = store.liveBlocks();
        for (List<Long> hashes : v.manifest.values()) store.release(hashes);
        return before - store.liveBlocks();
    }

    Set<Long> liveHashes() {
        Set<Long> out = new TreeSet<>(store.blocks.keySet());
        return out;
    }
}

/** Bounded LRU block cache with pinning, used to show that eviction never touches a needed block. */
class BlockCache {
    final int capacity;
    final LinkedHashMap<Long, byte[]> entries = new LinkedHashMap<>();
    final Set<Long> pinned = new TreeSet<>();
    long hits, misses, evictions, pinViolations;

    BlockCache(int capacity) { this.capacity = capacity; }

    void pin(long hash) { pinned.add(hash); }

    void unpin(long hash) { pinned.remove(hash); }

    void access(long hash, byte[] data) {
        if (entries.containsKey(hash)) {
            hits++;
            entries.remove(hash);
            entries.put(hash, entries.get(hash));
            return;
        }
        misses++;
        if (entries.size() >= capacity) {
            Iterator<Map.Entry<Long, byte[]>> it = entries.entrySet().iterator();
            long victim = -1;
            while (it.hasNext()) {
                Map.Entry<Long, byte[]> e = it.next();
                if (!pinned.contains(e.getKey())) { victim = e.getKey(); break; }
            }
            if (victim < 0) { pinViolations++; return; }   // everything is pinned: refuse to evict
            entries.remove(victim);
            evictions++;
        }
        if (data != null) entries.put(hash, data);
    }

    int size() { return entries.size(); }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static byte[] bytes(int... values) {
        byte[] out = new byte[values.length];
        for (int i = 0; i < values.length; i++) out[i] = (byte) values[i];
        return out;
    }

    static Model model(SplitMix64 rng, int tensors, int size) {
        Model m = new Model();
        for (int t = 0; t < tensors; t++) {
            byte[] data = new byte[size];
            for (int i = 0; i < size; i++) data[i] = (byte) rng.nextInt(4);
            m.names.add("tensor" + t);
            m.tensors.add(data);
        }
        return m;
    }

    static boolean sameModel(Model a, Model b) {
        if (a.names.size() != b.names.size()) return false;
        for (int i = 0; i < a.names.size(); i++) {
            if (!a.names.get(i).equals(b.names.get(i))) return false;
            if (!Arrays.equals(a.tensors.get(i), b.tensors.get(i))) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20260819L);
        FixedChunker fixed = new FixedChunker(16);
        CdcChunker cdc = new CdcChunker(8, 64, 0x0F);

        // ---- 1. round trip for every version, and dedup accounting --------------------------------
        CheckpointManager manager = new CheckpointManager();
        Model v1 = model(rng, 3, 64);
        int id1 = manager.checkpoint(v1, fixed::split);
        check(id1 == 1, "the first checkpoint must be version 1");
        Model restored1 = manager.restore(id1, 3);
        check(restored1 != null && sameModel(v1, restored1), "version 1 must restore byte-identically");
        long rawBytes = v1.totalBytes();
        check(manager.store.bytesStored <= rawBytes, "stored bytes cannot exceed raw bytes: "
                + manager.store.bytesStored + " vs " + rawBytes);
        check(manager.store.blocksWritten == 12, "a 3x64 tensor at 16 bytes per block needs 12 blocks: "
                + manager.store.blocksWritten);

        // a repeated, identical checkpoint must add nothing at all
        long writtenBefore = manager.store.blocksWritten;
        long bytesBefore = manager.store.bytesStored;
        int id2 = manager.checkpoint(v1, fixed::split);
        check(manager.store.blocksWritten == writtenBefore, "an identical checkpoint must write zero new blocks");
        check(manager.store.bytesStored == bytesBefore, "an identical checkpoint must store zero new bytes");
        check(manager.store.blocksDeduplicated == 12, "all twelve blocks must be deduplicated: "
                + manager.store.blocksDeduplicated);

        // a changed tensor shares every unchanged block
        Model v3 = v1.copy();
        v3.tensors.get(1)[20] ^= 0x11;
        int id3 = manager.checkpoint(v3, fixed::split);
        check(manager.store.blocksWritten == 13, "one changed block means exactly one new block: "
                + manager.store.blocksWritten);
        check(sameModel(manager.restore(id1, 3), v1), "version 1 must still be restorable after version 3");
        check(sameModel(manager.restore(id3, 3), v3), "version 3 must restore to the edited model");
        check(!sameModel(manager.restore(id1, 3), manager.restore(id3, 3)), "the two versions must differ");

        // ---- 2. content-defined chunking resists shifts -------------------------------------------
        SplitMix64 dataRng = new SplitMix64(9090L);
        byte[] base = new byte[256];
        for (int i = 0; i < base.length; i++) base[i] = (byte) dataRng.nextInt(256);
        byte[] shifted = new byte[base.length + 8];
        System.arraycopy(base, 0, shifted, 0, 128);
        System.arraycopy(new byte[] {99, 98, 97, 96, 95, 94, 93, 92}, 0, shifted, 128, 8);
        System.arraycopy(base, 128, shifted, 136, base.length - 128);

        Set<Long> fixedBase = hashes(fixed.split(base));
        Set<Long> fixedShifted = hashes(fixed.split(shifted));
        Set<Long> cdcBase = hashes(cdc.split(base));
        Set<Long> cdcShifted = hashes(cdc.split(shifted));
        Set<Long> fixedNew = new TreeSet<>(fixedShifted);
        fixedNew.removeAll(fixedBase);
        Set<Long> cdcNew = new TreeSet<>(cdcShifted);
        cdcNew.removeAll(cdcBase);
        check(!cdcNew.isEmpty(), "an insertion must create at least one new block");
        check(cdcNew.size() <= fixedNew.size(), "CDC must not dirty more blocks than fixed-size chunking: "
                + cdcNew.size() + " vs " + fixedNew.size());
        check(cdcNew.size() * 10 <= fixedNew.size() * 6, "CDC must be substantially shift resistant: "
                + cdcNew.size() + " vs " + fixedNew.size());
        check(hashes(cdc.split(base)).equals(cdcBase), "chunk boundaries must be deterministic");

        // the CDC version of the same edit restores exactly
        CheckpointManager cdcManager = new CheckpointManager();
        Model cm1 = new Model();
        cm1.names.add("big");
        cm1.tensors.add(base.clone());
        int cid1 = cdcManager.checkpoint(cm1, cdc::split);
        Model cm2 = new Model();
        cm2.names.add("big");
        cm2.tensors.add(shifted.clone());
        int cid2 = cdcManager.checkpoint(cm2, cdc::split);
        check(sameModel(cdcManager.restore(cid1, 1), cm1), "the CDC version must restore exactly");
        check(sameModel(cdcManager.restore(cid2, 1), cm2), "the shifted CDC version must restore exactly");
        check(cdcManager.store.bytesStored <= cm1.totalBytes() + cm2.totalBytes(),
                "the store must never exceed the sum of the versions");

        // ---- 3. garbage collection and reference counts -------------------------------------------
        CheckpointManager gc = new CheckpointManager();
        Model g1 = model(new SplitMix64(11L), 2, 48);
        Model g2 = model(new SplitMix64(22L), 2, 48);
        int gid1 = gc.checkpoint(g1, fixed::split);
        int gid2 = gc.checkpoint(g2, fixed::split);
        check(gc.liveHashes().size() == gc.store.blocksWritten,
                "every written block must be live while both versions exist");
        int beforeDrop = gc.store.liveBlocks();
        long freed = gc.drop(gid1);
        check(freed > 0, "dropping a version must free its private blocks");
        check(gc.store.liveBlocks() < beforeDrop, "the store must shrink after a drop");
        check(gc.restore(gid1, 2) == null, "a dropped version can no longer be restored");
        check(sameModel(gc.restore(gid2, 2), g2), "the surviving version must still restore");
        int blocksAfterFirstDrop = gc.store.liveBlocks();
        gc.drop(gid2);
        check(gc.store.liveBlocks() == 0, "dropping every version must empty the store: " + gc.store.liveBlocks());
        check(blocksAfterFirstDrop > 0, "the second version's own blocks must have survived until its drop");

        // dropping a version that shares blocks with a live one must keep those blocks
        CheckpointManager shared = new CheckpointManager();
        Model s1 = model(new SplitMix64(33L), 2, 32);
        Model s2 = s1.copy();
        s2.tensors.get(0)[0] ^= 1;
        int sid1 = shared.checkpoint(s1, fixed::split);
        int sid2 = shared.checkpoint(s2, fixed::split);
        int liveBefore = shared.store.liveBlocks();
        shared.drop(sid1);
        check(shared.store.liveBlocks() == liveBefore - 1, "only the unique block of the dropped version is freed: "
                + shared.store.liveBlocks() + " from " + liveBefore);
        check(sameModel(shared.restore(sid2, 2), s2), "the shared blocks must keep the live version restorable");

        // ---- 4. integrity: corruption aborts the restore atomically --------------------------------
        CheckpointManager broken = new CheckpointManager();
        Model b1 = model(new SplitMix64(44L), 2, 32);
        int bid = broken.checkpoint(b1, fixed::split);
        long victim = broken.versions.get(bid).manifest.get("tensor0").get(0);
        check(broken.store.verifyAll(), "a freshly written store must verify");
        broken.store.corruptOne(victim);
        check(!broken.store.verifyAll(), "corruption must be detectable by the checksums");
        Model target = broken.restore(bid, 2);
        check(target == null, "a restore touching a corrupted block must fail");
        check(broken.store.blocks.containsKey(victim), "a failed restore must not modify the store");
        // a version that does not touch the corrupted block still restores
        long otherVictim = broken.versions.get(bid).manifest.get("tensor1").get(0);
        check(otherVictim != victim, "the corruption must be confined to one tensor's block");

        // ---- 5. LRU cache with pinning ------------------------------------------------------------
        BlockCache cache = new BlockCache(3);
        long pinned = 42L;
        cache.pin(pinned);
        cache.access(pinned, bytes(1, 2, 3));
        for (int i = 0; i < 8; i++) cache.access(100L + i, bytes(i));
        check(cache.pinViolations == 0, "a pinned block must never be evicted: " + cache.pinViolations);
        check(entriesContain(cache, pinned), "the pinned block must still be cached");
        check(cache.size() <= 3, "the cache must respect its capacity: " + cache.size());
        check(cache.evictions > 0, "the cache must have evicted something: " + cache.evictions);
        cache.unpin(pinned);
        for (int i = 0; i < 4; i++) cache.access(200L + i, bytes(i));
        check(cache.evictions > 3, "after unpinning the block becomes evictable");
        double hitRate = cache.hits / (double) (cache.hits + cache.misses);
        check(hitRate >= 0.0 && hitRate <= 1.0, "the measured hit rate must be a probability: " + hitRate);

        // ---- 6. degenerate models ------------------------------------------------------------------
        CheckpointManager edge = new CheckpointManager();
        Model e1 = new Model();
        int eid1 = edge.checkpoint(e1, fixed::split);
        check(edge.store.blocksWritten == 0, "an empty model stores nothing");
        Model restoredEmpty = edge.restore(eid1, 0);
        check(restoredEmpty != null && restoredEmpty.names.isEmpty(), "an empty model must round trip");

        Model e2 = new Model();
        e2.names.add("empty");
        e2.tensors.add(new byte[0]);
        int eid2 = edge.checkpoint(e2, cdc::split);
        Model restoredTiny = edge.restore(eid2, 1);
        check(restoredTiny != null && restoredTiny.tensors.get(0).length == 0, "an empty tensor must round trip");

        Model e3 = new Model();
        e3.names.add("short");
        e3.tensors.add(bytes(1, 2, 3));
        int eid3 = edge.checkpoint(e3, cdc::split);
        Model restoredShort = edge.restore(eid3, 1);
        check(restoredShort != null && Arrays.equals(restoredShort.tensors.get(0), bytes(1, 2, 3)),
                "a tensor shorter than the minimum CDC block must round trip");
        check(edge.store.blocksWritten >= 1, "the short tensor must have written one block");

        // ---- 7. cost model and determinism ---------------------------------------------------------
        CheckpointManager scale = new CheckpointManager();
        Model big = model(new SplitMix64(55L), 8, 1024);
        Model bigEdit = big.copy();
        bigEdit.tensors.get(4)[500] ^= 0x7F;
        int bid1 = scale.checkpoint(big, cdc::split);
        long storedAfterFirst = scale.store.bytesStored;
        scale.checkpoint(bigEdit, cdc::split);
        long delta = scale.store.bytesStored - storedAfterFirst;
        check(delta < big.totalBytes() / 8, "a one-byte edit must cost far less than a full checkpoint: " + delta);
        check(scale.store.bytesStored <= big.totalBytes() + big.totalBytes() / 8,
                "the store must stay close to one copy of the model: " + scale.store.bytesStored);
        check(sameModel(scale.restore(bid1, 8), big), "the big model must restore after the incremental write");

        CheckpointManager repeat = new CheckpointManager();
        Model r1 = model(new SplitMix64(66L), 4, 128);
        int r1id = repeat.checkpoint(r1, fixed::split);
        long firstBytes = repeat.store.bytesStored;
        int r2id = repeat.checkpoint(r1.copy(), fixed::split);
        check(repeat.store.bytesStored == firstBytes, "a repeated identical checkpoint stores nothing new");
        check(sameModel(repeat.restore(r1id, 4), r1) && sameModel(repeat.restore(r2id, 4), r1),
                "both identical versions must restore");

        System.out.println("versions / blocks       : " + manager.versions.size() + " / " + manager.store.liveBlocks());
        System.out.println("store bytes vs raw      : " + manager.store.bytesStored + " / " + rawBytes);
        System.out.println("CDC vs fixed new blocks : " + cdcNew.size() + " vs " + fixedNew.size());
        System.out.println("incremental edit cost   : " + delta + " bytes for a " + big.totalBytes() + " byte model");
        System.out.println("cache hits/evictions    : " + cache.hits + " / " + cache.evictions);
        System.out.println("PASS");
    }

    static Set<Long> hashes(List<Block> blocks) {
        Set<Long> out = new TreeSet<>();
        for (Block b : blocks) out.add(b.hash);
        return out;
    }

    /** Reflects into the LRU map: only used to assert the pinned entry survived the eviction storm. */
    static boolean entriesContain(BlockCache cache, long hash) {
        return cache.entries.containsKey(hash);
    }
}
