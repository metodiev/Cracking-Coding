/** Task 941 - Implement continuous profiling pipeline that attributes CPU/memory hotspots to code and inputs.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A sampled-profiler ingest: every sample carries a stack (root -> leaf frames), a weight in CPU
 *  nanoseconds and a weight in allocated bytes, plus the input class of the request. The pipeline folds
 *  the samples into a flame-graph tree where each frame tracks its self cost (charged to the leaf) and its
 *  inclusive cost (charged to every frame on the stack), extracts the top-k CPU and memory hotspots with
 *  their shares, and breaks each frame's self cost down by input class, so a hotspot can be blamed both on
 *  code and on the traffic that triggered it.
 *  Simplified: offline batch ingest of a fixed sample set instead of a streaming collector, fixed frame
 *  names, and two weight columns instead of a typed metric schema.
 *  Verified: cost is conserved end to end, inclusive cost never falls below self cost, the tree-derived
 *  per-function costs and the per-input breakdown match direct loops over the raw samples, the hotspot
 *  ranking matches a brute-force sort, CPU and memory disagree about the hottest frame, and degenerate or
 *  64-frame-deep profiles are handled.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final int INPUT_CLASSES = 3;      // e.g. small / medium / bulk payloads

    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        long nextLong() {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }
        int nextInt(int bound) { return (int) Long.remainderUnsigned(nextLong(), bound); }
    }

    static final class Sample {
        final String[] frames;
        final int inputClass;
        final long cpu, memory;
        Sample(String[] frames, int inputClass, long cpu, long memory) {
            this.frames = frames; this.inputClass = inputClass; this.cpu = cpu; this.memory = memory;
        }
    }

    static final class FrameNode {
        final String name;
        final Map<String, FrameNode> children = new TreeMap<>();
        long cpuSelf, memorySelf, cpuTotal, memoryTotal;
        int samples;
        FrameNode(String name) { this.name = name; }
    }

    static final class Profile {
        final FrameNode root = new FrameNode("<root>");
        final Map<String, long[]> perFunction = new TreeMap<>();   // frame -> {cpuSelf, memorySelf, samples}
        final Map<String, long[]> attribution = new TreeMap<>();   // "frame#inputClass" -> {cpu, memory}
        long totalCpu, totalMemory, hash = 0xcbf29ce484222325L;
        int samples, maxDepth;

        void add(Sample s) {
            FrameNode node = root;
            node.cpuTotal += s.cpu;
            node.memoryTotal += s.memory;
            for (String frame : s.frames) {                        // inclusive cost for every frame on the stack
                node = node.children.computeIfAbsent(frame, FrameNode::new);
                node.cpuTotal += s.cpu;
                node.memoryTotal += s.memory;
            }
            node.cpuSelf += s.cpu;
            node.memorySelf += s.memory;
            node.samples++;
            String leaf = s.frames[s.frames.length - 1];
            long[] function = perFunction.computeIfAbsent(leaf, k -> new long[3]);
            function[0] += s.cpu;
            function[1] += s.memory;
            function[2]++;
            long[] cell = attribution.computeIfAbsent(leaf + "#" + s.inputClass, k -> new long[2]);
            cell[0] += s.cpu;
            cell[1] += s.memory;
            totalCpu += s.cpu;
            totalMemory += s.memory;
            samples++;
            maxDepth = Math.max(maxDepth, s.frames.length);
            hash = hash * 1000003L + leaf.hashCode() * 31L + s.inputClass * 7L + s.cpu;
        }

        /** Top-k frames by self cost for one metric (0 = cpu, 1 = memory). */
        List<String> hotspots(int metric, int k) {
            List<String> names = new ArrayList<>(perFunction.keySet());
            names.sort((a, b) -> {
                long diff = perFunction.get(b)[metric] - perFunction.get(a)[metric];
                return diff != 0 ? (diff > 0 ? 1 : -1) : a.compareTo(b);
            });
            return names.subList(0, Math.min(k, names.size()));
        }

        String render(int metric, int k) {
            long total = Math.max(metric == 0 ? totalCpu : totalMemory, 1);
            StringBuilder sb = new StringBuilder();
            for (String name : hotspots(metric, k)) {
                sb.append(name).append(' ').append(perFunction.get(name)[metric]).append(" (")
                        .append(String.format("%.1f", 100.0 * perFunction.get(name)[metric] / total)).append("%) ");
            }
            return sb.toString().trim();
        }
    }

    /** Fixture: a decode-heavy CPU path and an allocation-heavy memory path, driven by input class. */
    static List<Sample> fixture() {
        Rng rng = new Rng(0x9E3779B97F4A7C15L);
        String[][] paths = {
            {"serve.request", "parse.json", "decode.frame", "checksum.crc"},      // CPU bound
            {"serve.request", "parse.json", "decode.frame", "copy.byte"},         // CPU + memory
            {"serve.request", "parse.json", "decode.frame"},                      // pure CPU spin
            {"serve.request", "parse.json", "alloc.buffer"},                      // memory bound
            {"serve.request", "cache.lookup", "hash.key"},                        // cheap cache hit
            {"serve.request", "cache.lookup", "render.template", "escape.html"},  // mixed
        };
        List<Sample> samples = new ArrayList<>();
        for (int i = 0; i < 5000; i++) {
            int path = rng.nextInt(paths.length);
            int inputClass = rng.nextInt(INPUT_CLASSES);
            long cpu, memory;
            if (path <= 2) {
                cpu = 40_000 + inputClass * 18_000 + rng.nextInt(8_000);
                memory = 4_000 + rng.nextInt(2_000);
            } else if (path == 3) {
                cpu = 6_000 + rng.nextInt(3_000);
                memory = 90_000 + inputClass * 40_000 + rng.nextInt(20_000);
            } else {
                cpu = 12_000 + rng.nextInt(6_000);
                memory = 8_000 + rng.nextInt(4_000);
            }
            samples.add(new Sample(paths[path], inputClass, cpu, memory));
        }
        return samples;
    }

    static Profile ingest(List<Sample> samples) {
        Profile profile = new Profile();
        for (Sample s : samples) profile.add(s);
        return profile;
    }

    /** Oracle: per-function self cost straight from the raw samples, without any tree. */
    static Map<String, long[]> directFunctionCost(List<Sample> samples) {
        Map<String, long[]> direct = new TreeMap<>();
        for (Sample s : samples) {
            long[] cell = direct.computeIfAbsent(s.frames[s.frames.length - 1], k -> new long[3]);
            cell[0] += s.cpu;
            cell[1] += s.memory;
            cell[2]++;
        }
        return direct;
    }

    static Map<String, long[]> directAttribution(List<Sample> samples) {
        Map<String, long[]> direct = new TreeMap<>();
        for (Sample s : samples) {
            long[] cell = direct.computeIfAbsent(s.frames[s.frames.length - 1] + "#" + s.inputClass, k -> new long[2]);
            cell[0] += s.cpu;
            cell[1] += s.memory;
        }
        return direct;
    }

    /** Independent audit of the tree: re-accumulate totals from the leaves upwards. */
    static long[] audit(FrameNode node, long[] violations) {
        long cpu = node.cpuSelf, memory = node.memorySelf;
        for (FrameNode child : node.children.values()) {
            long[] childTotals = audit(child, violations);
            cpu += childTotals[0];
            memory += childTotals[1];
        }
        if (cpu != node.cpuTotal || memory != node.memoryTotal
                || node.cpuSelf > node.cpuTotal || node.memorySelf > node.memoryTotal) violations[0]++;
        return new long[] {cpu, memory};
    }

    static int treeDepth(FrameNode node) {
        int deepest = 0;
        for (FrameNode child : node.children.values()) deepest = Math.max(deepest, treeDepth(child));
        return deepest + 1;
    }

    static int nodes(FrameNode node) {
        int count = 1;
        for (FrameNode child : node.children.values()) count += nodes(child);
        return count;
    }

    public static void main(String[] args) {
        List<Sample> samples = fixture();
        Profile profile = ingest(samples);

        long expectedCpu = 0, expectedMemory = 0;
        for (Sample s : samples) {
            expectedCpu += s.cpu;
            expectedMemory += s.memory;
        }
        long functionCpu = 0, functionMemory = 0;
        for (long[] cell : profile.perFunction.values()) {
            functionCpu += cell[0];
            functionMemory += cell[1];
        }
        check(profile.totalCpu == expectedCpu && profile.totalMemory == expectedMemory && functionCpu == expectedCpu
                        && functionMemory == expectedMemory && profile.root.cpuTotal == expectedCpu
                        && profile.samples == samples.size(),
                "cost must be conserved from samples to flame graph: " + functionCpu + " vs " + expectedCpu);

        long[] violations = new long[1];
        audit(profile.root, violations);
        check(violations[0] == 0 && profile.root.cpuSelf == 0 && profile.maxDepth == 4
                        && treeDepth(profile.root) == 5 && nodes(profile.root) == 11
                        && profile.perFunction.size() == 6,
                "inclusive cost must cover self cost everywhere: " + violations[0] + " violations over "
                        + nodes(profile.root) + " nodes");

        Map<String, long[]> direct = directFunctionCost(samples);
        int mismatches = direct.size() == profile.perFunction.size() ? 0 : 1;
        for (Map.Entry<String, long[]> e : direct.entrySet()) {
            long[] mine = profile.perFunction.get(e.getKey());
            if (mine == null || mine[0] != e.getValue()[0] || mine[1] != e.getValue()[1]
                    || mine[2] != e.getValue()[2]) mismatches++;
        }
        check(mismatches == 0, mismatches + " per-function costs disagree with a direct loop over the samples");

        List<String> cpuTop = profile.hotspots(0, 5);
        List<String> memoryTop = profile.hotspots(1, 5);
        List<String> reference = new ArrayList<>(direct.keySet());
        reference.sort((a, b) -> {
            long diff = direct.get(b)[0] - direct.get(a)[0];
            return diff != 0 ? (diff > 0 ? 1 : -1) : a.compareTo(b);
        });
        check(cpuTop.equals(reference.subList(0, 5)) && cpuTop.get(0).equals("decode.frame")
                        && memoryTop.get(0).equals("alloc.buffer") && !cpuTop.get(0).equals(memoryTop.get(0)),
                "the hotspot ranking must match a brute-force sort and CPU must differ from memory: " + cpuTop
                        + " vs " + memoryTop);

        Map<String, long[]> referenceAttribution = directAttribution(samples);
        int attributionMismatches = referenceAttribution.size() == profile.attribution.size() ? 0 : 1;
        long attributedCpu = 0, attributedMemory = 0;
        for (Map.Entry<String, long[]> e : referenceAttribution.entrySet()) {
            long[] mine = profile.attribution.get(e.getKey());
            if (mine == null || mine[0] != e.getValue()[0] || mine[1] != e.getValue()[1]) attributionMismatches++;
        }
        for (long[] cell : profile.attribution.values()) {
            attributedCpu += cell[0];
            attributedMemory += cell[1];
        }
        int heaviest = 0;
        for (int c = 1; c < INPUT_CLASSES; c++) {
            if (profile.attribution.getOrDefault("decode.frame#" + c, new long[2])[0]
                    > profile.attribution.getOrDefault("decode.frame#" + heaviest, new long[2])[0]) heaviest = c;
        }
        check(attributionMismatches == 0 && attributedCpu == expectedCpu && attributedMemory == expectedMemory
                        && heaviest == 2 && profile.attribution.size() == 6 * INPUT_CLASSES,
                "input attribution must reconcile with the raw samples: " + attributionMismatches + " mismatches, "
                        + attributedCpu + " vs " + expectedCpu + ", heaviest class " + heaviest);

        Profile empty = ingest(List.of());
        check(empty.samples == 0 && empty.totalCpu == 0 && empty.totalMemory == 0 && empty.hotspots(0, 5).isEmpty()
                        && empty.render(1, 3).isEmpty() && treeDepth(empty.root) == 1,
                "an empty profile must yield no hotspots and no arithmetic surprises");

        Profile tiny = ingest(List.of(new Sample(new String[] {"solo.frame"}, 0, 11, 22)));
        Profile deepProfile = new Profile();
        String[] deep = new String[64];
        for (int i = 0; i < deep.length; i++) deep[i] = "frame." + i;
        deepProfile.add(new Sample(deep, 1, 100, 200));
        check(tiny.samples == 1 && tiny.hotspots(0, 5).equals(List.of("solo.frame")) && tiny.root.cpuTotal == 11
                        && tiny.root.cpuSelf == 0 && deepProfile.maxDepth == 64 && treeDepth(deepProfile.root) == 65
                        && deepProfile.perFunction.get("frame.63")[0] == 100 && deepProfile.root.cpuTotal == 100,
                "single-sample and 64-deep profiles must ingest correctly: tiny self " + tiny.root.cpuSelf
                        + ", deep depth " + treeDepth(deepProfile.root));

        Profile again = ingest(samples);
        check(again.hash == profile.hash && again.render(0, 5).equals(profile.render(0, 5))
                        && again.attribution.size() == profile.attribution.size()
                        && again.hotspots(1, 3).equals(memoryTop.subList(0, 3)),
                "the same sample set must produce an identical profile");

        System.out.println("profile: " + profile.samples + " samples, " + nodes(profile.root) + " flame nodes, "
                + profile.perFunction.size() + " leaf functions, max depth " + profile.maxDepth + ", cpu "
                + String.format("%.2f", profile.totalCpu / 1e9) + " s, memory "
                + String.format("%.1f", profile.totalMemory / 1e6) + " MB");
        System.out.println("cpu hotspots: " + profile.render(0, 3));
        System.out.println("memory hotspots: " + profile.render(1, 3));
        System.out.println("PASS");
    }
}
