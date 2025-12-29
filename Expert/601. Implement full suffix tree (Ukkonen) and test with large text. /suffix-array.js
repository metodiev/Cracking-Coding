/**
 * Suffix Array Construction
 * Using Doubling Method (suitable for medium N).
 */

function buildSuffixArrayDoubling(s) {
    const n = s.length;
    let sa = Array.from({length: n}, (_, i) => i);
    let rank = Array.from(s).map(c => c.charCodeAt(0));
    let k = 1;

    while (k < n) {
        sa.sort((a, b) => {
            if (rank[a] !== rank[b]) return rank[a] - rank[b];
            const rankA = a + k < n ? rank[a + k] : -1;
            const rankB = b + k < n ? rank[b + k] : -1;
            return rankA - rankB;
        });

        const tmp = Array(n);
        tmp[sa[0]] = 0;
        for (let i = 1; i < n; i++) {
            tmp[sa[i]] = tmp[sa[i-1]] +
                (rank[sa[i-1]] !== rank[sa[i]] ||
                ((sa[i-1]+k<n ? rank[sa[i-1]+k] : -1) !== (sa[i]+k<n ? rank[sa[i]+k] : -1)) ? 1 : 0);
        }
        rank = tmp;
        k <<= 1;
    }

    return sa;
}

// Example Usage
if (require.main === module) {
    const s = "banana";
    const sa = buildSuffixArrayDoubling(s);
    console.log(sa); // [5,3,1,0,4,2]
    console.log(sa.map(i => s.substring(i))); // ["a","ana","anana","banana","na","nana"]
}

module.exports = buildSuffixArrayDoubling;
