/**
 * Two-Sum Unique Pairs
 * Return all unique pairs in an array that sum up to a target.
 */

function twoSumUniquePairs(nums, target) {
    const seen = new Set();
    const pairs = new Set();

    for (let num of nums) {
        const complement = target - num;
        if (seen.has(complement)) {
            const sortedPair = [num, complement].sort((a,b) => a-b);
            pairs.add(sortedPair.toString());
        }
        seen.add(num);
    }

    return Array.from(pairs).map(p => p.split(',').map(Number));
}

// Example Usage
if (require.main === module) {
    const nums = [1, 2, 3, 2, 4, 5];
    const target = 5;
    console.log(twoSumUniquePairs(nums, target)); // [[1,4],[2,3]]
}

module.exports = twoSumUniquePairs;
