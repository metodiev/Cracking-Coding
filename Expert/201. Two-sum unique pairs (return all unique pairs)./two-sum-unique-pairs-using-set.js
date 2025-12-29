function twoSumUniquePairs(nums, target) {
    const seen = new Set();
    const pairs = new Set();

    for(let num of nums) {
        const complement = target - num;
        if (seen.has(complement)) {
            //Sort pair to avoid duplicates

            const sortedPair = [num, complement].sort((a, b) => a - b);
            pairs.add(sortedPair.toString());
        }
        seen.add(num);
        }
    return Array.from(pairs).map(pair => pair.split(',').map(Number));

}

//Example
nums = [1,2,3,4,5];
target = 5;
console.log(twoSumUniquePairs(nums, target));