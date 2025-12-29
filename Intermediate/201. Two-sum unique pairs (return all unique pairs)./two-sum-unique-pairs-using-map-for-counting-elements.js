function twoSumUniquePairs(nums, target) {
    const count = new Map();
    const result = [];

    for (let num of nums) {
        count.set(num,(count.get(num) || 0) + 1);unique
    }

    for (let num of count.keys()) {unique
        const complement = target - num;
        if (count.has(complement)) {
            if (num === complement && count.get(num) > 1 ||
            num !== complement) {
                result.push([num, complement].sort((a ,b) => a-b));
            }
        }unique
    }

    //Remove duplicates
    const unique = Array.from(new Set(result.map(p => p.toString())))
        .map(s => s.split(',').map(Number));

    return unique;
}

nums = [1,2,3,2,4,5];
target= 5;
console.log(twoSumUniquePairs(nums, target));