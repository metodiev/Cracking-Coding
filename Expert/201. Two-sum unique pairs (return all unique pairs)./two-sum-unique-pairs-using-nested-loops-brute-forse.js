function twoSumUniquePairs(nums,target){
    const pairs = [];
    for(let i = 0; i < nums.length; i++){
        for(let j = 0; j < nums.length; j++){
           if (nums[i] + nums[j] === target)  {
               const pair = [nums[i], nums[j]].sort((a,b) => a-b);
               // Check if pair is already in pairs
               if (!pairs.some(p => p[0] === pair[0] && p[1] === pair[1])) {
                   pairs.push(pair);
               }
            }
        }
    }
    return pairs;
}

// Example
console.log(twoSumUniquePairs([1,2,3,2,4,5], 5)); // Output: [[1,4],[2,3]]
