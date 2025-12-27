def two_sum_unique_brutoforce(nums, target):
    result = set()
    n = len(nums)

    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                pair = tuple(sorted((nums[i], nums[j])))
                result.add(pair)
    
    return list(result)

nums = [1, 2, 3, 4, 3, 2]
target = 5

print(two_sum_unique_brutoforce(nums, target))