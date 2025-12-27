def two_sum_unique_hashset(nums, target):
    seen = set()
    pairs = set()

    for num in nums:
        complement = target - num
        if complement in seen:
            pairs.add(tuple(sorted((num, complement))))
        seen.add(num)

    return list(pairs)

nums = [1, 2, 3, 4, 3, 2]
target = 6

print(two_sum_unique_hashset(nums, target))