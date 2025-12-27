def two_sum_uniqeu_two_pointers(nums, target):
    nums.sort()
    left, right = 0, len(nums) - 1
    result = [] 

    while left < right:
        s = nums[left] + nums[right]

        if s == target:
            result.append((nums[left], nums[right])) 
            left +=1
            right -=1

            while left < right and nums[left] == nums[left -1]:
                left +=1
            while left < right and nums[right] == nums[right + 1]:
                right -= 1
        elif s < target:
            left += 1
        else: 
            right -= 1
    return result


nums = [1, 2, 3, 4, 3, 2]
target = 5

print(two_sum_uniqeu_two_pointers(nums, target))
