from typing import *
import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        Approach:

        Convert into a max heap and pop k elements
        """
        heapq.heapify_max(nums)
        kth_largest = 0
        for i in range(k):
            kth_largest = heapq.heappop_max(nums)
            print(kth_largest)

        return kth_largest

def main():
    s = Solution()
    print(s.findKthLargest([3,2,1,5,6,4], 2))

if __name__ == "__main__":
    main()