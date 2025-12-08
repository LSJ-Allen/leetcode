#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>

using namespace std;

namespace Solution {
    // 287. Find the Duplicate Number
    int findDuplicate(vector<int>& nums) {
        /**
         * Approach:
         * Reimagine the array as a singly linked list with nums.size() nodes. Node i's next is nums[i].
         * There are n + 1 nodes from 0 to n. However, there are only n numbers meaning at least one node
         * must contain a repeated number, introducing a cycle in the linked list.
         *
         * Then the problem becomes finding where the cycle occurs without modifying the array.
         *
         * Use Floyd's algorithm with a fast pointer and a slow pointer both staring at position 0.
         * Fast pointer moves 2 at a time while slow pointer moves 1 at a time, When they intersect,
         * reset fast to 0, and it moves 1 at a time. when they intersect again, return slow or fast.
         */

        // edge case
        if (nums.size() == 1) {
            return nums[0];
        }

        int fast = 0, slow = 0;

        while (true) {
            slow = nums[slow];
            fast = nums[nums[fast]];

            if (slow == fast) break;
        }

        // reset fast to 0 and traverse again
        fast = 0;
        while (true) {
            fast = nums[fast];
            slow = nums[slow];
            if (slow == fast) break;
        }

        return fast;
    }

    // 16. 3sum closest
    int threeSumClosest(vector<int>& nums, int target) {
        /**
         * Approach:
         *
         * Two pointer. Sort the array first. Iterate the array starting at index 0, and use two pointers i+1 and
         * nums.size()-1 to find the two numbers that produce the closest result. If sum > target, move right pointer
         * so sum is smaller. If sum < target, move left pointer.
         *
         * Time: O(n^2)
         * Space: O(1)
         */

        sort(nums.begin(), nums.end());
        int closest = 999999999;

        for (int i = 0; i < nums.size(); i++) {
            int left = i + 1, right = (int) nums.size() - 1;

            // loop ends when left == right because one number cannot be used twice
            while (left < right) {
                int sum = nums[left] + nums[right] + nums[i];

                // update closest
                closest = abs(sum - target) < abs(closest - target) ? sum : closest;

                if (sum < target) {
                    left++;
                } else if (sum > target) {
                    right--;
                } else {
                    // if an exact match appears, return it
                    return sum;
                }
            }
        }

        return closest;
    }


    // 18 4Sum
    vector<vector<int>> fourSum(vector<int>& nums, int target) {

        // edge case
        if (nums.size() < 4) {
            return {};
        }

        sort(nums.begin(), nums.end());
        vector<vector<int>> result;

        for (int i = 0; i < nums.size() - 3; i++) {

            // skip dup
            if (i > 0 && nums[i] == nums[i-1]) continue;

            for (int j = i + 1; j < nums.size() - 2; j++) {
                // skip dup
                if (j > i + 1 && nums[j] == nums[j-1]) continue;

                long complement = (long) target - (long) nums[i] - (long) nums[j];
                int left = j + 1, right = (int) nums.size() - 1;

                while (left < right) {
                    long sum = (long) nums[left] + (long) nums[right];
                    if (sum < complement) {
                        left++;
                    } else if (sum > complement) {
                        right--;
                    } else {
                        result.push_back({nums[i], nums[j], nums[left], nums[right]});

                        // move left and right to skip all dups
                        while (left < right && nums[left] == nums[left+1]) {
                            left++;
                        }
                        while (left < right && nums[right] == nums[right-1]) {
                            right--;
                        }

                        // advance left and right
                        left++;
                        right--;
                    }
                }
            }
        }

        return result;
    }
};


int main() {
    vector<int> input = {-2,-1,-1,1,1,2,2};
    vector<vector<int>> result = Solution::fourSum(input, 0);
    return 0;
}
