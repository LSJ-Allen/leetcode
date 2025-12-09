#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>

using namespace std;

namespace Solution {
    // 77. combinations
    namespace combinations {
        vector<vector<int>> result;
        void backtrack(vector<int>& current, int k, int start, int n) {

            // base case when current has k numbers, it is a valid combination
            // add it to result
            if (current.size() == k) {
                result.push_back(current);
                return;
            }

            for (int i = start; i <= n; i++) {
                current.push_back(i);
                backtrack(current, k, i+1, n);

                // remove i to progress to the next number
                current.pop_back();
            }
        }

        vector<vector<int>> combine(int n, int k) {
            /**
             * Approach:
             * This approach is the classic backtrack approach when it comes to combination
             * Each recursion pick 1 number that goes into a combination.
             * The base case is reached when the combination reaches required size.
             * The backtrack function contains a loop that iterates the list of number,
             * each i is the number that will be picked. A current array is maintained throughout
             * backtrack to keep track of the current combination. Remember to remove i once the recursion finishes
             * as the combination is already saved to the result.
             */
            vector<int> cur = {};
            backtrack(cur, k, 1, n);
            return result;
        }
    }

    // 46. Permutations
    namespace permutations {
        vector<vector<int>> result;
        void backtrack(vector<int>& nums, int start_index, int k) {
            // base case
            if (start_index == k) {
                result.emplace_back(nums.begin(), nums.begin() + k);
            }

            for (int i = start_index; i < nums.size(); i++) {
                swap(nums[start_index], nums[i]);
                backtrack(nums, start_index+1, k);

                // swapping back
                swap(nums[i], nums[start_index]);
            }

        }

        vector<vector<int>> permute(vector<int>& nums) {
            /**
             * Approach:
             * Use the Swapping method for space efficient solution.
             * The algorithm builds permutations by swapping elements into the "chosen" prefix of the array.
             * Positions [0, start-1]: Already chosen (locked in)
             * Position start: Currently choosing
             * Positions [start+1, n-1]: Still available
             */
            int n = nums.size();
            vector<int> cur;
            backtrack(nums, 0, n);
            return result;
        }
    }

    // 39. Combination Sum
    namespace combinationSum {
        vector<vector<int>> result;
        void backtrack(vector<int>& current, const vector<int>& nums, int sum, int target, int start) {
            if (sum == target) {
                result.push_back(current);
                return;
            }

            for (int i = start; i < nums.size(); i++) {
                current.push_back(nums[i]);
                sum += nums[i];

                if (sum <= target) {
                    backtrack(current, nums, sum, target, i);
                }
                sum -= nums[i];
                current.pop_back();
            }
        }
        vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
            vector<int> current = {};
            backtrack(current, candidates, 0, target, 0);
            return result;
        }
    }

}
int main(int argc, char* argv[]) {
    vector<int> input = {2,3,6,7};
    Solution::combinationSum::combinationSum(input, 7);
    return 0;
}