#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

namespace Solution {
    int coinChange(vector<int> &coins, int amount) {
        /**
         * Approach:
         *
         * len(dp) = amount + 1 to store the minimum number of coin needed for each i.
         * For example, dp[3] represents the minimum number of coins to create amount 3.
         *
         * dp[0] is initialized to 0.
         * Iterate throught dp table to populate it one by one, for each i, go through each coin
         * to find the best coin to use for i. For example, for i = 5 coin = 1, remaining value = 4 so
         * dp[5] would be 1 (the 1 dollar coin we are using) + dp[4] (the best solution with amount 4)
         *
         * Go through each coin to find the best coin to use.
         *
         * Finally return dp[amount]
         *
         */

        vector<int> dp(amount + 1, -1);
        dp[0] = 0;
        for (int i = 1; i < amount + 1; i++) {
            // iterate each coin
            for (int coin: coins) {
                if (coin > i) continue;

                int remainValue = i - coin;
                int remainValueNum = dp[remainValue];

                if (remainValueNum == -1) continue;

                dp[i] = dp[i] > 0 ? min(dp[i], 1 + remainValueNum) : 1 + remainValueNum;
            }
        }

        return dp[amount];
    }

    int rob(vector<int> &nums) {
        /**
         * Approach:
         *
         * dp[i] = max value when robbing up till house i
         * dp[i] = max(dp[i-1], dp[i-2] + nums[i]), here we are choosing between whether to rob
         * the previous house or not. If we choose to rob prev house, we can not rob the current one
         * and the max amount we can get is dp[i-1] the max house we obtained at previous house.
         * If we choose to not rob prev house, the max value is dp[i-2] (the max value prev prev house) +
         * the value of the current house.
         */

        vector<int> dp(nums.size() + 2, 0);
        for (int i = 2; i < dp.size(); i++) {
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i - 2]);
        }

        return dp[dp.size() - 1];
    }

    // 1411. Number of Ways to Paint N * 3 Grid
    int numOfWays(int n) {
        /**
         * Approach:
         *
         * This is a 1D dp with state transition between rows.
         * There are two types of patterns: ABC patterns (diff colors) and ABA patterns (Green Red Green)
         *
         * Type 1 pattern:
         * GRY, GYR, RGY, RYG, YGR, YRG 6 patterns
         *
         * type 2 pattern:
         * GRG, GYG, RGR, RYG, YGY, YRY 6 patterns
         *
         * The transition from type 1 to type 2 is as follows:
         * GRY -> GRG, YRY (middle one can't change)
         *
         * type 1 -> type 1 (2 transitions):
         * GRY -> RYG, YGR (no vertical conflicts)
         *
         * type 2 -> type 2 (3 transitions):
         * ABA -> BAB, BCB, CAC
         *
         * type 2 -> type 1 (2 transition states):
         * ABA -> BAC, CAB
         */
        const int MOD = 1e9 + 7;

        long long type1 = 6; // ABC patterns
        long long type2 = 6; // ABA patterns

        for (int i = 2; i <= n; i++) {
            long long type1new = 2 * type1 + 2 * type2;
            long long type2new = 3 * type2 + 2 * type1;
            type1 = type1new % MOD;
            type2 = type2new % MOD;
        }

        return (type1 + type2) % MOD;
    }

    int maxProfit(vector<int> &prices) {
        /**
         * Approach:
         *
         * 1D DP with state machine.
         * States: Rest, Hold, Sold
         * Transitions (prev day to today):
         * Rest -> Rest (do nothing)
         * Sold -> Rest (sold prev day, now resting today)
         *
         * Rest -> Hold (buy a stock today)
         * Hold -> Hold (keep holding)
         *
         * Hold -> Sold (sell stock today)
         *
         * Create 3 1d dp arrays to represents each state. dp[i] is the max profit with the day ends up in that
         * particular state. For example hold[i] is the max profit of day i you can have by ending day i
         * in hold state.
         *
         */
        int n = static_cast<int>(prices.size());
        vector<int> hold(n, 0);
        vector<int> sold(n, 0);
        vector<int> rest(n, 0);

        // on day 1 cant sell so sold[0] = 0
        // on day 1, enter hold state by buying immediately
        hold[0] = -prices[0];

        for (int i = 1; i < prices.size(); i++) {
            // choose the max between two available transitions
            rest[i] = max(rest[i - 1], sold[i - 1]);

            // choosing between keep holding or transition from rest to purchase today
            hold[i] = max(hold[i - 1], rest[i - 1] - prices[i]);

            // sell today
            sold[i] = hold[i - 1] + prices[i];
        }

        // must not hold on last day for best result, thus max between sold and rest
        return max(rest[n - 1], sold[n - 1]);
    }

    // 1931. Painting a Grid With Three Different Colors
    int colorTheGrid(int m, int n) {
        /**
         * Approach:
         *
         * State encoding:
         * Imagine each column is an m digit ternary number where each digit represents the color of that
         * cell. Red: 0, Green: 1, Blue: 2. A row of length 5 [R G R G R] would be [0 1 0 1 0].
         * Thus there are total 3^m such combinations, and we can use an integer mask in range [0,3^m - 1) to
         * represent each combination. Due to the same-color-not-adjacent constraint, the number of valid combinations is
         * actually 3*2^(m-1).
         *
         * State Transition:
         * To verify a valid state transition, need to check if any digits of the same significance have the same
         * value between the current column and the previous column.
         *
         * Implementation:
         * 2D dp of dimension n * 3^m. dp[i][mask] represents the number of ways to color column i whose coloring
         * scheme is represented by the integer mask.
         * dp[i][mask] = sum([dp[i-1][valid_mask] for valid_mask in range(0, 3^m) and valid_mask can be adjacent);
         * The final result is the sum of dp's last row.
         *
         * For easy access of states, use a hash table with key=mask and value = [valid colorings for next column]
         * Since dp[i] is only dependent on dp[i-1] we can simplify this to a 1D dp.
         */
        const int mod = 1e9 + 7;
        const int range = pow(3, m);
        // step 1. find all valid colorings in [0, 3^m) and store their corresponding ternary representation
        // O(3^m)
        unordered_map<int, vector<int> > valid_colorings;

        for (int mask = 0; mask < range; mask++) {
            // get the ternary representation
            vector<int> color;
            int copy = mask;
            for (int i = 0; i < m; i++) {
                color.push_back(copy % 3);
                copy /= 3;
            }

            // verify if the coloring scheme is correct
            bool is_valid = true;
            for (int i = 1; i < m; i++) {
                if (color[i] == color[i - 1]) {
                    is_valid = false;
                    break;
                }
            }

            if (is_valid) {
                valid_colorings[mask] = move(color);
            }
        }

        // step 2. create a hash map of all valid colorings and their respective valid neighbor states
        // O((3^m)^2) = O(3^(2m))
        unordered_map<int, vector<int> > valid_states_map;
        for (const auto &[int_encoding, ternary_encoding]: valid_colorings) {
            vector<int> valid_states;
            for (const auto &[int_encoding_2, ternary_encoding_2]: valid_colorings) {
                bool is_valid = true;
                for (int i = 0; i < m; i++) {
                    if (ternary_encoding[i] == ternary_encoding_2[i]) {
                        is_valid = false;
                        break;
                    }
                }

                if (is_valid) {
                    valid_states_map[int_encoding].push_back(int_encoding_2);
                }
            }
        }

        // step 3. start dp.
        // 1D array to represent dp[i]
        vector<int> dp(range);

        // initialize dp[0], all valid encoding has value 1 because there is no previous states.
        for (const auto &[int_encoding, _]: valid_colorings) {
            dp[int_encoding] = 1;
        }

        // build dp from i = 1 to i = n
        // O(n*(3^m)*(3^m)) = O(n*3^(2m))
        for (int i = 1; i < n; i++) {
            // represents dp[i]
            vector<int> next(range);

            // populate dp[i]
            for (const auto &[int_encoding, _]: valid_colorings) {
                // dp[i][mask] = sum of all valid neighbor states from dp[i-1]
                int sum = 0;
                for (int valid_neighbor: valid_states_map[int_encoding]) {
                    sum = (sum + dp[valid_neighbor]) % mod;
                }

                next[int_encoding] = sum;
            }

            dp = move(next);
        }

        // calculate sum
        int sum = 0;
        for (int i: dp) {
            sum = (sum + i) % mod;
        }

        return sum;
    }
}

int main() {
    int result = Solution::colorTheGrid(1, 1);
    return 0;
}
