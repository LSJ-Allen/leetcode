#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>

using namespace std;

namespace Solution {
    int coinChange(vector<int>& coins, int amount) {
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

        vector<int>dp(amount + 1, -1);
        dp[0] = 0;
        for (int i = 1; i < amount + 1; i++) {
            // iterate each coin
            for (int coin : coins) {
                if (coin > i) continue;

                int remainValue = i - coin;
                int remainValueNum = dp[remainValue];

                if (remainValueNum == -1) continue;

                dp[i] = dp[i] > 0 ? min(dp[i], 1 + remainValueNum) : 1 + remainValueNum;
            }
        }

        return dp[amount];
    }

    int rob(vector<int>& nums) {
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

        vector<int>dp(nums.size() + 2, 0);
        for (int i = 2; i < dp.size(); i++) {
            dp[i] = max(dp[i-1], dp[i-2] + nums[i-2]);
        }

        return dp[dp.size()-1];
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

        long long type1 = 6;  // ABC patterns
        long long type2 = 6;  // ABA patterns

        for (int i = 2; i <= n; i++) {
            long long type1new = 2 * type1 + 2 * type2;
            long long type2new = 3 * type2 + 2 * type1;
            type1 = type1new % MOD;
            type2 = type2new % MOD;
        }

        return (type1 + type2) % MOD;
    }

    int maxProfit(vector<int>& prices) {
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
        int n = static_cast<int> (prices.size());
        vector<int> hold(n, 0);
        vector<int> sold(n, 0);
        vector<int> rest(n, 0);

        // on day 1 cant sell so sold[0] = 0
        // on day 1, enter hold state by buying immediately
        hold[0] = -prices[0];

        for (int i = 1; i < prices.size(); i++) {
            // choose the max between two available transitions
            rest[i] = max(rest[i-1], sold[i-1]);

            // choosing between keep holding or transition from rest to purchase today
            hold[i] = max(hold[i-1], rest[i-1] - prices[i]);

            // sell today
            sold[i] = hold[i-1] + prices[i];
        }

        // must not hold on last day for best result, thus max between sold and rest
        return max(rest[n-1], sold[n-1]);

    }
}

int main() {
    vector<int> input = {1,2,3,1};
    Solution::rob(input);
    return 0;
}