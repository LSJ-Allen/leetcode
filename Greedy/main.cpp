#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>

using namespace std;

class Solution {
public:
    // 767
    string reorganizeString(string s) {
        /**
         * This is a greedy problem with max heap
         * 1. count the frequency of each character with a freq table
         * 2. check if any character appears more thank (n+1)/2 times, if so
         * its impossible
         * 3. use a max heap to always get the two characters with the highest remaining frequencies.
         * 4. repeatedly take the two most frequent characters, add them to result
         * decrease coutns, put them back in the heap
         * 5. handle the last char if necessary
         *
         * Time complexity: O(nlogn)
         */

        // build frequency table
        unordered_map<char, int> charCount;
        for (char c : s) {
            charCount[c]++;
        }

        // insert each char into a max heap
        priority_queue<pair<int, char>> maxHeap;

        for (auto [c, count] : charCount) {
            // check if the count of any char is greater than (n+1)/2
            if (count > (s.length() + 1)/2) {
                return "";
            }
            maxHeap.emplace(count, c);
        }

        string result = "";
        // repeatedly intertwine the two most frequent chars
        while (maxHeap.size() >= 2) {
            auto [firstCount, firstChar] = maxHeap.top();
            maxHeap.pop();
            auto [secondCount, secondChar] = maxHeap.top();
            maxHeap.pop();

            result += firstChar;
            result += secondChar;

            // decrement the count
            firstCount--;
            secondCount--;

            if (firstCount > 0) {
                maxHeap.emplace(firstCount, firstChar);
            }

            if (secondCount > 0) {
                maxHeap.emplace(secondCount, secondChar);
            }
        }

        // deal with the remaining char
        if (maxHeap.size() > 0) {
            auto [lastCount, lastChar] = maxHeap.top();
            result += lastChar;
        }


        return result;
    }

    // 121. Best Time to Buy and Sell Stock
    int maxProfit(vector<int>& prices) {
        // iterate through the prices
        // if price lower than previous optimal buy price, update buy
        // if higher, calculate profit and update max profit if greater

        int maxProf = 0;
        int buyPrice = prices[0];
        for (int price : prices) {
            if (price > buyPrice) {
                maxProf = max(maxProf, price - buyPrice);
            } else {
                buyPrice = price;
            }
        }

        return maxProf;
    }
};

int main() {
    Solution s = Solution();

    cout << s.reorganizeString("abbabbaaab") << endl;
    return 0;
}
