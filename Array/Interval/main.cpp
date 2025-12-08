#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>

using namespace std;

namespace Solution {
    // 56. Mege Intervals
    vector<vector<int> > merge(vector<vector<int> > &intervals) {
        /**
         * Sort all intervals based on the beginning first.
         * Iterate sorted intervals and update an merged interval along the way.
         * If curInterval's start smaller than merged interval's end, cur interval can be merged.
         * Else, store merged interval and start anew.
         */

        vector<vector<int> > result;

        // sort based on the start of the intervals
        ranges::sort(intervals, [](const vector<int> &a, const vector<int> &b) {
            return a[0] < b[0];
        });

        vector<int> merged = intervals[0];

        for (int i = 1; i < intervals.size(); i++) {
            const vector<int> &cur = intervals[i];

            // if cur's beginning is smaller than or equal to merged's end, cur can be merged
            // merge by extending the end interval
            if (cur[0] <= merged[1]) {
                merged[1] = max(cur[1], merged[1]);
            } else {
                result.push_back(merged);
                merged = cur;
            }
        }

        // push the last interval
        result.push_back(merged);
        return result;
    }
};


int main(int argc, char *argv[]) {
    cout << "hello" << endl;
    return 0;
}
