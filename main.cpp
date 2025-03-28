#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <set>

using namespace std;

struct UserRecord {
    string username;
    string website;
    int timestamp;

    UserRecord(string pUsername, string pWebsite, int pTimestamp) {
        username = pUsername;
        website = pWebsite;
        timestamp = pTimestamp;
    }
};

class Solution {
public:
    vector<string> mostVisitedPattern(vector<string>& username, vector<int>& timestamp, vector<string>& website) {
        // firstly, sort the username, timestamp, and websites all based on timestamp
        vector<UserRecord*> userRecords;

        for (int i = 0; i < username.size(); i++) {
            auto* record = new UserRecord(username[i], website[i], timestamp[i]);
            userRecords.push_back(record);
        }

        auto comp = [](UserRecord* o1, UserRecord* o2) {
            return o1->timestamp < o2->timestamp;
        };

        sort(userRecords.begin(), userRecords.end(), comp);

        // summarize all users into a hash table with the list of their visited websites
        // being the value
        unordered_map<string, vector<string>> visitTable;

        for (auto record : userRecords) {
            const string& user = record->username;
            const string& web = record->website;

            if (visitTable.find(user) != visitTable.end()) {
                visitTable[user].push_back(web);
            } else {
                visitTable[user] = {web};
            }
        }

        // generate all 3 sequence for each user and count occurances
        // key: pattern, 3 web concat
        // value: set of all users that have this pattern.
        unordered_map<string, set<string>> patternUsers;

        for (const auto& [name, visitedWebs] : visitTable) {
            int n = visitedWebs.size();
            set<string> userPatterns;

            // get all permutation of three
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    for (int k = j + 1; k < n; k++) {
                        string pattern = visitedWebs[i] + ":" + visitedWebs[j] + ":" + visitedWebs[k];
                        userPatterns.insert(pattern);
                    }
                }
            }

            // now I have all the patterns, add all the patterns to the pattern map
            for (const string& pattern : userPatterns) {
                patternUsers[pattern].insert(name);
            }
        }

        // find the one with the most users
        string bestPattern;
        int bestPatternCount = 0;

        for (const auto& [pattern, count] : patternUsers) {
            if (count.size() > bestPatternCount) {
                bestPattern = pattern;
                bestPatternCount = count.size();
            } else if (count.size() == bestPatternCount) {
                bestPattern = bestPattern < pattern ? bestPattern : pattern;
            }
        }

        // finally parse pattern into vector
        string w;
        vector<string> result;
        for (char c : bestPattern) {
            if (c == ':') {
                result.push_back(w);
                w = "";
            } else {
                w += c;
            }
        }

        result.push_back(w);
        return result;
    }
};

int main() {
    Solution s = Solution();
    vector<string> username = {"ua","ua","ua","ub","ub","ub"};
    vector<string> website = {"a","b","a","a","b","c"};
    vector<int> timestamp = {1,2,3,4,5,6};
    s.mostVisitedPattern(username, timestamp, website);
    return 0;
}