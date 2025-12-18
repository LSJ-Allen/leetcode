#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sstream>

using namespace std;

namespace Solution {
    // 8. String to Integer
    int myAtoi(string s) {
        int i = 0;
        int n = s.length();

        // step 1. skip leading white spaces
        while (i < n && s[i] == ' ') {
            i++;
        }

        // step 2. get sign
        int sign = 1;
        if (i < n && (s[i] == '-' || s[i] == '+')) {
            sign = s[i] == '-' ? -1 : 1;
            i++;
        }

        // step 3. convert to digits
        long result = 0;
        while (i < n && isdigit(s[i])) {
            result = result*10 + (s[i] - '0');

            // Check for overflow before applying sign
            if (result * sign > INT_MAX) return INT_MAX;
            if (result * sign < INT_MIN) return INT_MIN;

            i++;
        }

        return result * sign;
    }

    namespace ReverseWords {
        vector<string> split(const string& s, char delimiter) {
            vector<string> tokens;
            string token;
            stringstream ss(s);

            // read string from stringstream ss until delimiter and save it to token
            while (getline(ss, token, delimiter)) {
                // skip empty ones
                if (token.empty()) {
                    continue;
                }
                tokens.push_back(token);
            }

            return tokens;
        }

        string reverseWords(string s) {
            vector<string> words = split(s, ' ');
            reverse(words.begin(), words.end());

            string result = accumulate(words.begin(), words.end(), string(""),
                [](const string& s1, const string& s2) {
                    return s1.empty() ? s2 : s1 + " " + s2;
                });
            return result;
        }
    }

    namespace TextJustification {
        string justifyLine(vector<string>& words, int wordsLength, int maxWidth, bool isLastLine) {
            // Handle single word or last line (left-justified)
            if (words.size() == 1 || isLastLine) {
                string line = words[0];
                for (int i = 1; i < words.size(); i++) {
                    line += " " + words[i];
                }
                // Pad remaining spaces at the end
                line += string(maxWidth - line.length(), ' ');
                return line;
            }

            // Full justification for middle lines
            int totalSpaces = maxWidth - wordsLength;
            int gaps = words.size() - 1;
            int spacesPerGap = totalSpaces / gaps;
            int extraSpaces = totalSpaces % gaps;

            string line = words[0];
            for (int i = 1; i < words.size(); i++) {
                // Add base spaces
                line += string(spacesPerGap, ' ');
                // Distribute extra spaces to leftmost gaps
                if (i <= extraSpaces) {
                    line += " ";
                }
                line += words[i];
            }

            return line;
        }

        vector<string> fullJustify(vector<string>& words, int maxWidth) {
            vector<string> result;
            vector<string> currentLine;
            int currentLength = 0;  // Length of words (without spaces)

            for (const string& word : words) {
                // Check if adding this word exceeds maxWidth
                // currentLength + currentLine.size() accounts for minimum spaces between words
                if (currentLength + currentLine.size() + word.length() > maxWidth) {
                    // Process current line
                    result.push_back(justifyLine(currentLine, currentLength, maxWidth, false));
                    currentLine.clear();
                    currentLength = 0;
                }

                currentLine.push_back(word);
                currentLength += word.length();
            }

            // Handle last line (left-justified)
            result.push_back(justifyLine(currentLine, currentLength, maxWidth, true));

            return result;
        }
    }

}

int main(int argc, char *argv[]) {
    string s = "hello world";
    string result = Solution::ReverseWords::reverseWords(s);
    cout << result << endl;
    return 0;
}
