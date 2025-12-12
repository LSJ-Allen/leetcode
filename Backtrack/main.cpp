#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>
#include <unordered_set>

using namespace std;

namespace Solution {
    // 77. combinations
    namespace combinations {
        vector<vector<int> > result;

        void backtrack(vector<int> &current, int k, int start, int n) {
            // base case when current has k numbers, it is a valid combination
            // add it to result
            if (current.size() == k) {
                result.push_back(current);
                return;
            }

            for (int i = start; i <= n; i++) {
                current.push_back(i);
                backtrack(current, k, i + 1, n);

                // remove i to progress to the next number
                current.pop_back();
            }
        }

        vector<vector<int> > combine(int n, int k) {
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
        vector<vector<int> > result;

        void backtrack(vector<int> &nums, int start_index, int k) {
            // base case
            if (start_index == k) {
                result.emplace_back(nums.begin(), nums.begin() + k);
            }

            for (int i = start_index; i < nums.size(); i++) {
                swap(nums[start_index], nums[i]);
                backtrack(nums, start_index + 1, k);

                // swapping back
                swap(nums[i], nums[start_index]);
            }
        }

        vector<vector<int> > permute(vector<int> &nums) {
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
        vector<vector<int> > result;

        void backtrack(vector<int> &current, const vector<int> &nums, int sum, int target, int start) {
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

        vector<vector<int> > combinationSum(vector<int> &candidates, int target) {
            vector<int> current = {};
            backtrack(current, candidates, 0, target, 0);
            return result;
        }
    }

    // 52. N-Queens II
    namespace NQueens2 {
        int totalSolution = 0;

        void backtrack(unordered_set<int> &cols,
                       unordered_set<int> &rightDiagonal,
                       unordered_set<int> &antiDiagonal,
                       int row,
                       int n) {
            // base case
            if (row == n) {
                totalSolution++;
                return;
            }

            for (int col = 0; col < n; col++) {
                // check if current position is under attack
                if (cols.contains(col) || rightDiagonal.contains(row - col) || antiDiagonal.contains(row + col)) {
                    continue;
                }

                cols.insert(col);
                rightDiagonal.insert(row - col);
                antiDiagonal.insert(row + col);

                backtrack(cols, rightDiagonal, antiDiagonal, row + 1, n);

                // remove queen to continue explore solutions within this row
                cols.erase(col);
                rightDiagonal.erase(row - col);
                antiDiagonal.erase(row + col);
            }
        }

        int totalNQueens(int n) {
            /**
             * Approach:
             * Use backtrack to explore positions to place queens.
             * Each backtrack explore the option to place a queen on one row and the
             * depth of recursion is equal to n. Since the board size is n by n, it
             * is guarenteed to have one queen at each row. In the optimized solution
             * use sets to track columns and diagonals for O(1) look up.
             *
             * Diagonal Representation:
             * ↘ Diagonals can be represented by the value row - col.
             * Example:
             * col: 0   1   2   3
                row 0:   (0)  1   2   3
                row 1:   -1  (0)  1   2
                row 2:   -2  -1  (0)  1
                row 3:   -3  -2  -1  (0)
             * The difference on each diagonal is the same. Thus, all ↘ diagonals can be
             * represented by an array where index i represents the diagonal that has
             * row - column = i and arr[i] represents whether that diagonal is attaced by a
             * queen.
             *
             * Similarly, the ↙ diagonal can be represented by row + col
             *      col: 0   1   2   3
                row 0:   0   1   2  (3)
                row 1:   1   2  (3)  4
                row 2:   2  (3)  4   5
                row 3:  (3)  4   5   6
             *
             * Thus, all columns and diagonals can be represented by a set for O(1) look up
             *
             * Time comlexity: O(n!)
             * row 0: n positions (n columns available)
             * row 1: roughly (n-2 columns available cuz prev queen taks away a column and a diagonal)
             * row 2: roughly n-4
             *  ...
             * Space complexity: O(n)
             */

            unordered_set<int> cols, rightDiagonal, antiDiagonal;
            backtrack(cols, rightDiagonal, antiDiagonal, 0, n);
            return totalSolution;
        }
    }

    // 22. Generate Parentheses
    namespace generateParentheses {
        vector<string> result;

        void backtrack(int numOpen, int numClose, int n, string &s) {
            // base case
            if (s.length() == 2 * n) {
                result.push_back(s);
                return;
            }

            // add opening
            if (numOpen < n) {
                s.push_back('(');
                backtrack(numOpen + 1, numClose, n, s);
                s.pop_back();
            }

            // add closing
            if (numClose < numOpen) {
                s.push_back(')');
                backtrack(numOpen, numClose + 1, n, s);
                s.pop_back();
            }
        }

        vector<string> generateParenthesis(int n) {
            /**
             * Approach:
             *
             * Track the number of opening parentheses and closing
             * parenthese used. We can add an opening parenthesis as
             * long as we haven't used all n of them. We can add a closing
             * parenthesis only lif it wouldn't exceed the number of openig parentheses
             * used so far.
             *
             * Each backtrack level add one parentheses to the string.
             */
            string s;
            backtrack(0, 0, n, s);
            return result;
        }
    }

    // 79. Word Search
    namespace wordSearch {
        bool search(vector<vector<char> > &board, int i, int j, int index, const string &word) {
            // argument index is the index of the next char

            // base case
            if (index == word.size()) {
                return true;
            }

            // turn the current element to -1 so backtrack does not come back to it
            char val = board[i][j];
            board[i][j] = -1;

            // check top neighbor
            if (i - 1 >= 0 && board[i - 1][j] == word[index]) {
                if (search(board, i - 1, j, index + 1, word)) {
                    return true;
                };
            }

            // check right neighbor
            if (j + 1 < board[0].size() && board[i][j + 1] == word[index]) {
                if (search(board, i, j + 1, index + 1, word)) {
                    return true;
                };
            }

            // check down neighbor
            if (i + 1 < board.size() && board[i + 1][j] == word[index]) {
                if (search(board, i + 1, j, index + 1, word)) {
                    return true;
                };
            }

            // check left neighbor
            if (j - 1 >= 0 && board[i][j - 1] == word[index]) {
                if (search(board, i, j - 1, index + 1, word)) {
                    return true;
                };
            }

            // undo board modification
            board[i][j] = val;
            // if the code progressed here, it means none of the 4 paths lead to a valid solution,
            // return false;
            return false;
        }

        bool exist(vector<vector<char> > &board, string word) {
            /**
             * Approach:
             *
             * Use backtrack to explore different paths, starting at a cell that contains the first character.
             * Each call stack adds one character to the path.
             */
            int m = (int) board.size();
            int n = (int) board[0].size();

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (board[i][j] == word[0]) {
                        if (search(board, i, j, 1, word)) {
                            return true;
                        }
                    }
                }
            }

            return false;
        }
    }
}

int main(int argc, char *argv[]) {
    vector<vector<char> > board = {{'C', 'A', 'A'}, {'A', 'A', 'A'}, {'B', 'C', 'D'}};
    bool result = Solution::wordSearch::exist(board, "AAB");
    return 0;
}