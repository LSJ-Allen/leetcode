#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>
#include "TreeNode.h"

namespace Solution {
    int maxSum = INT_MIN;
    // 124. Binary Tree Maximum Path Sum
    int maxPathSumDFS(TreeNode *root) {
        // recursion base case
        if (!root) {
            return 0;
        }

        int leftGain = max(maxPathSumDFS(root->left), 0);
        int rightGain = max(maxPathSumDFS(root->right), 0);

        // to update the global max
        int pathSum = leftGain + rightGain + root->val;
        maxSum = max(maxSum, pathSum);

        return max(leftGain, rightGain) + root->val;
    }

    int maxPathSum(TreeNode *root) {
        /**
         * Approach:
         * DFS approach while mainting a global value maxSum.
         * At each node, we are concerned only with the path that passes through node from its
         * left subtree to its right subtree.
         * Paths that do not pass node will be handled during the recursion of its subtrees.
         * Path that pass through node to its parent will be handled in parent's level.
         * Therefore, the path sum of node = max(leftGain, 0) + node->val + max(rightGain,0)
         * where the gain of a node = max(all paths that starts at node).
         */
        maxPathSumDFS(root);
        return maxSum;
    }
}

int main(int argc, char *argv[]) {
    vector<int> input = {2, -1};
    Tree *tree = new Tree(input);
    int result = Solution::maxPathSum(tree->root);
    cout << result << endl;

    delete tree;
    return 0;
}
