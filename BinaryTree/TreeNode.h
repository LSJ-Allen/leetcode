#ifndef LEETCODE_TREE_H
#define LEETCODE_TREE_H

#include <vector>
#include <climits>
#include <queue>

using namespace std;


struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {
    }

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {
    }

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {
    }
};

struct Tree {
    TreeNode *root;

    Tree() : root(nullptr) {
    }

    Tree(TreeNode *rootNode) : root(rootNode) {
    }

    Tree(const vector<int> &arr, int nullValue = INT_MIN) {
        if (arr.empty() || arr[0] == nullValue) {
            root = nullptr;
            return;
        }

        root = new TreeNode(arr[0]);
        queue<TreeNode *> q;
        q.push(root);

        int i = 1;
        while (!q.empty() && i < arr.size()) {
            TreeNode *curr = q.front();
            q.pop();

            // Left child
            if (i < arr.size() && arr[i] != nullValue) {
                curr->left = new TreeNode(arr[i]);
                q.push(curr->left);
            }
            i++;

            // Right child
            if (i < arr.size() && arr[i] != nullValue) {
                curr->right = new TreeNode(arr[i]);
                q.push(curr->right);
            }
            i++;
        }
    }
};
#endif //LEETCODE_TREE_H
