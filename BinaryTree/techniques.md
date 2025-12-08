# Binary Tree Techniques - Complete LeetCode Guide

## Table of Contents

1. [Tree Traversals](#tree-traversals)
2. [Depth-First Search (DFS)](#depth-first-search-dfs)
3. [Breadth-First Search (BFS)](#breadth-first-search-bfs)
4. [Binary Search Tree (BST)](#binary-search-tree-bst)
5. [Tree Construction](#tree-construction)
6. [Lowest Common Ancestor (LCA)](#lowest-common-ancestor-lca)
7. [Path Problems](#path-problems)
8. [Tree Modification](#tree-modification)
9. [Serialization & Deserialization](#serialization--deserialization)
10. [Advanced Techniques](#advanced-techniques)

---

## Tree Traversals

### Core Concept

Tree traversals are fundamental patterns for visiting every node in a tree. The three main DFS traversals differ in when
the root is processed relative to its children.

**Visual Representation:**

```
        1
       / \
      2   3
     / \
    4   5

Preorder:  1 -> 2 -> 4 -> 5 -> 3  (Root, Left, Right)
Inorder:   4 -> 2 -> 5 -> 1 -> 3  (Left, Root, Right)
Postorder: 4 -> 5 -> 2 -> 3 -> 1  (Left, Right, Root)
Level:     1 -> 2 -> 3 -> 4 -> 5  (BFS by level)
```

**Time Complexity:** O(n) - visit each node once
**Space Complexity:** O(h) for recursion stack, O(n) worst case for skewed tree

### Variations

#### 1. **Recursive Traversals**

- When to use: Simple, clean code; when stack space isn't a concern
- Implementation pattern: Natural recursive structure

```cpp
// Basic TreeNode definition
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// Preorder: Root -> Left -> Right
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);  // Process root first
    preorder(root->left, result);
    preorder(root->right, result);
}

// Inorder: Left -> Root -> Right
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);   // Process root in middle
    inorder(root->right, result);
}

// Postorder: Left -> Right -> Root
void postorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    postorder(root->left, result);
    postorder(root->right, result);
    result.push_back(root->val);   // Process root last
}
```

**Common Pitfalls:**

- Forgetting null check at the beginning
- Stack overflow with very deep trees
- Modifying tree structure during traversal without careful planning

#### 2. **Iterative Traversals**

- When to use: When recursion depth is a concern, more control over execution
- Implementation pattern: Use explicit stack

```cpp
// Iterative Preorder
vector<int> preorderIterative(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    
    stack<TreeNode*> st;
    st.push(root);
    
    while (!st.empty()) {
        TreeNode* node = st.top();
        st.pop();
        result.push_back(node->val);
        
        // Push right first, then left (stack is LIFO)
        if (node->right) st.push(node->right);
        if (node->left) st.push(node->left);
    }
    return result;
}

// Iterative Inorder
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;
    
    while (curr || !st.empty()) {
        // Go to leftmost node
        while (curr) {
            st.push(curr);
            curr = curr->left;
        }
        
        // Process node
        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        
        // Move to right subtree
        curr = curr->right;
    }
    return result;
}

// Iterative Postorder (Two stacks method)
vector<int> postorderIterative(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    
    stack<TreeNode*> st1, st2;
    st1.push(root);
    
    while (!st1.empty()) {
        TreeNode* node = st1.top();
        st1.pop();
        st2.push(node);
        
        if (node->left) st1.push(node->left);
        if (node->right) st1.push(node->right);
    }
    
    while (!st2.empty()) {
        result.push_back(st2.top()->val);
        st2.pop();
    }
    return result;
}
```

#### 3. **Morris Traversal**

- When to use: O(1) space requirement, modifying tree temporarily acceptable
- Implementation pattern: Threaded binary tree approach

```cpp
// Morris Inorder Traversal - O(1) space
vector<int> morrisInorder(TreeNode* root) {
    vector<int> result;
    TreeNode* curr = root;
    
    while (curr) {
        if (!curr->left) {
            // No left subtree, process current and go right
            result.push_back(curr->val);
            curr = curr->right;
        } else {
            // Find inorder predecessor
            TreeNode* pred = curr->left;
            while (pred->right && pred->right != curr) {
                pred = pred->right;
            }
            
            if (!pred->right) {
                // Create thread
                pred->right = curr;
                curr = curr->left;
            } else {
                // Remove thread, process current
                pred->right = nullptr;
                result.push_back(curr->val);
                curr = curr->right;
            }
        }
    }
    return result;
}
```

### LeetCode Problems

#### Easy

**[144] Binary Tree Preorder Traversal**

- Difficulty: Easy
- Pattern: Tree Traversal
- Solution approach: Recursive or iterative with stack

```cpp
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> result;
    preorder(root, result);
    return result;
}
```

**[94] Binary Tree Inorder Traversal**

- Difficulty: Easy
- Pattern: Tree Traversal
- Key insight: Inorder of BST gives sorted array

**[145] Binary Tree Postorder Traversal**

- Difficulty: Easy
- Pattern: Tree Traversal
- Common mistake: Iterative implementation is trickier than pre/inorder

#### Medium

**[102] Binary Tree Level Order Traversal**

- Difficulty: Medium
- Pattern: BFS
- Related: [103] Zigzag, [107] Level Order II, [199] Right Side View

---

## Depth-First Search (DFS)

### Core Concept

DFS explores as far as possible along each branch before backtracking. Used for path finding, tree properties
calculation, and subtree problems.

**Time Complexity:** O(n)
**Space Complexity:** O(h) where h is height

### Variations

#### 1. **Top-Down DFS (Preorder-like)**

- When to use: Passing information from parent to children
- Implementation pattern: Process node before recursing

```cpp
// Example: Maximum Depth
int maxDepth(TreeNode* root) {
    return maxDepthHelper(root, 0);
}

int maxDepthHelper(TreeNode* node, int depth) {
    if (!node) return depth;
    depth++;
    return max(maxDepthHelper(node->left, depth),
               maxDepthHelper(node->right, depth));
}

// Path Sum - checking if path exists with target sum
bool hasPathSum(TreeNode* root, int targetSum) {
    if (!root) return false;
    
    // Leaf node check
    if (!root->left && !root->right) {
        return root->val == targetSum;
    }
    
    int remaining = targetSum - root->val;
    return hasPathSum(root->left, remaining) || 
           hasPathSum(root->right, remaining);
}
```

#### 2. **Bottom-Up DFS (Postorder-like)**

- When to use: Computing properties based on children's results
- Implementation pattern: Recurse first, then process

```cpp
// Example: Maximum Depth (bottom-up)
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    
    int leftDepth = maxDepth(root->left);
    int rightDepth = maxDepth(root->right);
    
    return 1 + max(leftDepth, rightDepth);
}

// Diameter of Binary Tree
int diameter = 0;

int diameterOfBinaryTree(TreeNode* root) {
    height(root);
    return diameter;
}

int height(TreeNode* node) {
    if (!node) return 0;
    
    int leftHeight = height(node->left);
    int rightHeight = height(node->right);
    
    // Update diameter at each node
    diameter = max(diameter, leftHeight + rightHeight);
    
    return 1 + max(leftHeight, rightHeight);
}
```

#### 3. **DFS with State Tracking**

- When to use: Need to track path, ancestors, or cumulative values
- Implementation pattern: Pass state through recursion

```cpp
// Path Sum II - find all paths with target sum
vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    vector<vector<int>> result;
    vector<int> path;
    dfs(root, targetSum, path, result);
    return result;
}

void dfs(TreeNode* node, int remaining, vector<int>& path, 
         vector<vector<int>>& result) {
    if (!node) return;
    
    path.push_back(node->val);
    
    // Check if leaf and sum matches
    if (!node->left && !node->right && remaining == node->val) {
        result.push_back(path);
    }
    
    dfs(node->left, remaining - node->val, path, result);
    dfs(node->right, remaining - node->val, path, result);
    
    path.pop_back(); // Backtrack
}
```

### Problem Patterns

#### Pattern 1: **Tree Properties**

- Recognition: Calculate height, depth, size, balance
- Solution template: Bottom-up DFS

```cpp
// Check if tree is balanced
bool isBalanced(TreeNode* root) {
    return checkBalance(root) != -1;
}

int checkBalance(TreeNode* node) {
    if (!node) return 0;
    
    int left = checkBalance(node->left);
    if (left == -1) return -1;
    
    int right = checkBalance(node->right);
    if (right == -1) return -1;
    
    if (abs(left - right) > 1) return -1;
    
    return 1 + max(left, right);
}
```

#### Pattern 2: **Path Problems**

- Recognition: Find paths, sum paths, count paths
- Solution template: Top-down or bottom-up with backtracking

```cpp
// Binary Tree Maximum Path Sum
int maxSum = INT_MIN;

int maxPathSum(TreeNode* root) {
    maxGain(root);
    return maxSum;
}

int maxGain(TreeNode* node) {
    if (!node) return 0;
    
    // Only take positive gains
    int leftGain = max(maxGain(node->left), 0);
    int rightGain = max(maxGain(node->right), 0);
    
    // Price to start new path where node is highest
    int priceNewPath = node->val + leftGain + rightGain;
    maxSum = max(maxSum, priceNewPath);
    
    // For recursion, return max gain if continue path
    return node->val + max(leftGain, rightGain);
}
```

### LeetCode Problems

#### Easy

**[104] Maximum Depth of Binary Tree**

- Difficulty: Easy
- Pattern: Bottom-up DFS
- Solution: Return 1 + max(left, right)
- Common mistakes: Forgetting the +1 for current node

**[111] Minimum Depth of Binary Tree**

- Difficulty: Easy
- Pattern: DFS or BFS
- Key insight: Must reach a leaf node (both children null)

```cpp
int minDepth(TreeNode* root) {
    if (!root) return 0;
    if (!root->left) return 1 + minDepth(root->right);
    if (!root->right) return 1 + minDepth(root->left);
    return 1 + min(minDepth(root->left), minDepth(root->right));
}
```

**[100] Same Tree**

- Difficulty: Easy
- Pattern: Simultaneous DFS

```cpp
bool isSameTree(TreeNode* p, TreeNode* q) {
    if (!p && !q) return true;
    if (!p || !q) return false;
    if (p->val != q->val) return false;
    return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
}
```

**[101] Symmetric Tree**

- Difficulty: Easy
- Pattern: Mirror DFS

```cpp
bool isSymmetric(TreeNode* root) {
    return isMirror(root, root);
}

bool isMirror(TreeNode* t1, TreeNode* t2) {
    if (!t1 && !t2) return true;
    if (!t1 || !t2) return false;
    return (t1->val == t2->val) 
        && isMirror(t1->right, t2->left) 
        && isMirror(t1->left, t2->right);
}
```

#### Medium

**[112] Path Sum**

- Difficulty: Easy/Medium
- Pattern: Top-down DFS with accumulator

**[113] Path Sum II**

- Difficulty: Medium
- Pattern: DFS with backtracking
- Key technique: Track path, backtrack after exploring

**[543] Diameter of Binary Tree**

- Difficulty: Easy/Medium
- Pattern: Bottom-up DFS with global variable
- Common mistake: Forgetting diameter might not pass through root

**[124] Binary Tree Maximum Path Sum**

- Difficulty: Hard
- Pattern: Bottom-up DFS with global max
- Key insight: At each node, decide to extend path or start new

---

## Breadth-First Search (BFS)

### Core Concept

BFS explores nodes level by level using a queue. Essential for level-order problems, shortest path in unweighted trees,
and problems requiring level information.

**Visual Representation:**

```
        1           Level 0
       / \
      2   3         Level 1
     / \   \
    4   5   6       Level 2

Process order: 1 -> 2 -> 3 -> 4 -> 5 -> 6
```

**Time Complexity:** O(n)
**Space Complexity:** O(w) where w is maximum width

### Variations

#### 1. **Basic Level Order**

- When to use: Need to process nodes level by level
- Implementation pattern: Queue with level size tracking

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            currentLevel.push_back(node->val);
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        
        result.push_back(currentLevel);
    }
    return result;
}
```

**Common Pitfalls:**

- Not capturing level size before the loop
- Modifying queue size during iteration
- Forgetting to check null before pushing children

#### 2. **BFS with Level Information**

- When to use: Need level number or level-specific processing
- Implementation pattern: Queue with (node, level) pairs

```cpp
// Right Side View - see rightmost node at each level
vector<int> rightSideView(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            
            // Add last node of each level
            if (i == levelSize - 1) {
                result.push_back(node->val);
            }
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return result;
}
```

#### 3. **Zigzag Level Order**

- When to use: Alternate direction at each level
- Implementation pattern: Use flag or reverse alternate levels

```cpp
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;
    
    queue<TreeNode*> q;
    q.push(root);
    bool leftToRight = true;
    
    while (!q.empty()) {
        int levelSize = q.size();
        vector<int> currentLevel(levelSize);
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            
            // Find position to fill node's value
            int index = leftToRight ? i : (levelSize - 1 - i);
            currentLevel[index] = node->val;
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        
        leftToRight = !leftToRight;
        result.push_back(currentLevel);
    }
    return result;
}
```

### Problem Patterns

#### Pattern 1: **Level-wise Processing**

- Recognition: "each level", "level by level", "row by row"
- Solution template: Standard BFS with level size

```cpp
// Average of Levels
vector<double> averageOfLevels(TreeNode* root) {
    vector<double> result;
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        double sum = 0;
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            sum += node->val;
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        
        result.push_back(sum / levelSize);
    }
    return result;
}
```

#### Pattern 2: **Finding Specific Nodes**

- Recognition: "rightmost", "leftmost", "minimum depth"
- Solution template: BFS with early termination or filtering

```cpp
// Minimum Depth (BFS approach - more efficient)
int minDepth(TreeNode* root) {
    if (!root) return 0;
    
    queue<TreeNode*> q;
    q.push(root);
    int depth = 1;
    
    while (!q.empty()) {
        int levelSize = q.size();
        
        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = q.front();
            q.pop();
            
            // First leaf found is at minimum depth
            if (!node->left && !node->right) {
                return depth;
            }
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        depth++;
    }
    return depth;
}
```

### LeetCode Problems

#### Medium

**[102] Binary Tree Level Order Traversal**

- Difficulty: Medium
- Pattern: Basic BFS
- Foundation for many BFS problems

**[107] Binary Tree Level Order Traversal II**

- Difficulty: Medium
- Pattern: BFS + reverse result
- Solution: Standard BFS then reverse, or use deque

**[103] Binary Tree Zigzag Level Order Traversal**

- Difficulty: Medium
- Pattern: BFS with direction flag
- Common mistake: Trying to reverse queue instead of result

**[199] Binary Tree Right Side View**

- Difficulty: Medium
- Pattern: BFS - take last node per level
- Alternative: DFS with level tracking (root->right first)

**[515] Find Largest Value in Each Tree Row**

- Difficulty: Medium
- Pattern: BFS with max tracking per level

**[116] Populating Next Right Pointers in Each Node**

- Difficulty: Medium
- Pattern: BFS or constant space level-linking

```cpp
Node* connect(Node* root) {
    if (!root) return root;
    
    queue<Node*> q;
    q.push(root);
    
    while (!q.empty()) {
        int levelSize = q.size();
        
        for (int i = 0; i < levelSize; i++) {
            Node* node = q.front();
            q.pop();
            
            if (i < levelSize - 1) {
                node->next = q.front();
            }
            
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return root;
}
```

**[637] Average of Levels in Binary Tree**

- Difficulty: Easy
- Pattern: BFS with sum calculation

---

## Binary Search Tree (BST)

### Core Concept

BST maintains ordering property: left subtree < node < right subtree. This enables O(log n) average case operations and
in-order traversal produces sorted output.

**Visual Representation:**

```
        5
       / \
      3   7
     / \ / \
    2  4 6  8

Inorder: 2, 3, 4, 5, 6, 7, 8 (sorted!)
```

**Time Complexity:** O(log n) average, O(n) worst case (skewed tree)
**Space Complexity:** O(h) for recursion

### Variations

#### 1. **BST Validation**

- When to use: Verify BST property
- Implementation pattern: Pass valid range down tree

```cpp
bool isValidBST(TreeNode* root) {
    return validate(root, LONG_MIN, LONG_MAX);
}

bool validate(TreeNode* node, long minVal, long maxVal) {
    if (!node) return true;
    
    // Current node must be within range
    if (node->val <= minVal || node->val >= maxVal) {
        return false;
    }
    
    // Left subtree: all values < node->val
    // Right subtree: all values > node->val
    return validate(node->left, minVal, node->val) &&
           validate(node->right, node->val, maxVal);
}

// Alternative: Inorder traversal should be strictly increasing
bool isValidBST_Inorder(TreeNode* root) {
    TreeNode* prev = nullptr;
    return inorderCheck(root, prev);
}

bool inorderCheck(TreeNode* node, TreeNode*& prev) {
    if (!node) return true;
    
    if (!inorderCheck(node->left, prev)) return false;
    
    if (prev && prev->val >= node->val) return false;
    prev = node;
    
    return inorderCheck(node->right, prev);
}
```

#### 2. **BST Search**

- When to use: Finding values, checking existence
- Implementation pattern: Binary search on tree structure

```cpp
// Search in BST
TreeNode* searchBST(TreeNode* root, int val) {
    if (!root || root->val == val) return root;
    
    return val < root->val ? 
        searchBST(root->left, val) : 
        searchBST(root->right, val);
}

// Iterative version (more efficient)
TreeNode* searchBSTIterative(TreeNode* root, int val) {
    while (root && root->val != val) {
        root = val < root->val ? root->left : root->right;
    }
    return root;
}
```

#### 3. **BST Insertion & Deletion**

- When to use: Modifying BST structure
- Implementation pattern: Find position, maintain BST property

```cpp
// Insert into BST
TreeNode* insertIntoBST(TreeNode* root, int val) {
    if (!root) return new TreeNode(val);
    
    if (val < root->val) {
        root->left = insertIntoBST(root->left, val);
    } else {
        root->right = insertIntoBST(root->right, val);
    }
    return root;
}

// Delete from BST
TreeNode* deleteNode(TreeNode* root, int key) {
    if (!root) return nullptr;
    
    if (key < root->val) {
        root->left = deleteNode(root->left, key);
    } else if (key > root->val) {
        root->right = deleteNode(root->right, key);
    } else {
        // Node to delete found
        // Case 1: No children or one child
        if (!root->left) return root->right;
        if (!root->right) return root->left;
        
        // Case 2: Two children
        // Find inorder successor (smallest in right subtree)
        TreeNode* minNode = root->right;
        while (minNode->left) {
            minNode = minNode->left;
        }
        
        // Replace value and delete successor
        root->val = minNode->val;
        root->right = deleteNode(root->right, minNode->val);
    }
    return root;
}
```

#### 4. **BST to Sorted Array/List**

- When to use: Need sorted output
- Implementation pattern: Inorder traversal

```cpp
// Kth Smallest Element in BST
int kthSmallest(TreeNode* root, int k) {
    int count = 0;
    int result = -1;
    inorderKth(root, k, count, result);
    return result;
}

void inorderKth(TreeNode* node, int k, int& count, int& result) {
    if (!node || count >= k) return;
    
    inorderKth(node->left, k, count, result);
    
    count++;
    if (count == k) {
        result = node->val;
        return;
    }
    
    inorderKth(node->right, k, count, result);
}

// Convert BST to Greater Tree
// Each node's value = sum of all greater values
TreeNode* convertBST(TreeNode* root) {
    int sum = 0;
    reverseInorder(root, sum);
    return root;
}

void reverseInorder(TreeNode* node, int& sum) {
    if (!node) return;
    
    reverseInorder(node->right, sum); // Right first
    sum += node->val;
    node->val = sum;
    reverseInorder(node->left, sum);
}
```

### Problem Patterns

#### Pattern 1: **BST Property Exploitation**

- Recognition: Find element, range queries, k-th element
- Solution template: Use BST ordering to prune search space

```cpp
// Find Mode in BST (most frequent values)
vector<int> findMode(TreeNode* root) {
    vector<int> result;
    int maxCount = 0, currentCount = 0;
    TreeNode* prev = nullptr;
    
    inorderMode(root, prev, currentCount, maxCount, result);
    return result;
}

void inorderMode(TreeNode* node, TreeNode*& prev, int& currentCount,
                 int& maxCount, vector<int>& result) {
    if (!node) return;
    
    inorderMode(node->left, prev, currentCount, maxCount, result);
    
    // Count frequency
    currentCount = (prev && prev->val == node->val) ? currentCount + 1 : 1;
    
    if (currentCount > maxCount) {
        maxCount = currentCount;
        result.clear();
        result.push_back(node->val);
    } else if (currentCount == maxCount) {
        result.push_back(node->val);
    }
    
    prev = node;
    inorderMode(node->right, prev, currentCount, maxCount, result);
}
```

#### Pattern 2: **BST Construction**

- Recognition: Build BST from sorted/preorder array
- Solution template: Choose root, recursively build subtrees

```cpp
// Convert Sorted Array to BST
TreeNode* sortedArrayToBST(vector<int>& nums) {
    return buildBST(nums, 0, nums.size() - 1);
}

TreeNode* buildBST(vector<int>& nums, int left, int right) {
    if (left > right) return nullptr;
    
    int mid = left + (right - left) / 2;
    TreeNode* root = new TreeNode(nums[mid]);
    
    root->left = buildBST(nums, left, mid - 1);
    root->right = buildBST(nums, mid + 1, right);
    
    return root;
}
```

### LeetCode Problems

#### Easy

**[700] Search in a Binary Search Tree**

- Difficulty: Easy
- Pattern: BST Search
- Key insight: Use BST property to eliminate half the tree

**[701] Insert into a Binary Search Tree**

- Difficulty: Medium
- Pattern: BST Insertion
- Common mistake: Forgetting to return the modified tree

**[530] Minimum Absolute Difference in BST**

- Difficulty: Easy
- Pattern: Inorder traversal
- Key insight: Min difference is between adjacent inorder nodes

#### Medium

**[98] Validate Binary Search Tree**

- Difficulty: Medium
- Pattern: Range validation
- Common mistake: Only comparing with immediate children

```cpp
// WRONG approach
bool isValidBST_Wrong(TreeNode* root) {
    if (!root) return true;
    if (root->left && root->left->val >= root->val) return false;
    if (root->right && root->right->val <= root->val) return false;
    return isValidBST_Wrong(root->left) && isValidBST_Wrong(root->right);
}
// This fails for tree: [5,1,6,null,null,3,7]
```

**[230] Kth Smallest Element in a BST**

- Difficulty: Medium
- Pattern: Inorder traversal with counter
- Follow-up: How to optimize for frequent kth queries? (Augment tree with size)

**[235] Lowest Common Ancestor of a BST**

- Difficulty: Medium
- Pattern: BST property exploitation

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    // Both in left subtree
    if (p->val < root->val && q->val < root->val) {
        return lowestCommonAncestor(root->left, p, q);
    }
    // Both in right subtree
    if (p->val > root->val && q->val > root->val) {
        return lowestCommonAncestor(root->right, p, q);
    }
    // Split point - root is LCA
    return root;
}
```

**[450] Delete Node in a BST**

- Difficulty: Medium
- Pattern: BST deletion with three cases
- Key technique: Replace with inorder successor/predecessor

**[108] Convert Sorted Array to Binary Search Tree**

- Difficulty: Easy
- Pattern: Divide and conquer
- Key insight: Middle element as root ensures balance

**[538] Convert BST to Greater Tree**

- Difficulty: Medium
- Pattern: Reverse inorder with accumulator
- Key technique: Process right subtree first

---

## Tree Construction

### Core Concept

Constructing trees from traversals or other representations. Key insight: different traversals provide different
information about tree structure.

**Key Principles:**

- Preorder: First element is root
- Inorder: Splits left and right subtrees
- Postorder: Last element is root
- Need at least 2 traversals (one must be inorder) to uniquely construct tree

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### Variations

#### 1. **From Inorder + Preorder/Postorder**

- When to use: Classic construction problems
- Implementation pattern: Use one traversal for root, other for splitting

```cpp
// Construct Binary Tree from Preorder and Inorder
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    unordered_map<int, int> inorderMap;
    for (int i = 0; i < inorder.size(); i++) {
        inorderMap[inorder[i]] = i;
    }
    int preIdx = 0;
    return build(preorder, inorder, preIdx, 0, inorder.size() - 1, inorderMap);
}

TreeNode* build(vector<int>& preorder, vector<int>& inorder, int& preIdx,
                int inLeft, int inRight, unordered_map<int, int>& inorderMap) {
    if (inLeft > inRight) return nullptr;
    
    // Root is next element in preorder
    int rootVal = preorder[preIdx++];
    TreeNode* root = new TreeNode(rootVal);
    
    // Find root position in inorder
    int inRoot = inorderMap[rootVal];
    
    // Build left subtree (before root in inorder)
    root->left = build(preorder, inorder, preIdx, inLeft, inRoot - 1, inorderMap);
    
    // Build right subtree (after root in inorder)
    root->right = build(preorder, inorder, preIdx, inRoot + 1, inRight, inorderMap);
    
    return root;
}

// Construct Binary Tree from Inorder and Postorder
TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
    unordered_map<int, int> inorderMap;
    for (int i = 0; i < inorder.size(); i++) {
        inorderMap[inorder[i]] = i;
    }
    int postIdx = postorder.size() - 1;
    return buildPost(inorder, postorder, postIdx, 0, inorder.size() - 1, inorderMap);
}

TreeNode* buildPost(vector<int>& inorder, vector<int>& postorder, int& postIdx,
                    int inLeft, int inRight, unordered_map<int, int>& inorderMap) {
    if (inLeft > inRight) return nullptr;
    
    // Root is from end of postorder
    int rootVal = postorder[postIdx--];
    TreeNode* root = new TreeNode(rootVal);
    
    int inRoot = inorderMap[rootVal];
    
    // Build RIGHT first (postorder processed from back)
    root->right = buildPost(inorder, postorder, postIdx, inRoot + 1, inRight, inorderMap);
    root->left = buildPost(inorder, postorder, postIdx, inLeft, inRoot - 1, inorderMap);
    
    return root;
}
```

**Common Pitfalls:**

- Forgetting to increment/decrement traversal index
- Building subtrees in wrong order (matters for postorder)
- Not using map for O(1) inorder lookups (leads to O(n²))

#### 2. **From String/Array Representation**

- When to use: Serialization, special encodings
- Implementation pattern: Parse and recursively construct

```cpp
// Construct Binary Tree from String with brackets
// Input: "4(2(3)(1))(6(5))"
TreeNode* str2tree(string s) {
    int i = 0;
    return parse(s, i);
}

TreeNode* parse(string& s, int& i) {
    if (i >= s.length()) return nullptr;
    
    // Parse number (handle negative)
    int start = i;
    if (s[i] == '-') i++;
    while (i < s.length() && isdigit(s[i])) i++;
    
    int val = stoi(s.substr(start, i - start));
    TreeNode* node = new TreeNode(val);
    
    // Parse left child
    if (i < s.length() && s[i] == '(') {
        i++; // skip '('
        node->left = parse(s, i);
        i++; // skip ')'
    }
    
    // Parse right child
    if (i < s.length() && s[i] == '(') {
        i++; // skip '('
        node->right = parse(s, i);
        i++; // skip ')'
    }
    
    return node;
}
```

#### 3. **From Linked List**

- When to use: Convert linear structure to tree
- Implementation pattern: Find middle, recursively build

```cpp
// Convert Sorted List to Binary Search Tree
TreeNode* sortedListToBST(ListNode* head) {
    if (!head) return nullptr;
    if (!head->next) return new TreeNode(head->val);
    
    // Find middle using slow/fast pointers
    ListNode *slow = head, *fast = head, *prev = nullptr;
    while (fast && fast->next) {
        prev = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    
    // slow is middle
    TreeNode* root = new TreeNode(slow->val);
    
    // Cut left part
    if (prev) prev->next = nullptr;
    
    // Build subtrees
    root->left = sortedListToBST(prev ? head : nullptr);
    root->right = sortedListToBST(slow->next);
    
    return root;
}
```

### LeetCode Problems

#### Medium

**[105] Construct Binary Tree from Preorder and Inorder Traversal**

- Difficulty: Medium
- Pattern: Preorder root + inorder split
- Key optimization: HashMap for inorder indices

**[106] Construct Binary Tree from Inorder and Postorder Traversal**

- Difficulty: Medium
- Pattern: Postorder root + inorder split
- Common mistake: Building left before right (should be right first)

**[889] Construct Binary Tree from Preorder and Postorder Traversal**

- Difficulty: Medium
- Pattern: Both give root info
- Key insight: Not unique (need to handle carefully)

**[109] Convert Sorted List to Binary Search Tree**

- Difficulty: Medium
- Pattern: Find middle + recursion
- Follow-up: O(n) using inorder simulation

---

## Lowest Common Ancestor (LCA)

### Core Concept

LCA is the deepest node that is an ancestor of both given nodes. Critical for many tree problems involving relationships
between nodes.

**Visual Representation:**

```
        3
       / \
      5   1
     / \ / \
    6  2 0  8
      / \
     7   4

LCA(5, 1) = 3
LCA(5, 4) = 5
LCA(7, 4) = 2
```

**Time Complexity:** O(n)
**Space Complexity:** O(h)

### Variations

#### 1. **LCA in Binary Tree**

- When to use: General binary tree
- Implementation pattern: Postorder traversal returning found nodes

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    // Base case: empty tree or found one of the nodes
    if (!root || root == p || root == q) return root;
    
    // Search in left and right subtrees
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    
    // If both found in different subtrees, current node is LCA
    if (left && right) return root;
    
    // Otherwise return whichever is not null
    return left ? left : right;
}
```

#### 2. **LCA in BST**

- When to use: Binary Search Tree (more efficient)
- Implementation pattern: Use BST property to navigate

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    // Iterative approach
    while (root) {
        // Both in left subtree
        if (p->val < root->val && q->val < root->val) {
            root = root->left;
        }
        // Both in right subtree
        else if (p->val > root->val && q->val > root->val) {
            root = root->right;
        }
        // Split point found
        else {
            return root;
        }
    }
    return nullptr;
}
```

#### 3. **LCA with Parent Pointers**

- When to use: Nodes have parent pointers
- Implementation pattern: Path to root + intersection

```cpp
// With parent pointers
Node* lowestCommonAncestor(Node* p, Node * q) {
    unordered_set<Node*> ancestors;
    
    // Store all ancestors of p
    while (p) {
        ancestors.insert(p);
        p = p->parent;
    }
    
    // Find first ancestor of q in set
    while (q) {
        if (ancestors.count(q)) return q;
        q = q->parent;
    }
    
    return nullptr;
}
```

#### 4. **Distance Between Nodes**

- When to use: Need path length or distance calculations
- Implementation pattern: Find LCA + calculate distances

```cpp
// Find distance between two nodes
int findDistance(TreeNode* root, int p, int q) {
    TreeNode* lca = findLCA(root, p, q);
    int distP = findDepth(lca, p, 0);
    int distQ = findDepth(lca, q, 0);
    return distP + distQ;
}

int findDepth(TreeNode* node, int target, int depth) {
    if (!node) return -1;
    if (node->val == target) return depth;
    
    int left = findDepth(node->left, target, depth + 1);
    if (left != -1) return left;
    
    return findDepth(node->right, target, depth + 1);
}
```

### Problem Patterns

#### Pattern 1: **LCA Variants**

- Recognition: Find common ancestor with constraints
- Solution template: Modify basic LCA algorithm

```cpp
// LCA of Deepest Leaves
TreeNode* lcaDeepestLeaves(TreeNode* root) {
    pair<TreeNode*, int> result = helper(root, 0);
    return result.first;
}

// Returns {lca, depth}
pair<TreeNode*, int> helper(TreeNode* node, int depth) {
    if (!node) return {nullptr, depth};
    
    auto left = helper(node->left, depth + 1);
    auto right = helper(node->right, depth + 1);
    
    // Both subtrees same depth -> current is LCA
    if (left.second == right.second) {
        return {node, left.second};
    }
    
    // Return deeper side
    return left.second > right.second ? left : right;
}
```

### LeetCode Problems

#### Medium

**[236] Lowest Common Ancestor of a Binary Tree**

- Difficulty: Medium
- Pattern: Postorder DFS
- Key insight: LCA is split point where nodes diverge

**[235] Lowest Common Ancestor of a Binary Search Tree**

- Difficulty: Medium
- Pattern: BST property exploitation
- Optimization: Can be done iteratively

**[1644] Lowest Common Ancestor of a Binary Tree II**

- Difficulty: Medium
- Pattern: LCA with existence check
- Difference: Nodes might not exist in tree

**[1650] Lowest Common Ancestor of a Binary Tree III**

- Difficulty: Medium
- Pattern: LCA with parent pointers
- Similar to: Linked list intersection problem

**[1123] Lowest Common Ancestor of Deepest Leaves**

- Difficulty: Medium
- Pattern: LCA + depth tracking
- Key technique: Return depth along with LCA

---

## Path Problems

### Core Concept

Path problems involve finding, counting, or calculating properties of paths in trees. A path is a sequence of nodes
where each node is connected to the next.

**Path Types:**

1. **Root-to-leaf paths**: Must start at root and end at leaf
2. **Any path**: Can start and end anywhere
3. **Parent-to-child paths**: Must follow parent-child relationships

**Time Complexity:** O(n) typically
**Space Complexity:** O(h) for recursion

### Variations

#### 1. **Root-to-Leaf Paths**

- When to use: Path must start at root
- Implementation pattern: DFS with path accumulation

```cpp
// Path Sum - check if path exists with target sum
bool hasPathSum(TreeNode* root, int targetSum) {
    if (!root) return false;
    
    // Leaf node
    if (!root->left && !root->right) {
        return root->val == targetSum;
    }
    
    int remaining = targetSum - root->val;
    return hasPathSum(root->left, remaining) || 
           hasPathSum(root->right, remaining);
}

// Path Sum II - find all root-to-leaf paths with target sum
vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    vector<vector<int>> result;
    vector<int> path;
    dfsPath(root, targetSum, path, result);
    return result;
}

void dfsPath(TreeNode* node, int remaining, vector<int>& path,
             vector<vector<int>>& result) {
    if (!node) return;
    
    path.push_back(node->val);
    
    // Check leaf
    if (!node->left && !node->right && remaining == node->val) {
        result.push_back(path);
    }
    
    dfsPath(node->left, remaining - node->val, path, result);
    dfsPath(node->right, remaining - node->val, path, result);
    
    path.pop_back(); // Backtrack
}

// Sum Root to Leaf Numbers
// Paths represent numbers (e.g., 1->2->3 = 123)
int sumNumbers(TreeNode* root) {
    return dfsSum(root, 0);
}

int dfsSum(TreeNode* node, int currentSum) {
    if (!node) return 0;
    
    currentSum = currentSum * 10 + node->val;
    
    // Leaf node
    if (!node->left && !node->right) {
        return currentSum;
    }
    
    return dfsSum(node->left, currentSum) + dfsSum(node->right, currentSum);
}
```

#### 2. **Any-to-Any Paths**

- When to use: Path can start/end anywhere
- Implementation pattern: DFS with path sum tracking

```cpp
// Path Sum III - count paths with target sum (not necessarily root-to-leaf)
int pathSum(TreeNode* root, int targetSum) {
    if (!root) return 0;
    
    // Count paths starting from root
    int pathsFromRoot = countPaths(root, targetSum);
    
    // Count paths in left and right subtrees
    int pathsLeft = pathSum(root->left, targetSum);
    int pathsRight = pathSum(root->right, targetSum);
    
    return pathsFromRoot + pathsLeft + pathsRight;
}

int countPaths(TreeNode* node, long long remaining) {
    if (!node) return 0;
    
    int count = (node->val == remaining) ? 1 : 0;
    
    // Continue paths through this node
    count += countPaths(node->left, remaining - node->val);
    count += countPaths(node->right, remaining - node->val);
    
    return count;
}

// Optimized with prefix sum
int pathSum_Optimized(TreeNode* root, int targetSum) {
    unordered_map<long long, int> prefixSum;
    prefixSum[0] = 1; // Base case
    return dfs(root, 0, targetSum, prefixSum);
}

int dfs(TreeNode* node, long long currSum, int target,
        unordered_map<long long, int>& prefixSum) {
    if (!node) return 0;
    
    currSum += node->val;
    
    // Number of paths ending here with target sum
    int count = prefixSum[currSum - target];
    
    // Add current prefix sum
    prefixSum[currSum]++;
    
    // Recurse
    count += dfs(node->left, currSum, target, prefixSum);
    count += dfs(node->right, currSum, target, prefixSum);
    
    // Backtrack
    prefixSum[currSum]--;
    
    return count;
}
```

#### 3. **Maximum Path Sum**

- When to use: Optimize path value
- Implementation pattern: Bottom-up with global variable

```cpp
// Binary Tree Maximum Path Sum
int maxSum = INT_MIN;

int maxPathSum(TreeNode* root) {
    maxGain(root);
    return maxSum;
}

int maxGain(TreeNode* node) {
    if (!node) return 0;
    
    // Only consider positive gains
    int leftGain = max(maxGain(node->left), 0);
    int rightGain = max(maxGain(node->right), 0);
    
    // Price of new path through node
    int priceNewPath = node->val + leftGain + rightGain;
    
    // Update global maximum
    maxSum = max(maxSum, priceNewPath);
    
    // For recursion, return max gain if continue same path
    return node->val + max(leftGain, rightGain);
}
```

#### 4. **Longest Path**

- When to use: Maximize path length
- Implementation pattern: Track path length during traversal

```cpp
// Longest Univalue Path (all nodes same value)
int longestPath = 0;

int longestUnivaluePath(TreeNode* root) {
    dfsUnival(root);
    return longestPath;
}

int dfsUnival(TreeNode* node) {
    if (!node) return 0;
    
    int left = dfsUnival(node->left);
    int right = dfsUnival(node->right);
    
    int leftPath = 0, rightPath = 0;
    
    // Extend left path if values match
    if (node->left && node->left->val == node->val) {
        leftPath = left + 1;
    }
    
    // Extend right path if values match
    if (node->right && node->right->val == node->val) {
        rightPath = right + 1;
    }
    
    // Update global with path through this node
    longestPath = max(longestPath, leftPath + rightPath);
    
    // Return longer arm for parent
    return max(leftPath, rightPath);
}
```

### Problem Patterns

#### Pattern 1: **Path Constraints**

- Recognition: Specific path requirements (sum, length, values)
- Solution template: DFS with constraint checking

```cpp
// Check if path exists with all decreasing values
bool hasDecreasingPath(TreeNode* root) {
    return dfs(root, INT_MAX);
}

bool dfs(TreeNode* node, int prevVal) {
    if (!node) return false;
    if (node->val >= prevVal) return false;
    
    // Leaf reached with valid path
    if (!node->left && !node->right) return true;
    
    return dfs(node->left, node->val) || dfs(node->right, node->val);
}
```

### LeetCode Problems

#### Easy

**[112] Path Sum**

- Difficulty: Easy
- Pattern: Root-to-leaf path sum
- Key: Check at leaf nodes only

**[257] Binary Tree Paths**

- Difficulty: Easy
- Pattern: Find all root-to-leaf paths

```cpp
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> result;
    if (!root) return result;
    
    string path = to_string(root->val);
    if (!root->left && !root->right) {
        result.push_back(path);
        return result;
    }
    
    if (root->left) {
        dfsPath(root->left, path, result);
    }
    if (root->right) {
        dfsPath(root->right, path, result);
    }
    return result;
}

void dfsPath(TreeNode* node, string path, vector<string>& result) {
    path += "->" + to_string(node->val);
    
    if (!node->left && !node->right) {
        result.push_back(path);
        return;
    }
    
    if (node->left) dfsPath(node->left, path, result);
    if (node->right) dfsPath(node->right, path, result);
}
```

#### Medium

**[113] Path Sum II**

- Difficulty: Medium
- Pattern: Root-to-leaf with backtracking
- Key technique: Remember to pop after recursion

**[437] Path Sum III**

- Difficulty: Medium
- Pattern: Any-to-any path sum
- Optimization: Prefix sum map for O(n)

**[129] Sum Root to Leaf Numbers**

- Difficulty: Medium
- Pattern: Path represents number
- Key: Build number during traversal

**[988] Smallest String Starting From Leaf**

- Difficulty: Medium
- Pattern: Root-to-leaf string comparison

#### Hard

**[124] Binary Tree Maximum Path Sum**

- Difficulty: Hard
- Pattern: Any-to-any path optimization
- Key insight: Choose whether to extend path or start new

**[687] Longest Univalue Path**

- Difficulty: Medium
- Pattern: Path with constraint (same values)
- Technique: Track path length in both directions

---

## Tree Modification

### Core Concept

Problems involving changing tree structure or values while maintaining certain properties. Includes flipping, inverting,
flattening, and pruning operations.

**Time Complexity:** O(n) typically
**Space Complexity:** O(h) for recursion

### Variations

#### 1. **Tree Inversion/Mirroring**

- When to use: Swap left and right subtrees
- Implementation pattern: Recursive swap

```cpp
// Invert Binary Tree
TreeNode* invertTree(TreeNode* root) {
    if (!root) return nullptr;
    
    // Swap children
    TreeNode* temp = root->left;
    root->left = root->right;
    root->right = temp;
    
    // Recursively invert subtrees
    invertTree(root->left);
    invertTree(root->right);
    
    return root;
}

// Iterative version using queue
TreeNode* invertTree_Iterative(TreeNode* root) {
    if (!root) return nullptr;
    
    queue<TreeNode*> q;
    q.push(root);
    
    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();
        
        // Swap children
        swap(node->left, node->right);
        
        if (node->left) q.push(node->left);
        if (node->right) q.push(node->right);
    }
    
    return root;
}
```

#### 2. **Tree Flattening**

- When to use: Convert tree to linked list structure
- Implementation pattern: Modified preorder/postorder

```cpp
// Flatten Binary Tree to Linked List (preorder)
void flatten(TreeNode* root) {
    if (!root) return;
    
    // Flatten left and right subtrees
    flatten(root->left);
    flatten(root->right);
    
    // Store right subtree
    TreeNode* rightSubtree = root->right;
    
    // Move left subtree to right
    root->right = root->left;
    root->left = nullptr;
    
    // Attach original right subtree to end
    TreeNode* curr = root;
    while (curr->right) {
        curr = curr->right;
    }
    curr->right = rightSubtree;
}

// Morris-like O(1) space approach
void flatten_Optimized(TreeNode* root) {
    TreeNode* curr = root;
    
    while (curr) {
        if (curr->left) {
            // Find rightmost node of left subtree
            TreeNode* rightmost = curr->left;
            while (rightmost->right) {
                rightmost = rightmost->right;
            }
            
            // Connect it to current right
            rightmost->right = curr->right;
            
            // Move left to right
            curr->right = curr->left;
            curr->left = nullptr;
        }
        
        curr = curr->right;
    }
}
```

#### 3. **Tree Pruning**

- When to use: Remove nodes based on conditions
- Implementation pattern: Postorder with null returns

```cpp
// Binary Tree Pruning (remove subtrees with all 0s)
TreeNode* pruneTree(TreeNode* root) {
    if (!root) return nullptr;
    
    // Prune left and right subtrees
    root->left = pruneTree(root->left);
    root->right = pruneTree(root->right);
    
    // If current is 0 and both children pruned, prune this node
    if (root->val == 0 && !root->left && !root->right) {
        return nullptr;
    }
    
    return root;
}

// Remove Leaf Nodes with specific value
TreeNode* removeLeafNodes(TreeNode* root, int target) {
    if (!root) return nullptr;
    
    root->left = removeLeafNodes(root->left, target);
    root->right = removeLeafNodes(root->right, target);
    
    // If leaf with target value
    if (!root->left && !root->right && root->val == target) {
        return nullptr;
    }
    
    return root;
}
```

#### 4. **Tree Trimming (BST)**

- When to use: Remove nodes outside range
- Implementation pattern: BST property + pruning

```cpp
// Trim BST to be within [low, high]
TreeNode* trimBST(TreeNode* root, int low, int high) {
    if (!root) return nullptr;
    
    // Current node too small, all left subtree too small
    if (root->val < low) {
        return trimBST(root->right, low, high);
    }
    
    // Current node too large, all right subtree too large
    if (root->val > high) {
        return trimBST(root->left, low, high);
    }
    
    // Current node in range, trim both subtrees
    root->left = trimBST(root->left, low, high);
    root->right = trimBST(root->right, low, high);
    
    return root;
}
```

#### 5. **Adding Nodes**

- When to use: Augment tree with new nodes
- Implementation pattern: Insertion based on rules

```cpp
// Add One Row to Tree at depth d
TreeNode* addOneRow(TreeNode* root, int val, int depth) {
    // Special case: add at root
    if (depth == 1) {
        TreeNode* newRoot = new TreeNode(val);
        newRoot->left = root;
        return newRoot;
    }
    
    addRow(root, val, depth, 1);
    return root;
}

void addRow(TreeNode* node, int val, int targetDepth, int currDepth) {
    if (!node) return;
    
    if (currDepth == targetDepth - 1) {
        // Insert at this level
        TreeNode* newLeft = new TreeNode(val);
        TreeNode* newRight = new TreeNode(val);
        
        newLeft->left = node->left;
        newRight->right = node->right;
        
        node->left = newLeft;
        node->right = newRight;
    } else {
        addRow(node->left, val, targetDepth, currDepth + 1);
        addRow(node->right, val, targetDepth, currDepth + 1);
    }
}
```

### Problem Patterns

#### Pattern 1: **Structure Transformation**

- Recognition: Change tree shape/structure
- Solution template: Recursive restructuring

```cpp
// Convert BST to Greater Sum Tree
TreeNode* bstToGst(TreeNode* root) {
    int sum = 0;
    reverseInorder(root, sum);
    return root;
}

void reverseInorder(TreeNode* node, int& sum) {
    if (!node) return;
    
    reverseInorder(node->right, sum);
    sum += node->val;
    node->val = sum;
    reverseInorder(node->left, sum);
}
```

### LeetCode Problems

#### Easy

**[226] Invert Binary Tree**

- Difficulty: Easy
- Pattern: Recursive swap
- Famous: "Google interview question"

**[617] Merge Two Binary Trees**

- Difficulty: Easy
- Pattern: Simultaneous traversal + modification

```cpp
TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
    if (!t1) return t2;
    if (!t2) return t1;
    
    t1->val += t2->val;
    t1->left = mergeTrees(t1->left, t2->left);
    t1->right = mergeTrees(t1->right, t2->right);
    
    return t1;
}
```

#### Medium

**[114] Flatten Binary Tree to Linked List**

- Difficulty: Medium
- Pattern: Preorder traversal + restructuring
- Follow-up: Do it in-place with O(1) space

**[116] Populating Next Right Pointers**

- Difficulty: Medium
- Pattern: Level-order linking
- Key: Can use O(1) space by using next pointers

**[669] Trim a Binary Search Tree**

- Difficulty: Medium
- Pattern: BST pruning with range
- Key: Use BST property to prune entire subtrees

**[814] Binary Tree Pruning**

- Difficulty: Medium
- Pattern: Postorder pruning
- Similar to: Remove leaf nodes

**[1325] Delete Leaves With a Given Value**

- Difficulty: Medium
- Pattern: Postorder pruning with value check

---

## Serialization & Deserialization

### Core Concept

Converting tree structure to/from string or other representations. Essential for tree persistence, transmission, and
cloning.

**Approaches:**

1. **Level-order with nulls**: Most intuitive
2. **Preorder with markers**: Compact
3. **Bracket notation**: Human-readable

**Time Complexity:** O(n)
**Space Complexity:** O(n)

### Variations

#### 1. **Level-Order Serialization**

- When to use: Preserve tree structure explicitly
- Implementation pattern: BFS with null markers

```cpp
class Codec {
public:
    // Serialize tree to string
    string serialize(TreeNode* root) {
        if (!root) return "[]";
        
        string result = "[";
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            TreeNode* node = q.front();
            q.pop();
            
            if (node) {
                result += to_string(node->val) + ",";
                q.push(node->left);
                q.push(node->right);
            } else {
                result += "null,";
            }
        }
        
        // Remove trailing comma and add closing bracket
        result.back() = ']';
        return result;
    }

    // Deserialize string to tree
    TreeNode* deserialize(string data) {
        if (data == "[]") return nullptr;
        
        // Parse values
        vector<string> values = parse(data);
        
        TreeNode* root = new TreeNode(stoi(values[0]));
        queue<TreeNode*> q;
        q.push(root);
        
        int i = 1;
        while (!q.empty() && i < values.size()) {
            TreeNode* node = q.front();
            q.pop();
            
            // Left child
            if (values[i] != "null") {
                node->left = new TreeNode(stoi(values[i]));
                q.push(node->left);
            }
            i++;
            
            // Right child
            if (i < values.size() && values[i] != "null") {
                node->right = new TreeNode(stoi(values[i]));
                q.push(node->right);
            }
            i++;
        }
        
        return root;
    }
    
private:
    vector<string> parse(string data) {
        vector<string> result;
        data = data.substr(1, data.size() - 2); // Remove brackets
        
        stringstream ss(data);
        string item;
        
        while (getline(ss, item, ',')) {
            result.push_back(item);
        }
        
        return result;
    }
};
```

#### 2. **Preorder Serialization**

- When to use: More compact representation
- Implementation pattern: DFS with delimiter

```cpp
class Codec {
public:
    string serialize(TreeNode* root) {
        string result;
        serializeHelper(root, result);
        return result;
    }
    
    void serializeHelper(TreeNode* node, string& result) {
        if (!node) {
            result += "null,";
            return;
        }
        
        result += to_string(node->val) + ",";
        serializeHelper(node->left, result);
        serializeHelper(node->right, result);
    }
    
    TreeNode* deserialize(string data) {
        stringstream ss(data);
        return deserializeHelper(ss);
    }
    
    TreeNode* deserializeHelper(stringstream& ss) {
        string val;
        getline(ss, val, ',');
        
        if (val == "null") return nullptr;
        
        TreeNode* node = new TreeNode(stoi(val));
        node->left = deserializeHelper(ss);
        node->right = deserializeHelper(ss);
        
        return node;
    }
};
```

#### 3. **BST Optimized Serialization**

- When to use: Leverage BST property
- Implementation pattern: Preorder without nulls

```cpp
// Serialize BST - more compact (no nulls needed)
class CodecBST {
public:
    string serialize(TreeNode* root) {
        string result;
        serializeHelper(root, result);
        return result;
    }
    
    void serializeHelper(TreeNode* node, string& result) {
        if (!node) return;
        
        result += to_string(node->val) + ",";
        serializeHelper(node->left, result);
        serializeHelper(node->right, result);
    }
    
    TreeNode* deserialize(string data) {
        if (data.empty()) return nullptr;
        
        stringstream ss(data);
        vector<int> values;
        string val;
        
        while (getline(ss, val, ',')) {
            if (!val.empty()) {
                values.push_back(stoi(val));
            }
        }
        
        int idx = 0;
        return buildBST(values, idx, INT_MIN, INT_MAX);
    }
    
    TreeNode* buildBST(vector<int>& values, int& idx, int minVal, int maxVal) {
        if (idx >= values.size()) return nullptr;
        
        int val = values[idx];
        
        // Check if current value fits in valid range
        if (val < minVal || val > maxVal) return nullptr;
        
        TreeNode* node = new TreeNode(val);
        idx++;
        
        // Build left and right with updated ranges
        node->left = buildBST(values, idx, minVal, val);
        node->right = buildBST(values, idx, val, maxVal);
        
        return node;
    }
};
```

#### 4. **String Encoding with Brackets**

- When to use: Human-readable format
- Implementation pattern: Recursive with parentheses

```cpp
// Serialize to string like: "1(2(4)(5))(3)"
string tree2str(TreeNode* root) {
    if (!root) return "";
    
    string result = to_string(root->val);
    
    // Left child exists or right child exists (need to preserve structure)
    if (root->left || root->right) {
        result += "(" + tree2str(root->left) + ")";
    }
    
    // Right child exists
    if (root->right) {
        result += "(" + tree2str(root->right) + ")";
    }
    
    return result;
}

// Deserialize from string
TreeNode* str2tree(string s) {
    if (s.empty()) return nullptr;
    
    int idx = 0;
    return parseTree(s, idx);
}

TreeNode* parseTree(string& s, int& idx) {
    if (idx >= s.length()) return nullptr;
    
    // Parse number
    int start = idx;
    if (s[idx] == '-') idx++;
    while (idx < s.length() && isdigit(s[idx])) idx++;
    
    TreeNode* node = new TreeNode(stoi(s.substr(start, idx - start)));
    
    // Parse left child
    if (idx < s.length() && s[idx] == '(') {
        idx++; // skip '('
        node->left = parseTree(s, idx);
        idx++; // skip ')'
    }
    
    // Parse right child
    if (idx < s.length() && s[idx] == '(') {
        idx++; // skip '('
        node->right = parseTree(s, idx);
        idx++; // skip ')'
    }
    
    return node;
}
```

### LeetCode Problems

#### Medium

**[297] Serialize and Deserialize Binary Tree**

- Difficulty: Hard
- Pattern: Level-order or preorder with nulls
- Key: Choose consistent format for ser/deser

**[449] Serialize and Deserialize BST**

- Difficulty: Medium
- Pattern: Preorder without nulls (more efficient)
- Optimization: Use BST property to avoid null markers

**[428] Serialize and Deserialize N-ary Tree**

- Difficulty: Hard
- Pattern: Include child count or sentinel
- Challenge: Variable number of children

**[606] Construct String from Binary Tree**

- Difficulty: Easy
- Pattern: Bracket notation
- Key: Omit unnecessary empty parentheses

**[652] Find Duplicate Subtrees**

- Difficulty: Medium
- Pattern: Serialize subtrees + hash map

```cpp
vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
    unordered_map<string, int> count;
    vector<TreeNode*> result;
    serialize(root, count, result);
    return result;
}

string serialize(TreeNode* node, unordered_map<string, int>& count,
                 vector<TreeNode*>& result) {
    if (!node) return "#";
    
    string serial = to_string(node->val) + "," +
                    serialize(node->left, count, result) + "," +
                    serialize(node->right, count, result);
    
    if (++count[serial] == 2) {
        result.push_back(node);
    }
    
    return serial;
}
```

---

## Advanced Techniques

### Morris Traversal (O(1) Space)

```cpp
// Morris Inorder - O(1) space
vector<int> morrisInorder(TreeNode* root) {
    vector<int> result;
    TreeNode* curr = root;
    
    while (curr) {
        if (!curr->left) {
            result.push_back(curr->val);
            curr = curr->right;
        } else {
            // Find predecessor
            TreeNode* pred = curr->left;
            while (pred->right && pred->right != curr) {
                pred = pred->right;
            }
            
            if (!pred->right) {
                // Create thread
                pred->right = curr;
                curr = curr->left;
            } else {
                // Remove thread
                pred->right = nullptr;
                result.push_back(curr->val);
                curr = curr->right;
            }
        }
    }
    return result;
}
```

### Tree DP

```cpp
// House Robber III - can't rob adjacent nodes
int rob(TreeNode* root) {
    pair<int, int> result = robHelper(root);
    return max(result.first, result.second);
}

// Returns {robThis, skipThis}
pair<int, int> robHelper(TreeNode* node) {
    if (!node) return {0, 0};
    
    auto left = robHelper(node->left);
    auto right = robHelper(node->right);
    
    // Rob this node: can't rob children
    int robThis = node->val + left.second + right.second;
    
    // Skip this node: take max of children
    int skipThis = max(left.first, left.second) + 
                   max(right.first, right.second);
    
    return {robThis, skipThis};
}
```

### Segment Tree

```cpp
// Range Sum Query - Mutable (Segment Tree)
class NumArray {
private:
    vector<int> tree;
    int n;
    
    void buildTree(vector<int>& nums, int node, int start, int end) {
        if (start == end) {
            tree[node] = nums[start];
        } else {
            int mid = start + (end - start) / 2;
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;
            
            buildTree(nums, leftChild, start, mid);
            buildTree(nums, rightChild, mid + 1, end);
            
            tree[node] = tree[leftChild] + tree[rightChild];
        }
    }
    
    void updateTree(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = start + (end - start) / 2;
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;
            
            if (idx <= mid) {
                updateTree(leftChild, start, mid, idx, val);
            } else {
                updateTree(rightChild, mid + 1, end, idx, val);
            }
            
            tree[node] = tree[leftChild] + tree[rightChild];
        }
    }
    
    int queryTree(int node, int start, int end, int left, int right) {
        if (right < start || left > end) return 0;
        if (left <= start && end <= right) return tree[node];
        
        int mid = start + (end - start) / 2;
        int leftChild = 2 * node + 1;
        int rightChild = 2 * node + 2;
        
        return queryTree(leftChild, start, mid, left, right) +
               queryTree(rightChild, mid + 1, end, left, right);
    }
    
public:
    NumArray(vector<int>& nums) {
        n = nums.size();
        tree.resize(4 * n);
        buildTree(nums, 0, 0, n - 1);
    }
    
    void update(int index, int val) {
        updateTree(0, 0, n - 1, index, val);
    }
    
    int sumRange(int left, int right) {
        return queryTree(0, 0, n - 1, left, right);
    }
};
```

---

## Common Mistakes & Tips

### Common Mistakes

1. **Not checking for null**
   ```cpp
   // WRONG
   int height(TreeNode* root) {
       return 1 + max(height(root->left), height(root->right));
   }
   
   // CORRECT
   int height(TreeNode* root) {
       if (!root) return 0;
       return 1 + max(height(root->left), height(root->right));
   }
   ```

2. **Confusing leaf nodes**
   ```cpp
   // WRONG: null is not a leaf
   bool isLeaf = !root->left || !root->right;
   
   // CORRECT
   bool isLeaf = !root->left && !root->right;
   ```

3. **Stack overflow in recursion**
    - Always consider iterative approach for very deep trees
    - Use tail recursion when possible

4. **Forgetting to backtrack**
   ```cpp
   // WRONG
   void dfs(TreeNode* node, vector<int>& path) {
       path.push_back(node->val);
       dfs(node->left, path);
       dfs(node->right, path);
       // Missing: path.pop_back();
   }
   ```

5. **Incorrect BST validation**
    - Don't just compare with immediate children
    - Pass valid range down the tree

6. **Modifying tree during traversal carelessly**
    - Store references before modifying structure
    - Be careful with pointer invalidation

### Best Practices

1. **Choose right traversal**
    - Preorder: Copy tree, prefix expression
    - Inorder: BST sorted output, expression tree
    - Postorder: Delete tree, postfix expression
    - Level-order: Level-wise problems, shortest path

2. **Space optimization**
    - Use iteration instead of recursion when possible
    - Morris traversal for O(1) space
    - Reuse existing pointers

3. **Edge cases to test**
    - Empty tree (nullptr)
    - Single node
    - Only left children (skewed left)
    - Only right children (skewed right)
    - Complete binary tree
    - Perfect binary tree

4. **Debugging tips**
    - Print tree level by level
    - Visualize small examples
    - Check base cases first
    - Use assertions

5. **Optimization techniques**
    - Memoization for overlapping subproblems
    - Early termination when answer found
    - Hash maps for O(1) lookups
    - Avoid repeated calculations

6. **Code organization**
    - Separate helper functions
    - Clear variable names
    - Comments for complex logic
    - Consistent style

### Interview Tips

1. **Clarify the problem**
    - Is it a BST or general binary tree?
    - Are values unique?
    - Can tree be empty?
    - What's the expected size?

2. **Start with examples**
    - Draw small tree
    - Trace through algorithm
    - Consider edge cases

3. **Discuss approach**
    - Explain traversal choice
    - Mention time/space complexity
    - Discuss trade-offs

4. **Write clean code**
    - Handle null checks first
    - Use meaningful names
    - Add comments for clarity

5. **Test thoroughly**
    - Walk through with example
    - Check edge cases
    - Verify complexity

---

## Problem Categories Summary

### Time Complexity Patterns

- Single traversal: O(n)
- With map/set: O(n log n) or O(n)
- Nested traversal: O(n²) - usually can optimize
- BST operations: O(h) average, O(n) worst

### Space Complexity Patterns

- Recursion: O(h) - height of tree
- BFS queue: O(w) - width of tree
- Extra storage: O(n) - for all nodes
- Morris/Optimized: O(1)

---

This comprehensive guide covers all major binary tree techniques needed for LeetCode problems. Practice problems in each
category to build intuition and pattern recognition. Focus on understanding the core concepts and adapting them to new
problems rather than memorizing solutions.