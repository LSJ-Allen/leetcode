# Binary Search Techniques - Complete Guide

## Table of Contents

1. [Core Concept](#core-concept)
2. [Variations](#variations)
    - [Classic Binary Search](#1-classic-binary-search)
    - [Binary Search on Answer](#2-binary-search-on-answer)
    - [Rotated Array Search](#3-rotated-array-search)
    - [2D Matrix Search](#4-2d-matrix-search)
    - [Search with Duplicates](#5-search-with-duplicates)
3. [Problem Patterns](#problem-patterns)
    - [Pattern 1: Find Exact Match](#pattern-1-find-exact-match)
    - [Pattern 2: Find Boundary (First/Last Occurrence)](#pattern-2-find-boundary-firstlast-occurrence)
    - [Pattern 3: Minimize/Maximize Answer](#pattern-3-minimizemaximize-answer)
    - [Pattern 4: Search in Modified Sorted Array](#pattern-4-search-in-modified-sorted-array)
4. [LeetCode Problems](#leetcode-problems)
    - [Easy](#easy)
    - [Medium](#medium)
    - [Hard](#hard)
5. [Common Mistakes & Tips](#common-mistakes--tips)

---

## Core Concept

Binary search is a divide-and-conquer algorithm that finds a target value or condition in a sorted space by repeatedly
halving the search interval.

**Visual Representation:**

```
Array: [1, 3, 5, 7, 9, 11, 13, 15]
Target: 7

Step 1: [1, 3, 5, 7, 9, 11, 13, 15]
         L        M              R
         7 < 9, search left

Step 2: [1, 3, 5, 7, 9, 11, 13, 15]
         L     M  R
         7 > 5, search right

Step 3: [1, 3, 5, 7, 9, 11, 13, 15]
               L  M
               Found!
```

**Complexity:**

- **Time:** O(log n) - halves search space each iteration
- **Space:** O(1) for iterative, O(log n) for recursive

**Core Template:**

```cpp
int binarySearch(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;  // Avoid overflow
        
        if (nums[mid] == target) {
            return mid;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;  // Not found
}
```

---

## Variations

### 1. Classic Binary Search

**When to use:** Find exact target in sorted array

**Implementation Pattern:**

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) return mid;
        if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    
    return -1;
}
```

**Common Pitfalls:**

- Using `(left + right) / 2` can cause integer overflow for large arrays
- Wrong loop condition: `while (left < right)` vs `while (left <= right)`
- Off-by-one errors in boundary updates

---

### 2. Binary Search on Answer

**When to use:** Find minimum/maximum value that satisfies a condition (not searching in array, searching in answer
space)

**Implementation Pattern:**

```cpp
// Template for "minimize maximum" or "maximize minimum"
int binarySearchOnAnswer(/* parameters */) {
    int left = minPossible, right = maxPossible;
    int answer = right;  // or left, depending on problem
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (isValid(mid)) {  // Can we achieve this answer?
            answer = mid;
            // Try to optimize further
            right = mid - 1;  // For minimize problems
            // left = mid + 1;  // For maximize problems
        } else {
            left = mid + 1;  // Need larger value
            // right = mid - 1;  // Need smaller value
        }
    }
    
    return answer;
}
```

**Example Code Template:**

```cpp
// LeetCode 410: Split Array Largest Sum
bool canSplit(vector<int>& nums, int k, int maxSum) {
    int subarrays = 1, currentSum = 0;
    for (int num : nums) {
        if (currentSum + num > maxSum) {
            subarrays++;
            currentSum = num;
            if (subarrays > k) return false;
        } else {
            currentSum += num;
        }
    }
    return true;
}

int splitArray(vector<int>& nums, int k) {
    int left = *max_element(nums.begin(), nums.end());
    int right = accumulate(nums.begin(), nums.end(), 0);
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (canSplit(nums, k, mid)) {
            right = mid;  // Can achieve, try smaller
        } else {
            left = mid + 1;  // Can't achieve, need larger
        }
    }
    
    return left;
}
```

**Common Pitfalls:**

- Not identifying the search space correctly (what are min/max possible answers?)
- Wrong `isValid()` logic
- Choosing wrong half to search when valid answer found

---

### 3. Rotated Array Search

**When to use:** Search in sorted array that has been rotated

**Implementation Pattern:**

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) return mid;
        
        // Determine which half is sorted
        if (nums[left] <= nums[mid]) {
            // Left half is sorted
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;  // Target in sorted left half
            } else {
                left = mid + 1;   // Target in right half
            }
        } else {
            // Right half is sorted
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;   // Target in sorted right half
            } else {
                right = mid - 1;  // Target in left half
            }
        }
    }
    
    return -1;
}
```

**Common Pitfalls:**

- Forgetting `nums[left] <= nums[mid]` needs `<=` (for single element case)
- Wrong range checking in sorted half
- Not handling duplicates properly (requires different approach)

---

### 4. 2D Matrix Search

**When to use:** Search in row-wise and column-wise sorted 2D matrix

**Implementation Pattern (Staircase Search):**

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;
    
    int row = 0, col = matrix[0].size() - 1;  // Start top-right
    
    while (row < matrix.size() && col >= 0) {
        if (matrix[row][col] == target) {
            return true;
        } else if (matrix[row][col] > target) {
            col--;  // Move left (decrease values)
        } else {
            row++;  // Move down (increase values)
        }
    }
    
    return false;
}
```

**Alternative (Binary Search on Each Row):**

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    for (auto& row : matrix) {
        if (binarySearch(row, target)) return true;
    }
    return false;
}
// Time: O(m * log n) vs O(m + n) for staircase
```

**Common Pitfalls:**

- Starting from top-left or bottom-right (both directions same, can't eliminate)
- Not understanding why top-right/bottom-left work (opposite direction properties)
- Choosing less optimal O(m log n) over O(m + n) approach

---

### 5. Search with Duplicates

**When to use:** Binary search with duplicate elements

**Implementation Pattern:**

```cpp
// Find first occurrence
int findFirst(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) {
            result = mid;
            right = mid - 1;  // Continue searching left
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

// Find last occurrence
int findLast(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) {
            result = mid;
            left = mid + 1;  // Continue searching right
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}
```

**Common Pitfalls:**

- Returning immediately when target found (need to find boundary)
- Wrong direction after finding target (left for first, right for last)

---

## Problem Patterns

### Pattern 1: Find Exact Match

**Recognition Features:**

- Sorted array given
- Need to find specific target value
- Return index or -1

**Solution Template:**

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

**Example Problems:**

- LeetCode 704: Binary Search
- LeetCode 35: Search Insert Position

---

### Pattern 2: Find Boundary (First/Last Occurrence)

**Recognition Features:**

- Array may have duplicates
- Need to find first/last position of target
- Or find insertion point

**Solution Template:**

```cpp
int findBoundary(vector<int>& nums, int target, bool findFirst) {
    int left = 0, right = nums.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) {
            result = mid;
            if (findFirst) right = mid - 1;
            else left = mid + 1;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}
```

**Example Problems:**

- LeetCode 34: Find First and Last Position
- LeetCode 35: Search Insert Position

---

### Pattern 3: Minimize/Maximize Answer

**Recognition Features:**

- Not searching for an exact match in the array
- Questions like "minimize the maximum" or "maximize the minimum"
- Need to check if answer is valid (monotonic property)
- The middle element is included in left or right pointer update which leads to the loop being terminated on left <
  right instead of left <= right.

**Solution Template:**

```cpp
bool isValid(int answer) {
    // Check if this answer satisfies constraints
}

int binarySearchAnswer() {
    int left = minPossible, right = maxPossible;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (isValid(mid)) {
            right = mid;  // For minimize problems
        } else {
            left = mid + 1;
        }
    }
    
    return left;
}
```

- Need to think about whether mid should be included. For example, in a find min rotated array problem:

```cpp
if (nums[mid] < nums[right]) {
    right = mid;
} else {
    left = mid + 1;
}
```

**Example Problems:**

- LeetCode 410: Split Array Largest Sum
- LeetCode 875: Koko Eating Bananas
- LeetCode 1011: Capacity To Ship Packages

---

### Pattern 4: Search in Modified Sorted Array

**Recognition Features:**

- Array was sorted but then modified (rotated, partially sorted)
- Still need O(log n) solution
- Need to identify which portion is sorted

**Solution Template:**

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        
        // Identify sorted half
        if (nums[left] <= nums[mid]) {
            // Left half sorted
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            // Right half sorted
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return -1;
}
```

**Example Problems:**

- LeetCode 33: Search in Rotated Sorted Array
- LeetCode 153: Find Minimum in Rotated Sorted Array
- LeetCode 81: Search in Rotated Sorted Array II (with duplicates)

---

## LeetCode Problems

### Easy

#### **[704] Binary Search**

- **Difficulty:** Easy
- **Pattern:** Find Exact Match
- **Solution Approach:** Classic binary search implementation
- **Code Template:**

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

- **Common Mistakes:**
    - Using `(left + right) / 2` - can overflow
    - Wrong loop condition `while (left < right)`
- **Related:** 35, 374, 278

---

#### **[35] Search Insert Position**

- **Difficulty:** Easy
- **Pattern:** Find Boundary
- **Solution Approach:** Binary search returns insertion point when not found
- **Code Template:**

```cpp
int searchInsert(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    
    return left;  // left is the insertion position
}
```

- **Common Mistakes:**
    - Returning wrong index when not found
    - Not understanding why `left` is insertion point
- **Related:** 704, 34

---

#### **[278] First Bad Version**

- **Difficulty:** Easy
- **Pattern:** Find Boundary
- **Solution Approach:** Binary search for first occurrence with API call
- **Code Template:**

```cpp
int firstBadVersion(int n) {
    int left = 1, right = n;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (isBadVersion(mid)) {
            right = mid;  // First bad is at or before mid
        } else {
            left = mid + 1;  // First bad is after mid
        }
    }
    
    return left;
}
```

- **Common Mistakes:**
    - Using `while (left <= right)` causes infinite loop
    - Not minimizing API calls
- **Related:** 34, 69

---

### Medium

#### **[4] Median of Two Sorted Arrays**

- **Difficulty:** Hard (but classic)
- **Pattern:** Binary Search on Index
- **Solution Approach:** Binary search on partition point
- **Code Template:**

```cpp
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    if (nums1.size() > nums2.size()) {
        return findMedianSortedArrays(nums2, nums1);
    }
    
    int m = nums1.size(), n = nums2.size();
    int left = 0, right = m;
    
    while (left <= right) {
        int i = left + (right - left) / 2;
        int j = (m + n + 1) / 2 - i;
        
        int maxLeft1 = (i == 0) ? INT_MIN : nums1[i - 1];
        int minRight1 = (i == m) ? INT_MAX : nums1[i];
        int maxLeft2 = (j == 0) ? INT_MIN : nums2[j - 1];
        int minRight2 = (j == n) ? INT_MAX : nums2[j];
        
        if (maxLeft1 <= minRight2 && maxLeft2 <= minRight1) {
            if ((m + n) % 2 == 0) {
                return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2.0;
            } else {
                return max(maxLeft1, maxLeft2);
            }
        } else if (maxLeft1 > minRight2) {
            right = i - 1;
        } else {
            left = i + 1;
        }
    }
    
    return 0.0;
}
```

- **Common Mistakes:**
    - Wrong formula: `j = (m + n) / 2 - i` (should be `(m + n + 1) / 2`)
    - Out of bounds access without checks
    - Wrong median for odd length (use max of left, not min of right)
- **Related:** 240, 378

---

#### **[33] Search in Rotated Sorted Array**

- **Difficulty:** Medium
- **Pattern:** Modified Sorted Array
- **Solution Approach:** Identify sorted half, check if target in range
- **Code Template:**

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) return mid;
        
        if (nums[left] <= nums[mid]) {
            // Left half sorted
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            // Right half sorted
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return -1;
}
```

- **Common Mistakes:**
    - Using `nums[left] < nums[mid]` instead of `<=`
    - Wrong range checks in sorted half
    - Confusing which half to search
- **Related:** 81, 153, 154

---

#### **[34] Find First and Last Position**

- **Difficulty:** Medium
- **Pattern:** Find Boundary
- **Solution Approach:** Two binary searches - one for first, one for last
- **Code Template:**

```cpp
int findBound(vector<int>& nums, int target, bool isFirst) {
    int left = 0, right = nums.size() - 1;
    int result = -1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) {
            result = mid;
            if (isFirst) {
                right = mid - 1;  // Continue left
            } else {
                left = mid + 1;   // Continue right
            }
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return result;
}

vector<int> searchRange(vector<int>& nums, int target) {
    return {findBound(nums, target, true), 
            findBound(nums, target, false)};
}
```

- **Common Mistakes:**
    - Returning immediately when found (need boundaries)
    - Searching wrong direction for first/last
- **Related:** 278, 704

---

#### **[74] Search a 2D Matrix** (fully sorted)

- **Difficulty:** Medium
- **Pattern:** Classic Binary Search (treat as 1D)
- **Solution Approach:** Map 2D index to 1D and binary search
- **Code Template:**

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;
    
    int m = matrix.size(), n = matrix[0].size();
    int left = 0, right = m * n - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        int val = matrix[mid / n][mid % n];  // Convert to 2D
        
        if (val == target) return true;
        if (val < target) left = mid + 1;
        else right = mid - 1;
    }
    
    return false;
}
```

- **Common Mistakes:**
    - Wrong 2D index calculation
    - Not recognizing it's fully sorted (different from 240)
- **Related:** 240

---

#### **[240] Search a 2D Matrix II** (row & col sorted)

- **Difficulty:** Medium
- **Pattern:** 2D Matrix Search (Staircase)
- **Solution Approach:** Start from top-right or bottom-left corner
- **Code Template:**

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    if (matrix.empty() || matrix[0].empty()) return false;
    
    int row = 0, col = matrix[0].size() - 1;  // Top-right
    
    while (row < matrix.size() && col >= 0) {
        if (matrix[row][col] == target) {
            return true;
        } else if (matrix[row][col] > target) {
            col--;  // Go left (decrease)
        } else {
            row++;  // Go down (increase)
        }
    }
    
    return false;
}
```

- **Common Mistakes:**
    - Starting from top-left or bottom-right (doesn't work!)
    - Not understanding the BST property from corners
    - Using O(m log n) instead of O(m + n) approach
- **Key Insight:** Top-right acts like BST root: left=smaller, down=larger
- **Related:** 74, 378

---

#### **[153] Find Minimum in Rotated Sorted Array**

- **Difficulty:** Medium
- **Pattern:** Modified Sorted Array
- **Solution Approach:** Binary search comparing mid with right boundary
- **Code Template:**

```cpp
int findMin(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] > nums[right]) {
            // Minimum is in right half
            left = mid + 1;
        } else {
            // Minimum is in left half (including mid)
            right = mid;
        }
    }
    
    return nums[left];
}
```

- **Common Mistakes:**
    - Comparing with `nums[left]` instead of `nums[right]`
    - Using `while (left <= right)` causes issues
- **Related:** 33, 154

---

#### **[162] Find Peak Element**

- **Difficulty:** Medium
- **Pattern:** Binary Search on Answer
- **Solution Approach:** Move towards higher neighbor
- **Code Template:**

```cpp
int findPeakElement(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] > nums[mid + 1]) {
            // Peak is on left (including mid)
            right = mid;
        } else {
            // Peak is on right
            left = mid + 1;
        }
    }
    
    return left;
}
```

- **Common Mistakes:**
    - Out of bounds when checking `mid + 1`
    - Not understanding why moving to higher side works
- **Related:** 852

---

#### **[410] Split Array Largest Sum**

- **Difficulty:** Hard
- **Pattern:** Binary Search on Answer
- **Solution Approach:** Binary search on maximum subarray sum
- **Code Template:**

```cpp
bool canSplit(vector<int>& nums, int k, int maxSum) {
    int subarrays = 1, currentSum = 0;
    
    for (int num : nums) {
        if (currentSum + num > maxSum) {
            subarrays++;
            currentSum = num;
            if (subarrays > k) return false;
        } else {
            currentSum += num;
        }
    }
    
    return true;
}

int splitArray(vector<int>& nums, int k) {
    int left = *max_element(nums.begin(), nums.end());
    int right = accumulate(nums.begin(), nums.end(), 0);
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (canSplit(nums, k, mid)) {
            right = mid;  // Can achieve, try smaller
        } else {
            left = mid + 1;  // Can't achieve, need larger
        }
    }
    
    return left;
}
```

- **Common Mistakes:**
    - Wrong search space (min should be max element, not 0)
    - Incorrect `canSplit()` greedy logic
    - Using `while (left <= right)` with wrong updates
- **Related:** 875, 1011, 1482

---

#### **[875] Koko Eating Bananas**

- **Difficulty:** Medium
- **Pattern:** Binary Search on Answer
- **Solution Approach:** Binary search on eating speed
- **Code Template:**

```cpp
bool canFinish(vector<int>& piles, int h, int k) {
    long long hours = 0;
    for (int pile : piles) {
        hours += (pile + k - 1) / k;  // Ceiling division
        if (hours > h) return false;
    }
    return true;
}

int minEatingSpeed(vector<int>& piles, int h) {
    int left = 1, right = *max_element(piles.begin(), piles.end());
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (canFinish(piles, h, mid)) {
            right = mid;  // Can finish, try slower
        } else {
            left = mid + 1;  // Too slow, need faster
        }
    }
    
    return left;
}
```

- **Common Mistakes:**
    - Not using ceiling division `(pile + k - 1) / k`
    - Wrong search space (should start at 1, not 0)
    - Integer overflow in hours calculation
- **Related:** 410, 1011

---

### Hard

#### **[4] Median of Two Sorted Arrays**

(See Medium section - listed here as it's marked Hard on LeetCode)

---

#### **[154] Find Minimum in Rotated Sorted Array II** (with duplicates)

- **Difficulty:** Hard
- **Pattern:** Modified Sorted Array with Duplicates
- **Solution Approach:** Binary search with special handling for duplicates
- **Code Template:**

```cpp
int findMin(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] > nums[right]) {
            left = mid + 1;
        } else if (nums[mid] < nums[right]) {
            right = mid;
        } else {
            // nums[mid] == nums[right], can't determine which half
            right--;  // Reduce search space by 1
        }
    }
    
    return nums[left];
}
```

- **Common Mistakes:**
    - Not handling duplicates case (worst case O(n))
    - Wrong duplicate handling strategy
- **Related:** 153, 33, 81

---

#### **[81] Search in Rotated Sorted Array II** (with duplicates)

- **Difficulty:** Medium (but harder than 33)
- **Pattern:** Modified Sorted Array with Duplicates
- **Solution Approach:** Binary search with duplicate handling
- **Code Template:**

```cpp
bool search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (nums[mid] == target) return true;
        
        // Handle duplicates
        if (nums[left] == nums[mid] && nums[mid] == nums[right]) {
            left++;
            right--;
            continue;
        }
        
        if (nums[left] <= nums[mid]) {
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    
    return false;
}
```

- **Common Mistakes:**
    - Not handling the case when left, mid, right all equal
    - Worst case degrades to O(n)
- **Related:** 33, 154

---

## Common Mistakes & Tips

### Common Mistakes

1. **Integer Overflow**
   ```cpp
   // ❌ WRONG - can overflow
   int mid = (left + right) / 2;
   
   // ✅ CORRECT
   int mid = left + (right - left) / 2;
   ```

2. **Wrong Loop Condition**
   ```cpp
   // For finding exact match
   while (left <= right)  // ✅ Correct
   
   // For finding boundary/minimum
   while (left < right)   // ✅ Correct (avoids infinite loop)
   ```

3. **Off-by-One Errors**
   ```cpp
   // When found target but need first occurrence
   if (nums[mid] == target) {
       result = mid;
       right = mid - 1;  // ✅ Continue searching left
       // NOT: right = mid (infinite loop!)
   }
   ```

4. **Accessing Out of Bounds**
   ```cpp
   // ❌ WRONG - mid can equal size
   int minRight = nums[mid];
   
   // ✅ CORRECT
   int minRight = (mid == nums.size()) ? INT_MAX : nums[mid];
   ```

5. **Wrong Median for Odd Length (LeetCode 4)**
   ```cpp
   // ❌ WRONG
   return min(minRight1, minRight2);
   
   // ✅ CORRECT - left partition has one more element
   return max(maxLeft1, maxLeft2);
   ```

6. **Starting from Wrong Corner (LeetCode 240)**
   ```cpp
   // ❌ WRONG - both directions increase/decrease
   int row = 0, col = 0;  // Top-left
   int row = m-1, col = n-1;  // Bottom-right
   
   // ✅ CORRECT - opposite directions
   int row = 0, col = n-1;  // Top-right
   int row = m-1, col = 0;  // Bottom-left
   ```

### Best Practices

1. **Always use `left + (right - left) / 2`** to avoid overflow

2. **Clearly define search space**
    - What are valid values for left and right?
    - Is right inclusive or exclusive?

3. **Choose correct loop condition**
    - `while (left <= right)` for finding exact match
    - `while (left < right)` for finding boundary/minimum

4. **Handle edge cases**
    - Empty array
    - Single element
    - All duplicates
    - Target at boundaries

5. **For "binary search on answer" problems**
    - Identify what you're searching for (the answer space)
    - Write `isValid()` function first
    - Determine if minimizing or maximizing

6. **Verify boundary updates**
    - When moving `left = mid + 1`, always +1
    - When moving `right = mid - 1`, always -1
    - When keeping mid as potential answer: `right = mid` or `left = mid`

7. **Test with examples**
    - Single element: `[5]`
    - Two elements: `[1, 2]`
    - Target at start/end
    - Target not present

8. **For rotated arrays**
    - Always identify which half is sorted first
    - Use `<=` in comparison: `nums[left] <= nums[mid]`
    - Check if target is in sorted half's range

9. **For 2D matrix problems**
    - Understand the sorting properties
    - Choose appropriate starting corner
    - Remember: top-right/bottom-left work, top-left/bottom-right don't

10. **Debug strategy**
    - Print left, mid, right each iteration
    - Verify which branch is taken
    - Check boundary updates are correct

---

## Quick Reference Table

| Problem Type     | Loop Condition  | When Found                      | Example |
|------------------|-----------------|---------------------------------|---------|
| Exact match      | `left <= right` | `return mid`                    | LC 704  |
| First occurrence | `left <= right` | `result = mid; right = mid - 1` | LC 34   |
| Last occurrence  | `left <= right` | `result = mid; left = mid + 1`  | LC 34   |
| Minimize answer  | `left < right`  | `right = mid` if valid          | LC 410  |
| Maximize answer  | `left < right`  | `left = mid` if valid           | LC -    |
| Rotated array    | `left <= right` | Check sorted half               | LC 33   |
| 2D matrix        | While in bounds | `return true`                   | LC 240  |

---

## Interview Tips

1. **Always clarify**
    - Is array sorted? Fully or partially?
    - Are there duplicates?
    - What to return if not found?

2. **Start with brute force**
    - Explain O(n) solution first
    - Then optimize to O(log n)

3. **Draw examples**
    - Visualize the search space
    - Show how it shrinks

4. **State complexity**
    - Time and space
    - Explain why it's O(log n)

5. **Handle edge cases**
    - Empty array
    - Single element
    - All same values
    - Target outside range