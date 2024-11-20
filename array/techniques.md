# Array Problem-Solving Techniques Guide

## 1. Two Pointers Technique

### Core Concept
A method that uses two references to traverse an array, processing elements by comparing or manipulating them based on their relationship.

### Variations & Use Cases

1. **Fast & Slow Pointers (Floyd's Technique)**
   - **When to use:** 
     - Finding middle element
     - Cycle detection
     - Finding duplicates
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(fast && fast->next) {
         slow = slow->next;
         fast = fast->next->next;
     }
     ```

2. **Left & Right Pointers (Converging Pointers)**
   - **When to use:**
     - Searching pairs in sorted array
     - Partitioning arrays
     - Two-sum variations
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(left < right) {
         if(condition) left++;
         else right--;
     }
     ```

3. **Same Direction Pointers**
   - **When to use:**
     - In-place array modifications
     - Removing duplicates
     - Maintaining window
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     for(fast = 0; fast < n; fast++) {
         if(condition) {
             swap(nums[slow++], nums[fast]);
         }
     }
     ```

### LeetCode Problems

#### Easy
1. **[167] Two Sum II - Input Array Is Sorted**
   - **Approach:** Left & Right pointers
   - **Pattern:** Compare sum with target
   ```cpp
   while(left < right) {
       sum = nums[left] + nums[right];
       if(sum == target) return {left+1, right+1};
       sum < target ? left++ : right--;
   }
   ```

2. **[283] Move Zeroes**
   - **Approach:** Same Direction pointers
   - **Pattern:** Partition non-zeros and zeros
   ```cpp
   for(int i = 0; i < n; i++) {
       if(nums[i] != 0) {
           swap(nums[nonZeroPos++], nums[i]);
       }
   }
   ```

#### Medium
1. **[11] Container With Most Water**
   - **Approach:** Left & Right pointers
   - **Pattern:** Area maximization
   ```cpp
   while(left < right) {
       maxArea = max(maxArea, min(height[left], height[right]) * (right-left));
       height[left] < height[right] ? left++ : right--;
   }
   ```

2. **[75] Sort Colors**
   - **Approach:** Three pointers (Dutch Flag)
   - **Pattern:** Three-way partitioning
   ```cpp
   while(mid <= high) {
       if(nums[mid] == 0) swap(nums[low++], nums[mid++]);
       else if(nums[mid] == 2) swap(nums[mid], nums[high--]);
       else mid++;
   }
   ```

#### Hard
1. **[42] Trapping Rain Water**
   - **Approach:** Left & Right pointers with max tracking
   - **Pattern:** Two-way comparison
   ```cpp
   while(left < right) {
       if(height[left] < height[right]) {
           height[left] >= leftMax ? leftMax = height[left] : water += leftMax - height[left];
           left++;
       } else {
           // Similar for right side
       }
   }
   ```

## 2. Sliding Window Technique

### Core Concept
A technique to process array elements in a window that slides through the array, maintaining a set of elements that satisfy certain conditions.

### Variations & Use Cases

1. **Fixed Size Window**
   - **When to use:**
     - Subarray of exact size k
     - Average/sum calculations
     - Pattern matching of fixed length
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     for(int i = 0; i < k; i++) windowSum += nums[i];
     for(int i = k; i < n; i++) {
         windowSum += nums[i] - nums[i-k];
         // Process window
     }
     ```

2. **Variable Size Window**
   - **When to use:**
     - Longest/shortest subarray meeting condition
     - Sum/product constraints
     - String pattern matching
   - **Time:** O(n), **Space:** O(1) or O(k)
   - **Example Pattern:**
     ```cpp
     while(right < n) {
         // Add element to window
         while(!valid()) {
             // Remove element from window
             left++;
         }
         // Update result
         right++;
     }
     ```

## 3. Prefix Sum Technique

### Core Concept
A preprocessing technique that builds an auxiliary array containing cumulative sums, enabling efficient range queries and subarray calculations.

### Variations & Use Cases

1. **1D Prefix Sum**
   - **When to use:**
     - Range sum queries
     - Subarray sum calculations
     - Cumulative statistics
   - **Time:** O(1) per query after O(n) preprocessing
   - **Example Pattern:**
     ```cpp
     for(int i = 1; i < n; i++) {
         prefix[i] = prefix[i-1] + nums[i];
     }
     // Query: sum(i,j) = prefix[j] - prefix[i-1]
     ```

2. **2D Prefix Sum**
   - **When to use:**
     - Rectangle sum queries
     - 2D range calculations
     - Matrix region processing
   - **Time:** O(1) per query after O(m*n) preprocessing
   - **Example Pattern:**
     ```cpp
     for(int i = 1; i <= m; i++) {
         for(int j = 1; j <= n; j++) {
             dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i-1][j-1];
         }
     }
     ```

3. **Difference Array**
   - **When to use:**
     - Range update operations
     - Interval modifications
     - Frequency counting in ranges
   - **Time:** O(1) per update, O(n) for final array
   - **Example Pattern:**
     ```cpp
     void addToRange(int l, int r, int val) {
         diff[l] += val;
         diff[r + 1] -= val;
     }
     ```

### LeetCode Problems

#### Easy
1. **[303] Range Sum Query - Immutable**
   - **Approach:** 1D Prefix Sum
   - **Pattern:** Range query optimization
   ```cpp
   vector<int> prefixSum;
   NumArray(vector<int>& nums) {
       prefixSum = vector<int>(nums.size() + 1);
       for(int i = 0; i < nums.size(); i++)
           prefixSum[i + 1] = prefixSum[i] + nums[i];
   }
   int sumRange(int i, int j) {
       return prefixSum[j + 1] - prefixSum[i];
   }
   ```

#### Medium
1. **[304] Range Sum Query 2D - Immutable**
   - **Approach:** 2D Prefix Sum
   - **Pattern:** 2D range query
   ```cpp
   NumMatrix(vector<vector<int>>& matrix) {
       if(matrix.empty()) return;
       dp.resize(matrix.size() + 1, vector<int>(matrix[0].size() + 1));
       for(int i = 1; i <= matrix.size(); i++)
           for(int j = 1; j <= matrix[0].size(); j++)
               dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i-1][j-1];
   }
   ```

2. **[560] Subarray Sum Equals K**
   - **Approach:** Prefix Sum with Hash Map
   - **Pattern:** Count subarrays with sum
   ```cpp
   for(int i = 0; i < nums.size(); i++) {
       sum += nums[i];
       if(map.find(sum - k) != map.end())
           count += map[sum - k];
       map[sum]++;
   }
   ```
## 4. Kadane's Algorithm

### Core Concept
A dynamic programming approach for solving maximum subarray problems by maintaining local and global maximum values.

### Variations & Use Cases

1. **Basic Kadane's**
   - **When to use:**
     - Maximum sum subarray
     - Continuous sequence optimization
     - One-dimensional array optimization
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     int maxSoFar = nums[0], maxEndingHere = nums[0];
     for(int i = 1; i < n; i++) {
         maxEndingHere = max(nums[i], maxEndingHere + nums[i]);
         maxSoFar = max(maxSoFar, maxEndingHere);
     }
     ```

2. **Circular Array Kadane's**
   - **When to use:**
     - Maximum sum in circular array
     - Wrapping around array ends
     - Circular sequence optimization
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     int normalMax = kadane(nums);
     int totalSum = accumulate(nums.begin(), nums.end(), 0);
     int circularMax = totalSum + kadane(invertedArray);
     return max(normalMax, circularMax);
     ```

3. **Product Version**
   - **When to use:**
     - Maximum product subarray
     - Handle negative numbers
     - Product sequence optimization
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     int maxProd = nums[0], minProd = nums[0], result = nums[0];
     for(int i = 1; i < n; i++) {
         int temp = max({nums[i], maxProd*nums[i], minProd*nums[i]});
         minProd = min({nums[i], maxProd*nums[i], minProd*nums[i]});
         maxProd = temp;
         result = max(result, maxProd);
     }
     ```

### LeetCode Problems

#### Easy
1. **[53] Maximum Subarray**
   - **Approach:** Basic Kadane's
   - **Pattern:** Track local and global maximum
   ```cpp
   int maxSubArray(vector<int>& nums) {
       int maxSum = nums[0], currSum = nums[0];
       for(int i = 1; i < nums.size(); i++) {
           currSum = max(nums[i], currSum + nums[i]);
           maxSum = max(maxSum, currSum);
       }
       return maxSum;
   }
   ```

#### Medium
1. **[152] Maximum Product Subarray**
   - **Approach:** Product Version Kadane's
   - **Pattern:** Track both max and min products
   ```cpp
   int maxProduct(vector<int>& nums) {
       int maxProd = nums[0], minProd = nums[0], result = nums[0];
       for(int i = 1; i < nums.size(); i++) {
           int temp = maxProd;
           maxProd = max({nums[i], maxProd*nums[i], minProd*nums[i]});
           minProd = min({nums[i], temp*nums[i], minProd*nums[i]});
           result = max(result, maxProd);
       }
       return result;
   }
   ```

2. **[918] Maximum Sum Circular Subarray**
   - **Approach:** Circular Array Kadane's
   - **Pattern:** Consider both normal and circular cases
   ```cpp
   int maxSubarraySumCircular(vector<int>& nums) {
       int totalSum = 0, currMax = 0, currMin = 0;
       int maxSum = nums[0], minSum = nums[0];
       
       for(int num : nums) {
           totalSum += num;
           currMax = max(num, currMax + num);
           currMin = min(num, currMin + num);
           maxSum = max(maxSum, currMax);
           minSum = min(minSum, currMin);
       }
       
       return maxSum > 0 ? max(maxSum, totalSum - minSum) : maxSum;
   }
   ```

## 5. Binary Search on Arrays

### Core Concept
A divide-and-conquer technique that repeatedly divides the search interval in half, typically used on sorted arrays or for optimization problems.

### Variations & Use Cases

1. **Classic Binary Search**
   - **When to use:**
     - Finding element in sorted array
     - Finding insertion position
     - Checking element existence
   - **Time:** O(log n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(left <= right) {
         mid = left + (right - left) / 2;
         if(nums[mid] == target) return mid;
         if(nums[mid] < target) left = mid + 1;
         else right = mid - 1;
     }
     ```

2. **Binary Search on Answer**
   - **When to use:**
     - Optimization problems
     - Minimization/maximization
     - Finding threshold values
   - **Time:** O(log n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(left < right) {
         mid = left + (right - left) / 2;
         if(isValid(mid)) right = mid;
         else left = mid + 1;
     }
     ```

3. **Rotated Array Binary Search**
   - **When to use:**
     - Searching in rotated sorted array
     - Finding rotation point
     - Modified sorted arrays
   - **Time:** O(log n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(left <= right) {
         mid = left + (right - left) / 2;
         if(nums[mid] == target) return mid;
         if(nums[left] <= nums[mid]) {
             if(nums[left] <= target && target < nums[mid]) right = mid - 1;
             else left = mid + 1;
         } else {
             if(nums[mid] < target && target <= nums[right]) left = mid + 1;
             else right = mid - 1;
         }
     }
     ```

### LeetCode Problems

#### Easy
1. **[704] Binary Search**
   - **Approach:** Classic Binary Search
   - **Pattern:** Standard implementation
   ```cpp
   int search(vector<int>& nums, int target) {
       int left = 0, right = nums.size() - 1;
       while(left <= right) {
           int mid = left + (right - left) / 2;
           if(nums[mid] == target) return mid;
           if(nums[mid] < target) left = mid + 1;
           else right = mid - 1;
       }
       return -1;
   }
   ```

#### Medium
1. **[33] Search in Rotated Sorted Array**
   - **Approach:** Rotated Array Binary Search
   - **Pattern:** Handle two sorted subarrays
   ```cpp
   int search(vector<int>& nums, int target) {
       int left = 0, right = nums.size() - 1;
       while(left <= right) {
           int mid = left + (right - left) / 2;
           if(nums[mid] == target) return mid;
           if(nums[left] <= nums[mid]) {
               if(nums[left] <= target && target < nums[mid]) right = mid - 1;
               else left = mid + 1;
           } else {
               if(nums[mid] < target && target <= nums[right]) left = mid + 1;
               else right = mid - 1;
           }
       }
       return -1;
   }
   ```

2. **[875] Koko Eating Bananas**
   - **Approach:** Binary Search on Answer
   - **Pattern:** Minimize maximum value
   ```cpp
   int minEatingSpeed(vector<int>& piles, int h) {
       int left = 1, right = *max_element(piles.begin(), piles.end());
       while(left < right) {
           int mid = left + (right - left) / 2;
           if(canEatAll(piles, h, mid)) right = mid;
           else left = mid + 1;
       }
       return left;
   }
   ```

## 6. Stack-based Techniques

### Core Concept
Using a stack to maintain elements in a specific order, typically for finding next/previous greater/smaller elements or maintaining monotonic properties.

### Variations & Use Cases

1. **Monotonic Stack**
   - **When to use:**
     - Next/Previous greater/smaller element
     - Histogram problems
     - Temperature span problems
   - **Time:** O(n), **Space:** O(n)
   - **Example Pattern:**
     ```cpp
     for(int i = 0; i < n; i++) {
         while(!stack.empty() && stack.top() < nums[i]) {
             // Process smaller elements
             stack.pop();
         }
         stack.push(nums[i]);
     }
     ```

2. **Stack with Indices**
   - **When to use:**
     - When position information is needed
     - Calculate spans/distances
     - Rectangle problems
   - **Time:** O(n), **Space:** O(n)
   - **Example Pattern:**
     ```cpp
     for(int i = 0; i < n; i++) {
         while(!stack.empty() && nums[stack.top()] < nums[i]) {
             result[stack.top()] = i - stack.top();
             stack.pop();
         }
         stack.push(i);
     }
     ```

### LeetCode Problems

#### Easy
1. **[496] Next Greater Element I**
   - **Approach:** Monotonic Stack
   - **Pattern:** Find next greater element
   ```cpp
   vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
       stack<int> st;
       unordered_map<int, int> nextGreater;
       
       for(int num : nums2) {
           while(!st.empty() && st.top() < num) {
               nextGreater[st.top()] = num;
               st.pop();
           }
           st.push(num);
       }
       
       vector<int> result;
       for(int num : nums1) {
           result.push_back(nextGreater.count(num) ? nextGreater[num] : -1);
       }
       return result;
   }
   ```

#### Medium
1. **[739] Daily Temperatures**
   - **Approach:** Stack with Indices
   - **Pattern:** Calculate span/distance to next greater
   ```cpp
   vector<int> dailyTemperatures(vector<int>& temperatures) {
       int n = temperatures.size();
       vector<int> result(n);
       stack<int> st;
       
       for(int i = 0; i < n; i++) {
           while(!st.empty() && temperatures[st.top()] < temperatures[i]) {
               result[st.top()] = i - st.top();
               st.pop();
           }
           st.push(i);
       }
       return result;
   }
   ```

#### Hard
1. **[84] Largest Rectangle in Histogram**
   - **Approach:** Stack with Indices
   - **Pattern:** Calculate area using boundaries
   ```cpp
   int largestRectangleArea(vector<int>& heights) {
       stack<int> st;
       int maxArea = 0;
       heights.push_back(0); // Add sentinel
       
       for(int i = 0; i < heights.size(); i++) {
           while(!st.empty() && heights[st.top()] > heights[i]) {
               int height = heights[st.top()];
               st.pop();
               int width = st.empty() ? i : i - st.top() - 1;
               maxArea = max(maxArea, height * width);
           }
           st.push(i);
       }
       return maxArea;
   }
   ```

## 7. Cyclic Sort

### Core Concept
A sorting technique specifically designed for arrays containing numbers in a given range, typically 1 to N. It works by placing each number in its correct position.

### Variations & Use Cases

1. **Basic Cyclic Sort**
   - **When to use:**
     - Array contains numbers 1 to N
     - Need to sort in-place
     - Finding missing numbers
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(i < n) {
         if(nums[i] != i + 1) {
             swap(nums[i], nums[nums[i] - 1]);
         } else {
             i++;
         }
     }
     ```

2. **Modified Cyclic Sort**
   - **When to use:**
     - Array contains numbers 0 to N
     - Finding duplicates
     - Multiple missing numbers
   - **Time:** O(n), **Space:** O(1)
   - **Example Pattern:**
     ```cpp
     while(i < n) {
         if(nums[i] >= 0 && nums[i] < n && nums[i] != nums[nums[i]]) {
             swap(nums[i], nums[nums[i]]);
         } else {
             i++;
         }
     }
     ```

### LeetCode Problems

#### Easy
1. **[268] Missing Number**
   - **Approach:** Modified Cyclic Sort
   - **Pattern:** Find missing number in range [0,n]
   ```cpp
   int missingNumber(vector<int>& nums) {
       int i = 0, n = nums.size();
       while(i < n) {
           if(nums[i] < n && nums[i] != i) {
               swap(nums[i], nums[nums[i]]);
           } else {
               i++;
           }
       }
       
       for(i = 0; i < n; i++) {
           if(nums[i] != i) return i;
       }
       return n;
   }
   ```

#### Medium
1. **[287] Find the Duplicate Number**
   - **Approach:** Cyclic Sort
   - **Pattern:** Find duplicate in range [1,n]
   ```cpp
   int findDuplicate(vector<int>& nums) {
       while(nums[0] != nums[nums[0]]) {
           swap(nums[0], nums[nums[0]]);
       }
       return nums[0];
   }
   ```

2. **[442] Find All Duplicates in an Array**
   - **Approach:** Modified Cyclic Sort
   - **Pattern:** Mark visited positions
   ```cpp
   vector<int> findDuplicates(vector<int>& nums) {
       vector<int> result;
       for(int i = 0; i < nums.size(); i++) {
           int idx = abs(nums[i]) - 1;
           if(nums[idx] > 0) {
               nums[idx] = -nums[idx];
           } else {
               result.push_back(abs(nums[i]));
           }
       }
       return result;
   }
   ```

## 8. Hash Map Technique

### Core Concept
Using hash maps/sets for O(1) lookup time, frequency counting, and mapping relationships between elements.

### Variations & Use Cases

1. **Frequency Counter**
   - **When to use:**
     - Counting element occurrences
     - Finding duplicates
     - Anagram problems
   - **Time:** O(n), **Space:** O(n)
   - **Example Pattern:**
     ```cpp
     unordered_map<int, int> freq;
     for(int num : nums) freq[num]++;
     ```

2. **Value-Index Mapping**
   - **When to use:**
     - Two Sum variations
     - Finding pairs
     - Complementary elements
   - **Time:** O(n), **Space:** O(n)
   - **Example Pattern:**
     ```cpp
     unordered_map<int, int> map;  // value -> index
     for(int i = 0; i < n; i++) {
         if(map.count(target - nums[i])) {
             return {map[target - nums[i]], i};
         }
         map[nums[i]] = i;
     }
     ```

3. **Group/Bucket Organization**
   - **When to use:**
     - Grouping related elements
     - Organizing by property
     - Finding patterns
   - **Time:** O(n), **Space:** O(n)
   - **Example Pattern:**
     ```cpp
     unordered_map<string, vector<string>> groups;
     for(string& str : strs) {
         string key = getKey(str);
         groups[key].push_back(str);
     }
     ```

### LeetCode Problems

#### Easy
1. **[1] Two Sum**
   - **Approach:** Value-Index Mapping
   - **Pattern:** Find pair summing to target
   ```cpp
   vector<int> twoSum(vector<int>& nums, int target) {
       unordered_map<int, int> map;
       for(int i = 0; i < nums.size(); i++) {
           if(map.count(target - nums[i])) {
               return {map[target - nums[i]], i};
           }
           map[nums[i]] = i;
       }
       return {};
   }
   ```

#### Medium
1. **[49] Group Anagrams**
   - **Approach:** Group Organization
   - **Pattern:** Group strings by sorted form
   ```cpp
   vector<vector<string>> groupAnagrams(vector<string>& strs) {
       unordered_map<string, vector<string>> groups;
       for(string& s : strs) {
           string key = s;
           sort(key.begin(), key.end());
           groups[key].push_back(s);
       }
       
       vector<vector<string>> result;
       for(auto& pair : groups) {
           result.push_back(pair.second);
       }
       return result;
   }
   ```

2. **[560] Subarray Sum Equals K**
   - **Approach:** Prefix Sum with Hash Map
   - **Pattern:** Count subarrays with sum k
   ```cpp
   int subarraySum(vector<int>& nums, int k) {
       unordered_map<int, int> sumFreq {{0, 1}};
       int sum = 0, count = 0;
       
       for(int num : nums) {
           sum += num;
           count += sumFreq[sum - k];
           sumFreq[sum]++;
       }
       return count;
   }
   ```
