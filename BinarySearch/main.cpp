#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <cmath>

using namespace std;

namespace Solution {
    // 4. Median of Two Sorted Arrays
    namespace MedianofTwoSortedArrays {
        int binarySearch(const vector<int> &arr1, const vector<int> &arr2) {
            int left = 0, right = static_cast<int>(arr1.size());
            while (left <= right) {
                int mid = (left + right) / 2;
                int j = (arr1.size() + arr2.size()) / 2 - mid;
                int maxLeft1 = mid == 0 ? INT_MIN : arr1[mid - 1];
                int maxLeft2 = j == 0 ? INT_MIN : arr2[j - 1];
                int minRight1 = mid >= arr1.size() ? INT_MAX : arr1[mid];
                int minRight2 = j >= arr2.size() ? INT_MAX : arr2[j];

                if (maxLeft1 > minRight2) {
                    // condition 1 failed, search in left half
                    right = mid - 1;
                } else if (maxLeft2 > minRight1) {
                    // condition 2 failed, search right half
                    left = mid + 1;
                } else {
                    // none of the conditions failed, return
                    return mid;
                }
            }
            return -1;
        }

        double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {
            /**
             * Approach:
             *
             * The key idea is how we can partition each of the two arrays such that the left parts
             * constitute the left part of the merged array and the right parts constitute th right part
             * of the merged array without actually merging them.
             *
             * Example:
             * arr1 = [1, 3, 8, 9, 15] arr2 = [7, 11, 18, 19, 21, 25]
             * merged = [1, 3, 7, 8, 9, 11, 15, 18, 19, 21, 25]
             *                           ^
             *                         median
             * arr1 = [1, 3, 8, 9 | 15] arr2 = [7 | 11, 18, 19, 21, 25]
             *                       i               j
             * Such partition must satisfies:
             *  1. max(left1) <= min(right2)
             *  2. max(left2) <= min(right1)
             *  3. len(left1) + len(left2) - (len(right1) + len(right2) <= 1
             *
             * Let arr1 denotes the smaller array, and i be the index of
             * the partition in the arr1 such that arr1[0:i] = left1 and arr[i:] = right1.
             * len(arr1) = m and len(arr2) = n and m <= n.
             * We can calculate the position j to divide arr2 that satisfies condition 3.
             *      total_len = m + n
             *      half_len = (m + n)/2
             *      i + j = (m + n)/2
             *      j = (m + n)/2 - i
             *
             * Thus the goal is to find an index i with binary search such that
             * condition 1 and 2 are satisfied with j = (m + n)/2 - i. Note that the binary search
             * is on the index rather than the element in the array.
             *
             * If condition 1 fails, left1 contains too many elements and mid needs to be moved to the left
             * half. If condition 2 fails, left1 contains not enough elements and mid needs to be moved to the
             * right half.
             *
             * How to binary search:
             *
             * Example:
             *
             * 1st iteration:
             * arr1 = [1   3   8   9   15]
             * index = 0   1   2   3    4
             *         l
             *                          r
             *                  m
             * j = 5 - 2 = 3
             * arr2 = [7  11  18  19  21  25]
             *                    j
             * left1 = [1 3] left2 = [7 11 18]
             * right1 = [8 9 15] right2 = [19 21 25]
             *
             * condition 1 satisfied
             * contition 2 not satisfied -> means not enough element in left1, move m to right.
             *
             * 2nd iteration:
             * arr1 = [1   3   8   9   15]
             * index = 0   1   2   3    4
             *                     l
             *                          r
             *                     m
             * j = 5 - 3 = 2
             * arr2 = [7  11  18  19  21  25]
             *                 j
             * left1 = [0 1 2] left2 = [7 11]
             * right1 = [9 15] right2 = [18 19 21 25]
             *
             * condition 1 satisfied
             * contition 2 not satisfied -> means not enough element in left1, move m to right.
             *
             * 3rd iteration:
             * arr1 = [1   3   8   9   15]
             * index = 0   1   2   3    4
             *                          l
             *                          r
             *                          m
             * j = 5 - 4 = 1
             * arr2 = [7  11  18  19  21  25]
             *             j
             * left1 = [1 3 8 9] left2 = [7]
             * right1 = [15] right2 = [11  18  19  21  25]
             *
             * condition 1 satisfied
             * contition 2 satisfied
             *
             * A valid partition is found! The median is either min(right2[0], right1[0]) if m + n is odd
             * or (min(right2[0], right1[0]) + max(left1[end], left2[end]))/2 if m+n is even.
             * In index notation:
             * median = min(arr1[i], arr2[j]) if odd
             * median = (min(arr1[i], arr2[j]) + max(arr1[i-1], arr2[j-1]))/2
             *
             * Remember to handle a bunch edge cases when it comes to array indexing
             */
            vector<int> *arr1;
            vector<int> *arr2;
            if (nums1.size() > nums2.size()) {
                arr1 = &nums2;
                arr2 = &nums1;
            } else {
                arr1 = &nums1;
                arr2 = &nums2;
            }

            int m = arr1->size();
            int n = arr2->size();

            int i = binarySearch((*arr1), (*arr2));
            int j = (m + n) / 2 - i;
            double median;
            if ((m + n) % 2 == 0) {
                median = (static_cast<double>(min(i >= m ? INT_MAX : (*arr1)[i], j >= n ? INT_MAX : (*arr2)[j])) +
                          static_cast<double>(max(i == 0 ? INT_MIN : (*arr1)[i - 1],
                                                  j == 0 ? INT_MIN : (*arr2)[j - 1]))) / 2;
            } else {
                median = static_cast<double>(min(i >= m ? INT_MAX : (*arr1)[i], j >= n ? INT_MAX : (*arr2)[j]));
            }

            return median;
        }
    }

    // 240. Search a 2D Matrix II
    namespace Search2DMatrixII {
        bool searchMatrix(vector<vector<int> > &matrix, int target) {
            /**
             * Approach:
             *
             * The matrix is actually a binary search tree with its root at the top right or bottom left corner.
             * If starting from top right, the value on the left is smaller and  the value underneath is bigger,
             * making up the two children. Thus, finding the targeting element becomes finding a value in the binary
             * search tree and if out of bounds (aka visiting a null node) the target does not exist.
             */

            int i = 0, j = matrix[0].size() - 1;

            while (true) {
                // while traversing, i is always increasing and j is alwayse decreasing
                // base case
                if (i >= matrix.size() || j < 0) {
                    return false;
                }

                int node_val = matrix[i][j];
                if (node_val == target) {
                    return true;
                }

                if (node_val < target) {
                    // the value is in the right subtree, go to the right subtree by moving down
                    i++;
                } else {
                    // the value is in the left subtree, go to the left by moving left
                    j--;
                }
            }
        }
    }

    // 33. Search in Rotated Sorted Array
    namespace RotatedArray {
        int binarySearch(const vector<int> &nums, int left, int right, int target) {
            while (left <= right) {
                int mid = (left + right) / 2;
                if (nums[mid] == target) {
                    return mid;
                }

                if (nums[mid] > target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            return -1;
        }

        int search(vector<int> &nums, int target) {
            /**
             * Approach:
             *
             * After rotation, either the left half or the right half is sorted. Search the sorted part with
             * regular binary searchfirst, if not found, repeat the algorithm with the other half.
             *
             * Find the sorted half by comparing left with middle or right with middle. If arr[left] < arr[middle]
             * then left half is sorted, else right is sorted.
             *
             * Example:
             * [4   5   6   7   0   1   2]  target = 0
             *                  l
             *                           r
             *                      m
             *
             */
            int left = 0, right = nums.size() - 1;
            while (left <= right) {
                int mid = (left + right) / 2;
                int index;
                if (nums[left] <= nums[mid]) {
                    index = binarySearch(nums, left, mid, target);
                    if (index != -1) {
                        return index;
                    } else {
                        left = mid + 1;
                    }
                } else {
                    index = binarySearch(nums, mid, right, target);
                    if (index != -1) {
                        return index;
                    } else {
                        right = mid - 1;
                    }
                }
            }
            return -1;
        }
    }

    // 81. Rotated Array II
    namespace RotatedArray2 {
        bool search(vector<int> &nums, int target) {
            /**
             * Approach: similar to Rotated Array I
             * However when nums[left] == nums[right] == nums[mid] ambiguity occurs.
             * Shrink left and right, worst case O(n)
             */

            int left = 0, right = nums.size() - 1;
            while (left <= right) {
                int mid = (left + right) / 2;

                if (nums[mid] == target) return true;
                if (nums[left] == nums[right] && nums[right] == nums[mid]) {
                    left++;
                    right--;
                    continue;
                }

                if (nums[left] <= nums[mid]) {
                    // left half sorted
                    if (nums[left] <= target && target < nums[mid]) {
                        right = mid - 1;
                    } else {
                        left = mid + 1;
                    }
                } else {
                    // right half is sorted
                    if (nums[mid] < target && target <= nums[right]) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
            }
            return false;
        }
    }

    // 153. Find Minimum in Rotated Sorted Array
    namespace findMinInRotated {
        int findMin(vector<int> &nums) {
            /**
             * Approach similar to binary search in rotated array.
             *
             * Need to find the point of rotation.
             * if nums[left] <= nums[mid] left is sorted, point of rotation in right
             * if nums[right] > nums[mid] right is sorted, point of rotation in left.
             *
             * Example:
             * [3   4   5   1   2]
             *              l
             *              r
             *              m
             *
             */

            int left = 0, right = nums.size() - 1;
            while (left < right) {
                // loop terminates at left < right instead of left <= right because
                // when we update right, it is updated as right = mid instead of right = mid - 1;
                // left <= right would lead to inifinite loop.
                int mid = (left + right) / 2;
                if (nums[mid] > nums[right]) {
                    // left part sorted, search right part
                    // skip mid because mid is not the minimum
                    left = mid + 1;
                } else {
                    // case for nums[mid] <= nums[right]
                    // right part is sorted, search left part
                    // mid could be the minimum cuz the case here represents nums[mid] <= nums[right]
                    // so mid could be the smallest number.
                    // Therefore, mid is not skipped.
                    right = mid;
                }
            }

            // at the end, left would be max and right would be min
            return nums[right];
        }

        int findMax(vector<int> &nums) {
            /**
             * Approach:
             *
             * nums[left] < nums[mid] left side sorted, max value in right side including mid
             * nums[left] >= nums[mid] right side sorted, max value in left side excluding mid
             *
             * Note:
             *
             * during integer division, mid is default to the floor of (left + right)/2
             * when right = left + 1, mid will default the the left one, causing nums[left] < nums[mid]
             * to fail, change mid to be the ceil so mid is always the bigger one
             *
             */

            int left = 0, right = nums.size() - 1;
            while (left < right) {
                int mid = ceil(static_cast<double>(left + right) / 2);
                if (nums[left] < nums[mid]) {
                    // left side sorted, max value in right side
                    left = mid;
                } else {
                    // left >= mid, max value in left side
                    right = mid - 1;
                }
            }

            // at the end, left would be max and right would be min
            return nums[right];
        }

        int findMax2(vector<int> &nums) {
            int n = nums.size();
            int left = 0, right = n - 1;

            // If array is not rotated, last element is max
            if (nums[left] < nums[right]) {
                return nums[right];
            }

            while (left < right) {
                int mid = left + (right - left) / 2;

                // Check if mid is the max (drop point)
                if (nums[mid] > nums[mid + 1]) {
                    return nums[mid]; // Found it directly!
                }

                // Decide which half contains the max
                if (nums[mid] > nums[right]) {
                    // Max is in right half (including mid)
                    left = mid;
                } else {
                    // Max is in left portion
                    right = mid - 1;
                }
            }

            return nums[left];
        }
    }
}

int main() {
    vector<int> input = {11, 13};
    vector<int> input2 = {1};
    int result = Solution::findMinInRotated::findMax2(input2);
    cout << result << endl;
    return 0;
}
