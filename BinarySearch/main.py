from typing import *

class Solution:
    # 3453 Separate Squares I
    def separateSquares(self, squares: List[List[int]]) -> float:
        """
        Approach

        sort squares based on y coord and use binary search to find the line where
        it splits left and right area by half.

        How to calculate area efficiently?
        the area is defined by side^2 when a square is not divided by middle line.
        area = side * abs(middle - side) if it is cut
        so area = side * min(side, abs(middle - side)) for a rectangle

        """
        err = 1e-5
        squares.sort(key=lambda x: (x[1], x[2]))
        n = len(squares)
        left = squares[0][1]
        right = max([x[1]+x[2] for x in squares])

        while (right - left) > err:
            mid = (left + right) / 2
            
            # calculate left area
            leftArea = 0
            for square in squares:
                if square[1] > mid:
                    break

                leftArea += square[2] * min(square[2], abs(mid - square[1]))
            
            # calculate right area
            rightArea = 0
            for square in squares:
                if square[1] + square[2] < mid:
                    continue

                rightArea += square[2] * min(square[2], abs(mid - (square[1] + square[2])))
            
            if (rightArea > leftArea):
                left = mid
            else:
                right = mid
        
        return left

def main():
    s = Solution()
    print(s.separateSquares([[8,16,1],[6,15,10]]))

if __name__ == "__main__":
    main()