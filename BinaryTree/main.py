from typing import *
from .tree_node import TreeNode
from collections import deque
class Solution:
    def subtreeWithAllDeepest(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        """
        Approach:

        Each node will obtain the subtree of the deepest from left and right subtree,
        if both tree are full, then this subtree can be extended with the current node.
        else, do not extend
        """
        def dfs(node: TreeNode, depth: int) -> Tuple[TreeNode, int]:
            # base case
            if node == None:
                return node, depth
            
            leftTree, leftDepth = dfs(node.left, depth+1)
            rightTree, rightDepth = dfs(node.right, depth+1)

            # if left and right have same depth, merge and return the node itself
            if leftDepth == rightDepth:
               return node, leftDepth
            
            if leftDepth > rightDepth:
                return leftTree, leftDepth
            
            if rightDepth > leftDepth:
                return rightTree, rightDepth
            
        result, _ = dfs(root, 0)
        return result
    
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        """
        Approach:

        bfs with level traversing
        """
        queue = deque()
        queue.append(root)

        averageValues = []

        while len(queue) > 0:
            total = 0
            levelSize = len(queue)
            for i in range(levelSize):
                node = queue.popleft()
                total += node.val
                
                if node.left is not None:
                    queue.append(node.left)
                if node.right is not None:
                    queue.append(node.right)

            averageValues.append(total/levelSize)

        return averageValues
    
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        order = True    # true = left flase = right
        queue = deque()
        queue.append(root)

        result = []
        while len(queue) > 0:
            levelSize = len(queue)
            levelResult = []
            for i in range(levelSize):
                node = queue.popleft()

                if node is None:
                    continue

                levelResult.append(node.val)
                queue.append(node.left)
                queue.append(node.right)

            if not levelResult:
                continue
            
            if order:
                result.append(levelResult)
            else:
                levelResult.reverse()
                result.append(levelResult)

            # reverse traversal order
            order = not order
        
        return result

def main():
    pass

if __name__ == "__main__":
    main()