from typing import *
from .tree_node import TreeNode
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
    
def main():
    pass

if __name__ == "__main__":
    main()