from typing import *

# 208 Implement Trie
class Trie:
    class Node:
        def __init__(self, val):
            self.val = val
            self.is_word = False
            # use a hash map to store children char to children object relationship
            self.children = {}
    
        def add_child(self, child):
            self.children[child.val] = child
    
    def __init__(self):
        self.root = Trie.Node('')

    def insert(self, word: str) -> None:
        curNode = self.root
        for c in word:
            # if the char is in children, traverse down the Trie
            if c in curNode.children:
                curNode = curNode.children[c]
                continue
            
            # else add it to children
            newNode = Trie.Node(c)
            curNode.add_child(newNode)
            curNode = newNode
        
        curNode.is_word = True

    def search(self, word: str) -> bool:
        curNode = self.root

        for c in word:
            if c not in curNode.children:
                return False
            curNode = curNode.children[c]
        
        # if no more children, return true
        return curNode.is_word

    def startsWith(self, prefix: str) -> bool:
        curNode = self.root

        for c in prefix:
            if c not in curNode.children:
                return False
            curNode = curNode.children[c]
        
        return True

# 211. Design Add and Search Words Data Structure
class WordDictionary:
    # trie with a dfs to expore different paths

    class Node:
        def __init__(self, val):
            self.val = val
            self.isWord = False
            # use a hash map to store children char to children object relationship
            self.children = {}

    def __init__(self):
        self.root = WordDictionary.Node('')

    def addWord(self, word: str) -> None:
        curNode = self.root
        for c in word:
            # if the char is in children, traverse down the Trie
            if c in curNode.children:
                curNode = curNode.children[c]
                continue
            
            # else add it to children
            curNode.children[c] = WordDictionary.Node(c)
            curNode = curNode.children[c]
        
        curNode.isWord = True

    # dfs through the trie to find if s exists
    # a node return true if dfs of any of its children yield true and the value at current node
    # is equal to s[index]
    def _dfs(self, node: WordDictionary.Node, s: str, index: int) -> bool:
        # base case: we have reached the last character of s
        if index + 1 == len(s):
            # return true if the last char match and the current node is a word end, not 
            # just a prefix
            return node.isWord and ((node.val == s[index]) or s[index] == '.')

        # pruning: if node's val is not equal to s[index] return false immediately
        isValid = (node.val == s[index] or s[index] == '.')
        if not isValid:
            return False
        
        for child in node.children.values():
            # if any of the subtree yield true, return immediately
            if(self._dfs(child, s, index + 1)):
                return True
    
    def search(self, word: str) -> bool:
        for node in self.root.children.values():
            if self._dfs(node, word, 0):
                return True
        
        return False

def main():
    wordDictionary = WordDictionary()
    wordDictionary.addWord("bad")
    wordDictionary.addWord("dad")
    wordDictionary.addWord("mad")
    print(wordDictionary.search("pad"))
    print(wordDictionary.search("bad"))
    print(wordDictionary.search(".ad"))
    print(wordDictionary.search("b.."))
if __name__ == "__main__":
    main()