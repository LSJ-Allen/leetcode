#include <iostream>
#include <string>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <set>

using namespace std;

namespace Solution {
    namespace LRU {
        struct Node {
            Node *next = nullptr;
            Node *prev = nullptr;
            int key;
            int val;

            Node(int key, int value) : key(key), val(value) {
            };
        };

        class LRUCache {
            /**
             * Use a doublely linked list and a hash map to build the LRU cache.
             * The order in the linked list would represent elements from least recently
             * used to most recently used. When full, evict head node.
             *
             * The hash map <key, Node*> stores references to the nodes for o(1) look up.
             * when a node is removed, it is removed from the hash map as well.
             */
        private:
            std::unordered_map<int, Node *> map;
            int maxCapacity = 0;
            // dummy head and tail node to handle edge cases
            Node *head = new Node(0, 0);
            Node *tail = new Node(0, 0);

            void remove(Node *node) {
                // remove a node
                Node *prev = node->prev;
                Node *next = node->next;

                // remove node from the link
                prev->next = next;
                next->prev = prev;

                // remove from map
                map.erase(node->key);
                delete node;
            }

            void insert(Node *newNode, Node *node) {
                // insert a new node at node's position
                // prev -> node -> next
                //              ^
                //          insert here

                Node *next = node->next;
                node->next = newNode;
                newNode->prev = node;

                newNode->next = next;
                next->prev = newNode;

                // insert into map
                map[newNode->key] = newNode;
            }

            void move(Node *n1, Node *n2) {
                if (n1 == n2) {
                    return;
                }
                // move n1 to n2's next
                Node *n1_prev = n1->prev;
                Node *n1_next = n1->next;
                Node *n2_next = n2->next;

                // place n1
                n2->next = n1;
                n2_next->prev = n1;

                n1->prev = n2;
                n1->next = n2_next;

                // cut n1
                n1_prev->next = n1_next;
                n1_next->prev = n1_prev;
            }

        public:
            LRUCache(int capacity) {
                this->maxCapacity = capacity;
                this->head->next = tail;
                this->tail->prev = head;
            }

            int get(int key) {
                // query the hash map
                if (this->map.contains(key)) {
                    this->move(map[key], this->tail->prev);
                    return map[key]->val;
                } else {
                    return -1;
                }
            }

            void put(int key, int value) {
                // if key exists, placed it at head.
                if (map.contains(key)) {
                    Node *existingNode = map[key];
                    existingNode->val = value;
                    this->move(existingNode, this->tail->prev);
                    return;
                }

                // if current capacity is at max, evict the least recently used first
                if (this->map.size() == this->maxCapacity) {
                    this->remove(this->head->next);
                }

                // if key does not exist
                Node *newNode = new Node(key, value);
                this->insert(newNode, this->tail->prev);
            }


            // destructor
            ~LRUCache() {
                Node *curr = head;
                while (curr) {
                    Node *next = curr->next;
                    delete curr;
                    curr = next;
                }
            }
        };
    }
}

int main() {
    Solution::LRU::LRUCache cache = Solution::LRU::LRUCache(2);
    cache.get(2);
    cache.put(2, 6);
    cache.get(1);
    cache.put(1, 5);
    cache.put(1, 2);
    cache.get(1);
    cout << cache.get(2) << endl;;
    return 0;
}
