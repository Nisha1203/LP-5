# -*- coding: utf-8 -*-
"""Welcome To Colab

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/notebooks/intro.ipynb
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %%writefile p1.cpp
# #include <iostream>
# #include <vector>
# #include <queue>
# #include <stack>
# #include <omp.h>
# #include <chrono>
# 
# using namespace std;
# using namespace std::chrono;
# 
# const int MAX = 100; // Maximum size for the graph
# 
# vector<int> graph[MAX]; // Graph representation
# bool visited[MAX];      // Array to mark visited nodes
# 
# // Breadth-First Search function
# bool bfs(int start_node, int search_node) {
#     queue<int> q;
#     q.push(start_node);
#     visited[start_node] = true;
# 
#     while (!q.empty()) {
#         int current_node = q.front();
#         q.pop();
#         if (current_node == search_node) {
#             return true; // Node found
#         }
# 
#         #pragma omp parallel for
#         for (int i = 0; i < graph[current_node].size(); i++) {
#             int adj_node = graph[current_node][i];
#             if (!visited[adj_node]) {
#                 #pragma omp critical
#                 {
#                     visited[adj_node] = true;
#                     q.push(adj_node);
#                 }
#             }
#         }
#     }
#     return false; // Node not found
# }
# 
# // Depth-First Search function
# bool dfs(int start_node, int search_node) {
#     stack<int> s;
#     s.push(start_node);
# 
#     while (!s.empty()) {
#         int current_node = s.top();
#         s.pop();
#         if (current_node == search_node) {
#             return true; // Node found
#         }
#         if (!visited[current_node]) {
#             visited[current_node] = true;
# 
#             #pragma omp parallel for
#             for (int i = 0; i < graph[current_node].size(); i++) {
#                 int adj_node = graph[current_node][i];
#                 if (!visited[adj_node]) {
#                     #pragma omp critical
#                     {
#                         s.push(adj_node);
#                     }
#                 }
#             }
#         }
#     }
#     return false; // Node not found
# }
# 
# int main() {
#     int n, m; // n: number of nodes, m: number of edges
#     cout << "Enter the number of nodes and edges: ";
#     cin >> n >> m;
# 
#     // Input edges
#     cout << "Enter the edges (node pairs):\n";
#     for (int i = 0; i < m; i++) {
#         int u, v;
#         cin >> u >> v;
#         graph[u].push_back(v);
#         graph[v].push_back(u); // For undirected graph
#     }
# 
#     // Initialize visited array
#     #pragma omp parallel for
#     for (int i = 0; i < n; i++) {
#         visited[i] = false;
#     }
# 
#     // Get number of threads
#     int num_threads;
#     #pragma omp parallel
#     {
#         #pragma omp single
#         num_threads = omp_get_num_threads();
#     }
#     cout << "Number of threads: " << num_threads << endl;
# 
#     // Search for a node using BFS
#     int search_node;
#     cout << "Enter the node to search: ";
#     cin >> search_node;
# 
#     auto bfs_start = high_resolution_clock::now();
#     bool found_bfs = bfs(0, search_node); // Start BFS from node 0
#     auto bfs_stop = high_resolution_clock::now();
#     auto bfs_duration = duration_cast<microseconds>(bfs_stop - bfs_start);
#     if (found_bfs) {
#         cout << "Node " << search_node << " found using BFS\n";
#     } else {
#         cout << "Node " << search_node << " not found using BFS\n";
#     }
#     cout << "BFS Execution Time: " << bfs_duration.count() << " microseconds" << endl;
# 
#     // Reset visited array
#     #pragma omp parallel for
#     for (int i = 0; i < n; i++) {
#         visited[i] = false;
#     }
# 
#     // Search for a node using DFS
#     auto dfs_start = high_resolution_clock::now();
#     bool found_dfs = dfs(0, search_node); // Start DFS from node 0
#     auto dfs_stop = high_resolution_clock::now();
#     auto dfs_duration = duration_cast<microseconds>(dfs_stop - dfs_start);
#     if (found_dfs) {
#         cout << "Node " << search_node << " found using DFS\n";
#     } else {
#         cout << "Node " << search_node << " not found using DFS\n";
#     }
#     cout << "DFS Execution Time: " << dfs_duration.count() << " microseconds" << endl;
# 
#     return 0;
# }
#

!g++ -fopenmp p1.cpp -o Myexe
!./Myexe
