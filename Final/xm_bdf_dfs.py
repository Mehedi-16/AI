graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'H'],
    'D': ['B', 'F', 'G'],
    'F': ['D'],
    'H': ['C', 'K'],
    'K': ['H'],
    'G': ['D', 'I', 'J'],
    'I': ['G', 'L'],
    'J': ['G', 'L'],
    'L': ['I', 'J']
}

def bfs(start):
    visited, queue = set(), [start]
    order = []
    while queue:
        node = queue.pop(0)
        if node not in visited:
            order.append(node)
            visited.add(node)
            queue.extend([n for n in graph[node] if n not in visited])
    return order

def dfs(start, visited=None, order=None):
    if visited is None:
        visited = set()
    if order is None:
        order = []
    order.append(start)
    visited.add(start)
    for n in graph[start]:
        if n not in visited:
            dfs(n, visited, order)
    return order

# Run traversals
bfs_result = bfs('A')
dfs_result = dfs('A')

# Example branching factor (b) and depth (d)
b, d = 3, 4
bfs_complexity = b ** d       # BFS → O(b^d)
dfs_complexity = b * d        # DFS → O(b·d)

print("BFS Traversal Order:", " -> ".join(bfs_result))
print("DFS Traversal Order:", " -> ".join(dfs_result))
print("\n--- Time Complexity ---")
print(f"BFS: O(b^d) = O({b}^{d}) = {bfs_complexity}")
print(f"DFS: O(b·d) = O({b}×{d}) = {dfs_complexity}")
