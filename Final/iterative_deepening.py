def depth_limited_search(graph, start, goal, limit):
    def dfs(node, path, depth):
        if depth > limit:     # limit cross করলে থেমে যাবে
            return None
        path = path + [node]
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            result = dfs(neighbor, path, depth + 1)
            if result:
                return result
        return None
    return dfs(start, [], 0)


def iterative_deepening_search(graph, start, goal, max_depth):
    for limit in range(max_depth + 1):    # limit ধাপে ধাপে বাড়বে
        result = depth_limited_search(graph, start, goal, limit)
        if result:
            return result, limit
    return None, None


# --- ইনপুট ---
s = input("Enter graph (ex: A:B,C B:D C:D D:E E:): ")
graph = {}
for p in s.split():
    node, neighbor = p.split(':')
    graph[node] = neighbor.split(',') if neighbor else []

start = 'A'
goal = 'E'
max_depth = 5

path, used_depth = iterative_deepening_search(graph, start, goal, max_depth)

if path:
    print("Goal is ", path, "Depth =", used_depth)
else:
    print("Goal not found (max depth cross)")
