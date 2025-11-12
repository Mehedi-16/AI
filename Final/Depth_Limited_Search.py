def depth_limited_search(graph, start, goal, limit):
    def dfs(node, path, depth):
        if depth > limit:
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

str=input("enter string")
graph={}
for p in str.split():
    node,neighbor= p.split(":")
    if neighbor:
        graph[node]= neighbor.split(",")
    else:
        graph[node]=[]
start_node = 'A'
goal_node = 'E'
depth_limit = 2
result = depth_limited_search(graph, start_node, goal_node, depth_limit)
if result:
    print("Path found:", result)
else:
    print("No path found within depth limit.")
