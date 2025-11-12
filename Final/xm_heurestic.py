from queue import PriorityQueue

graph = {
    'A': {'B': 2, 'E': 11},
    'B': {'C': 1, 'G': 9},
    'C': {},
    'E': {'D': 6},
    'D': {'G': 1},
    'G': {}
}

heuristic = {
    'A': 11,
    'B': 6,
    'C': 99,
    'D': 1,
    'E': 7,
    'G': 0
}

def a_star(start, goal):
    pq = PriorityQueue()
    pq.put((0 + heuristic[start], 0, start, [start]))
    visited = set()

    while not pq.empty():
        f, g, node, path = pq.get()
        if node == goal:
            print("Path:", " -> ".join(path))
            print("Total cost:", g)
            return
        if node in visited:
            continue
        visited.add(node)

        for neighbor, cost in graph[node].items():
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                pq.put((new_f, new_g, neighbor, path + [neighbor]))

a_star('A', 'G')
