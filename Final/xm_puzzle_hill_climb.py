import copy, random

goal = [[1,2,3],[4,5,6],[7,8,0]]
init = [[3,8,5],[7,1,0],[2,6,4]]

def h(state):
    dist = 0
    for i in range(3):
        for j in range(3):
            v = state[i][j]
            if v == 0:
                continue
            goal_r = (v - 1) // 3
            goal_c = (v - 1) % 3
            dist += abs(i - goal_r) + abs(j - goal_c)
    return dist


def h_misplaced(state):
    return sum(state[i][j] != 0 and state[i][j] != goal[i][j] for i in range(3) for j in range(3))

def blank_pos(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def neighbors(state):
    i, j = blank_pos(state)
    moves = []
    for x, y in [(-1,0),(1,0),(0,-1),(0,1)]:
        a, b = i+x, j+y
        if 0 <= a < 3 and 0 <= b < 3:
            n = copy.deepcopy(state)
            n[i][j], n[a][b] = n[a][b], n[i][j]
            moves.append(n)
    return moves

def hill_climb_once(start_state, max_steps=1000, max_sideways=100):
    current = copy.deepcopy(start_state)
    current_h = h(current)
    sideways = 0

    for _ in range(max_steps):
        nbrs = neighbors(current)
        nbrs_h = [h(n) for n in nbrs]
        best_h = min(nbrs_h)
        best_candidates = [n for n, hv in zip(nbrs, nbrs_h) if hv == best_h]
        best = random.choice(best_candidates)

        if best_h < current_h:
            current, current_h = best, best_h
            sideways = 0
        elif best_h == current_h and sideways < max_sideways:
            current, current_h = best, best_h
            sideways += 1
        else:
            break

        if current_h == 0:
            return current, current_h, True

    return current, current_h, False


def scramble_from(state, moves=20):
    s = copy.deepcopy(state)
    for _ in range(moves):
        nbrs = neighbors(s)
        s = random.choice(nbrs)
    return s


def hill_climb_with_restarts(initial, restarts=50, scramble_moves=20, max_steps=1000):
    best_overall = None
    best_h = float('inf')

    for attempt in range(restarts):
        if attempt == 0:
            start = initial
        else:
            start = scramble_from(initial, moves=scramble_moves)

        state, state_h, reached = hill_climb_once(start, max_steps=max_steps)

        if state_h < best_h:
            best_h = state_h
            best_overall = state
    print("Final State:")
    for row in best_overall:
        print(row)


if __name__ == '__main__':
    random.seed(0)
    print("Initial misplaced-tiles heuristic:", h_misplaced(init))
    hill_climb_with_restarts(init, restarts=200, scramble_moves=30, max_steps=1000)
