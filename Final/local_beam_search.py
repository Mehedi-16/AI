import random

def objective_function(x):
    return -(x**2) + 10*x + 5   # f(x) = -x² + 10x + 5

def local_beam_search(objective, bounds, k=3, max_iter=100, step_size=0.5):
    population = [random.uniform(bounds[0], bounds[1]) for _ in range(k)]

    for i in range(max_iter):
        neighbors = []
        for x in population:
            for _ in range(2): 
                neighbor = x + random.uniform(-step_size, step_size)
                neighbor = max(bounds[0], min(bounds[1], neighbor))  # boundary check
                neighbors.append(neighbor)
        
        all_candidates = population + neighbors
        scores = [(x, objective(x)) for x in all_candidates]

        scores.sort(key=lambda t: t[1], reverse=True)
        population = [t[0] for t in scores[:k]]

    best_x = population[0]
    best_value = objective(best_x)
    return best_x, best_value


best_x, best_value = local_beam_search(objective_function, bounds=[-10, 10], k=3, max_iter=100)
print(f"Best Solution: x = {best_x:.4f}, f(x) = {best_value:.4f}")
