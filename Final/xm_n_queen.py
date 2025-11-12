import random

N = 8 
def print_board(board):
    print(f"\nBoard State (h={calculate_h(board)}):")
    for row in range(N):
        row_str = ""
        for col in range(N):
            if board[col] == row:
                row_str += "Q "
            else:
                row_str += ". "
        print(row_str)
    print("-" * (N*2))

def calculate_h(board):
   
    h = 0
    for i in range(N):
        for j in range(i + 1, N):
           
            if board[i] == board[j]:
                h += 1
            
           
            if abs(board[i] - board[j]) == abs(i - j):
                h += 1
    return h


def hill_climbing():
  
    current_board = [random.randint(0, N - 1) for _ in range(N)]
    current_h = calculate_h(current_board)
    
    print(f"Starting new attempt with h={current_h}")

    while True:
        best_neighbor = current_board
        best_h = current_h
        for col_to_move in range(N):
            original_row = current_board[col_to_move]
            
            for row_to_move in range(N):
                if row_to_move == original_row:
                    continue  
                
                neighbor = list(current_board)
                neighbor[col_to_move] = row_to_move
                
                neighbor_h = calculate_h(neighbor)
                
                if neighbor_h < best_h:
                    best_h = neighbor_h
                    best_neighbor = neighbor
        
        if best_h >= current_h:
           
            if current_h == 0:
                
                return current_board, True, current_h
            else:
               
                return current_board, False, current_h
       
        current_board = best_neighbor
        current_h = best_h


def random_restart_hill_climbing():
  
    solution_found = False
    attempts = 0
    
    while not solution_found:
        attempts += 1
        print(f"\n--- ATTEMPT {attempts} ---")
        
      
        final_board, is_solution, final_h = hill_climbing()
        
        if is_solution:
            print("\n=====================")
            print(f"SOLUTION FOUND! (in {attempts} attempts)")
            print_board(final_board)
            solution_found = True
        else:
            print(f"Stuck at a local minimum with h={final_h}. Restarting...")
         
if __name__ == "__main__":
    random_restart_hill_climbing()