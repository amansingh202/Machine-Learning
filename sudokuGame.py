import time

class SudokuSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.n = len(puzzle)
        self.sqrt_n = int(self.n ** 0.5)
        self.total_nodes = 0
        self.start_time = None

    def find_empty_cell(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.puzzle[i][j] == 0:
                    return i, j
        return None, None

    def is_valid(self, row, col, num):
        # Check if 'num' is not present in the current row, column, and 3x3 subgrid
        return (not self.used_in_row(row, num) and
                not self.used_in_col(col, num) and
                not self.used_in_subgrid(row - row % 3, col - col % 3, num))

    def used_in_row(self, row, num):
        return num in self.puzzle[row]

    def used_in_col(self, col, num):
        return any(row[col] == num for row in self.puzzle)

    def used_in_subgrid(self, start_row, start_col, num):
        for i in range(3):
            for j in range(3):
                if self.puzzle[i + start_row][j + start_col] == num:
                    return True
        return False

    def solve_brute_force(self):
        self.start_time = time.time()  # Start timing
        row, col = self.find_empty_cell()
        if row is None:
            return True  # Puzzle solved
        for num in range(1, self.n + 1):
            if self.is_valid(row, col, num):
                self.puzzle[row][col] = num
                self.total_nodes += 1
                if self.solve_brute_force():
                    return True  # Solution found
                self.puzzle[row][col] = 0  # Undo the placement
        return False  # No solution found

    def print_solution(self):
        for row in self.puzzle:
            print(row)

# Example usage:
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

solver = SudokuSolver(puzzle)
start_time = time.time()
if solver.solve_brute_force():
    print("Sudoku puzzle solved successfully:")
    solver.print_solution()
    print("Total nodes visited:", solver.total_nodes)
    print("Time taken:", round(time.time() - start_time, 6), "seconds")
else:
    print("No solution exists for the given puzzle.")
