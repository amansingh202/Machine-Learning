import time

class SudokuSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.n = len(puzzle)
        self.start_time = None
        self.total_nodes = 0

    def find_empty_cell(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.puzzle[i][j] == 'X':
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
        self.start_time = time.time()
        if self._solve_brute_force():
            return True
        else:
            return False

    def _solve_brute_force(self):
        row, col = self.find_empty_cell()
        if row is None:
            return True  # Puzzle solved
        for num in range(1, self.n + 1):
            if self.is_valid(row, col, num):
                self.puzzle[row][col] = num
                self.total_nodes += 1
                if self._solve_brute_force():
                    return True  # Solution found
                self.puzzle[row][col] = 'X'  # Undo the placement
        return False  # No solution found

    def print_solution(self):
        print("Sudoku puzzle solved successfully in {:.6f} seconds:".format(time.time() - self.start_time))
        for row in self.puzzle:
            print(row)


# Example usage:
puzzle = [
    ['5', '3', 'X', 'X', '7', 'X', 'X', 'X', 'X'],
    ['6', 'X', 'X', '1', '9', '5', 'X', 'X', 'X'],
    ['X', '9', '8', 'X', 'X', 'X', 'X', '6', 'X'],
    ['8', 'X', 'X', 'X', '6', 'X', 'X', 'X', '3'],
    ['4', 'X', 'X', '8', 'X', '3', 'X', 'X', '1'],
    ['7', 'X', 'X', 'X', '2', 'X', 'X', 'X', '6'],
    ['X', '6', 'X', 'X', 'X', 'X', '2', '8', 'X'],
    ['X', 'X', 'X', '4', '1', '9', 'X', 'X', '5'],
    ['X', 'X', 'X', 'X', '8', 'X', 'X', '7', '9']
]

solver = SudokuSolver(puzzle)
if solver.solve_brute_force():
    solver.print_solution()
    print("Total nodes visited:", solver.total_nodes)
else:
    print("No solution exists for the given puzzle.")
