import csv
import sys
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
                if self.puzzle[i][j] == 'X':
                    return i, j
        return -1, -1  # Return -1, -1 when no empty cell is found


    def is_valid(self, row, col, num):
        # Check if 'num' is not present in the current row, column, and 3x3 subgrid
        return (not self.used_in_row(row, num) and
                not self.used_in_col(col, num) and
                not self.used_in_subgrid(row - row % 3, col - col % 3, num))

    def used_in_row(self, row, num):
        return str(num) in self.puzzle[row]

    def used_in_col(self, col, num):
        return str(num) in [self.puzzle[row][col] for row in range(self.n)]

    def used_in_subgrid(self, start_row, start_col, num):
        for row in range(3):
            for col in range(3):
                if self.puzzle[row + start_row][col + start_col] == str(num):
                    return True
        return False

    def solve_brute_force(self):
        self.start_time = time.time()
        if self._solve():
            print("Sudoku puzzle solved successfully.")
            print("Time taken: {:.6f} seconds".format(time.time() - self.start_time))
            self._print_solution()
        else:
            print("No solution exists for the Sudoku puzzle.")

    def _solve(self):
        empty_cell = self.find_empty_cell()
        if not empty_cell:
            return True  # Puzzle solved
        row, col = empty_cell
        for num in range(1, self.n + 1):
            if self.is_valid(row, col, str(num)):
                self.puzzle[row][col] = str(num)
                self.total_nodes += 1
                if self._solve():
                    return True
                self.puzzle[row][col] = 'X'  # Backtrack
        return False

    def print_solution(self, filename):
        print("Kumar, Aman, A20538809 solution:")
        print("Input file:", sys.argv[2])
        if sys.argv[1] == '1':
            print("Algorithm: Brute Force")
        elif sys.argv[1] == '2':
            print("Algorithm: Backtracking Search")
        elif sys.argv[1] == '3':
            print("Algorithm: Forward Checking with MRV Heuristics")
        print("\nInput puzzle:")
        
        data = load_puzzle(filename)
        
        formatted_data = []
        for set_index, puzzle_set in enumerate(data, start=1):
            if set_index == 1:
                for puzzle in puzzle_set:
                    formatted_data.append(",".join(puzzle))
                formatted_data.append("")

        print("\n".join(formatted_data))
        
        print("\nNumber of search tree nodes generated:", self.total_nodes)
        print("Search time: {:.6f}".format(time.time() - self.start_time))
        print("Solved puzzle:")
        for row in self.puzzle:
            print(','.join(str(cell) if cell != 'X' else 'X' for cell in row))
        print("\nSaving solution to", sys.argv[2].split('.')[0] + "_SOLUTION.csv...")
        with open(sys.argv[2].split('.')[0] + "_SOLUTION.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.puzzle:
                writer.writerow(row)

def load_puzzle(filename):
    puzzle = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            puzzle.append(row)
    return puzzle, puzzle

def main():
    if len(sys.argv) != 3:
        print("ERROR: Not enough/too many/illegal input arguments.")
        sys.exit()
    mode = sys.argv[1]
    filename = sys.argv[2]
    if mode not in ['1', '2', '3', '4']:
        print("ERROR: Illegal mode input. Mode should be 1, 2, 3, or 4.")
        sys.exit()
    try:
        puzzle, raw_data = load_puzzle(filename)
    except FileNotFoundError:
        print("ERROR: File not found.")
        sys.exit()
    
    solver = SudokuSolver(puzzle)

    if mode == '1':
        solver.solve_brute_force()
        solver.print_solution(filename)
    elif mode == '2':
        solver.solve_backtracking()
        solver.print_solution(filename)
    elif mode == '3':
        solver.solve_forward_checking_mrv()
        solver.print_solution(filename)
    elif mode == '4':
        solver.test_solution()

if __name__ == "__main__":
    main()
