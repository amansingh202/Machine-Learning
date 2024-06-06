#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import csv
import time

class SudokuSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.n = len(puzzle)
        self.sqrt_n = int(self.n ** 0.5)
        self.total_nodes = 0
        self.start_time = 0

    def is_valid(self, row, col, num):
        for x in range(self.n):
            if self.puzzle[row][x] == num or self.puzzle[x][col] == num:
                return False
        start_row, start_col = self.sqrt_n * (row // self.sqrt_n), self.sqrt_n * (col // self.sqrt_n)
        for i in range(self.sqrt_n):
            for j in range(self.sqrt_n):
                if self.puzzle[i + start_row][j + start_col] == num:
                    return False
        return True

    def find_empty_cell(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.puzzle[i][j] == 'X':
                    return i, j
        return None, None

    def solve_brute_force(self):
        self.start_time = time.time()
        row, col = self.find_empty_cell()
        if row is None:
            return True
        for num in range(1, self.n + 1):
            if self.is_valid(row, col, str(num)):
                self.puzzle[row][col] = str(num)
                self.total_nodes += 1
                if self.solve_brute_force():
                    return True
                self.puzzle[row][col] = 'X'
        return False

    def solve_backtracking(self):
        self.start_time = time.time()
        if self.backtrack():
            return True
        else:
            return False

    def backtrack(self):
        row, col = self.find_empty_cell()
        if row is None:
            return True
        for num in range(1, self.n + 1):
            if self.is_valid(row, col, str(num)):
                self.puzzle[row][col] = str(num)
                self.total_nodes += 1
                if self.backtrack():
                    return True
                self.puzzle[row][col] = 'X'
        return False

    def solve_forward_checking_mrv(self):
        self.start_time = time.time()
        if self.forward_checking_mrv():
            return True
        else:
            return False

    def forward_checking_mrv(self):
        row, col = self.find_empty_cell()
        if row is None:
            return True
        domain = self.get_domain(row, col)
        for num in domain:
            self.puzzle[row][col] = str(num)
            self.total_nodes += 1
            if self.forward_checking_mrv():
                return True
            self.puzzle[row][col] = 'X'
        return False

    def get_domain(self, row, col):
        domain = set(str(i) for i in range(1, self.n + 1))
        for i in range(self.n):
            domain.discard(self.puzzle[row][i])
            domain.discard(self.puzzle[i][col])
        start_row, start_col = self.sqrt_n * (row // self.sqrt_n), self.sqrt_n * (col // self.sqrt_n)
        for i in range(self.sqrt_n):
            for j in range(self.sqrt_n):
                domain.discard(self.puzzle[i + start_row][j + start_col])
        return domain

    def print_solution(self):
        print("Last Name, First Name, AXXXXXXXX solution:")
        print("Input file:", sys.argv[2])
        if sys.argv[1] == '1':
            print("Algorithm: Brute Force")
        elif sys.argv[1] == '2':
            print("Algorithm: Backtracking Search")
        elif sys.argv[1] == '3':
            print("Algorithm: Forward Checking with MRV Heuristics")
        print("\nInput puzzle:")
        for row in self.puzzle:
            print(','.join(row))
        print("\nNumber of search tree nodes generated:", self.total_nodes)
        print("Search time:", round(time.time() - self.start_time, 2), "seconds\n")
        print("Solved puzzle:")
        for row in self.puzzle:
            print(','.join(row))
        print("\nSaving solution to", sys.argv[2].split('.')[0] + "_SOLUTION.csv...")
        with open(sys.argv[2].split('.')[0] + "_SOLUTION.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.puzzle:
                writer.writerow(row)

    def test_solution(self):
        solved_puzzle = [[int(num) if num != 'X' else num for num in row] for row in self.puzzle]
        for row in solved_puzzle:
            if 'X' in row:
                print("ERROR: This is NOT a solved Sudoku puzzle.")
                return
        for row in solved_puzzle:
            if len(set(row)) != self.n:
                print("ERROR: This is NOT a solved Sudoku puzzle.")
                return
        for col in range(self.n):
            if len(set(row[col] for row in solved_puzzle)) != self.n:
                print("ERROR: This is NOT a solved Sudoku puzzle.")
                return
        for i in range(0, self.n, self.sqrt_n):
            for j in range(0, self.n, self.sqrt_n):
                box = []
                for x in range(self.sqrt_n):
                    for y in range(self.sqrt_n):
                        box.append(solved_puzzle[i + x][j + y])
                if len(set(box)) != self.n:
                    print("ERROR: This is NOT a solved Sudoku puzzle.")
                    return
        print("This is a valid, solved, Sudoku puzzle.")

def load_puzzle(filename):
    with open(filename, 'r') as file:
        puzzle = list(csv.reader(file))
    return puzzle

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
        puzzle = load_puzzle(filename)
    except FileNotFoundError:
        print("ERROR: File not found.")
        sys.exit()

    solver = SudokuSolver(puzzle)

    if mode == '1':
        solver.solve_brute_force()
        solver.print_solution()
    elif mode == '2':
        solver.solve_backtracking()
        solver.print_solution()
    elif mode == '3':
        solver.solve_forward_checking_mrv()
        solver.print_solution()
    elif mode == '4':
        solver.test_solution()

if __name__ == "__main__":
    main()


# In[ ]:




