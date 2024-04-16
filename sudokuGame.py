#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sys
import csv
import time

class SudokuSolver:


    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.n = len(puzzle)
        self.sqrt_n = int(self.n ** 0.5)
        self.total_nodes = 0
        self.start_time = None

    def is_sudoku_solved(self):
        # Check if each row contains all nums
        all_nums = [str(i) for i in range(1, self.n + 1)]  # Generate list of strings from 1 to 9
        for row in self.puzzle:
            row_str = [str(cell) for cell in row]  # Convert all elements in the row to strings
            if sorted(row_str) != all_nums:
                return False
        return True

        # Check if each column contains all nums
        for col in range(9):
            if sorted(self.puzzle[:, col]) != all_nums:
                # print(sorted(board[:, col]))
                # print("issue2")
                return False

        # Check if each 3x3 subgrid contains all nums
        for row in range(0, 9, 3):
            for col in range(0, 9, 3):
                subgrid = self.puzzle[row:row+3, col:col+3]
                if sorted(subgrid.flatten()) != all_nums:
                    # print("issue3")
                    return False
        # print("Solved puzzle:")
        # print_sudoku(board)
        return True


    def find_empty_cell(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.puzzle[i][j] == 'X':
                    return i, j
        return None # Return -1, -1 when no empty cell is found


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

    # def solve_brute_force(self, count):
    #     # #self.start_time = time.time()
    #     # row, col = self.find_empty_cell()
    #     # if row == -1 and col == -1:
    #     #     return True  # Puzzle solved

    #     # for num in range(1, self.n + 1):
    #     #     if self.is_valid(row, col, str(num)):
    #     #         self.puzzle[row][col] = str(num)
    #     #         self.total_nodes += 1
    #     #         if self.solve_brute_force():
    #     #             return True  # Solution found
    #     #         self.puzzle[row][col] = 'X'  # Undo the placement
    #     # return False 
    #     # print()
    #     # print("Step")
    #     # print_sudoku(board)

    #     empty_location = self.find_empty_cell()
    #     # print(empty_location)
    #     # count[0] = count[0] + 1
    #     # If there is no empty location, the sudoku is solved
    #     # if count[0]%1000000==0:
    #     #     print_sudoku(board)
    #     #     print(count[0])
    #     count[0] = count[0] + 1
    #     if not empty_location:

    #         if self.is_sudoku_solved():
    #             print("solved")
    #             return True
    #         else:
    #             # print("not solved")
    #             return False

    #     row, col = empty_location

    #     # Try placing numbers from 1 to 9
    #     for num in range(1, 10):
    #         # if is_valid(board, row, col, num):
    #             # Place the number if it's valid
    #         self.puzzle[row][col] = num

    #             # Recursively try to solve the remaining board
    #         if self.solve_brute_force(count):
    #             # print("in if")
    #             return True


    #             # If placing the number leads to an invalid solution, backtrack
    #         self.puzzle[row][col] = 'X'

    #     return False
    def is_sudoku_solved(self):
    # Check if each row contains all nums
        all_nums = set(range(1, self.n + 1))  # Generate set of integers from 1 to 9
        for row in self.puzzle:
            row_set = set(row)  # Convert row to set to remove duplicates
            if row_set != all_nums:
                return False

        # Check if each column contains all nums
        for col in range(self.n):
            col_set = set(self.puzzle[row][col] for row in range(self.n))  # Extract elements from each row in the column
            if col_set != all_nums:
                return False

        # Check if each 3x3 subgrid contains all nums
        for row in range(0, self.n, 3):
            for col in range(0, self.n, 3):
                subgrid_set = set(self.puzzle[i][j] for i in range(row, row + 3) for j in range(col, col + 3))  # Extract elements from subgrid
                if subgrid_set != all_nums:
                    return False

        return True

    def solve_sudoku_bruteforce(self,count):
    # print()
    # print("Step")
    # print_sudoku(board)

        empty_location = self.find_empty_cell()
        # print(empty_location)
        # count[0] = count[0] + 1
        # If there is no empty location, the sudoku is solved
        # if count[0]%1000000==0:
        #     print_sudoku(board)
        #     print(count[0])
        count[0] = count[0] + 1
        if not empty_location:

            if self.is_sudoku_solved():
                print("solved")
                return True
            else:
                # print("not solved")
                return False

        row, col = empty_location

        # Try placing numbers from 1 to 9
        for num in range(1, 10):
            # if is_valid(board, row, col, num):
                # Place the number if it's valid
            self.puzzle[row][col] = num

                # Recursively try to solve the remaining board
            if self.solve_sudoku_bruteforce(count):
                # print("in if")
                return True


                # If placing the number leads to an invalid solution, backtrack
            self.puzzle[row][col] = 'X'

        return False


    

    def solve_backtracking(self):
        #self.start_time = time.time()
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
        #self.start_time = time.time()
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

    def print_solution(self, filename):
        start_time = time.time()
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
        
        print("Solved puzzle:")
        for row in self.puzzle:
            print(','.join(str(cell) if cell != 'X' else 'X' for cell in row))
        print("\nSaving solution to", sys.argv[2].split('.')[0] + "_SOLUTION.csv...")
        with open(sys.argv[2].split('.')[0] + "_SOLUTION.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in self.puzzle:
                writer.writerow(row)


        print("Search time: %s" % (time.time() - start_time))




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
    count = [0]
    if mode == '1':
        solver.solve_sudoku_bruteforce(count)
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


# In[ ]:




