import csv
import numpy as np
import time
import sys

# method to print  algoname
def get_algoname(mode):
    if mode==1:
        print("Brute Force Search")
    elif mode==2:
        print("CSP Back-Tracking search")
    elif mode==3:
        print("CSP with Forward-Checking and MRV heuristics")
    elif mode==4:
        print("Test")

# method to check assigntment at board[row][col] of num is valid or not
def is_valid(board, row, col, num):
    # Check if the number is not present in row or column
    for i in range(9):
        # print("check" ,board[row][i] , " " , board[i][col] , " " ,num )
        if board[row][i] == num and board[row][i]!='X':
            # print("issue")
            return False
        if board[i][col] == num and board[i][col]!='X':
            # print("issue")
            return False

    # Check if the number is not present in the 3x3 grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            # print("check", board[i + start_row][j + start_col] , " ", num)
            if board[i + start_row][j + start_col] == num:
                # print("issue")
                return False

    return True

# method to check if sudoku solved or not
def is_sudoku_solved(board):
    # Check if each row contains all nums
    all_nums=['1','2','3','4','5','6','7','8','9']
    for row in board:
        if sorted(row) != all_nums:
            # print(sorted(row))
            # print("issue1")
            return False

    # Check if each column contains all nums
    for col in range(9):
        if sorted(board[:, col]) != all_nums:
            # print(sorted(board[:, col]))
            # print("issue2")
            return False

    # Check if each 3x3 subgrid contains all nums
    for row in range(0, 9, 3):
        for col in range(0, 9, 3):
            subgrid = board[row:row+3, col:col+3]
            if sorted(subgrid.flatten()) != all_nums:
                # print("issue3")
                return False
    # print("Solved puzzle:")
    # print_sudoku(board)
    return True

# method to find empty first location returns tuple
def find_empty_location(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 'X':
                return i, j
    return None
# method to return all empty locations in list of tuple
def find_empty_location_all(board):
    # Find all empty cells
    empty_cells = [(i, j) for i in range(9) for j in range(9) if board[i][j] == 'X']
    # print(unassigned_cells)
    return empty_cells

# method to solve sudoku with bruteforce search
def solve_sudoku_bruteforce(board,count):
    # print()
    # print("Step")
    # print_sudoku(board)

    empty_location = find_empty_location(board)
    # print(empty_location)
    # count[0] = count[0] + 1
    # If there is no empty location, the sudoku is solved
    # if count[0]%1000000==0:
    #     print_sudoku(board)
    #     print(count[0])
    count[0] = count[0] + 1
    if not empty_location:

        if is_sudoku_solved(board):
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
        board[row][col] = num

            # Recursively try to solve the remaining board
        if solve_sudoku_bruteforce(board,count):
            # print("in if")
            return True


            # If placing the number leads to an invalid solution, backtrack
        board[row][col] = 'X'

    return False

# method to solve sudoku with csp
def solve_sudoku_csp(board,count):
    empty_location = find_empty_location(board)
    # print("step")
    count[0] = count[0] + 1
    # print_sudoku(board)
    # If there is no empty location, the sudoku is solved
    if not empty_location:
        return True
    # count[0] = count[0] + 1
    row, col = empty_location

    # Try placing numbers from 1 to 9
    for num in range(1, 10):
        if is_valid(board, row, col, str(num)):
            # Place the number if it's valid
            board[row][col] = str(num)
            # print_sudoku(board)
            # Recursively try to solve the remaining board
            if solve_sudoku_csp(board,count):
                return True

            # If placing the number leads to an invalid solution, backtrack
            board[row][col] = 'X'

    return False

# method to get remaining values for board[row][col] cell from 1 to 9 [forward checking]

def get_remaining_values(board, row, col):
    # values = set(range(1, 10))
    values = set(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    # Remove values in the same row and column
    values -= set(board[row][i] for i in range(9))
    values -= set(board[i][col] for i in range(9))

    # Remove values in the 3x3 grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    values -= set(board[i + start_row][j + start_col] for i in range(3) for j in range(3))

    return list(values)


def solve_sudoku_csp_mrv(board,count):
    # print("step")
    count[0]=count[0]+1
    unassigned_cells = find_empty_location_all(board)
    # print(unassigned_cells)
    if not unassigned_cells:
        return True  # Solution found

    # Select the unassigned cell with the Minimum Remaining Values (MRV)
    # for cel in unassigned_cells:
    #     print(cel , get_remaining_values(board, cel[0], cel[1]))
    row, col = min(unassigned_cells, key=lambda cell: len(get_remaining_values(board, cell[0], cell[1])))
    # print(row, " " ,col)
    for num in get_remaining_values(board, row, col):
        if is_valid(board, row, col, num):
            board[row][col] = num

            # Forward checking: Remove num from remaining values in related cells
            if solve_sudoku_csp_mrv(board,count):
                return True  # Solution found

            # If placing the number leads to an invalid solution, backtrack
            board[row][col] = 'X'

    return False
def print_sudoku(board):
    for i in range(9):
        for j in range(9):
            print(board[i][j], end=" ")
        print()

def read_sudoku_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        sudoku_board = [list(row) for row in reader]
    return sudoku_board

def main(mode, filename):
    # Replace 'input.csv' with the path to your CSV file
    start_time = time.time()
    csv_file_path = filename
    sudoku_board = read_sudoku_from_csv(csv_file_path)
    sudoku_board=np.array(sudoku_board)

    # print(is_sudoku_solved(sudoku_board))
    print(" Input Sudoku:")
    print_sudoku(sudoku_board)
    count=[0]
    if mode==1:
        if solve_sudoku_bruteforce(sudoku_board,count):
            print("\nSolved Sudoku:")
            print_sudoku(sudoku_board)
            print("Number of search tree nodes generated :  ",count)
        else:
            print("Number of search tree nodes generated :  ", count)
            print("\nNo solution exists.")
    if mode==2:
        if solve_sudoku_csp(sudoku_board,count):
            print("\nSolved Sudoku:")
            print_sudoku(sudoku_board)
            print("Number of search tree nodes generated :  ",count)
        else:
            print("Number of search tree nodes generated :  ", count)
            print("\nNo solution exists.")
    if mode==3:
        if solve_sudoku_csp_mrv(sudoku_board,count):
            print("\nSolved Sudoku:")
            print_sudoku(sudoku_board)
            print("Number of search tree nodes generated : ",count)
        else:
            print("Number of search tree nodes generated : ", count)
            print("\nNo solution exists.")
    if mode==4:
        if is_sudoku_solved(sudoku_board):
            print("This is a valid, solved, Sudoku puzzle.")
        else:
            print("ERROR: This is NOT a solved Sudoku puzzle.")
    print("Search time: %s" % (time.time() - start_time))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Not enough/too many/illegal input arguments.")
        sys.exit(1)

    mode = int(sys.argv[1])
    filename = sys.argv[2]
    if 1>mode or mode>4:
        print("ERROR: Invalid mode. Mode should be 1, 2, 3, or 4.")
        sys.exit(1)
    print("Patel, Nirmal A20531914 solution:")
    print("Input file: ",filename)
    print("Algorithm: ",get_algoname(mode))
    main(mode, filename)
