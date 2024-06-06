'''
Name: Aman Kumar
Hawk Id: A20538809
Cs480
Programming Assignment 02
'''

import csv
import numpy as np
import time
import sys


#display the sudoku solved and unsolved
def display(puzzle):
    formatted_puzzle = []
    for row in puzzle:
        formatted_row = [str(cell) if cell != 'X' else 'X' for cell in row]
        formatted_puzzle.append(','.join(formatted_row))
    print( '\n'.join(formatted_puzzle))

#load the sudoku values through this function
def load_sudoku(file_path):
    with open(file_path, 'r') as file:
        puzzle = [list(row) for row in csv.reader(file)]
    return puzzle

#find the coordinates of first empty cell ie.. X
def empty_cell(puzzle):
    for row in range(len(puzzle)):
        for col in range(len(puzzle)):
            if puzzle[row][col] == 'X':
                return row, col
    return None

#checking the validity is a given number can be 
#placed at a specified position in the sudoku puzzle
def is_valid(puzzle, row, col, num):
    return (not used_in_row(puzzle,row, num) and
                not used_in_col(puzzle,col, num) and
                not used_in_subgrid(puzzle,row - row % 3, col - col % 3, num))


#if a number is already present in a row 
def used_in_row(puzzle, row, num):
        return str(num) in puzzle[row]


#if a number is already present in the column
def used_in_col(puzzle, col, num):
        return str(num) in [puzzle[row][col] for row in range(len(puzzle))]


#check if a given number is already present in the subgrid starting from the 
#start row and start col
def used_in_subgrid(puzzle, start_row, start_col, num):
    for row in range(3):
        for col in range(3):
            if puzzle[row + start_row][col + start_col] == str(num):
                return True
    return False



#for brute force search algorithm
def bruteforce_solve(puzzle,count):
    
    #checks if there are any empty cells left
    emptyCell = empty_cell(puzzle)
    
    #if there are no empty cells 

    #if there are still empty cells
    #it continues with the solving process
    count[0] = count[0] + 1
    if not emptyCell:
        
        #check the current configuration if it gives a valid solution
        if test_sudoku(puzzle):
            return True
    
        else:
            return False

    row, col = emptyCell

    #trying all possible numbers using brute force aproach 
    #in empty cells
    for i in range(1, len(puzzle)+1):
        
        puzzle[row][col] = i

        #recursilvely searching for a solution 
        #if not found it backtracks and tries a different number
        if bruteforce_solve(puzzle,count):
            return True

        puzzle[row][col] = 'X'

    return False



#method to determine the backtracking for CSP backtracking search
def backtracking(puzzle, nodes):
    try:
        #co ordinates of next empty cell
        emptyCell = empty_cell(puzzle)
        
        #number of nodes visited during each search
        nodes[0] += 1
        
        if not emptyCell:
            return True
        
        #current empty cell
        row, col = emptyCell
        
        for i in range(1, len(puzzle)+1):
            if is_valid(puzzle, row, col, str(i)):
                
                puzzle[row][col] = str(i)
                
                #calls backtracking function with updated parameters
                #if true means puzzle is solved 
                if backtracking(puzzle, nodes):
                    return True

                
                puzzle[row][col] = 'X'
        
        return False

    except Exception as e:
        print("An error occurred:", e)



#for Constraint Satisfaction Problem back-tracking search,
def solve_csp_backtracking(puzzle, nodes):
        
    if backtracking(puzzle, nodes):
        return True
    else:
        return False





#function to determine the possible values to be placed in the sudoku cell without violating the rules
def leftover_sudoku(puzzle, row, col):
    
    try:

        #to filter out the values that are already present in same row and column
        #thus it leaves only the possible values that can be placed in a specific cell
        val = set(str(i) for i in range(1, len(puzzle)+1))
        val -= {puzzle[row][i] for i in range(len(puzzle))}
        val -= {puzzle[i][col] for i in range(len(puzzle))}


        #starting row and column of the puzzle in the subgrid
        start_row = (row // 3) * 3
        start_col = (col // 3) * 3


        #identifies values already present in 3x3 subgrid 
        #returns the possible remaining values 
        subgrid_values = set()
        for i in range(3):
            for j in range(3):
                subgrid_values.add(puzzle[i + start_row][j + start_col])
        val -= subgrid_values

        return list(val)
    except Exception as e:
        print("An error occurred:", e)
        return None

#function to find all empty cells
def all_empty_cells(puzzle):
    
    empty_cells = [(i, j) for i in range(9) for j in range(9) if puzzle[i][j] == 'X']
    
    return empty_cells

#CSP with forward-checking and MRV heuristics
def csp_mrv_frwdCheck(puzzle,nodes):
   
    try:
        #count nodes at each search
        nodes[0] += 1

        #retreives all empty cells in sudoku puzzle
        emptyCell = all_empty_cells(puzzle)

        #checks if no empty cells left in the puzzle
        if not emptyCell:
            return True


        #finding empty cell with minimum number of remaining values
        min_leftover = float('inf')

        #iterate through every cell in the empty cell
        for cell in emptyCell:
            row, col = cell
            leftover = len(leftover_sudoku(puzzle, row, col))
            if leftover < min_leftover:
                min_leftover = leftover
                min_row, min_col = row, col
        #return the row and column with min leftover values
        row, col = min_row, min_col

        #iterate over each possible number 
        for num in leftover_sudoku(puzzle, row, col):
            #check validity of each number by calling the is_valid method
            if is_valid(puzzle, row, col, num):
                #if valid, place the number in the cell
                puzzle[row][col] = num
                #recursively call the function to solve the puzzle further
                if csp_mrv_frwdCheck(puzzle, nodes):
                    #solution found 
                    return True
                puzzle[row][col] = 'X'
        #if no solution is found 
        #return false
        return False
    except Exception as e:
        print("An error occurred:", e)
        return False



def algorithm(mode):
    try:
        if mode == 1:
            return "Brute Force Search"
        elif mode == 2:
            return "CSP Back-Tracking search"
        elif mode == 3:
            return "CSP with Forward-Checking and MRV heuristics"
        elif mode == 4:
            return "Test puzzle"
    except Exception as e:
        print("An error occurred:", e)
        return None

    
#test if the completed puzzle is correct
def test_sudoku(board):
    
    try:
        n = len(board[0])
        sqrt_n = int(n ** 0.5)

        solved_puzzle = [[int(num) if num != 'X' else num for num in row] for row in board]

        for row in solved_puzzle:
            if 'X' in row:
                return False
            
        for row in solved_puzzle:
            if len(set(row)) != n:
                return False
            
        for col in range(len(board)):
            if len(set(row[col] for row in solved_puzzle)) != n:
                return False
            
        for i in range(0, n, sqrt_n):
            for j in range(0, n, sqrt_n):
                box = []
                for x in range(sqrt_n):
                    for y in range(sqrt_n):
                        box.append(solved_puzzle[i + x][j + y])
                if len(set(box)) != n:
                    return False
        
        return True

    except Exception as e:
        print("An error occurred:", e)
        return False

    

def main(mode, filename):
    
    start_time = time.time()
    filepath = filename
    puzzle = load_sudoku(filepath)
    puzzle=np.array(puzzle)

    
    print("Input Puzzle:")
    display(puzzle)
    print('\n')
    nodes = [0]

    #for brute force search algorithm
    if mode==1:
        if bruteforce_solve(puzzle,nodes):
            print("Number of search tree nodes generated :  ",nodes)
            print("Search time: %s seconds" % (time.time() - start_time))
            print("\nSolved Puzzle:")
            display(puzzle)
            
        else:
            print("Number of search tree nodes generated :  ", nodes)
            print("\nNo solution exists.")

    #for Constraint Satisfaction Problem back-tracking search,
    if mode==2:
        if solve_csp_backtracking(puzzle, nodes):
        
            print("Number of search tree nodes generated :  ",nodes)
            print("Search time: %s seconds" % (time.time() - start_time))
            print("\nSolved Puzzle:")
            display(puzzle)
            
        else:
            print("Number of search tree nodes generated :  ", nodes)
            print("\nNo solution exists.")

    # CSP with forward-checking and MRV heuristics
    if mode==3:
        if csp_mrv_frwdCheck(puzzle,nodes):
            print("Number of search tree nodes generated :  ",nodes)
            print("Search time: %s seconds" % (time.time() - start_time))
            print("\nSolved Puzzle:")
            display(puzzle)
            
        else:
            print("Number of search tree nodes generated : ", nodes)
            print("\nNo solution exists.")

    #test if the completed puzzle is correct
    if mode==4:
        if test_sudoku(puzzle):
            print("This is a valid, solved, Sudoku puzzle.")
            print("Search time: %s seconds" % (time.time() - start_time))
        else:
            print("ERROR: This is NOT a solved Sudoku puzzle.")
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Not enough/too many/illegal input arguments.")
        sys.exit(1)

    mode = int(sys.argv[1])
    filename = sys.argv[2]
    if mode not in [1,2,3,4]:
        print("ERROR: Illegal mode input. Mode should be 1, 2, 3, or 4.")
        sys.exit()
    print("Kumar, Aman, A20538809 solution:")
    print("Input file: ",filename)
    print("Algorithm: ",algorithm(mode))
    main(mode, filename)
