# !/usr/bin/env python
# coding:utf-8
"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8
"""
import sys
import time
from statistics import mean, stdev
import filecmp

ROW = "ABCDEFGHI"
COL = "123456789"


def square_index(row, col, i, j):
    r = ROW.find(row)
    c = (int(col) - 1)
    return ROW[r // 3 * 3 + i], COL[c // 3 * 3 + j]


def print_board(board):
    """Helper function to print board in a square."""
    print("-----------------")
    for i in ROW:
        row = ''
        for j in COL:
            row += (str(board[i + j]) + " ")
        print(row)


def board_to_string(board):
    """Helper function to convert board dictionary to string for writing."""
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return ''.join(ordered_vals)


def is_consistent(board, value, row, col):
    for i in range(9):
        if (board[row + COL[i]] == value) or (board[ROW[i] + col] == value):
            return False
    for i in range(3):
        for j in range(3):
            r, c = square_index(row, col, i, j)
            if board[r + c] == value:
                return False
    return True


def arc_consistency(board, domain, row, col):
    for c in range(9):
        if board[row + COL[c]] in domain[row + col]:
            domain[row + col].remove(board[row + COL[c]])
            if not domain[row + col]:
                return False
        if board[ROW[c] + col] in domain[row + col]:
            domain[row + col].remove(board[ROW[c] + col])
            if not domain[row + col]:
                return False

    for i in range(3):
        for j in range(3):
            r, c = square_index(row, col, i, j)
            if board[r + c] in domain[row + col]:
                domain[row + col].remove(board[r + c])
                if not domain[row + col]:
                    return False
    return True


def revised(board, x1, x2):
    pass

def backtracking(board):
    """Takes a board and returns solved board."""
    # TODO: implement this
    li = [_ for _ in range(1, 10)]  # Forming domain
    csp = {(r + c): list(li) for r in ROW for c in COL if board[r + c] == 0}

    for key in csp.keys():
        if not arc_consistency(board, csp, key[0], key[1]):
            return False

    solved_board = backtracking_search(board, csp)
    return solved_board


def backtracking_search(board, csp):
    if 0 not in board.values():
        return board

    #  find mrv heuristic
    vsize = 10
    variable = ''
    for k, v in csp.items():
        if board[k] == 0 and vsize > len(v):
            variable = k
            vsize = len(v)

    row, col = variable[0], variable[1]
    for value in csp[variable]:
        if is_consistent(board, value, row, col):
            forward_checking = {k: v[:] for k, v in csp.items()}
            flag = True
            for i in range(9):
                if COL[i] != col and row + COL[i] in forward_checking.keys():
                    temp = forward_checking[row + COL[i]]
                    if value in temp:
                        temp.remove(value)
                        if not temp:
                            flag = False
                if ROW[i] != row and ROW[i] + col in forward_checking.keys():
                    temp = forward_checking[ROW[i] + col]
                    if value in temp:
                        temp.remove(value)
                        if not temp:
                            flag =  False
            for i in range(3):
                for j in range(3):
                    r, c = square_index(row, col, i, j)
                    if r != row and c != col and r + c in forward_checking.keys():
                        temp = forward_checking[r + c]
                        if value in temp:
                            temp.remove(value)
                            if not temp:
                                flag = False
            if flag:
                board[row + col] = value
                result = backtracking_search(board, forward_checking)
                if result:
                    return result
                else:
                    board[row + col] = 0
    return False

if __name__ == '__main__':
    if len(sys.argv) > 1:

        # Running sudoku solver with one board $python3 sudoku.py <input_string>.
        print(sys.argv[1])
        # Parse boards to dict representation, scanning board L to R, Up to Down
        board = {ROW[r] + COL[c]: int(sys.argv[1][9 * r + c])
                 for r in range(9) for c in range(9)}
        solved_board = backtracking(board)

        # Write board to file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        outfile.write(board_to_string(solved_board))
        outfile.write('\n')
    else:
        # Running sudoku solver for boards in sudokus_start.txt $python3 sudoku.py
        #  Read boards from source.
        src_filename = 'sudokus_start.txt'
        try:
            srcfile = open(src_filename, "r")
            sudoku_list = srcfile.read()
        except:
            print("Error reading the sudoku file %s" % src_filename)
            exit()
        # Setup output file
        out_filename = 'output.txt'
        outfile = open(out_filename, "w")
        # Solve each board using backtracking

        for line in sudoku_list.split("\n"):
            if len(line) < 9:
                continue
                # Parse boards to dict representation, scanning board L to R, Up to Down
            board = {ROW[r] + COL[c]: int(line[9 * r + c])
                     for r in range(9) for c in range(9)}
            # Print starting board. TODO: Comment this out when timing runs.
            # print_board(board)
            # Solve with backtracking
            solved_board = backtracking(board)
            # Print solved board. TODO: Comment this out when timing runs.
            # print_board(solved_board)
            # Write board to file
            outfile.write(board_to_string(solved_board))
            outfile.write('\n')
    print("Finishing all boards in file.")
