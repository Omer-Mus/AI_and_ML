from __future__ import division
from __future__ import print_function

import sys
import math
import time
import queue as Q
from collections import deque
import psutil


#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """

    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n * n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n * n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.config = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

        #  Depth variable
        self.depth = 0

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3 * i: 3 * (i + 1)])

    def move_up(self):
        """
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index - self.n < 0:  # if cant move
            pass
        else:
            nState = PuzzleState(list(self.config), self.n, self, 'Up', 0)
            currBlank = nState.blank_index
            nState.config[nState.blank_index] = nState.config[nState.blank_index - nState.n]
            nState.config[currBlank - nState.n] = 0
            nState.blank_index = currBlank - nState.n
            nState.depth = self.depth + 1
            return nState

    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index + self.n > self.n ** 2 - 1:  # if cant move
            pass
        else:
            nState = PuzzleState(list(self.config), self.n, self, 'Down', 0)
            currBlank = nState.blank_index
            nState.config[self.blank_index] = nState.config[nState.blank_index + nState.n]
            nState.config[currBlank + nState.n] = 0
            nState.blank_index = currBlank + nState.n
            nState.depth = self.depth + 1
            return nState

    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        if self.blank_index % self.n == 0:  # if cant move
            pass
        else:
            nState = PuzzleState(list(self.config), self.n, self, 'Left', 0)
            currBlank = nState.blank_index
            nState.config[nState.blank_index] = nState.config[nState.blank_index - 1]
            nState.config[currBlank - 1] = 0
            nState.blank_index = currBlank - 1
            nState.depth = self.depth + 1
            return nState

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        # For eight puzzle, index 2,5,8 -> 3,6,9
        if (self.blank_index + 1) % self.n == 0:  # if cant move
            pass
        else:
            nState = PuzzleState(list(self.config), self.n, self, 'Right', 0)
            currBlank = nState.blank_index
            nState.config[nState.blank_index] = nState.config[nState.blank_index + 1]
            nState.config[currBlank + 1] = 0
            nState.blank_index = currBlank + 1
            nState.depth = self.depth + 1
            return nState

    def expand(self):
        """ Generate the child nodes of this node """

        # Node has already been expanded
        if len(self.children) != 0:
            return self.children

        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children


# Students need to change the method to have the corresponding parameters
def writeOutput(goal, depth, max_depth, expanded):
    directions = []
    while goal.action != 'Initial':
        directions.append(goal.action)
        goal = goal.parent
    original_stdout = sys.stdout
    with open('output.txt', 'w') as f:
        sys.stdout = f
        print(f"""path_to_goal: {directions[::-1]}
cost_of_path: {len(directions)}
nodes_expanded: {expanded}
search_depth: {depth}
max_search_depth: {max_depth}""")
        sys.stdout = original_stdout


def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###

    if test_goal(initial_state):
        writeOutput(initial_state, 0, 0, 0)
        return True

    # frontier = []
    # frontier.append(initial_state)
    frontier = {}
    frontier[tuple(initial_state.config)] = initial_state
    explored = {}
    max_depth = 0
    expanded = 0
    while frontier:
        curr = frontier.pop(list(frontier.keys())[0])
        explored[tuple(curr.config)] = curr
        search_depth = curr.depth
        if not test_goal(curr):
            expanded += 1
            max_depth = search_depth + 1
            exp = curr.expand()
            for i in exp:
                if i.depth > max_depth:
                    max_depth = i.depth
                if tuple(i.config) not in frontier.keys():
                    if tuple(i.config) not in explored.keys():
                        # frontier.append(i)
                        frontier[tuple(i.config)] = i
        else:
            writeOutput(curr, search_depth, max(search_depth, max_depth), expanded)
            # curr.display()
            return True
    return False


def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    #  Case 1
    if test_goal(initial_state):
        writeOutput(initial_state, 0, 0, 0)
        return True

    frontier = {}
    # frontier.append(initial_state)
    frontier[tuple(initial_state.config)] = initial_state
    explored = {}
    max_depth = 0
    expanded = 0
    while frontier:
        curr = frontier.popitem()[1]
        # curr = frontier.popleft()
        explored[tuple(curr.config)] = curr
        search_depth = curr.depth
        if not test_goal(curr):
            expanded += 1
            max_depth = max(search_depth + 1, max_depth)
            exp = curr.expand()
            exp.reverse()
            for i in exp:
                if i.depth > max_depth:
                    max_depth = i.depth
                if tuple(i.config) not in frontier.keys():
                    if tuple(i.config) not in explored.keys():
                        frontier[tuple(i.config)] = i
                        # frontier.append(i)
        else:
            writeOutput(curr, search_depth, max(search_depth, max_depth), expanded)
            return True
    return False


def A_star_search(initial_state):
    """A * search"""
    ### STUDENT CODE GOES HERE ###

    if test_goal(initial_state):
        writeOutput(initial_state, 0, 0, 0)
        return True

    frontier = Q.PriorityQueue()
    explored = {}
    total_cost = {}
    cost = calculate_total_cost(initial_state)
    frontier.put((cost, initial_state.config))
    total_cost[(cost, tuple(initial_state.config))] = initial_state
    max_depth = 0
    expanded = 0
    while frontier:
        item = frontier.get()
        curr = total_cost[(item[0], tuple(item[1]))]
        explored[tuple(curr.config)] = curr
        search_depth = curr.depth
        if not test_goal(curr):
            max_depth = max(search_depth + 1, max_depth)
            expanded += 1
            exp = curr.expand()
            for i in exp:
                if tuple(i.config) not in explored.keys():
                    new_cost = calculate_total_cost(i)
                    frontier.put((new_cost, i.config))
                    total_cost[(new_cost, tuple(i.config))] = i
        else:
            writeOutput(curr, search_depth, max(search_depth, max_depth), expanded)
            return True

    return False


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    cost = state.depth
    for i in range(state.n ** 2):
        if state.config[i] != 0:
            cost += calculate_manhattan_dist(i, state.config[i], state.n)
    return cost


def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    return abs(idx % n - value % n) + abs(idx // n - value // n)


def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    ### STUDENT CODE GOES HERE ###
    return puzzle_state.config == sorted(puzzle_state.config)


# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, board_size)
    start_time = time.time()

    if search_mode == "bfs":
        bfs_search(hard_state)
    elif search_mode == "dfs":
        dfs_search(hard_state)
    elif search_mode == "ast":
        A_star_search(hard_state)
    else:
        print("Enter valid command arguments !")

    original_stdout = sys.stdout
    with open('output.txt', 'a') as f:
        sys.stdout = f
        end_time = time.time()
        #  Changed 3 to 8
        print("Program completed in %.8f" % (end_time - start_time))
        process = psutil.Process()
        max_ram_usage = (process.memory_full_info().uss) * (10 ** -6)  # Compiling on macOS Big Sur.
        print("max_ram_usage: %.8f" % (max_ram_usage / 1000))
        sys.stdout = original_stdout
if __name__ == '__main__':
    main()
