# Intro
he N-puzzle game consists of a board holding N = m2−1 distinct movable tiles, plus one empty space. There is\
one tile for each number in the set {0, 1,..., m2−1}. In this assignment, we will represent the blank space with the \
number 0 and focus on the m = 3 case (8-puzzle).
In this combinatorial search problem, the aim is to get from any initial board state to the configuration with all\
tiles arranged in ascending order {0, 1,..., m2−1} – this is your goal state. The search space is the set of all possible\
states reachable from the initial state. Each move consists of swapping the empty space with a component in one of\
the four directions {‘Up’, ‘Down’, ‘Left’, ‘Right’}. Give each move a cost of one. Thus, the total cost of a path will\
be equal to the number of moves made.\
## Algorithm Review
Recall from lecture that search begins by visiting the root node of the search tree, given by the initial state. Three /
main events occur when visiting a node:/
• First, we remove a node from the frontier set.\
• Second, we check if this node matches the goal state.\
• If not, we then expand the node. To expand a node, we generate all of its immediate successors and add them\
to the frontier, if they (i) are not yet already in the frontier, and (ii) have not been visited yet.\
This describes the life cycle of a visit, and is the basic order of operations for search agents in this assignment–(1)\
remove, (2) check, and (3) expand. We will implement the assignment algorithms as described here. Please refer to\
lecture notes for further details, and review the lecture pseudo-code before you begin.

### Output
output.txt, containing the following statistics:\
path to goal: the sequence of moves taken to reach the goal\
cost of path: the number of moves taken to reach the goal\
nodes expanded: the number of nodes that have been expanded\
search depth: the depth within the search tree when the goal node is found\
max search depth: the maximum depth of the search tree in the lifetime of the algorithm\
running time: the total running time of the search instance, reported in seconds\
max ram usage: the maximum RAM usage in the lifetime of the process as measured by the ru maxrss attribute\
in the resource module, reported in megabytes\
