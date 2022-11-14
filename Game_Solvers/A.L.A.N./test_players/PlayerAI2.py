import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid
from Utils import manhattan_distance


TRAP = -1
MAX_DEPTH = 5
# TO BE IMPLEMENTED
# 
class PlayerAI2(BaseAI):

    def __init__(self,player_num=1, opp_num=2) -> None:
        # You may choose to add attributes to your player - up to you!
        super().__init__()
        self.pos = None
        self.player_num = player_num
        self.opp_num = opp_num
    
    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position 

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        # available_moves = grid.get_neighbors(self.pos, only_available = True)

        # # make random move
        # new_pos = random.choice(available_moves) if available_moves else None
        child, _, mvs = self.maximizeMove(grid, float('-inf'), float('inf'))
        # print('possible moves')
        # print(mvs)
        return self.get_move(child, grid.map)
        return new_pos

    def getOpponentPos(self, grid):
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                # print(grid[i][j])
                if(grid[i][j] == self.opp_num):
                    # print(grid)
                    return (i,j)
        # print('space')
        # print(grid)
        return -1

    def getMyPos(self, grid):
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                # print(grid[i][j])
                if(grid[i][j] == self.player_num):
                    # print(grid)
                    return (i,j)
        # print('space')
        # print(grid)
        return -1

    def getChildrenThrow(self, _grid, player):
        grid = _grid.map
        grids = []
        probs=  []
        if(player == self.player_num):
            x,y = self.getOpponentPos(grid)
            x1, y2 = self.getMyPos(grid)
            # print('grid: \n',grid)
        else:
            x,y = self.getMyPos(grid)
            x1, y2 = self.getOpponentPos(grid)
        
        for i in range(max(0, x-2), min(len(grid), x+3)):
            for j in range(max(0, y-2), min(len(grid[i]), y+3)):
                if(grid[i][j] == self.opp_num or grid[i][j] == self.player_num):
                    continue
                if(grid[i][j] != TRAP):
                    neighbors = _grid.get_neighbors((i,j), only_available=True)
                    prob = 1 - 0.05 * (manhattan_distance([x1,y2],[i,j]) - 1)
                    aux = _grid.clone()
                    # aux = np.copy(grid)
                    aux.map[i][j] = TRAP
                    grids.append(aux)
                    probs.append(prob)
        grids = np.array(grids)
        # if(player == self.player_num):
        #     print(np.array([g.map for g in grids]))
        return (grids, probs)

    def getChildrenMove(self, _grid, player):
        grid = _grid.map
        grids = []
        if(player == self.opp_num):
            x,y = self.getOpponentPos(grid)
            # print('opposite position:', '(',x, y,')')
        else:
            x,y = self.getMyPos(grid)
            # print(grid)
            # print('my position', '(',x, y,')')

        for i in range(max(0, x-1), min(len(grid), x+2)):
            for j in range(max(0, y-1), min(len(grid[i]), y+2)):
                if(grid[i][j] == self.player_num or grid[i][j] == self.opp_num):
                    continue
                if(grid[i][j] != TRAP):
                    # aux = np.copy(grid)
                    aux = _grid.clone()
                    aux.map[i][j] = player
                    aux.map[x][y] = 0
                    grids.append(aux)
        grids = np.array(grids)
        # if(player == self.player_num):
        # print(grids)    
        return grids

    def utilityFn(self, _grid):
        grid = _grid.map
        x,y = self.getOpponentPos(grid)
        # print(grid)
        # print(x,y)
        # i,j = self.getMyPos(grid)
        pos = (x,y)
        return 8 - len(_grid.get_neighbors(pos,only_available=True))
        counter = 0
        for i in range(max(0, x-2), min(len(grid), x+3)):
            for j in range(max(0, y-2), min(len(grid[i]), y+3)):
                if(grid[i][j] == TRAP):
                    counter += 5/manhattan_distance([x,y], [i,j])
        if(x ==0 or x == 6):
            counter += 2
        if(y == 0 or y == 6):
            counter += 2
        # print('counter: ',counter)
        return counter

    def moveUtilityFn(self, _grid):
        grid = _grid.map
        x,y = self.getMyPos(grid)
        # x1,y1 = self.getOpponentPos(grid)
        # return 10 - manhattan_distance([x,y], [x1,y1])
        counter = 0
        for i in range(max(0, x-2), min(len(grid), x+3)):
            for j in range(max(0, y-2), min(len(grid[i]), y+3)):
                if(grid[i][j] == TRAP):
                    counter += 1 + 2*manhattan_distance([x,y], [i,j])

        return counter

    def isFinalState(self, _grid):
        grid = _grid.map
        x,y = self.getOpponentPos(grid)
        for i in range(max(0, x-1), min(len(grid), x+2)):
            for j in range(max(0, y-1), min(len(grid[i]), y+2)):
                if(i == x and j == y):
                    continue
                if(grid[i][j] != TRAP):
                    return False
        return True

    def isFinalMoveState(self, _grid):
        # grid = _grid.map
        # x,y = self.getMyPos(grid)
        # for i in range(max(0, x-1), min(len(grid), x+2)):
        #     for j in range(max(0, y-1), min(len(grid[i]), y+2)):
        #         if(i == x and j == y):
        #             continue
        #         if(grid[i][j] != TRAP):
        #             return False
        # return True
        neighbors = _grid.get_neighbors(self.getMyPos(_grid.map), only_available = True)
        if len(neighbors) == 0:
            return True
        return False

    def maximizeMove(self, grid, alpha, beta, depth=0):
        if(depth == MAX_DEPTH):
            return (None, self.moveUtilityFn(grid), None)
        if(self.isFinalMoveState(grid)):
            print('depth:', depth)
            return (None, self.moveUtilityFn(grid), None)

        max_child, max_utility = (None, float('-inf'))
        mvs = self.getChildrenMove(grid, self.player_num)
        if(len(mvs) == 0):
            print(grid)
            exit('SOMETHING AWFUL')
        for child in mvs:
            _, utility = self.minimizeThrow(child, alpha, beta, depth+1)
            if(utility > max_utility):
                max_utility, max_child = utility, child
            if(max_utility >= beta):
                break
            if(max_utility > alpha):
                alpha = max_utility
        return (max_child, max_utility, mvs)

    def minimizeThrow(self, grid, alpha, beta, depth=1):
        if(depth == MAX_DEPTH):
            return (None, self.moveUtilityFn(grid))
        if(self.isFinalMoveState(grid)):
            return (None, self.moveUtilityFn(grid))
        min_child, min_utility = (None, float('inf'))
        thrs, probs = self.getChildrenThrow(grid, self.opp_num)
        for child, prob in zip(thrs, probs):
            _, utility, _ = self.maximizeMove(child, alpha, beta, depth+1)
            if(utility < min_utility):
                min_utility, min_child = utility, child
            if(min_utility <= alpha):
                break
            if(min_utility < beta):
                beta = min_utility
        return (min_child, min_utility)

    def minimizeMove(self, grid, alpha, beta, depth=1):
        # spaces = "\t"*depth
        # print(f'{spaces} min')
        if(depth == MAX_DEPTH):
            return (None, self.utilityFn(grid))
        if(self.isFinalState(grid)):
            return (None, self.utilityFn(grid))
        min_child, min_utility = (None, float('inf'))
        for child in self.getChildrenMove(grid, self.opp_num):
            _, utility, _ = self.maximizeThrow(child, alpha, beta, depth+1)
            if(utility < min_utility):
                min_utility, min_child = utility, child
            if(min_utility <= alpha):
                break
            if(min_utility < beta):
                beta = min_utility
        # print('min move')
        # print(min_child.map)
        # print(min_utility)
        return (min_child, min_utility)

    def maximizeThrow(self, grid, alpha, beta,depth=0):
        # spaces = "\t"*depth
        # print(f'{spaces} max')
        if(depth == MAX_DEPTH):
            return (None, self.utilityFn(grid), None)
        if(self.isFinalState(grid)):
            return (None, self.utilityFn(grid) + 100, None)

        max_child, max_utility = (None, float('-inf'))
        thrs, probs = self.getChildrenThrow(grid, self.player_num)
        for child, prob in zip(thrs, probs):
            _, utility = self.minimizeMove(child, alpha, beta, depth+1)
            utility = utility*prob
            if(utility > max_utility):
                max_utility, max_child = utility, child
            if(max_utility >= beta):
                break
            if(max_utility > alpha):
                alpha = max_utility
        # print('max throw')
        # print(max_child.map, max_utility)
        return (max_child, max_utility, thrs)

    def minProbThrow(self, grid, alpha, beta):
        pass

    def maxProbThrow(self, grid, alpha, beta):
        pass

    def get_move(self, child, grid):
        if(child == None):
            print(grid)
            exit()
        aux = np.where(child.map==self.player_num)
        # print(aux)
        return (aux[0][0], aux[1][0])

    def get_trap(self, child, grid):
        aux = np.where(child.map!=grid)
        # print(child.map)
        # print(grid)
        # print(aux)
        return (aux[0][0], aux[1][0])

    def getTrap(self, grid : Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        child, _, thrs = self.maximizeThrow(grid, float('-inf'), float('inf'))
        # print('throw options:')
        # print(np.array([t.map for t in thrs]))
        return self.get_trap(child, grid.map)
        return child
        

    