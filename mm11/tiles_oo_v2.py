import queue
from collections import deque
import numpy as np
import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0  # To handle cases with equal priorities

    def put(self, item, priority):
        entry = (priority, self.counter, item)
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def get(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item


class Stack:
    """ Notice we name the functions like in queue.Queue.
    A better architecture was to have those data structure implement the same abstract functions"""
    def __init__(self):
        self.items = deque()

    def put(self, item):
        self.items.append(item)

    def get(self):
        if not self.empty():
            return self.items.pop()
        else:
            raise IndexError("pop from an empty stack")

    def empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


class Node():
    def __init__(self, state, parent=None, action='', actioned_tile=None, cost=1, solved=False, **kwargs):
        """ parent => previous state,
        action => action made on parent to get us to current state.
        state => current board state
        actioned_tile => the number of the tile that moved (1-8) (for printing solution)
        cost => utility cost func
        solved => is this end node solves the problem (it's the end of a winning path)"""

        self.parent = parent
        self.action = action
        self.actioned_tile = actioned_tile
        self.state = state
        self.cost = cost
        self.solved = solved
        # Adding kwargs makes the Node dynamic for anything I want to do with it
        # (so I can use it across all algorithms)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_path(self) -> list:
        path = list()
        curr_node = self
        while curr_node:
            if curr_node.actioned_tile != None:
                path.append(curr_node.actioned_tile)
            curr_node = curr_node.parent

        return list(reversed(path))


class TilesGame():
    def __init__(self):
        self.winning_boards = self.get_winning_boards()
        arr = np.repeat(10, 9)
        pwr = np.arange(0, 9)
        self.mults = np.power(arr, pwr).reshape((3, 3))


    def is_final_state(self, board):
        """ return True if board == final state """
        for winning_board in self.winning_boards:
            if  np.all(board == winning_board):
                return True
        return False


    def get_winning_boards(self):
        """ return list of winning positions """
        win1 = np.arange(0, 9).reshape((3, 3))
        win2 = np.concatenate((np.arange(1, 9), [0])).reshape((3, 3))
        return [win1, win2]

    def hash_board(self, board):
        """ given board returns hashed value """
        _hash = np.sum(board * self.mults)
        return _hash


    @staticmethod
    def init_board(user_input):
        """ init a board game from user string. """
        board = np.fromstring(user_input, dtype=np.uint8, sep=' ')
        board = board.reshape((3, 3))
        return board

    @staticmethod
    def get_valid_moves(board, prev_move='') -> list:
        """ Valid moves are l, r, d, u (left, right, down, up)
        0 blocks from y:left and x:up, 2 blocks from x:down and y:right
        prev move cannot be countered (if u moved left, you cannot go right)
        returns dictionary of valid moves.
        """
        valid_moves = {'l', 'r', 'd', 'u'}
        opposite_moves = {'': '',
                          'l': 'r',
                          'r': 'l',
                          'd': 'u',
                          'u': 'd'}
        valid_moves.discard(opposite_moves[prev_move])
        empty_cell_loc = np.where(board == 0)

        # case x
        if empty_cell_loc[0] == 0:
            valid_moves.discard('u')
        if empty_cell_loc[0] == 2:
            valid_moves.discard('d')

        # case y
        if empty_cell_loc[1] == 0:
            valid_moves.discard('l')
        if empty_cell_loc[1] == 2:
            valid_moves.discard('r')

        return list(valid_moves)

    @staticmethod
    def get_swapping_indices(empty_cell_loc, move):
        """ given empty cell and the move, return the indices (x, y)
        of the cell needed to be swapped"""

        x, y = 0, 0

        ec_x = empty_cell_loc[0][0]
        ec_y = empty_cell_loc[1][0]

        if move == 'l':
            x = ec_x
            y = ec_y - 1
        if move == 'r':
            x = ec_x
            y = ec_y + 1

        if move == 'u':
            x = ec_x - 1
            y = ec_y
        if move == 'd':
            x = ec_x + 1
            y = ec_y

        return x, y

    @staticmethod
    def make_move(_board, move):
        """ only accept valid moves. validity responsibility is on the user
        change the board according to the move.
        returns new board, moved_tile """
        board = _board.copy()
        empty_cell_loc = np.where(board == 0)
        x, y = TilesGame.get_swapping_indices(empty_cell_loc, move)
        moved_tile = board[x, y]

        #board[empty_cell_loc], board[x, y] = board[x, y], board[empty_cell_loc]
        board[empty_cell_loc], board[x, y] = board[x, y].item(), board[empty_cell_loc].item()

        return board, moved_tile


class TilesAgent():
    def __init__(self, name, state, game):
        self.name = name
        self.explored = set()
        self.counter_expends = 0
        self.init_state = state
        self.game = game

class GBFSTilesAgent(TilesAgent):
    """ agent state is the games board """
    def __init__(self, state: np.ndarray, game: TilesGame):
        super().__init__(name="GBFS", state=state, game=game)
        # We use deque (which is 2 side queue. We push and pop from same side to simulate a stack)
        self.frontier = PriorityQueue()


class IDDFSTilesAgent(TilesAgent):
    """ agent state is the games board """
    def __init__(self, state: np.ndarray, game: TilesGame):
        super().__init__(name="IDDFS", state=state, game=game)
        # We use deque (which is 2 side queue. We push and pop from same side to simulate a stack)
        self.frontier = Stack()

    def solve(self, max_depth=20):
        for i in range(1, max_depth+1):
            solution = self.dfs_traverse(i)
            if solution is not None:
                return solution
        return None

    def dfs_traverse(self, depth=1):
        self.explored = set()

        state_hash = self.game.hash_board(self.init_state)
        #self.explored.add(state_hash)

        init_node = Node(self.init_state, depth=0, _hash=state_hash)
        self.frontier.put(init_node)

        while not self.frontier.empty():
            curr_node: Node = self.frontier.get()
            self.counter_expends += 1

            if curr_node.depth >= depth:
                #self.explored.remove(curr_node._hash)
                continue

            if self.game.is_final_state(curr_node.state):
                curr_node.solved = True
                return curr_node

            valid_moves = self.game.get_valid_moves(curr_node.state, curr_node.action)
            for move in valid_moves:
                new_state, actioned_tile = self.game.make_move(curr_node.state, move)
                new_state_hash = self.game.hash_board(new_state)
                if new_state_hash not in self.explored:
                    #self.explored.add(new_state_hash)
                    new_node = Node(new_state, parent=curr_node, action=move, actioned_tile=actioned_tile,
                                    depth=curr_node.depth+1, _hash=new_state_hash)
                    self.frontier.put(new_node)



class BFSTilesAgent(TilesAgent):
    """ agent state is the games board """
    def __init__(self, state: np.ndarray, game: TilesGame):
        super().__init__(name="BFS", state=state, game=game)
        self.frontier = queue.Queue()


    def solve(self):
        state_hash = self.game.hash_board(self.init_state)
        self.explored.add(state_hash)

        init_node = Node(self.init_state)
        self.frontier.put(init_node)
        while not self.frontier.empty():
            curr_node: Node = self.frontier.get()
            self.counter_expends += 1

            if self.game.is_final_state(curr_node.state):
                curr_node.solved = True
                return curr_node

            valid_moves = self.game.get_valid_moves(curr_node.state, curr_node.action)
            for move in valid_moves:
                new_state, actioned_tile = self.game.make_move(curr_node.state, move)
                new_state_hash = self.game.hash_board(new_state)
                if new_state_hash not in self.explored:
                    self.explored.add(new_state_hash)
                    new_node = Node(new_state, parent=curr_node, action=move, actioned_tile=actioned_tile)
                    self.frontier.put(new_node)




def main():
    user_input = "1 4 0 5 8 2 3 6 7"
    #user_input = "1 2 3 0 4 6 7 5 8"
    #
    game = TilesGame()
    board = game.init_board(user_input)
    bfs_agent = BFSTilesAgent(board, game)
    winning_node = bfs_agent.solve()

    if winning_node is not None:
        print(bfs_agent.name)
        print(bfs_agent.counter_expends)
        print(winning_node.get_path())
        print()

    game = TilesGame()
    board = game.init_board(user_input)
    iddfs_agent = IDDFSTilesAgent(board, game)


    winning_node = iddfs_agent.solve(15)

    if winning_node is not None:

        print(iddfs_agent.name)
        print(iddfs_agent.counter_expends)
        print(winning_node.get_path())
        print()

    else:
        print("no solution")




if __name__ == '__main__':
    main()