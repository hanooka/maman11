import sys
import queue
from collections import deque
import numpy as np
import heapq


def manhattan_distance(board1, board2):
    size = board1.shape[0]
    distance_matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            value = board1[i, j]
            pos_board1 = np.array(np.where(board1 == value)).flatten()
            pos_board2 = np.array(np.where(board2 == value)).flatten()
            distance_matrix[i, j] = np.abs(pos_board1[0] - pos_board2[0]) + np.abs(pos_board1[1] - pos_board2[1])
    return distance_matrix


def count_direct_reversal(board1, board2):
    """ All pairs of neighbors in the 8-tile problem. """
    indices = [((1, 1), (2, 1)), ((1, 0), (2, 0)), ((1, 0), (1, 1)), ((0, 0), (0, 1)),
               ((0, 2), (1, 2)), ((1, 1), (1, 2)), ((2, 1), (2, 2)), ((2, 0), (2, 1)),
               ((0, 1), (0, 2)), ((0, 1), (1, 1)), ((0, 0), (1, 0)), ((1, 2), (2, 2))]
    count = 0
    for ind1, ind2 in indices:
        if board1[ind1] == board2[ind2] and board2[ind1] == board1[ind2]:
            count += 1
    return count


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0  # case priority equal, first in is first out. counter track this

    def put(self, item, priority):
        entry = (priority, self.counter, item)
        heapq.heappush(self.heap, entry)
        self.counter += 1

    def get(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def empty(self):
        return len(self.heap) == 0


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


class Node:
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
            if np.all(board == winning_board):
                return True
        return False

    def is_solvable(self, _board):
        """ Check if given board is winnable (vs the winning boards).
        Theoretically, if there are even swaps, board is solvable.
        (looks like this works only for 2k X 2k board) """
        board = _board.copy()
        for winning_board in self.winning_boards:
            swaps = self.get_swaps_to_win(board, winning_board)
            if swaps % 2 == 0:
                return True
        return False

    def get_winning_boards(self):
        """ return list of winning positions """
        win1 = np.arange(0, 9).reshape((3, 3))
        win2 = np.concatenate((np.arange(1, 9), [0])).reshape((3, 3))
        return [win1]  # , win2]

    def hash_board(self, board):
        """ given board returns hashed value """
        _hash = np.sum(board * self.mults)
        return _hash

    @staticmethod
    def is_input_correct(input_str):
        numbers = input_str.split()
        len_9 = len(numbers) == 9
        no_repeats = len(set(numbers)) == len(numbers)
        all_digits_0_8 = all(0 <= int(num) <= 8 for num in numbers if num.isdigit())
        return len_9 and no_repeats and all_digits_0_8

    @staticmethod
    def init_board(user_input):
        """ init a board game from user string. """
        is_input_correct = TilesGame.is_input_correct(user_input)
        if is_input_correct:
            board = np.fromstring(user_input, dtype=np.int8, sep=' ')
            board = board.reshape((3, 3))
            return board
        else:
            raise ValueError(f"Input {user_input} is incorrect.")

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

        # board[empty_cell_loc], board[x, y] = board[x, y], board[empty_cell_loc]
        board[empty_cell_loc], board[x, y] = board[x, y].item(), board[empty_cell_loc].item()

        return board, moved_tile

    @staticmethod
    def get_swaps_to_win(board: np.ndarray, win_board: np.ndarray):
        """ returns the number of swaps in order to get from board to win_board
        mark on EVEN swaps which imply solution exists. """

        board_arr = board.reshape(9, )
        win_board_arr = win_board.reshape(9, )

        swaps = 0
        for ind, val in np.ndenumerate(win_board_arr):
            exch_idx = np.where(board_arr == val)[0]

            if board_arr[ind] != win_board_arr[ind]:
                board_arr[ind], board_arr[exch_idx] = board_arr[exch_idx].item(), board_arr[ind].item()
                swaps += 1

        return swaps

    @staticmethod
    def calculate_distance(board1, board2, dist_metric='l2'):
        if str(dist_metric).lower() == 'l2':
            """ fail implementation.. this count tiles instead of game rules.. """
            return np.sqrt(np.sum(np.power(board1 - board2, 2)))

        elif str(dist_metric).lower() == 'l1':
            """ lol what is this? this is not admissible :> """
            return np.sum(np.abs(board1 - board2))

        elif str(dist_metric).lower() == 'diff':
            return np.sum(np.array(board2 != board1).astype('int8'))

        elif str(dist_metric).lower() == 'manh':
            dist_mat = manhattan_distance(board1, board2)
            return np.sum(dist_mat)

        elif str(dist_metric).lower() == 'crazy_manh':
            """ not admissible :> """
            dist_mat = manhattan_distance(board1, board2)
            dist_mat = np.where(dist_mat >= 1, dist_mat * 3, dist_mat)
            return np.sum(dist_mat)

        elif str(dist_metric).lower() == 'impr_manh':
            """ This is a deal breaker! """
            manh_dist = np.sum(manhattan_distance(board1, board2))
            count = count_direct_reversal(board1, board2)
            return manh_dist + count * 2

        else:
            raise ValueError(f"{dist_metric} is not supported.")


class TilesAgent():
    def __init__(self, name, state: np.ndarray, game):
        self.name = name
        self.explored = set()
        self.counter_expends = 0
        self.init_state = state.copy()
        self.game = game


class ASTARTilesAgent(TilesAgent):
    """ agent state is the games board """

    def __init__(self, state: np.ndarray, game: TilesGame, alpha=0.5, dist_metric='l1'):
        super().__init__(name="A_STAR", state=state, game=game)
        # We use deque (which is 2 side queue. We push and pop from same side to simulate a stack)
        self.frontier = PriorityQueue()
        self.dist_metric = dist_metric
        self.alpha = alpha

    def get_correct_winning_board(self, init_state):
        """ Math properties of the problem. in order for a board to be solvable (getting from state 1 to 2,
        you need even numbers of `swaps`. where `swap` is exchange between two tiles, until we get from state 1
        to 2) """
        win_boards = self.game.get_winning_boards()
        for win_board in win_boards:
            swaps = self.game.get_swaps_to_win(win_board, init_state)
            if swaps % 2 == 0:
                return win_board
        return None

    def get_priority(self, state, accum_cost):
        """ This function represents f(n) = h(n) + g(n) `ThE HeUrIsTic` + `ThE AccumulateD CoSt`
        also added alpha and 1-alpha as weights, so we can fiddle around.
        for example if alpha = 0, we get a greedy-first-best-search. """
        winning_board = self.game.get_winning_boards()[0]
        priority = 2 * ((1 - self.alpha) * self.game.calculate_distance(state, winning_board, self.dist_metric)
                        + self.alpha * accum_cost)
        return priority

    def solve(self):
        state_hash = self.game.hash_board(self.init_state)
        self.explored.add(state_hash)

        init_node = Node(self.init_state, accum_cost=0)
        priority = self.get_priority(init_node.state, init_node.accum_cost)
        self.frontier.put(init_node, priority)
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
                    new_node = Node(new_state, parent=curr_node, action=move, actioned_tile=actioned_tile,
                                    accum_cost=curr_node.accum_cost + 1)
                    priority = self.get_priority(new_node.state, new_node.accum_cost)
                    self.frontier.put(new_node, priority)


class GBFSTilesAgent(TilesAgent):
    """ agent state is the games board """

    def __init__(self, state: np.ndarray, game: TilesGame, dist_metric='l1'):
        super().__init__(name="GBFS", state=state, game=game)
        # We use deque (which is 2 side queue. We push and pop from same side to simulate a stack)
        self.frontier = PriorityQueue()
        self.dist_metric = 'l1'

    def get_correct_winning_board(self, init_state):
        """ Math properties of the problem. in order for a board to be solvable (getting from state 1 to 2,
        you need even numbers of `swaps`. where `swap` is exchange between two tiles, until we get from state 1
        to 2) """
        win_boards = self.game.get_winning_boards()
        for win_board in win_boards:
            swaps = self.game.get_swaps_to_win(win_board, init_state)
            if swaps % 2 == 0:
                return win_board
        return None

    def get_priority(self, state):
        """ This function represents f(n) = h(n). `ThE HeUrIsTic` """
        winning_board = self.game.get_winning_boards()[0]
        priority = self.game.calculate_distance(state, winning_board, self.dist_metric)
        return priority

    def solve(self):
        state_hash = self.game.hash_board(self.init_state)
        self.explored.add(state_hash)

        init_node = Node(self.init_state)
        priority = self.get_priority(init_node.state)
        self.frontier.put(init_node, priority)
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
                    priority = self.get_priority(new_node.state)
                    self.frontier.put(new_node, priority)


class IDDFSTilesAgent(TilesAgent):
    """ agent state is the games board """

    def __init__(self, state: np.ndarray, game: TilesGame):
        super().__init__(name="IDDFS", state=state, game=game)
        # We use deque (which is 2 side queue. We push and pop from same side to simulate a stack)
        self.frontier = Stack()

    def solve(self, max_depth=30):
        for i in range(1, max_depth + 1):
            solution = self.dfs_traverse(i)
            if solution is not None:
                return solution
        return None

    def dfs_traverse(self, depth=1):
        self.explored = set()

        state_hash = self.game.hash_board(self.init_state)
        # self.explored.add(state_hash)

        init_node = Node(self.init_state, depth=0, _hash=state_hash)
        self.frontier.put(init_node)

        while not self.frontier.empty():
            curr_node: Node = self.frontier.get()
            self.counter_expends += 1

            if curr_node.depth >= depth:
                # self.explored.remove(curr_node._hash)
                continue

            if self.game.is_final_state(curr_node.state):
                curr_node.solved = True
                return curr_node

            valid_moves = self.game.get_valid_moves(curr_node.state, curr_node.action)
            for move in valid_moves:
                new_state, actioned_tile = self.game.make_move(curr_node.state, move)
                new_state_hash = self.game.hash_board(new_state)
                if new_state_hash not in self.explored:
                    # self.explored.add(new_state_hash)
                    new_node = Node(new_state, parent=curr_node, action=move, actioned_tile=actioned_tile,
                                    depth=curr_node.depth + 1, _hash=new_state_hash)
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
        return None


"""
Input examples
    user_input = "1 4 0 5 8 2 3 6 7"
    user_input = "1 2 4 5 3 6 7 8 0"
    user_input = "1 3 5 7 2 4 6 8 0"
    user_input = "8 2 1 5 3 7 4 6 0"
    user_input = "8 0 5 4 3 6 7 1 2"
    user_input = "8 2 1 5 3 7 0 4 6"
    user_input = "4 1 8 0 2 5 3 7 6"
    user_input = "1 2 4 3 8 7 5 6 0"
    user_input = "1 4 2 3 7 8 5 6 0"
"""


def main(argv):
    if len(argv) > 1:
        user_input = " ".join(argv[1:])
    else:
        raise ValueError("No argument were given.")

    game = TilesGame()
    board = game.init_board(user_input)
    dist_metric = 'impr_manh'
    alpha = 0.5

    bfs_agent = BFSTilesAgent(board, game)
    iddfs_agent = IDDFSTilesAgent(board, game)
    gbfs_agent = GBFSTilesAgent(board, game, dist_metric=dist_metric)
    a_star_agent = ASTARTilesAgent(board, game, alpha=alpha, dist_metric=dist_metric)

    agents = [bfs_agent, iddfs_agent, gbfs_agent, a_star_agent]

    for agent in agents:
        winning_node = agent.solve()
        if winning_node is not None:
            print(agent.name)
            print(agent.counter_expends)
            print(winning_node.get_path())
            print()
        else:
            print(f"{agent.name} failed to solve")


if __name__ == '__main__':
    main(sys.argv)
