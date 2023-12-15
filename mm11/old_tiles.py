import queue
import random
import numpy as np


def get_winning_boards():
    """ return list of winning positions """
    win1 = np.arange(0, 9).reshape((3, 3))
    win2 = np.concatenate((np.arange(1, 9), [0])).reshape((3, 3))
    return [win1, win2]


def check_win(board: np.ndarray, winning_board: np.ndarray):
    """ returns True if boards are equal """

    return np.all(board == winning_board)


def init_game_board_from_user_input(user_input):
    """ init a board game from user string. """
    board = np.fromstring(user_input, dtype=np.int8, sep=' ')
    board = board.reshape((3, 3))
    return board


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


def make_move(board, move):
    """ only accept valid moves. validity responsibility is on the user """
    empty_cell_loc = np.where(board == 0)
    x, y = get_swapping_indices(empty_cell_loc, move)

    board[empty_cell_loc], board[x, y] = board[x, y], board[empty_cell_loc]
    return board


def hash_board(board):
    """ given board returns hashed value """
    arr = np.repeat(10, 9)
    pwr = np.arange(0, 9)
    mults = np.power(arr, pwr).reshape((3, 3))
    _hash = np.sum(board * mults)
    return _hash


def solve_game(board):
    states = set()
    states_moves_queue = queue.Queue()
    winning_boards = get_winning_boards()
    # Case we got a winning board first move! :>
    for win_board in winning_boards:
        if check_win(board, win_board):
            return True, 0

    move = ''
    counter = 0
    while True:
        counter += 1
        valid_moves = get_valid_moves(board, move)
        _hash = hash_board(board)

        for move in valid_moves:
            if (move, _hash) in states:
                continue
            else:
                states_moves_queue.put((move, board.copy()))
                states.add((move, _hash))

        move, board = states_moves_queue.get()

        print(board)
        print(move)

        board = make_move(board, move)
        for win_board in winning_boards:
            if check_win(board, win_board):
                print(board)
                return True, counter


def testing_funcs():
    user_input = "1 4 0 5 8 2 3 6 7"

    board = init_game_board_from_user_input(user_input)
    print(board)
    valid_moves = get_valid_moves(board)
    print(valid_moves)
    move = random.sample(valid_moves, 1)[0]
    print(move)
    board = make_move(board, move)
    print(board)


def main():
    user_input = "1 4 0 5 8 2 3 6 7"
    user_input = "1 2 3 4 5 6 7 0 8"
    board = init_game_board_from_user_input(user_input)
    did_solve, counter = solve_game(board)
    print(did_solve)
    print(f'moves it took {counter}')


if __name__ == '__main__':
    main()

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
