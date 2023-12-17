import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import math
import random
from tqdm import tqdm

class TicTacToe:
    def __init__(self):
        self.board = [[" " for _ in range(4)] for _ in range(4)]
        self.current_player = -1


    def check_draw(self):
        for row in self.board:
            if " " in row:
                return False
        return True

    def print_board(self):
        # Prints a GUI-like representation of the board
        print("┌───┬───┬───┬───┐")
        for i, row in enumerate(self.board):
            print("│ " + " │ ".join(row) + " │")
            if i < 3:
                print("├───┼───┼───┼───┤")
        print("└───┴───┴───┴───┘")

    def check_winner(self, player):
        for row in self.board:
            if all([cell == player for cell in row]):
                return True
        for col in range(4):
            if all([self.board[row][col] == player for row in range(4)]):
                return True
        if all([self.board[i][i] == player for i in range(4)]) or all(
            [self.board[i][3 - i] == player for i in range(4)]
        ):
            return True
        return False

    def step(self, state):
        row = int(state / 4)
        col = int(state % 4)

        current_player_symbol = " "
        if self.current_player == -1:
            current_player_symbol = "X"
        else:
            current_player_symbol = "O"

        if self.board[row][col] == " ":
            self.board[row][col] = current_player_symbol

        if self.check_winner("O"):
            return self.board, self.current_player, True, -1
        elif self.check_winner("X"):
            return self.board, self.current_player, True, 1
        elif self.check_draw():
            return self.board, self.current_player, True, 0

        self.current_player *= -1

        return self.board, self.current_player, False, 0

with open("Q_strategys.pkl", "rb") as f:
    strategies = pkl.load(f)

# Initialized as a random policy for player 1

def checkrots(board, memo = None, path = None):
    if memo is None:
        memo = {}
    if path is None:
        path = tuple()

    key = "".join(map(str, board.flatten()))
    if(key in memo):
        return
    
    memo[key] = path
    # 0 = rot90 (counter clockwise)
    # 1 = left-right flip
    # 2 = up-down flip
    checkrots(np.rot90(board), memo, path + (0,))
    checkrots(np.fliplr(board), memo, path + (1,))
    checkrots(np.flipud(board), memo, path + (2,))

def get_all_rotations(state):
    all_rotations = {}
    temp_state = state.copy().reshape(4,4)
    checkrots(temp_state,all_rotations)
    return all_rotations

def policy_player1(board):

    # possible_actions = []

    # for i in range(4):
    #     for j in range(4):
    #         if board[i][j] == " ":
    #             possible_actions.append(i*4 + j)


    # return random.choice(possible_actions)
    board = np.array(board).flatten()
    player = "X" if sum(board == "X") == sum(board == "O") else "O"
    strategy = strategies[f"{player}_strategy"]
    
    all_rotations = get_all_rotations(board)
    # s_key = "".join(board)
    flag = True
    for rotation in all_rotations:
        if(rotation in strategy):
            # return rotation
            q_values = strategy[rotation]
            flag = False
            break
    if(flag):
        q_values = np.zeros(16).astype(float)
    # if s_key in strategy:
    #     q_values = strategy[s_key]
    # else:
    #     q_values = np.zeros(16).astype(float)
    
    available_actions = np.where(board == " ")[0]
    legal_q_vals = q_values[available_actions]
    max_q = legal_q_vals.max()
    idxs = np.where(legal_q_vals == max_q)[0]
    return available_actions[np.random.choice(idxs)]


# Initialized as a random policy for player 2
def policy_player2(board):

    possible_actions = []
    board = np.array(board).flatten()
    player = "X" if sum(board == "X") == sum(board == "O") else "O"
    
    strategy = strategies[f"{player}_strategy"]

    all_rotations = get_all_rotations(board)
    # s_key = "".join(board)
    q_values = None
    flag = True
    rotation_path = None
    for rotation in all_rotations.keys():
        if(rotation in strategy):
            # return rotation
            q_values = strategy[rotation]
            rotation_path = all_rotations[rotation]
            flag = False
            break
    if(flag):
        q_values = np.zeros(16).astype(float)

    # s_key = "".join(board)
    # if s_key in strategy:
    #     q_values = strategy[s_key]
    # else:
    #     q_values = np.zeros(16).astype(float)
    
    available_actions = np.where(board == " ")[0]
    legal_q_vals = q_values[available_actions]
    max_q = legal_q_vals.max()
    idxs = np.where(legal_q_vals == max_q)[0]
    return available_actions[np.random.choice(idxs)]
    


    #return random.choice(possible_actions)

def play_one_game(policy_player1, policy_player2):
    tictactoe = TicTacToe()


    terminated = 0
    board = [[" " for _ in range(4)] for _ in range(4)]

    for i in range(8):
        for turn in [-1, 1]:
            action = 0
            if turn == -1:
                action = policy_player1(board)
            else:
                action = policy_player2(board)

            board, player, terminated, reward = tictactoe.step(action)

            # Uncomment this if you want to see the board
            #tictactoe.print_board()

            if terminated:
                break

    return -1*reward # This is the player who won

def run_alternating_games(games=10):
    results = []
    for i in tqdm(range(games)):
        for j in range(2):
            if j==0:
                winner = play_one_game(policy_player1, policy_player2)

                match winner:
                    case -1:
                        results.append(1)
                    case 1:
                        results.append(2)
                    case 0:
                        results.append(0)

            if j==1:
                winner = play_one_game(policy_player2, policy_player1)

                match winner:
                    case -1:
                        results.append(2)
                    case 1:
                        results.append(1)
                    case 0:
                        results.append(0)


    return results

class QLearningAgent:
    def __init__(self, sign ,gamma=0.3, alpha=0.95):
        self.sign = sign
        self.gamma = gamma # also called learning rate
        self.alpha = alpha # also called discount factor
        self.Q_dict = {}

    def checkrots(self,board, memo = []):
        # print(board)
        key = "".join(map(str, board.flatten()))
        if(key in memo):
            return
        memo.append(key)
        self.checkrots(np.rot90(board),memo)
        self.checkrots(np.fliplr(board),memo)
        self.checkrots(np.flipud(board),memo)
        
    def create_key(self, state):
        return "".join(state)
    
    def get_all_rotations(self,state):
        all_rotations = []
        temp_state = state.copy().reshape(4,4)
        self.checkrots(temp_state,all_rotations)
        return all_rotations
    
    def get_stored_state(self, all_rotations):
        for rotation in all_rotations:
            if(rotation in self.Q_dict):
                return rotation
        
    def get_q_values(self, state):
        all_rotations = self.get_all_rotations(state)

        # this intersection might become costly over time. Since we can have at most 8 rotations, it might be worthwhile to check in a loop and exit early if needed
        # stored_state = set(all_rotations).intersection(set(self.Q_dict.keys()))
        stored_state = self.get_stored_state(all_rotations)
        
        if(stored_state):
            # stored_state = stored_state.pop()
            return self.Q_dict[stored_state]
        
        # s_key = self.create_key(state)
        # if s_key not in self.Q_dict:
        self.Q_dict[all_rotations[0]] = np.ones(16).astype(np.float16) * 0.5
        return self.Q_dict[all_rotations[0]]

    def choose_action(self, state, available_actions):
        q_values = self.get_q_values(state)
        illegal_state = np.array([i for i in range(16) if i not in available_actions])
        if len(illegal_state):
            q_values[illegal_state] = -1.0
        legal_q_vals = q_values[available_actions]
        
        max_q = legal_q_vals.max()
        idxs = np.where(legal_q_vals == max_q)[0]
        return available_actions[np.random.choice(idxs)]

    def update_q_values(self, state_action_history: list, reward):
        
        # updating the reward for the last state
        state, action = state_action_history[-1]
        all_rotations = self.get_all_rotations(state)
        # stored_state = set(all_rotations).intersection(set(self.Q_dict.keys())).pop()
        stored_state = self.get_stored_state(all_rotations)
        # s_key = self.create_key(state)
        self.Q_dict[stored_state][action] = reward

        reward = self.get_q_values(state).max()

        # iterating from last state and action to the first
        # updating the Q matrix based on discount factor
        for state, action in reversed(state_action_history[:-1]):
            all_rotations = self.get_all_rotations(state)
            # stored_state = set(all_rotations).intersection(set(self.Q_dict.keys())).pop()
            stored_state = self.get_stored_state(all_rotations)
            # s_key = self.create_key(state)
            self.Q_dict[stored_state][action] = ( (1 - self.gamma) * self.Q_dict[stored_state][action]) + (self.gamma*self.alpha*reward)
            reward = self.get_q_values(state).max()

# results = run_alternating_games(1000)
# print("Draws: ", results.count(0))
# print("Player 1 Wins:", results.count(1))
# print("Player 2 Wins:", results.count(2))

player1 = QLearningAgent(-1) # playing as X
player2 = QLearningAgent(1) # playing as O
exploration_ratio = 0.8
tictactoe = TicTacToe()

# Simulate a number of games
for i in tqdm(range(1000000)):
    # Initialize the game and the state
    tictactoe = TicTacToe()
    board = np.array([" "]*16)
    # to keep track of history to make the update
    state_action_history1 = [] # for player 1
    state_action_history2 = [] # for player 2
    while True:
        # The agent chooses an action
        # temp_board = board.copy().reshape(4,4)
        available_actions = np.where(np.array(board) == " ")[0]
        
        if np.random.rand() < exploration_ratio:
            action = np.random.choice(available_actions)
            player1.get_q_values(board)
        else:
            action = player1.choose_action(board, available_actions)

        # row = int(action / 4)
        # col = int(action % 4)
        # temp_board[row][col] = action
        # The agent performs the action and gets the reward
        state_action_history1.append([board, action])
        board, player, terminated, reward = tictactoe.step(action)
        # unflattened_board = np.copy(board)
        board = np.array(board).flatten()
        
        if not terminated:
            # The agent updates the Q-value of the previous state-action pair
            next_available_actions =  np.where(np.array(board) == " ")[0]
            if np.random.rand() < exploration_ratio:
                action = np.random.choice(next_available_actions)
                player2.get_q_values(board)
            else:
                action = player2.choose_action(board, next_available_actions)

            # The agent performs the action and gets the reward
            state_action_history2.append([board, action])
            board, player, terminated, reward = tictactoe.step(action)
            board = np.array(board).flatten()
            
        if terminated:
            break
    
    # When we reach here we will have final reward
    # -1 = O won, player will be 1
    # 1 = X won, player will be -1
    #if reward != 0:
    #    print(reward)
    player1.update_q_values(state_action_history=state_action_history1, reward=reward)
    player2.update_q_values(state_action_history=state_action_history2, reward=reward*-1)

Q_strategies = {
    "X_strategy": player1.Q_dict,
    "O_strategy": player2.Q_dict
}
with open("Q_strategys.pkl", "wb") as f:
    pkl.dump(Q_strategies, f)