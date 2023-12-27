
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import math
import random
import os
from tqdm import tqdm
from copy import deepcopy
class WinCondition:
    def __init__(self):
        self.win_player = "X"

    def check_win(self):


        layers = self.check_all_layers()
        z_check = self.check_all_z()
        diag = self.check_cross_diagonals() or self.check_all_vertical_diagonals()

        return diag or z_check or layers



    def check_all_layers(self):
        return any(self.check_layer(z) for z in range(4))

    def check_all_z(self):
        return any(self.check_z(x, y) for x in range(4) for y in range(4))

    def check_all_vertical_diagonals(self):

        xdiag = any(self.check_vertical_xdiagonals(x) for x in range(4))
        ydiag = any(self.check_vertical_ydiagonals(y) for y in range(4))

        return xdiag or ydiag


    def check_layer(self, z):
        x_checker = any(self.check_x(y, z) for y in range(4))
        y_checker = any(self.check_y(x, z) for x in range(4))
        diag_checker = self.check_diagonals(z)

        return x_checker or y_checker or diag_checker


    def check_cross_diagonals(self):
        first = all(self.board[c][c][c] == self.win_player for c in range(4))
        second = all(self.board[c][3-c][3-c] == self.win_player for c in range(4))
        third = all(self.board[c][c][3-c] == self.win_player for c in range(4))
        fourth = all(self.board[c][3-c][c] == self.win_player for c in range(4))



        return first or second or third or fourth

    def check_x(self, y, z):
        return all(self.board[x][y][z] == self.win_player for x in range(4))


    def check_y(self, x, z):
        return all(self.board[x][y][z] == self.win_player for y in range(4))



    def check_diagonals(self, z):
        if all(self.board[diag][diag][z] == self.win_player for diag in range(4)):
            return True

        if all(self.board[3-reverse_diag][reverse_diag][z] == self.win_player for reverse_diag in range(4)):
            return True

        return False

    def check_z(self, x, y):
        return all(self.board[x][y][z] == self.win_player for z in range(4))



    def check_vertical_xdiagonals(self, x):
        if all(self.board[x][diag][diag] == self.win_player for diag in range(4)):
            return True

        if all(self.board[x][reverse_diag][3-reverse_diag] == self.win_player for reverse_diag in range(4)):
            return True

        return False


    def check_vertical_ydiagonals(self, y):
        if all(self.board[diag][y][diag] == self.win_player for diag in range(4)):
            return True

        if all(self.board[reverse_diag][y][3-reverse_diag] == self.win_player for reverse_diag in range(4)):
            return True

        return False

class TicTacToe4x4x4(WinCondition):
    def __init__(self, render_mode="computer"):
        # 3D board: 4 layers of 4x4 grids
        super().__init__()
        self.board = [[[" " for _ in range(4)] for _ in range(4)] for _ in range(4)]
        self.current_player = "X"
        self.players = ["X", "O"]
        self.terminated = False
        self.winner = " "
        self.render_mode = render_mode
        self.fig = None

    def check_draw(self):
        # Check for any empty space in the entire 3D board
        return not any(
            " " in self.board[x][y][z] for x in range(4) for y in range(4) for z in range(4)
        )

    def get_action_space(self):
        action_space = []
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if self.board[x][y][z] == " ":
                        action_space.append(self.get_position(x, y, z))

        action_space.sort()
        return action_space

    def print_board(self):
        # Prints each layer of the 3D board
        num = 0
        for layer in range(4):
            print(f"Layer {layer + 1}:")
            print("┌───────────────┬───────────────┬───────────────┬───────────────┐")
            for i, row in enumerate(self.board[layer]):
                # temp_row = []
                # for _ in row:
                #     coordinates = get_coordinates(num)
                #     temp_row.append(str(coordinates) + ":" + str(num) + str(_))
                #     num += 1
                print("│ " + " │ ".join(row) + "  │")
                if i < 3:
                    print("├───────────────┼───────────────┼───────────────┼───────────────┤")
            print("└───────────────┴───────────────┴───────────────┴───────────────┘")
            if layer < 3:
                print()

    def create_visualization(self):
        if(not self.fig):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")

            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if self.board[x][y][z] == "X":
                            self.ax.scatter(x, y, z, color="r", marker="o")
                        if self.board[x][y][z] == "O":
                            self.ax.scatter(x, y, z, color="b", marker="o")

            cmin = 0
            cmax = 3

            self.ax.set_xticks(np.arange(cmin, cmax + 1, 1))
            self.ax.set_yticks(np.arange(cmin, cmax + 1, 1))
            self.ax.set_zticks(np.arange(cmin, cmax + 1, 1))

            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")

            if self.winner != " ":
                plt.title(f"Player {self.winner} Won!")
            else:
                plt.title(f"Player {self.current_player} Turn" )

            plt.show()
        else:
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if self.board[x][y][z] == "X":
                            self.ax.scatter(x, y, z, color="r", marker="o")
                        if self.board[x][y][z] == "O":
                            self.ax.scatter(x, y, z, color="b", marker="o")

    def change_player(self):
        if self.current_player == "X":
            self.current_player = "O"
        else:
            self.current_player = "X"

    def get_coordinates(self, position):
        x = int((position % 16) % 4)
        y = int((position % 16) / 4)
        z = int(position / 16)

        return x, y, z

    def get_position(self, x, y, z):
        return z * 16 + y * 4 + x

    def update_board(self, x, y, z):
        reward = 0

        if self.terminated:
            return self.board, reward, self.terminated, self.current_player

        if self.board[x][y][z] == " ":
            self.board[x][y][z] = self.current_player
        else:
            self.terminated = True
            return self.board, reward, self.terminated, self.current_player


        self.win_player = self.current_player
        win = self.check_win()
        draw = self.check_draw()

        self.terminated = win or draw

        if win:
            if self.current_player == "X":
                reward = -1
                self.winner = "X"
            else:
                reward = 1
                self.winner = "O"
        elif draw:
            reward = 0

        self.change_player()

        return self.board, reward, self.terminated, self.current_player

    def step_coordinates(self, x, y, z):
        # Output: Observation, reward, terminated, player_turn
        observation, reward, terminated, player_turn = self.update_board(x, y, z)

        if self.render_mode == "human":
            self.create_visualization()

        return observation, reward, terminated, player_turn

    def step(self, position):

        # Output: Observation, reward, terminated, player_turn
        x, y, z = self.get_coordinates(position)
        observation, reward, terminated, player_turn = self.update_board(x, y, z)

        if self.render_mode == "human":
            self.create_visualization()
        else:
            self.print_board()
        return observation, reward, terminated, player_turn

def convert_and_flatten_state(state):
    flattened_state = []
    for layer in state:
        for row in layer:
            for cell in row:
                if cell == 'X':
                    flattened_state.append(-1)
                elif cell == 'O':
                    flattened_state.append(1)
                else:
                    flattened_state.append(0)  # Represent empty cells as 0
    
    return np.array(flattened_state)

class TicTacToePlayer:
    def __init__(self, player):
        self.state_size = 64
        self.action_size = 64
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.9995
        self.epilon_lower_bound = 0.0001
        self.history = []
        self.player = player
        self.model = self.build_model()
    
    def build_model(self):
        if(os.path.isfile('model_player_{}.keras'.format(self.player))):
            model = tf.keras.models.load_model('model_player_{}.keras'.format(self.player))
            return model
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, input_dim = self.state_size),
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer='l2'),
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer='l2'),
            tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer='l2'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        if(os.path.isfile('weights_player_{}.h5'.format(self.player))):
            model.load_weights('weights_player_{}.h5'.format(self.player))
        else:
            raise Exception("Model weights not found, run ttt_3d.py to train model first")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5,
            decay_steps=10000,
            decay_rate=0.99)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = lr_schedule, clipvalue=0.5))
        return model

    def select_action(self, state, action_space):
       
        state_batch = np.expand_dims(convert_and_flatten_state(state), axis=0)
        q_values = self.model.predict(state_batch)[0]
        if(self.player == "O"):
            mask = np.ones_like(q_values) * -np.inf
            for action in action_space:
                mask[action] = 0
            masked_q_values = q_values + mask
            # masked_q_values = np.abs(masked_q_values)
            selected_action = np.argmax(masked_q_values)
            
            # Check Self win
            win_checker = WinCondition()
            win_checker.win_player = self.player
            for a in action_space:
                tmp_board = deepcopy(state)
                #tmp_board[x][y][z] = self.player
                x,y,z = get_coordinates(a)
                tmp_board[x][y][z] = self.player
                win_checker.board = tmp_board
                if win_checker.check_win():
                    selected_action = a
                    return selected_action

            # Check other win
            win_checker = WinCondition()
            win_checker.win_player = 'X' if self.player=='O' else "O"
            for a in action_space:
                tmp_board = deepcopy(state)
                x,y,z = get_coordinates(a)
                tmp_board[x][y][z] = 'X' if self.player=='O' else "O"
                win_checker.board = tmp_board
                if win_checker.check_win():
                    selected_action = a
                    return selected_action
               
            return selected_action
        else:
            mask = np.ones_like(q_values) * -np.inf
            for action in action_space:
                mask[action] = 0
            masked_q_values = q_values + mask
            selected_action = np.argmax(masked_q_values)

            # Check Self win
            win_checker = WinCondition()
            win_checker.win_player = self.player
            for a in action_space:
                tmp_board = deepcopy(state)
                #tmp_board[x][y][z] = self.player
                x,y,z = get_coordinates(a)
                tmp_board[x][y][z] = self.player
                win_checker.board = tmp_board
                if win_checker.check_win():
                    selected_action = a
                    return selected_action

            # Check other win
            win_checker = WinCondition()
            win_checker.win_player = 'X' if self.player=='O' else "O"
            for a in action_space:
                tmp_board = deepcopy(state)
                x,y,z = get_coordinates(a)
                
                tmp_board[x][y][z] = 'X' if self.player=='O' else "O"
                win_checker.board = tmp_board
                if win_checker.check_win():
                    selected_action = a
                    return selected_action
                
            return selected_action
    
player_X = TicTacToePlayer(player='X')
player_O = TicTacToePlayer(player='O')


def get_coordinates(position):
    x = int((position % 16) % 4)
    y = int((position % 16) / 4)
    z = int(position / 16)
    return x, y, z

def get_position(x, y, z):
    return z * 16 + y * 4 + x

def policy_player1(observation, action_space):
    # print("Input action")
    # action = int(input())
    # print("\n\n\n{} {}\n\n".format(action,get_coordinates(action)))
    # return action
    action = player_X.select_action(observation, action_space)
    print("X action: ")
    print(action, get_coordinates(action))
    print("\n")
    return action

def policy_player2(observation, action_space):
    # for action in action_space:
    #     print(action,get_coordinates(action))
    print("Input action")
    action = int(input())
    print("\n\n\n{} {}\n\n".format(action,get_coordinates(action)))
    return action
    # action = player_O.select_action(observation, action_space)
    # print("O action: ")
    # print(action, get_coordinates(action))
    # print("\n")
    # return action

def play_one_game(policy_player1, policy_player2, render_mode="computer"):
    env = TicTacToe4x4x4(render_mode)
    terminated = 0
    observation = [[[" " for _ in range(4)] for _ in range(4)] for _ in range(4)]
    reward = 0
    player_turn = "X"
    i = 0

    while not terminated:
        i += 1
        action_space = env.get_action_space()

        if player_turn == "X":
            action = policy_player1(observation, action_space)
        else:
            action = policy_player2(observation, action_space)
        
        observation, reward, terminated, player_turn = env.step(action)
        # for obs in observation:
        #     print(obs)
        # print({reward, terminated, player_turn})
    return reward

x,o,draw = 0,0,0
reward = play_one_game(policy_player1, policy_player2, render_mode="computer")
print(reward)
# num_steps = 10
# for i in tqdm(range(num_steps)):
#     reward = play_one_game(policy_player1, policy_player2, render_mode="computer")
#     if(reward == -1):
#         x += 1
#     elif(reward == 1):
#         o += 1
#     else:
#         draw += 1

# print("x: {}\no: {}\nd: {}".format(x/num_steps,o/num_steps,draw/num_steps))