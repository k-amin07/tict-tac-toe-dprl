import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import copy
import math
import random
import os


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
        for layer in range(4):
            print(f"Layer {layer + 1}:")
            print("┌───┬───┬───┬───┐")
            for i, row in enumerate(self.board[layer]):
                print("│ " + " │ ".join(row) + " │")
                if i < 3:
                    print("├───┼───┼───┼───┤")
            print("└───┴───┴───┴───┘")
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

        return observation, reward, terminated, player_turn


def convert_and_flatten_state(state):
    flattened_state = []
    for layer in state:
        for row in layer:
            for cell in row:
                if cell == 'X':
                    flattened_state.append(1)
                elif cell == 'O':
                    flattened_state.append(-1)
                else:
                    flattened_state.append(0)  # Represent empty cells as 0
    
    return np.array(flattened_state)

class TicTacToePlayer:
    def __init__(self, player):
        self.state_size = 64
        self.action_size = 64
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_decay_rate = 0.995
        self.epilon_lower_bound = 0.001
        self.history = []
        self.player = player
        self.model = self.build_model()
    
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.state_size, input_dim = self.state_size, activation='relu'),
            tf.keras.layers.Dense(self.state_size, activation='relu'),
            tf.keras.layers.Dense(self.state_size, activation='relu'),
            tf.keras.layers.Dense(self.state_size, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1-self.gamma))
        if(os.path.isfile('weights_player_{}.h5'.format(self.player))):
            model.load_weights('weights_player_{}.h5'.format(self.player))
        return model

    def select_action(self, state, action_space):
        if(np.random.rand() <= self.epsilon):
            return np.random.choice(action_space)
       
        state_batch = np.expand_dims(convert_and_flatten_state(state), axis=0)
        q_values = self.model.predict(state_batch)[0]
        if(self.player == "O"):
            mask = np.ones_like(q_values) * -np.inf
            for action in action_space:
                mask[action] = 0
            masked_q_values = q_values + mask
            selected_action = np.argmax(masked_q_values)
            return selected_action
        else:
            mask = np.ones_like(q_values) * np.inf
            for action in action_space:
                mask[action] = 0
            masked_q_values = q_values + mask
            selected_action = np.argmin(masked_q_values)
            return selected_action
        # q_value = np.argmax(self.model.predict(np.expand_dims(convert_and_flatten_state(state), axis=0))[0])
        # # q_values = self.model.predict(convert_and_flatten_state(state))
        # # return np.argmax(q_values[0])
        # return q_value
    
    def update_history(self,state,action,reward,next_state, done):
        self.history.append((state,action,reward,next_state, done))

    def train(self):
        batch_size = 32
        ## need some history to train. choose random actions until then
        ## will probably have to adjust this parameter later
        if(len (self.history) < batch_size):
            return
        # batch = 
        states, actions, rewards, next_states, dones = zip(*random.sample(self.history, batch_size))
        # states, actions, rewards, next_states, dones = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4]

        states = np.array(list(map(convert_and_flatten_state, np.array(states).reshape((-1, 64)))))
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(list(map(convert_and_flatten_state,np.array(next_states).reshape((-1,64)))))
        dones = np.array(dones)
        # states = states.reshape((-1, 64))
        # next_states = next_states.reshape((-1, 64))

        targets = self.model.predict(states)
        if(self.player == "O"):
            targets[np.arange(batch_size), actions.astype(int)] = rewards + self.gamma * np.max(self.model.predict(next_states), axis=1) * (1 - dones)
        else:
            targets[np.arange(batch_size), actions.astype(int)] = rewards + self.gamma * np.min(self.model.predict(next_states), axis=1) * (1 - dones)

        self.model.fit(states, targets, epochs=50, verbose=0)

        if self.epsilon > self.epilon_lower_bound:
            self.epsilon *= self.epsilon_decay_rate

    def save_model(self):
        self.model.save_weights('weights_player_{}.h5'.format(self.player))

player_X = TicTacToePlayer(player='X')
player_O = TicTacToePlayer(player='O')

def policy_player1(observation, action_space):
    position = player_X.select_action(observation, action_space)
    # position = random.choice(action_space)

    return position


# Initialized as a random policy for player 2
def policy_player2(observation, action_space):
    position = player_O.select_action(observation, action_space)
    # position = random.choice(action_space)

    return position

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
        prev_state = copy.deepcopy(observation)

        if player_turn == "X":
            action = policy_player1(observation, action_space)
        else:
            action = policy_player2(observation, action_space)

        observation, reward, terminated, player_turn = env.step(action)

        ## player has switched at this point but we need to update history for the player that performed the action.
        ## So the player that gets updated is the one opposite to player turn
        if player_turn == "O":
            player_X.update_history(prev_state,action,reward,observation, terminated)
            player_X.train()
        else:
            player_O.update_history(prev_state,action,reward,observation, terminated)
            player_O.train()
    return reward  # This is the player who won

x,o,draw = 0,0,0
for i in range(1000):
    reward = play_one_game(policy_player1, policy_player2, render_mode="computer")
    if(reward == -1):
        x += 1
    elif(reward == 1):
        o += 1
    else:
        draw += 1

print("x: {}\no: {}\nd: {}".format(x/1000,o/1000,draw/1000))

player_X.save_model()
player_O.save_model()